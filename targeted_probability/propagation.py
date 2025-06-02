import torch
from typing import Union
from common.propagate_mixture import propagate
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangularPartition, AvoidHyperRectangle, ReachHyperRectangle
from grid.utils import get_shell_loc, uniform_grid
from plotting.plotting import plot_partition
from targeted_probability.bounds import compute_targeted_bound, TargetedBounds, get_objective_targeted

def propagate_mixture_targeted_bounds(f: Dynamics,
                                      initial_distribution: Union[Gaussian, GaussianMixture],
                                      noise_distribution: Gaussian,
                                      shell: torch.Tensor,
                                      avoid_sets: AvoidHyperRectangle,
                                      reach_sets: ReachHyperRectangle,
                                      grid_size: int = 10,
                                      prediction_horizon: int = 2,
                                      num_samples: int = 1000,
                                      plot: bool = True,
                                      **kwargs):

    # Parameters
    num_avoid_sets = avoid_sets.lower.shape[0]
    num_reach_sets = reach_sets.lower.shape[0]

    # Initialize coarse partition
    loc_shell = get_shell_loc(shell)
    inner_partition = uniform_grid(shell[0], shell[1], grid_size)
    partition = HyperRectangularPartition(inner_partition=inner_partition,
                                          avoid_sets=avoid_sets,
                                          reach_sets=reach_sets,
                                          loc_shell=loc_shell,
                                          shell=shell)

    # Initialize mixture
    mixture_distribution = initial_distribution
    mixtures = [mixture_distribution]

    # Refine initial partition
    objective = get_objective_targeted(mixture=mixture_distribution)
    partition = partition.refine(objective=objective,
                                 contributions=mixture_distribution.compute_probabilities(partition),
                                 avoid_sets=avoid_sets,
                                 reach_sets=reach_sets,
                                 target=0.01,
                                 max_regions=5000)
    if plot:
        plot_partition(partition)

    # Compute mixture weights
    probs = mixture_distribution.compute_probabilities(partition)

    # Initialize bounds
    bounds_partition = TargetedBounds(torch.zeros(partition.lower.shape[0]), torch.zeros(partition.lower.shape[0]))
    bounds_avoid_sets = TargetedBounds(torch.zeros(num_avoid_sets), torch.zeros(num_avoid_sets))
    bounds_reach_sets = TargetedBounds(torch.zeros(num_reach_sets), torch.zeros(num_reach_sets))

    # Bound trajectories
    aps_avoid = [mixture_distribution.compute_probabilities(avoid_sets).sum()]
    lbs_avoid, ubs_avoid = [bounds_avoid_sets.lb_delta.sum()], [bounds_avoid_sets.ub_delta.sum()]

    aps_reach = [mixture_distribution.compute_probabilities(reach_sets).sum()]
    lbs_reach, ubs_reach = [bounds_reach_sets.lb_delta.sum()], [bounds_reach_sets.ub_delta.sum()]

    for t in range(prediction_horizon):

        mixture_distribution = propagate(f, probs, partition.locs, noise_distribution)

        # Initialize coarse grid
        inner_partition = uniform_grid(shell[0], shell[1], grid_size)
        next_partition = HyperRectangularPartition(inner_partition=inner_partition,
                                                   avoid_sets=avoid_sets,
                                                   reach_sets=reach_sets,
                                                   loc_shell=loc_shell,
                                                   shell=shell)

        # Refinement
        objective = get_objective_targeted(mixture=mixture_distribution)
        next_partition = next_partition.refine(objective=objective,
                                     contributions=mixture_distribution.compute_probabilities(next_partition),
                                               avoid_sets=avoid_sets,
                                               reach_sets=reach_sets,
                                     target=0.01,
                                     max_regions=5000)
        if plot:
            plot_partition(next_partition)

        mixtures.append(mixture_distribution)
        samples = mixture_distribution(num_samples)

        aps_avoid.append(mixture_distribution.compute_probabilities(avoid_sets).sum())
        aps_reach.append(mixture_distribution.compute_probabilities(reach_sets).sum())

        contributions, bounds_avoid_sets = compute_targeted_bound(f=f,
                                                                   partition_probs=probs,
                                                                   noise_distribution=noise_distribution,
                                                                   partition=partition,
                                                                   bounds_partition=bounds_partition,
                                                                   target=avoid_sets)

        contributions_reach, bounds_reach_sets = compute_targeted_bound(f=f,
                                                                   partition_probs=probs,
                                                                   noise_distribution=noise_distribution,
                                                                   partition=partition,
                                                                   bounds_partition=bounds_partition,
                                                                   target=reach_sets)

        contributions_partition, bounds_partition = compute_targeted_bound(f=f,
                                                                           partition_probs=probs,
                                                                           noise_distribution=noise_distribution,
                                                                           partition=partition,
                                                                           bounds_partition=bounds_partition,
                                                                           target=next_partition)

        # Update partition
        partition = next_partition

        # Update weights
        probs = mixture_distribution.compute_probabilities(partition)

        lbs_avoid.append(bounds_avoid_sets.lb_delta.sum())
        ubs_avoid.append(bounds_avoid_sets.ub_delta.sum())

        lbs_reach.append(bounds_reach_sets.lb_delta.sum())
        ubs_reach.append(bounds_reach_sets.ub_delta.sum())

        print(f'Partition size: {partition.lower.shape[0]}')
        print(f'End of computing for t={t}')

    lbs_avoid, aps_avoid, ubs_avoid = torch.tensor(lbs_avoid), torch.tensor(aps_avoid), torch.tensor(ubs_avoid)
    lbs_reach, aps_reach, ubs_reach = torch.tensor(lbs_reach), torch.tensor(aps_reach), torch.tensor(ubs_reach)

    # Clamping for valid probability bounds
    lbs_avoid = torch.maximum(lbs_avoid, -aps_avoid)
    ubs_avoid = torch.minimum(ubs_avoid, 1-aps_avoid)

    lbs_reach = torch.maximum(lbs_reach, -aps_reach)
    ubs_reach = torch.minimum(ubs_reach, 1-aps_reach)

    return mixtures, aps_avoid, lbs_avoid, ubs_avoid, aps_reach, lbs_reach, ubs_reach
