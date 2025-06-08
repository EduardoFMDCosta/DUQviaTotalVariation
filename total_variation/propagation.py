import torch
from typing import Union
from itertools import accumulate
from common.bounds import Bounds, ConfidenceInterval
from common.propagate_mixture import propagate_mixture
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangularPartition, AvoidHyperRectangle, ReachHyperRectangle
from grid.utils import get_shell_loc, uniform_grid, get_high_prob_set, propagate_high_prob_set
from plotting.plotting import plot_partition, plot_partition_bounds
from total_variation.bounds import get_objective_tv_bound, compute_bound_tv


def propagate_tv(f: Dynamics,
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

    # High probability set
    hpr = get_high_prob_set(mixture_distribution)

    # Refine initial partition
    objective = get_objective_tv_bound(mixture=mixture_distribution)
    partition = partition.refine(objective=objective,
                                 contributions=mixture_distribution.compute_probabilities(partition),
                                 avoid_sets=avoid_sets,
                                 reach_sets=reach_sets,
                                 target=0.005,
                                 max_regions=5000)
    if plot:
        plot_partition(partition=partition, avoid_sets=avoid_sets, reach_sets=reach_sets, high_prob_set=hpr)

    # Compute mixture weights
    probs = mixture_distribution.compute_probabilities(partition)

    # Initialize bounds
    lbs_avoid,  ubs_avoid = [0.0], [0.0]
    lbs_reach, ubs_reach = [0.0], [0.0]

    # Bound trajectories
    aps_avoid = [mixture_distribution.compute_probabilities(avoid_sets).sum()]
    aps_reach = [mixture_distribution.compute_probabilities(reach_sets).sum()]

    for t in range(prediction_horizon):

        mixture_distribution = propagate_mixture(f, probs, partition.locs, noise_distribution)

        # Initialize coarse grid
        inner_partition = uniform_grid(shell[0], shell[1], grid_size)
        next_partition = HyperRectangularPartition(inner_partition=inner_partition,
                                                   avoid_sets=avoid_sets,
                                                   reach_sets=reach_sets,
                                                   loc_shell=loc_shell,
                                                   shell=shell)

        # Refinement
        objective = get_objective_tv_bound(mixture=mixture_distribution)
        next_partition = next_partition.refine(objective=objective,
                                               contributions=mixture_distribution.compute_probabilities(next_partition),
                                               avoid_sets=avoid_sets,
                                               reach_sets=reach_sets,
                                               target=0.005,
                                               max_regions=5000)

        mixtures.append(mixture_distribution)
        samples = mixture_distribution(num_samples)

        aps_avoid.append(mixture_distribution.compute_probabilities(avoid_sets).sum())
        aps_reach.append(mixture_distribution.compute_probabilities(reach_sets).sum())

        contributions, tv = compute_bound_tv(f=f,
                                             mixture=mixture_distribution,
                                             noise_distribution=noise_distribution,
                                             partition=partition)

        # Update partition
        partition = next_partition

        # Update high probability set
        hpr = propagate_high_prob_set(f=f, hpr=hpr, noise_distribution=noise_distribution)
        if plot:
            plot_partition(partition=partition, avoid_sets=avoid_sets, reach_sets=reach_sets, high_prob_set=hpr)

        # Update weights
        probs = mixture_distribution.compute_probabilities(partition)

        lbs_avoid.append(-tv.item())
        ubs_avoid.append(tv.item())

        lbs_reach.append(-tv.item())
        ubs_reach.append(tv.item())

        print(f'Partition size: {partition.lower.shape[0]}')
        print(f'End of computing for t={t}')


    # Make bound cumulative
    lbs_avoid, ubs_avoid = list(accumulate(lbs_avoid)), list(accumulate(ubs_avoid))
    lbs_reach, ubs_reach = list(accumulate(lbs_reach)), list(accumulate(ubs_reach))

    # Convert to tensor
    lbs_avoid, aps_avoid, ubs_avoid = torch.tensor(lbs_avoid), torch.tensor(aps_avoid), torch.tensor(ubs_avoid)
    lbs_reach, aps_reach, ubs_reach = torch.tensor(lbs_reach), torch.tensor(aps_reach), torch.tensor(ubs_reach)

    # Clamping for valid probability bounds
    lbs_avoid = torch.clamp(lbs_avoid, min=-aps_avoid)
    ubs_avoid = torch.clamp(ubs_avoid, max=1-aps_avoid)

    lbs_reach = torch.clamp(lbs_reach, min=-aps_reach)
    ubs_reach = torch.clamp(ubs_reach, max=1-aps_reach)

    return mixtures, ConfidenceInterval(aps_avoid, Bounds(lbs_avoid, ubs_avoid)), ConfidenceInterval(aps_reach, Bounds(lbs_reach, ubs_reach))