import torch
from typing import Union
from common.monte_carlo import simulate_monte_carlo, simulate_mixtures
from common.propagate_mixture import propagate
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import LinearDynamics, Dynamics, DubinsCarDynamics
from grid.obstacles import compute_hitting_prob
from grid.regions import HyperRectangle, HyperRectangularPartition, AvoidHyperRectangle, ReachHyperRectangle
from grid.utils import get_shell, get_shell_loc, uniform_grid
from plotting.plotting import plot_confidence_interval, plot_partition, plot_samples
from targeted_probability.bounds import compute_targeted_bound, TargetedBounds, get_objective_targeted

def propagate_mixture_targeted_bounds(f: Dynamics,
                                      initial_distribution: Union[Gaussian, GaussianMixture],
                                      noise_distribution: Gaussian,
                                      shell: torch.Tensor,
                                      avoid_sets: AvoidHyperRectangle,
                                      reach_sets: ReachHyperRectangle,
                                      initial_grid_size: int = 10,
                                      prediction_horizon: int = 2,
                                      num_samples: int = 1000,
                                      plot: bool = True):

    # Parameters
    num_unsafe_sets = avoid_sets.lower.shape[0]

    # Initialize coarse partition
    loc_shell = get_shell_loc(shell)
    inner_partition = uniform_grid(shell[0], shell[1], initial_grid_size)
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
    if True:
        plot_partition(partition)

    # Compute mixture weights
    probs = mixture_distribution.compute_probabilities(partition)

    # Initialize bounds
    bounds_partition = TargetedBounds(torch.zeros(partition.lower.shape[0]), torch.zeros(partition.lower.shape[0]))
    bounds_unsafe_sets = TargetedBounds(torch.zeros(num_unsafe_sets), torch.zeros(num_unsafe_sets))

    # Bound trajectories
    aps = [mixture_distribution.compute_probabilities(avoid_sets).sum()]
    lbs, ubs = [bounds_unsafe_sets.lb_delta.sum()], [bounds_unsafe_sets.ub_delta.sum()]

    for t in range(prediction_horizon):

        mixture_distribution = propagate(f, probs, partition.locs, noise_distribution)

        # Initialize coarse grid
        inner_partition = uniform_grid(shell[0], shell[1], initial_grid_size)
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
        if True:
            plot_partition(next_partition)

        mixtures.append(mixture_distribution)
        samples = mixture_distribution(num_samples)
        mixture_hitting_prob = mixture_distribution.compute_probabilities(avoid_sets)
        mixture_hitting_prob_all = mixture_hitting_prob.sum()
        aps.append(mixture_hitting_prob_all)

        contributions, bounds_unsafe_sets = compute_targeted_bound(f=f,
                                                                   partition_probs=probs,
                                                                   noise_distribution=noise_distribution,
                                                                   partition=partition,
                                                                   bounds_partition=bounds_partition,
                                                                   target=avoid_sets)

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

        lb_hitting_prob_all = bounds_unsafe_sets.lb_delta.sum()
        lbs.append(lb_hitting_prob_all)

        ub_hitting_prob_all = bounds_unsafe_sets.ub_delta.sum()
        ubs.append(ub_hitting_prob_all)

        print(f'End of computing for t={t}')

    lbs, aps, ubs = torch.tensor(lbs), torch.tensor(aps), torch.tensor(ubs)

    # Clamping for valid probability bounds
    lbs = torch.maximum(lbs, -aps)
    ubs = torch.minimum(ubs, 1-aps)

    return mixtures, aps, lbs, ubs


if __name__ == '__main__':
    torch.manual_seed(0)

    # System parameters
    A = torch.Tensor(
        [
            [0.84, 0.10],
            [0.05, 0.72]
        ])
    f = LinearDynamics(A)
    #f = DubinsCarDynamics()

    # Initial distribution
    mean_initial = torch.Tensor([5., 5.])
    cov_initial = torch.diag(torch.Tensor([0.002, 0.002]))
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = torch.diag(torch.Tensor([0.01, 0.01]))
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Shell
    shell = torch.tensor([[-3, -3], [8, 8]])

    num_samples = 10000
    horizon = 15
    grid_size = 1600

    # Define unsafe set
    avoid_sets = AvoidHyperRectangle(torch.tensor([[3.8, 2], [2, 1]]), torch.tensor([[4.8, 3.5], [3, 2]]))
    reach_sets = ReachHyperRectangle(torch.tensor([[0., 0.]]), torch.tensor([[0.2, 0.2]]))

    mixtures, aps, lbs, ubs = propagate_mixture_targeted_bounds(f=f,
                                                                initial_distribution=initial_distribution,
                                                                noise_distribution=noise_distribution,
                                                                shell=shell,
                                                                avoid_sets=avoid_sets,
                                                                reach_sets=reach_sets,
                                                                initial_grid_size=grid_size,
                                                                prediction_horizon=horizon,
                                                                num_samples=num_samples,
                                                                plot= False)

    monte_carlo_samples = simulate_monte_carlo(f=f,
                                               initial_distribution=initial_distribution,
                                               noise_distribution=noise_distribution,
                                               prediction_horizon=horizon,
                                               num_samples=num_samples)

    hitting_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                            avoid_sets=avoid_sets)

    plot_confidence_interval(aps, lbs, ubs, hitting_probs_mc)

    gmm_samples = simulate_mixtures(mixtures=mixtures,
                                    num_samples=num_samples)

    check = compute_hitting_prob(gmm_samples, avoid_sets)

    plot_samples(monte_carlo_samples=monte_carlo_samples,
                 gmm_samples=gmm_samples,
                 avoid_sets=avoid_sets)

