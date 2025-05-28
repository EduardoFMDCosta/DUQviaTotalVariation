import torch
from typing import Union
from common.monte_carlo import simulate_monte_carlo, simulate_mixtures
from common.propagate_mixture import propagate
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import LinearDynamics, factory, Dynamics, DubinsCarDynamics
from grid.obstacles import compute_hitting_prob
from grid.regions import HyperRectangle, HyperRectangularPartition
from grid.utils import get_shell, get_shell_loc, uniform_grid
from plotting.plotting import plot_confidence_interval, plot_partition, plot_samples
from targeted_probability.bounds import compute_targeted_bound, TargetedBounds, get_objective_targeted
from targeted_probability.transition_kernel import bound_transition_kernel

def propagate_mixture_targeted_bounds(f: Dynamics,
                                      initial_distribution: Union[Gaussian, GaussianMixture],
                                      noise_distribution: Gaussian,
                                      unsafe_sets: HyperRectangle,
                                      initial_grid_size: int = 10,
                                      prediction_horizon: int = 2,
                                      num_samples: int = 1000,
                                      plot: bool = False):

    # Parameters
    num_unsafe_sets = unsafe_sets.lower.shape[0]

    shell = torch.tensor([[-3, -3], [8, 8]])
    loc_shell = get_shell_loc(shell)

    # Initialize bounds
    bounds_partition = TargetedBounds(torch.zeros(initial_grid_size+1), torch.zeros(initial_grid_size+1))
    bounds_unsafe_sets = TargetedBounds(torch.zeros(num_unsafe_sets), torch.zeros(num_unsafe_sets))

    mixtures = []
    probs = None
    partition = None

    # Bound trajectories
    aps = []
    lbs, ubs = [bounds_unsafe_sets.lb_delta.sum()], [bounds_unsafe_sets.ub_delta.sum()]

    for t in range(prediction_horizon + 1):

        if t == 0:
            mixture_distribution = initial_distribution
        else:
            mixture_distribution = propagate(f, probs, partition.locs, noise_distribution)

        mixtures.append(mixture_distribution)
        samples = mixture_distribution(num_samples)
        mixture_hitting_prob = mixture_distribution.compute_probabilities(unsafe_sets)
        mixture_hitting_prob_all = mixture_hitting_prob.sum()
        aps.append(mixture_hitting_prob_all)

        if t < prediction_horizon:
            #shell = get_shell(samples)
            #loc_shell = get_shell_loc(shell)

            inner_partition = uniform_grid(shell[0], shell[1], initial_grid_size)
            partition = HyperRectangularPartition(inner_partition, loc_shell, shell)
            if plot:
                plot_partition(partition)

            contributions, bounds_unsafe_sets = compute_targeted_bound(f=f,
                                                     mixture=mixture_distribution,
                                                     noise_distribution=noise_distribution,
                                                     partition=partition,
                                                     bounds_partition=bounds_partition,
                                                     target=unsafe_sets)

            contributions_partition, bounds_partition = compute_targeted_bound(f=f,
                                                                       mixture=mixture_distribution,
                                                                       noise_distribution=noise_distribution,
                                                                       partition=partition,
                                                                       bounds_partition=bounds_partition,
                                                                       target=partition)

            # Refinement
            # objective = get_objective_targeted(f=f,
            #                                    noise_distribution=noise_distribution,
            #                                    target=unsafe_sets)
            # contributions = bound_transition_kernel(f, partition, unsafe_sets, cov_noise, supremum=True).squeeze()
            # partition = partition.refine(objective=objective,
            #                              contributions=contributions,
            #                              target=1.02,
            #                              pareto=0.3,
            #                              max_regions=3000)

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
            [0.74, 0.10],
            [0.05, 0.82]
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

    num_samples = 10000
    horizon = 3
    grid_size = 225

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([[2, 2], [0.5, 0]]), torch.tensor([[3, 3], [1.5, 1]]))

    mixtures, aps, lbs, ubs = propagate_mixture_targeted_bounds(f=f,
                                      initial_distribution=initial_distribution,
                                      noise_distribution=noise_distribution,
                                      unsafe_sets=unsafe_set,
                                      initial_grid_size=grid_size,
                                      prediction_horizon=horizon,
                                      num_samples=num_samples,
                                      plot= False)

    monte_carlo_samples = simulate_monte_carlo(f=f,
                                               initial_distribution=initial_distribution,
                                               noise_distribution=noise_distribution,
                                               prediction_horizon=horizon,
                                               num_samples=num_samples)

    hitting_probs_mc = compute_hitting_prob(monte_carlo_samples, unsafe_set)

    plot_confidence_interval(aps, lbs, ubs, hitting_probs_mc)

    gmm_samples = simulate_mixtures(mixtures=mixtures,
                                    num_samples=num_samples)

    check = compute_hitting_prob(gmm_samples, unsafe_set)

    plot_samples(monte_carlo_samples=monte_carlo_samples,
                 gmm_samples=gmm_samples,
                 unsafe_sets=unsafe_set)

