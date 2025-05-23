import torch
from typing import Union
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import LinearDynamics, factory, Dynamics
from grid.regions import HyperRectangle, HyperRectangularPartition
from grid.utils import get_shell, get_shell_loc, uniform_grid
from plotting.plotting import plot_confidence_interval, plot_partition, plot_samples
from sampling.monte_carlo import hitting_prob, monte_carlo, sample_from_gmms
from targeted_probability.bounds import compute_targeted_bound, TargetedBounds


def propagate(f: Dynamics,
              weights: torch.Tensor,
              locs: torch.Tensor,
              noise_distribution: Gaussian):

    cov_noise = noise_distribution.covariance
    mixture_distribution = GaussianMixture(f(locs), cov_noise, weights)

    return mixture_distribution

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

    # Initialize bounds
    bounds_partition = TargetedBounds(torch.zeros(initial_grid_size+1), torch.zeros(initial_grid_size+1))
    bounds_unsafe_sets = TargetedBounds(torch.zeros(num_unsafe_sets), torch.zeros(num_unsafe_sets))

    mixtures = []
    probs = None
    partition = None

    # Bound trajectories
    lbs, aps, ubs = [bounds_unsafe_sets.lb_delta], [initial_distribution.compute_probabilities(unsafe_sets)], [bounds_unsafe_sets.ub_delta]

    for t in range(prediction_horizon + 1):

        if t == 0:
            mixture_distribution = initial_distribution
        else:
            mixture_distribution = propagate(f, probs, partition.locs, noise_distribution)

        mixtures.append(mixture_distribution)
        samples = mixture_distribution(num_samples)

        if t < prediction_horizon:
            shell = get_shell(samples)
            loc_shell = get_shell_loc(shell)

            inner_partition = uniform_grid(shell[0], shell[1], initial_grid_size)
            partition = HyperRectangularPartition(inner_partition, loc_shell, shell)
            if plot:
                plot_partition(partition)

            bounds_unsafe_sets = compute_targeted_bound(f=f,
                                                     mixture=mixture_distribution,
                                                     noise_distribution=noise_distribution,
                                                     partition=partition,
                                                     bounds_partition=bounds_partition,
                                                     target=unsafe_sets)

            # Update weights
            probs = mixture_distribution.compute_probabilities(partition)

            lbs.append(bounds_unsafe_sets.lb_delta)
            aps.append(mixture_distribution.compute_probabilities(unsafe_sets))
            ubs.append(bounds_unsafe_sets.ub_delta)

        print(f'End of computing for t={t}')

    lbs, aps, ubs = torch.tensor(lbs), torch.tensor(aps), torch.tensor(ubs)
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

    # Initial distribution
    mean_initial = torch.Tensor([10., 9.])
    cov_initial = torch.diag(torch.Tensor([0.002, 0.002]))
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = torch.diag(torch.Tensor([0.1, 0.1]))
    noise_distribution = Gaussian(mean_noise, cov_noise)

    n_samples = 5000

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([1.0, 1.0]).unsqueeze(0), torch.tensor([3.0, 3.0]).unsqueeze(0))

    mixtures, aps, lbs, ubs = propagate_mixture_targeted_bounds(f,
                                      initial_distribution,
                                      noise_distribution,
                                      unsafe_set,
                                      initial_grid_size= 25,
                                      prediction_horizon= 10,
                                      num_samples= 1000,
                                      plot= False)

    monte_carlo_samples = monte_carlo(f=f,
                                      initial_distribution=initial_distribution,
                                      noise_distribution=noise_distribution,
                                      prediction_horizon=10,
                                      num_samples=1000)

    hitting_probs_mc = hitting_prob(monte_carlo_samples, unsafe_set)

    plot_confidence_interval(aps, lbs, ubs, hitting_probs_mc)

    gmm_samples = sample_from_gmms(mixtures=mixtures,
                                  num_samples=1000)

    plot_samples(monte_carlo_samples=monte_carlo_samples,
                 gmm_samples=gmm_samples,
                 unsafe_sets=unsafe_set)

