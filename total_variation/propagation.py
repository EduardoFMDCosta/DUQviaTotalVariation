import torch
from typing import Union
from itertools import accumulate
from total_variation.bounds import compute_bound_tv, get_objective_tv_bound
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangularPartition
from grid.utils import get_shell, get_shell_loc, uniform_grid


def propagate(f: Dynamics,
              weights: torch.Tensor,
              locs: torch.Tensor,
              noise_distribution: Gaussian):

    cov_noise = noise_distribution.covariance
    mixture_distribution = GaussianMixture(f(locs), cov_noise, weights)

    return mixture_distribution

def propagate_mixture_tv_bounds(f: Dynamics,
                            initial_distribution: Union[Gaussian, GaussianMixture],
                            noise_distribution: Gaussian,
                            initial_grid_size: int = 10,
                            prediction_horizon: int = 2,
                            num_samples: int = 1000):

    tv_bounds = [0.0]
    mixtures = []
    probs = None
    partition = None

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

            inner_locs = uniform_grid(shell[0], shell[1], initial_grid_size)
            partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

            contributions, tv_bound = compute_bound_tv(f=f,
                                                       mixture=mixture_distribution,
                                                       noise_distribution=noise_distribution,
                                                       partition=partition)

            # Refinement
            objective = get_objective_tv_bound(f=f,
                                               mixture=mixture_distribution,
                                               noise_distribution=noise_distribution)

            partition = partition.refine(objective=objective,
                                         contributions=contributions,
                                         target=0.05,
                                         pareto=0.3,
                                         max_regions=5000)

            probs = mixture_distribution.compute_probabilities(partition)
            contributions, tv_bound = compute_bound_tv(f=f,
                                                       mixture=mixture_distribution,
                                                       noise_distribution=noise_distribution,
                                                       partition=partition)

            tv_bounds.append(tv_bound.item())

        print(f'End of computing for t={t}')

    # TV_{t+1} = min(TV_t + bound, 1)
    tv_bounds = [min(1, tv) for tv in accumulate(tv_bounds)]

    return mixtures, tv_bounds

def monte_carlo(f: Dynamics,
                initial_distribution: Union[Gaussian, GaussianMixture],
                noise_distribution: Gaussian,
                prediction_horizon: int = 2,
                num_samples: int = 1000):

    samples = initial_distribution(num_samples)

    monte_carlo_samples = [samples]

    for t in range(prediction_horizon):
        noise_samples = noise_distribution(num_samples)
        samples = f(samples) + noise_samples

        monte_carlo_samples.append(samples)

    return monte_carlo_samples

def sample_from_gmm(mixtures: list,
                    num_samples: int = 1000):

    gmm_samples = []

    for mixture in mixtures:
        samples = mixture(num_samples)
        gmm_samples.append(samples)

    return gmm_samples