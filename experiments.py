import torch
from typing import Union
from itertools import accumulate
from bounds import compute_bound_TV
from distributions import Gaussian, GaussianMixture
from dynamics import Dynamics
from regions import HyperRectangularPartition
from utils import get_shell, get_shell_loc, uniform_grid


def propagate(f: Dynamics,
              weights: torch.Tensor,
              locs: torch.Tensor,
              noise_distribution: Gaussian):

    covs_noise = noise_distribution.covariance.unsqueeze(0).expand(locs.size(0), -1, -1)
    mixture_distribution = GaussianMixture(f(locs), covs_noise, weights)

    return mixture_distribution

def approximation_scheme_tv(f: Dynamics,
                            initial_distribution: Union[Gaussian, GaussianMixture],
                            noise_distribution: Gaussian,
                            initial_grid_size: int = 10,
                            prediction_horizon: int = 2,
                            n_samples: int = 1000):

    tv_bounds = [0.0]
    mixtures = []

    for t in range(prediction_horizon + 1):

        if t == 0:
            mixture_distribution = initial_distribution
        else:
            mixture_distribution = propagate(f, probs, locs, noise_distribution)


        mixtures.append(mixture_distribution)
        samples = mixture_distribution(n_samples)

        if t < prediction_horizon:
            shell = get_shell(samples)
            loc_shell = get_shell_loc(shell)

            inner_locs = uniform_grid(shell[0], shell[1], initial_grid_size) #TODO: Change to Steven's discretization tool?
            partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

            locs = partition.locs
            probs = mixture_distribution.compute_probabilities(partition)

            tv_bound = compute_bound_TV(f, probs, noise_distribution, partition)

            #TODO: Add refinement algorithm

            tv_bounds.append(tv_bound.item())

    # TV_{t+1} = min(TV_t + bound, 1)
    tv_bounds = [min(1, tv) for tv in accumulate(tv_bounds)]

    return mixtures, tv_bounds

def monte_carlo(f: Dynamics,
                initial_distribution: Union[Gaussian, GaussianMixture],
                noise_distribution: Gaussian,
                prediction_horizon: int = 2,
                n_samples: int = 1000):

    samples = initial_distribution(n_samples)

    monte_carlo_samples = [samples]

    for t in range(prediction_horizon):
        noise_samples = noise_distribution(n_samples)
        samples = f(samples) + noise_samples

        monte_carlo_samples.append(samples)

    return monte_carlo_samples

def sample_from_gmm(mixtures: list,
                    n_samples: int = 1000):

    gmm_samples = []

    for mixture in mixtures:
        samples = mixture(n_samples)
        gmm_samples.append(samples)

    return gmm_samples