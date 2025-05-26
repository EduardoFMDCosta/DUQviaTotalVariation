import torch
from typing import Union
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics


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

    return torch.stack(monte_carlo_samples)


def sample_from_gmms(mixtures: list,
                     num_samples: int = 1000):

    gmm_samples = []

    for mixture in mixtures:
        samples = mixture(num_samples)
        gmm_samples.append(samples)

    return torch.stack(gmm_samples)