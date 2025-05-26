import torch
from typing import Union
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics


def simulate_monte_carlo(f: Dynamics,
                initial_distribution: Union[Gaussian, GaussianMixture],
                noise_distribution: Gaussian,
                prediction_horizon: int,
                num_samples: int):

    samples = initial_distribution(num_samples)

    monte_carlo_samples = [samples]

    for t in range(prediction_horizon):
        noise_samples = noise_distribution(num_samples)
        samples = f(samples) + noise_samples

        monte_carlo_samples.append(samples)

    return torch.stack(monte_carlo_samples)


def simulate_mixtures(mixtures: list,
                      num_samples: int):

    gmm_samples = []

    for mixture in mixtures:
        samples = mixture(num_samples)
        gmm_samples.append(samples)

    return torch.stack(gmm_samples)