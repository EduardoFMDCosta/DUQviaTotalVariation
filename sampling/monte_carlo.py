import torch
from typing import Union
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangle


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

def hitting_prob(samples: torch.Tensor,
                 unsafe_sets: HyperRectangle):

    b, n, d = samples.shape
    m = unsafe_sets.lower.shape[0]  # number of obstacles

    # Check if samples are within each unsafe region
    hits = torch.logical_and(
        (samples.unsqueeze(2) >= unsafe_sets.lower.unsqueeze(0).unsqueeze(0)),
        (samples.unsqueeze(2) <= unsafe_sets.upper.unsqueeze(0).unsqueeze(0))
    ).all(dim=-1)

    # Compute hitting probabilities
    hitting_probabilities = hits.float().sum(dim=1) / n

    return hitting_probabilities


def sample_from_gmms(mixtures: list,
                     num_samples: int = 1000):

    gmm_samples = []

    for mixture in mixtures:
        samples = mixture(num_samples)
        gmm_samples.append(samples)

    return torch.stack(gmm_samples)