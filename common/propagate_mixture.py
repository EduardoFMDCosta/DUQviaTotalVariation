import torch
from dynamics.dynamics import Dynamics
from distributions.distributions import Gaussian, GaussianMixture

def propagate_mixture(f: Dynamics,
              weights: torch.Tensor,
              locs: torch.Tensor,
              noise_distribution: Gaussian):

    cov_noise = noise_distribution.covariance_matrix
    mixture_distribution = GaussianMixture(f(locs), cov_noise, weights)

    return mixture_distribution