import torch
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics

def propagate(f: Dynamics,
              weights: torch.Tensor,
              locs: torch.Tensor,
              noise_distribution: Gaussian):

    cov_noise = noise_distribution.covariance
    mixture_distribution = GaussianMixture(f(locs), cov_noise, weights)

    return mixture_distribution