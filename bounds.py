import torch
from math import sqrt
from typing import Union
from torch.special import erf
from distributions import Gaussian, GaussianMixture
from dynamics import Dynamics
from regions import HyperRectangularPartition


def print_bound(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"TV bound at propag step: {result[0]}")
        return result
    return wrapper

# ----------------------------------------------------------------------------------------- #
# ------------------- TV upper bound (using the maximization approach) -------------------- #
# ----------------------------------------------------------------------------------------- #

def compute_h(f: Dynamics, noise_distribution: Gaussian, partition: HyperRectangularPartition):

    inverse_cov = torch.linalg.inv(noise_distribution.covariance)
    spectral_norm_inverse_cov = torch.linalg.norm(inverse_cov, ord=2)

    max_norm_regions = f.compute_local_maximum_distance(partition)

    return 1 / (2 * sqrt(2)) * spectral_norm_inverse_cov * max_norm_regions

@print_bound
def compute_bound_TV(f: Dynamics, mixture_distribution: Union[Gaussian, GaussianMixture], noise_distribution: Gaussian, partition: HyperRectangularPartition):

    h = compute_h(f, noise_distribution, partition)
    erf_h = erf(h)

    mixture_probs = mixture_distribution.compute_probabilities(partition)

    bound = torch.dot(erf_h, mixture_probs)

    return bound

