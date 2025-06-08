import torch
from math import sqrt
from typing import Union
from torch.special import erf
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangularPartition


# ----------------------------------------------------------------------------------------- #
# ------------------- TV upper bound (using the maximization approach) -------------------- #
# ----------------------------------------------------------------------------------------- #

def compute_h(f: Dynamics,
              noise_distribution: Gaussian,
              partition: HyperRectangularPartition):

    inverse_cov = torch.linalg.inv(noise_distribution.covariance_matrix)
    spectral_norm_inverse_cov = torch.linalg.norm(inverse_cov, ord=2)

    max_norm_regions = f.compute_local_maximum_distance(partition)

    return 1 / (2 * sqrt(2)) * spectral_norm_inverse_cov * max_norm_regions


def compute_bound_tv(f: Dynamics,
                     mixture: Union[Gaussian, GaussianMixture],
                     noise_distribution: Gaussian,
                     partition: HyperRectangularPartition):

    h = compute_h(f, noise_distribution, partition)
    erf_h = erf(h)

    probs = mixture.compute_probabilities(partition)
    contributions = erf_h * probs
    bound = contributions.sum()

    return contributions, bound

def get_objective_tv_bound(mixture: Union[Gaussian, GaussianMixture]):
    def objective_tv_bound(partition: HyperRectangularPartition):
        return mixture.compute_probabilities(partition), None
    return objective_tv_bound