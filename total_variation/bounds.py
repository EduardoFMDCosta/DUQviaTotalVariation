import torch
from math import sqrt
from typing import Union
from torch.special import erf
import bound_propagation as bp
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangularPartition, HyperRectangle
from distributions.distributions import Gaussian, GaussianMixture

from interval_bp import factory

def compute_centralized_ibp(dynamics: Dynamics,
                            partition: HyperRectangularPartition):
    net = factory.build(dynamics)
    ibp = net.ibp(bp.HyperRectangle(partition.lower, partition.upper))

    locs = partition.locs
    f_locs = dynamics(locs)

    centralized_lower = ibp.lower - f_locs
    centralized_upper = ibp.upper - f_locs

    centralized_lower[centralized_lower.isnan()] = float('inf')
    centralized_upper[centralized_upper.isnan()] = float('inf')

    return HyperRectangle(centralized_lower, centralized_upper)

def compute_local_maximum_distance(dynamics: Dynamics,
                                   partition: HyperRectangularPartition):

    centralized_hypercubes = compute_centralized_ibp(dynamics, partition)
    max_abs_values = torch.maximum(centralized_hypercubes.lower.abs(), centralized_hypercubes.upper.abs())

    return max_abs_values.norm(p=2, dim=1)

def compute_h(f: Dynamics,
              noise_distribution: Gaussian,
              partition: HyperRectangularPartition):

    inverse_cov = torch.linalg.inv(noise_distribution.covariance_matrix)
    spectral_norm_inverse_cov = torch.linalg.norm(inverse_cov, ord=2)

    max_norm_regions = compute_local_maximum_distance(f, partition)

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