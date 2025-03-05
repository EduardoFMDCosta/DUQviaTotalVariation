from typing import Union

import torch
import parameters
import grid_generation as grid
import bounds as tv

from distributions import Gaussian, GaussianMixture
from dynamics import Dynamics
from regions import HyperRectangularPartition
from utils import get_shell, get_shell_loc


def propagate(f: Dynamics,
              weights: torch.Tensor,
              locs: torch.Tensor,
              noise_distribution: Gaussian):

    mixture_means = f(locs)
    covs_noise = noise_distribution.covariance.unsqueeze(0).expand(mixture_means.size(0), -1, -1)
    mixture_distribution = GaussianMixture(mixture_means, covs_noise, weights)

    return mixture_distribution

def tv_bound_algorithm(f: Dynamics,
                       initial_distribution: Union[Gaussian, GaussianMixture],
                       noise_distribution: Gaussian,
                       grid_type,
                       prediction_horizon: int = 10,
                       n_samples: int = 1000):

    tv_bounds = [0.0]
    mixtures = []

    for t in range(prediction_horizon + 1):

        if t == 0:
            mixture_distribution = initial_distribution
        else:
            mixture_distribution = propagate(f, mixture_probs, mixture_locs, noise_distribution)


        mixtures.append(mixture_distribution)
        samples = mixture_distribution(n_samples)

        if t < prediction_horizon:
            shell = get_shell(samples)
            loc_shell = get_shell_loc(shell)

            partition = HyperRectangularPartition(locs, loc_shell, shell)
            locs = partition.locs

            mixture_probs = mixture_distribution.compute_probabilities(partition)

            tv_bound, contributions = tv.compute_bound_TV(f, mixture_distribution, noise_distribution, partition)


            for r in range(parameters.n_refinements):

                regions = regions[:-1]
                signatures = signatures[:-1]
                regions, signatures = grid.refine_regions(regions, signatures, contributions, parameters.threshold)

                double_hat_probs = hat_mixture.compute_regions_probabilities(regions) #TODO: Generalize for GMMs with different covariances
                regions, signatures = grid.add_unbounded_representations(regions, signatures, outer_signature)

                tv_bound, contributions = tv.compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, double_hat_probs, regions)


            tv_bounds.append(tv_bound.item())

    return torch.Tensor(tv_bounds), mixtures