import torch
from regions import HyperRectangularPartition, HyperRectangle
from typing import Union
import torch.distributions as dist

def gaussian_probabilities(means: torch.Tensor,
                           covariance: torch.Tensor,
                           regions: Union[HyperRectangularPartition, HyperRectangle]):
    lower = regions.lower
    upper = regions.upper

    sigma = torch.sqrt(torch.diag(covariance))

    if means.dim() == 1:
        means = means.unsqueeze(0)

    lower_norm = (lower.unsqueeze(0) - means.unsqueeze(1)) / sigma
    upper_norm = (upper.unsqueeze(0) - means.unsqueeze(1)) / sigma

    normal = torch.distributions.Normal(0.0, 1.0)
    lower_cdf = normal.cdf(lower_norm)
    upper_cdf = normal.cdf(upper_norm)

    probs = torch.prod(upper_cdf - lower_cdf, dim=2)

    if isinstance(regions, HyperRectangularPartition):
        probs[:, -1] = 1 - torch.sum(probs[:, :-1], dim=1)  # Last region represents the complement of shell

    if probs.shape[0] == 1:
        probs = probs.squeeze(0) # Squeeze it back to (n,) if means represent a sole Gaussian

    return torch.clamp(probs, 0, 1)


def kernel_probs_given_optimal_means(means_for_target: torch.Tensor,
                                     covariance: torch.Tensor,
                                     target_lower: torch.Tensor,
                                     target_upper: torch.Tensor,
                                     target_is_partition: bool):

    normal = dist.Normal(0, 1)
    std_devs = torch.sqrt(torch.diag(covariance))

    lower_std = (target_lower - means_for_target) / std_devs
    upper_std = (target_upper - means_for_target) / std_devs

    lower_std[lower_std.isnan()] = - float('inf') # Q? Should always be negative?
    upper_std[upper_std.isnan()] = float('inf')

    # Compute Gaussian probs for each pair (\barR_from, R_target)
    lower_cdf = normal.cdf(lower_std)
    upper_cdf = normal.cdf(upper_std)

    probs_per_dimension = upper_cdf - lower_cdf
    probs = probs_per_dimension.prod(dim=-1)

    if target_is_partition:
        probs[:, -1] = 1 - probs[:, -1] # As we had computed P(shell)

    return probs