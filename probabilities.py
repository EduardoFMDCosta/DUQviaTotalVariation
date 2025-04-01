import torch
from regions import HyperRectangularPartition, HyperRectangle
from typing import Union

def gaussian_probabilities(means: torch.Tensor,
                           covariance: torch.Tensor,
                           regions: Union[HyperRectangularPartition, HyperRectangle]):
    lower = regions.lower
    upper = regions.upper

    if lower.dim() == 1:
        lower = lower.unsqueeze(0)
        upper = upper.unsqueeze(0)

    sigma = torch.sqrt(torch.diag(covariance))

    if means.dim() == 1:
        means = means.unsqueeze(0)

    lower_norm = (lower.unsqueeze(1) - means.unsqueeze(0)) / sigma
    upper_norm = (upper.unsqueeze(1) - means.unsqueeze(0)) / sigma

    normal = torch.distributions.Normal(0.0, 1.0)
    lower_cdf = normal.cdf(lower_norm)
    upper_cdf = normal.cdf(upper_norm)

    probs = torch.prod(upper_cdf - lower_cdf, dim=2)

    if isinstance(regions, HyperRectangularPartition):
        probs[-1, :] = 1 - torch.sum(probs[:-1, :], dim=0)  # Last region represents the complement of shell

    if probs.shape[1] == 1:
        probs = probs.squeeze(1) # Squeeze it back to (n,) if means represent a sole Gaussian

    return probs