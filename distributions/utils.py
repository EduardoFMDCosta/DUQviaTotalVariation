import torch
from typing import Union
from grid.regions import HyperRectangularPartition, HyperRectangle

def gaussian_probabilities(mean: torch.Tensor,
                           covariance: torch.Tensor,
                           regions: Union[HyperRectangularPartition, HyperRectangle, torch.Tensor]):
    lower = regions.lower
    upper = regions.upper

    sigma = torch.sqrt(torch.diag(covariance))

    if mean.dim() == 1:
        mean = mean.unsqueeze(0)

    lower_norm = (lower.unsqueeze(0) - mean.unsqueeze(1)) / sigma
    upper_norm = (upper.unsqueeze(0) - mean.unsqueeze(1)) / sigma

    normal = torch.distributions.Normal(0.0, 1.0)
    lower_cdf = normal.cdf(lower_norm)
    upper_cdf = normal.cdf(upper_norm)

    probs = torch.prod(upper_cdf - lower_cdf, dim=2)

    if isinstance(regions, HyperRectangularPartition):
        probs[:, -1] = 1 - torch.sum(probs[:, :-1], dim=1)  # Last region represents the complement of shell

    if probs.shape[0] == 1:
        probs = probs.squeeze(0) # Squeeze it back to (n,) if means represent a sole Gaussian

    probs.clamp_(min=0.0, max=1.0) # avoid numerical issues
    probs /= torch.sum(probs)

    return probs