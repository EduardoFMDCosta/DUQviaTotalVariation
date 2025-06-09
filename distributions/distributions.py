import torch
from typing import Union
import discretize_distributions as ds
from torch.distributions.normal import Normal
from grid.regions import HyperRectangularPartition, HyperRectangle

class Gaussian(ds.MultivariateNormal):
    def __init__(self, mean: torch.Tensor, covariance_matrix: torch.Tensor):
        super().__init__(loc=mean, covariance_matrix=covariance_matrix)

    def __call__(self, num_samples: int):
        return self.sample((num_samples,))

    def compute_probabilities(self, regions: Union[HyperRectangularPartition, HyperRectangle]):
        return gaussian_probabilities(mean=self.loc,
                                      covariance=self.covariance_matrix,
                                      regions=regions)


class GaussianMixture(ds.MixtureMultivariateNormal):
    def __init__(self, means: torch.Tensor, covariance_matrix: torch.Tensor, weights: torch.Tensor):
        super().__init__(mixture_distribution=torch.distributions.Categorical(probs=weights),
                         component_distribution=ds.MultivariateNormal(loc=means, covariance_matrix=covariance_matrix))

    def __call__(self, num_samples: int):
        chosen_components = torch.multinomial(self.mixture_distribution.probs, num_samples, replacement=True)
        gaussian_distributions = ds.MultivariateNormal(self.component_distribution.loc, self.component_distribution.covariance_matrix)
        samples = gaussian_distributions.sample((num_samples,))
        return samples[torch.arange(num_samples), chosen_components]

    def compute_probabilities(self, regions):
        probs = gaussian_probabilities(mean=self.component_distribution.loc,
                                       covariance=self.component_distribution.covariance_matrix,
                                       regions=regions)
        return torch.sum(probs * self.mixture_distribution.probs.unsqueeze(1), dim=0)


### Auxiliary methods
def gaussian_probabilities(mean: torch.Tensor,
                           covariance: torch.Tensor,
                           regions: Union[HyperRectangularPartition, HyperRectangle, torch.Tensor]):
    lower = regions.lower
    upper = regions.upper
    sigma = torch.sqrt(covariance.diagonal(dim1=-2, dim2=-1))

    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
        sigma = sigma.unsqueeze(0)

    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0).expand_as(mean)

    lower_norm = (lower.unsqueeze(0) - mean.unsqueeze(1)) / sigma.unsqueeze(1)
    upper_norm = (upper.unsqueeze(0) - mean.unsqueeze(1)) / sigma.unsqueeze(1)

    normal = torch.distributions.Normal(0.0, 1.0)
    lower_cdf = normal.cdf(lower_norm)
    upper_cdf = normal.cdf(upper_norm)

    probs = torch.prod(upper_cdf - lower_cdf, dim=2)

    if isinstance(regions, HyperRectangularPartition):
        probs[:, -1] = 1 - torch.sum(probs[:, :-1], dim=1)  # Last region represents the complement of shell

    if probs.shape[0] == 1:
        probs = probs.squeeze(0)  # Squeeze it back to (n,) if means represent a sole Gaussian

    probs.clamp_(min=0.0, max=1.0)  # avoid numerical issues

    return probs
