import torch
from abc import abstractmethod
from typing import Union
from torch.distributions import MultivariateNormal
from grid.regions import HyperRectangularPartition, HyperRectangle


class Distributions:
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_probabilities(self,
                              regions: Union[HyperRectangularPartition, HyperRectangle, torch.Tensor]):
        pass


class Gaussian(Distributions):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.covariance = cov

    def __call__(self, num_samples: int):
        mvn = MultivariateNormal(loc=self.mean,
                                 covariance_matrix=self.covariance)
        return mvn.sample((num_samples,))

    def compute_probabilities(self,
                              regions: Union[HyperRectangularPartition, HyperRectangle]):
        return gaussian_probabilities(mean=self.mean,
                                      covariance=self.covariance,
                                      regions=regions)


class GaussianMixture(Distributions):
    def __init__(self, means: torch.Tensor, covariance: torch.Tensor, weights: torch.Tensor):
        self.means = means
        self.covariance = covariance
        self.weights = weights

    def __call__(self, num_samples: int):
        chosen_components = torch.multinomial(self.weights, num_samples, replacement=True)
        gaussian_distributions = MultivariateNormal(self.means, self.covariance)
        samples = gaussian_distributions.sample((num_samples,))
        return samples[torch.arange(num_samples), chosen_components]

    def compute_probabilities(self, regions):
        probs = gaussian_probabilities(mean=self.means,
                                       covariance=self.covariance,
                                       regions=regions)
        return torch.sum(probs * self.weights.unsqueeze(1), dim=0)


### Auxiliary methods
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

    return probs