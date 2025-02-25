import torch
from torch.distributions import MultivariateNormal
from abc import abstractmethod
import probability_mass_computation as proba
from regions import HyperRectangularVoronoiPartition, HyperRectangle
from typing import Union


class _Distributions:

    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_probabilities(self, regions: torch.Tensor):
        pass


class Gaussian(_Distributions):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.covariance = cov

    def __call__(self, n_samples: int):
        mvn = MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)
        return mvn.sample((n_samples,))

    def compute_probabilities(self, regions: Union[HyperRectangularVoronoiPartition, HyperRectangle]):

        lower = regions.lower
        upper = regions.upper

        if lower.dim() == 1:
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)

        sigma = torch.sqrt(torch.diag(self.covariance))

        lower_norm = (lower - self.mean) / sigma
        upper_norm = (upper - self.mean) / sigma

        normal = torch.distributions.Normal(0.0, 1.0)
        lower_cdf = normal.cdf(lower_norm)
        upper_cdf = normal.cdf(upper_norm)

        return torch.prod(upper_cdf - lower_cdf, dim=1)


class GaussianMixture(_Distributions):
    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        self.means = means
        self.covariances = covs
        self.weights = weights

    def __call__(self, n_samples: int):
        chosen_components = torch.multinomial(self.weights, n_samples, replacement=True)
        gaussian_distributions = MultivariateNormal(self.means, self.covariances)
        samples = gaussian_distributions.sample((n_samples,))
        return samples[torch.arange(n_samples), chosen_components]

    def compute_probabilities(self, regions):
        raise NotImplementedError("Implement for new framework.")