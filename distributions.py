import torch
from torch.distributions import MultivariateNormal
from abc import abstractmethod
from probabilities import gaussian_probabilities
from regions import HyperRectangularPartition, HyperRectangle
from typing import Union


class Distributions:

    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_probabilities(self, regions: torch.Tensor):
        pass


class Gaussian(Distributions):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.covariance = cov

    def __call__(self, n_samples: int):
        mvn = MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)
        return mvn.sample((n_samples,))

    def compute_probabilities(self, regions: Union[HyperRectangularPartition, HyperRectangle]):
        return gaussian_probabilities(self.mean, self.covariance, regions)


class GaussianMixture(Distributions):
    def __init__(self, means: torch.Tensor, covariance: torch.Tensor, weights: torch.Tensor):
        self.means = means
        self.covariance = covariance
        self.weights = weights

    def __call__(self, n_samples: int):
        chosen_components = torch.multinomial(self.weights, n_samples, replacement=True)
        gaussian_distributions = MultivariateNormal(self.means, self.covariance)
        samples = gaussian_distributions.sample((n_samples,))
        return samples[torch.arange(n_samples), chosen_components]

    def compute_probabilities(self, regions):
        probs = gaussian_probabilities(self.means, self.covariance, regions)
        return torch.sum(probs * self.weights, dim=1)
