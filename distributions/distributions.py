import torch
from abc import abstractmethod
from typing import Union
from torch.distributions import MultivariateNormal
from distributions.utils import gaussian_probabilities
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
