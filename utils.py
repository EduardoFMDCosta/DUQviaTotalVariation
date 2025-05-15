from typing import Union
import torch
from copy import deepcopy
from distributions import Gaussian
from dynamics import Dynamics, factory
from optimization import gradient_descent
from distributions.probabilities import kernel_probs_given_optimal_means, gaussian_probabilities
from grid.regions import HyperRectangle, Polytope, HyperRectangularPartition
import bound_propagation as bp

def get_shell(samples: torch.Tensor, eps:float | None = None):

    if eps is None: 
        eps = 3 * torch.std(samples, dim = 0)

    min_point = torch.min(samples, dim=0).values - eps
    max_point = torch.max(samples, dim=0).values + eps

    return torch.stack((min_point, max_point), dim=0)

def get_shell_loc(macro_region: torch.Tensor):
    dimensions = macro_region.size(1)

    # Calculate max and min coordinates along each dimension
    max_coords, _ = torch.max(macro_region, dim=0)
    min_coords, _ = torch.min(macro_region, dim=0)

    # Choose a point close to an arbitrary face
    outer_point = torch.where(torch.arange(dimensions) != dimensions - 1, (min_coords + max_coords) / 2, max_coords + 5.0)
    return outer_point


def uniform_grid(lower: torch.Tensor,
                 upper: torch.Tensor,
                 n: int):

    d = lower.shape[0]
    equidistant_spaces = [torch.linspace(lower[i], upper[i], steps=n) for i in range(d)]
    mesh = torch.meshgrid(*equidistant_spaces, indexing="ij")
    grid = torch.stack(mesh, dim=-1).reshape(-1, d)

    return grid