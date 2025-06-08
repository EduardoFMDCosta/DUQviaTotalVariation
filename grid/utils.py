import torch
import itertools
import bound_propagation as bp
from grid.regions import HyperRectangle
from dynamics.dynamics import Dynamics

from interval_bp import factory

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
                 grid_size: int):

    d = lower.shape[0]
    k = int(round(grid_size ** (1 / d)))
    assert k ** d == grid_size, "grid_size must be a perfect power of the number of dimensions"

    # Generate grid split points per dimension
    grids = [torch.linspace(lower[i], upper[i], steps=k + 1) for i in range(d)]

    # Get all combinations of lower corner indices
    corner_indices = list(itertools.product(*(range(k) for _ in range(d))))

    # Generate lower and upper bounds
    lowers = []
    uppers = []
    for idx in corner_indices:
        sub_lower = torch.tensor([grids[i][idx[i]] for i in range(d)])
        sub_upper = torch.tensor([grids[i][idx[i] + 1] for i in range(d)])
        lowers.append(sub_lower)
        uppers.append(sub_upper)

    grid = HyperRectangle(torch.stack(lowers), torch.stack(uppers))

    return grid

def get_high_prob_set(mixture):

    samples = mixture(5000)

    lower = samples.min(dim=0).values.unsqueeze(0)
    upper = samples.max(dim=0).values.unsqueeze(0)

    return HyperRectangle(lower, upper)

def propagate_high_prob_set(f: Dynamics,
                            hpr,
                            noise_distribution):
    net = factory.build(f)
    propagated_hpr = net.ibp(bp.HyperRectangle(lower=hpr.lower, upper=hpr.upper))

    radius = 3 * torch.sqrt(torch.diagonal(noise_distribution.covariance_matrix)).unsqueeze(0)

    return HyperRectangle(propagated_hpr.lower-radius, propagated_hpr.upper+radius) # Minkowski sum