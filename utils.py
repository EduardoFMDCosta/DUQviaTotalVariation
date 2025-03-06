import torch
import itertools
from copy import deepcopy

from distributions import Gaussian


def get_shell(samples: torch.Tensor):

    min_point = torch.min(samples, dim=0).values
    max_point = torch.max(samples, dim=0).values

    return torch.stack((min_point, max_point), dim=0)

def get_shell_loc(macro_region: torch.Tensor):
    dimensions = macro_region.size(1)

    # Calculate max and min coordinates along each dimension
    max_coords, _ = torch.max(macro_region, dim=0)
    min_coords, _ = torch.min(macro_region, dim=0)

    # Choose a point close to an arbitrary face
    outer_point = torch.where(torch.arange(dimensions) != dimensions - 1, (min_coords + max_coords) / 2, max_coords + 1e-1)
    return outer_point


def uniform_grid(lower: torch.Tensor,
                 upper: torch.Tensor,
                 n: int):

    d = lower.shape[0]
    equidistant_spaces = [torch.linspace(lower[i], upper[i], steps=n) for i in range(d)]
    mesh = torch.meshgrid(*equidistant_spaces, indexing="ij")
    grid = torch.stack(mesh, dim=-1).reshape(-1, d)

    return grid


def generate_hypercube_vertices(lower_point, upper_point):
    d = lower_point.shape[0]  # Number of dimensions
    vertices = torch.tensor(list(itertools.product(*zip(lower_point, upper_point))))
    return vertices

def compute_sup_inf_kernel(f, covariance, regions, set, N = 10000, low = -20, high=20):
    regions_to = deepcopy(regions)

    sup_probs_regions, inf_probs_regions = [], []
    sup_probs_set, inf_probs_set = [], []

    for lower_point_from, upper_point_from in zip(regions.lower, regions.upper):

        sup_prob_regions, sup_prob_set = torch.zeros(len(regions.lower)), torch.zeros(1)
        inf_prob_regions, inf_prob_set = torch.ones(len(regions.lower)), torch.ones(1)

        lower_sample_from = deepcopy(lower_point_from)
        lower_sample_from[torch.isinf(lower_sample_from)] = low
        upper_sample_from = deepcopy(upper_point_from)
        upper_sample_from[torch.isinf(upper_sample_from)] = high
        samples = torch.distributions.uniform.Uniform(lower_sample_from, upper_sample_from).sample([N]) 
        
        for (i, (lower_point_to, upper_point_to)) in enumerate(zip(regions_to.lower, regions_to.upper)):
            # take higher for unbounded
            lower_point_to[torch.isinf(lower_point_to)] = 2 * low
            upper_point_to[torch.isinf(upper_point_to)] = 2 * high  

        for sample in samples:
            kernel = Gaussian(f(sample), covariance) 
            prob_regions = kernel.compute_probabilities(regions_to)
            prob_set = kernel.compute_probabilities(set)
            sup_prob_regions = torch.maximum(sup_prob_regions, prob_regions)
            inf_prob_regions = torch.minimum(inf_prob_regions, prob_regions)
            sup_prob_set = torch.maximum(sup_prob_set, prob_set)
            inf_prob_set = torch.minimum(inf_prob_set, prob_set)

        sup_probs_regions.append(sup_prob_regions)
        inf_probs_regions.append(inf_prob_regions)

        sup_probs_set.append(sup_prob_set)
        inf_probs_set.append(inf_prob_set)

    sup_probs_regions = torch.stack(sup_probs_regions)
    inf_probs_regions = torch.stack(inf_probs_regions)
    sup_probs_set = torch.tensor(sup_probs_set)
    inf_probs_set = torch.tensor(inf_probs_set)

    return sup_probs_regions, inf_probs_regions, sup_probs_set, inf_probs_set

def compute_kernel_at_locs(f, locs, covariance, regions, set):

    probs_regions, probs_set = [], []

    for loc in locs:
        kernel = Gaussian(f(loc), covariance)

        prob_regions = kernel.compute_probabilities(regions)
        probs_regions.append(prob_regions)

        prob_set = kernel.compute_probabilities(set)
        probs_set.append(prob_set)

    return torch.stack(probs_regions), torch.tensor(probs_set)