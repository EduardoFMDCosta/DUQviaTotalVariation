import torch
import itertools
from copy import deepcopy

from distributions import Gaussian


def create_uniform_grid(lower: torch.Tensor, upper: torch.Tensor, n: int):

    d = lower.shape[0]
    equidistant_spaces = [torch.linspace(lower[i], upper[i], steps=n) for i in range(d)]
    mesh = torch.meshgrid(*equidistant_spaces, indexing="ij")
    grid = torch.stack(mesh, dim=-1).reshape(-1, d)

    return grid


def generate_hypercube_vertices(lower_point, upper_point):
    d = lower_point.shape[0]  # Number of dimensions
    vertices = torch.tensor(list(itertools.product(*zip(lower_point, upper_point))))
    return vertices

def compute_sup_inf_kernel(f, covariance, regions, set, N = 5000, low = -1000, high=1000):
    regions_to = deepcopy(regions)

    sup_probs_regions, inf_probs_regions = [], []
    sup_probs_set, inf_probs_set = [], []

    for lower_point_from, upper_point_from in zip(regions.lower, regions.upper):

        sup_prob_regions, sup_prob_set = torch.zeros(len(regions.lower)), torch.zeros(1)
        inf_prob_regions, inf_prob_set = torch.ones(len(regions.lower)), torch.ones(1)

        lower_point_from[torch.isinf(lower_point_from)] = low
        upper_point_from[torch.isinf(upper_point_from)] = high        
        samples = torch.distributions.uniform.Uniform(lower_point_from, upper_point_from).sample([N]) 
        
        for (i, (lower_point_to, upper_point_to)) in enumerate(zip(regions.lower, regions.upper)):
            # take higher for unbounded
            lower_point_to[torch.isinf(lower_point_to)] = 2 * low
            regions_to.set_lower(i, lower_point_to)
            upper_point_to[torch.isinf(upper_point_to)] = 2 * high  
            regions_to.set_upper(i, upper_point_to)

        for sample in samples:
            kernel = Gaussian(f(sample), covariance) 
            prob_regions = kernel.compute_probabilities(regions_to)
            prob_set = kernel.compute_probabilities(set)
            sup_prob_regions = torch.max(sup_prob_regions, prob_regions)
            inf_prob_regions = torch.min(inf_prob_regions, prob_regions)
            sup_prob_set = torch.max(sup_prob_set, prob_set)
            inf_prob_set = torch.min(inf_prob_set, prob_set)

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