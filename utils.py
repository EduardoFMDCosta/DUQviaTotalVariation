import torch
import itertools
from copy import deepcopy

from distributions import Gaussian
from regions import HyperRectangle, Polytope


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


def compute_inf_sup_kernel(f, covariance, regions, set):
    sup_probs_regions, inf_probs_regions = [], []
    sup_probs_set, inf_probs_set = [], []

    # TODO: suboptimal to create new objects in a for loop (HyperRectangle, Polytope and Gaussian)
    #       this can be at least reduced, e.g. for now, for the sake of readability, the regions 
    #       are twice casted as HyperRectangle

    for lower_from, upper_from in zip(regions.lower[:-1], regions.upper[:-1]):
        sup_prob_regions, sup_prob_set = torch.zeros(len(regions.lower)), torch.zeros(1)
        inf_prob_regions, inf_prob_set = torch.ones(len(regions.lower)), torch.ones(1)

        region_from = HyperRectangle(lower_from, upper_from)
        vertices_from = region_from.get_vertices()
        vertices_image = f(vertices_from)
        image = Polytope(vertices_image)

        for i, (lower_to, upper_to, center_to) in enumerate(zip(regions.lower[:-1], regions.upper[:-1], regions.center[:-1])):
            region_to = HyperRectangle(lower_to, upper_to)
            furthest, closest = image.furthest_and_closest_from(center_to)
            kernel_inf = Gaussian(f(furthest), covariance) 
            kernel_sup = Gaussian(f(closest), covariance) 
            inf_prob_regions[i] = kernel_inf.compute_probabilities(region_to)
            sup_prob_regions[i] = kernel_sup.compute_probabilities(region_to) 
        
        # To the outer shell
        shell = HyperRectangle(lower=regions.shell[0], upper=regions.shell[1])
        furthest, closest = image.furthest_and_closest_from(shell.center)
        kernel_inf = Gaussian(f(closest), covariance)   # closest from center minimizes
        kernel_sup = Gaussian(f(furthest), covariance)  # furthest from center maximizes
        inf_prob_regions[-1] = 1 - kernel_inf.compute_probabilities(shell)
        sup_prob_regions[-1] = 1 - kernel_sup.compute_probabilities(shell) 

        inf_probs_regions.append(inf_prob_regions)
        sup_probs_regions.append(sup_prob_regions)

        # To the unsafe set
        furthest, closest = image.furthest_and_closest_from(set.center)
        kernel_inf = Gaussian(f(furthest), covariance) 
        kernel_sup = Gaussian(f(closest), covariance) 
        inf_prob_set[0] = kernel_inf.compute_probabilities(set)
        sup_prob_set[0] = kernel_sup.compute_probabilities(set) 

        inf_probs_set.append(inf_prob_set)
        sup_probs_set.append(sup_prob_set)
    
    # Inf from outer shell is always zero
    inf_probs_regions.append(torch.zeros(len(regions.lower)))
    inf_probs_set.append(torch.zeros(1))

    # Sup from outer shell to regions
    sup_prob_regions = torch.zeros(len(regions.lower))
    shell = HyperRectangle(lower=regions.shell[0], upper=regions.shell[1])
    vertices_shell = shell.get_vertices()
    vertices_image_shell = f(vertices_shell)
    image_shell = Polytope(vertices_image_shell)   
    for i, (lower_to, upper_to, center_to) in enumerate(zip(regions.lower[:-1], regions.upper[:-1], regions.center[:-1])):
        region_to = HyperRectangle(lower_to, upper_to)
        closest = image_shell.closest_from_outside(center_to)
        kernel_sup = Gaussian(f(closest), covariance) 
        sup_prob_regions[i] = kernel_sup.compute_probabilities(region_to) 
    sup_probs_regions.append(sup_prob_regions) 

    # Sup from outer shell to unsafe set
    sup_prob_set = torch.zeros(1)
    closest = image_shell.closest_from_outside(set.center)
    kernel_sup = Gaussian(f(closest), covariance) 
    sup_prob_set[0] = kernel_sup.compute_probabilities(set)
    sup_probs_set.append(sup_prob_set) 

    inf_probs_regions = torch.stack(inf_probs_regions)
    sup_probs_regions = torch.stack(sup_probs_regions)
    inf_probs_set = torch.tensor(inf_probs_set)
    sup_probs_set = torch.tensor(sup_probs_set)
    return inf_probs_regions, sup_probs_regions, inf_probs_set, sup_probs_set

def compute_kernel_at_locs(f, locs, covariance, regions, set):

    probs_regions, probs_set = [], []

    for loc in locs:
        kernel = Gaussian(f(loc), covariance)

        prob_regions = kernel.compute_probabilities(regions)
        probs_regions.append(prob_regions)

        prob_set = kernel.compute_probabilities(set)
        probs_set.append(prob_set)

    return torch.stack(probs_regions), torch.tensor(probs_set)