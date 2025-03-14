from cmath import isinf
from typing import Union
import torch
import itertools
from copy import deepcopy
from distributions import Gaussian
from dynamics import Dynamics
from optimization import gradient_descent
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

def inf_from_S_to_V(f: Dynamics,
                    covariance: torch.Tensor,
                    S: Union[HyperRectangle, torch.Tensor],
                    V: Union[HyperRectangle, torch.Tensor]):

    # TODO: this is conservative for inf_{R_unbounded} T(R_unbounded)
    if torch.isinf(S.lower).any(): # inf_{R_unbounded} T(R_i) = 0
        return 0.0

    z_inf = gradient_descent(f, covariance, V, S, maximize=False)
    return Gaussian(f(z_inf), covariance).compute_probabilities(V)


def sup_from_S_to_V(f: Dynamics,
                    covariance: torch.Tensor,
                    S: Union[HyperRectangle, torch.Tensor],
                    V: Union[HyperRectangle, torch.Tensor]):

    if torch.isinf(S.lower).any() and torch.isinf(V.lower).any(): # sup_{R_unbounded} T(R_unbounded) = 1
        return 1.0

    z_sup = gradient_descent(f, covariance, V, S, maximize=True)
    return Gaussian(f(z_sup), covariance).compute_probabilities(V)


def get_inf_sup_for_target_set(f, covariance, partition, target_set):
    inf_for_target_set, sup_for_target_set = [], []
    for (lower, upper) in zip(partition.lower, partition.upper):
        S = HyperRectangle(lower, upper)
        inf_for_target_set.append(
            inf_from_S_to_V(f, covariance, S, target_set)
        )
        sup_for_target_set.append(
            sup_from_S_to_V(f, covariance, S, target_set)
        )

    return torch.tensor(inf_for_target_set), torch.tensor(sup_for_target_set)


def get_inf_sup_for_partition(f, covariance, partition):
    inf_for_partition_all, sup_for_partition_all = [], []
    for (lower, upper) in zip(partition.lower, partition.upper):
        S = HyperRectangle(lower, upper)
        inf_for_partition, sup_for_partition = [], []
        for (lower_target, upper_target) in zip(partition.lower, partition.upper):
            target_set = HyperRectangle(lower_target, upper_target)
            inf_for_partition.append(
                inf_from_S_to_V(f, covariance, S, target_set)
            )
            sup_for_partition.append(
                sup_from_S_to_V(f, covariance, S, target_set)
            )
        inf_for_partition_all.append(inf_for_partition)
        sup_for_partition_all.append(sup_for_partition)

    return torch.tensor(inf_for_partition_all), torch.tensor(sup_for_partition_all)



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
            kernel_inf = Gaussian(furthest, covariance) 
            kernel_sup = Gaussian(closest, covariance) 
            inf_prob_regions[i] = kernel_inf.compute_probabilities(region_to)
            sup_prob_regions[i] = kernel_sup.compute_probabilities(region_to) 
        
        # To the outer shell
        shell = regions.shell
        furthest, closest = image.furthest_and_closest_from(shell.center)
        kernel_inf = Gaussian(closest, covariance)   # closest from center minimizes
        kernel_sup = Gaussian(furthest, covariance)  # furthest from center maximizes
        inf_prob_regions[-1] = 1 - kernel_inf.compute_probabilities(shell)
        sup_prob_regions[-1] = 1 - kernel_sup.compute_probabilities(shell) 

        inf_probs_regions.append(inf_prob_regions)
        sup_probs_regions.append(sup_prob_regions)

        # To the unsafe set
        furthest, closest = image.furthest_and_closest_from(set.center)
        kernel_inf = Gaussian(furthest, covariance) 
        kernel_sup = Gaussian(closest, covariance) 

        inf_prob_set[0] = kernel_inf.compute_probabilities(set)
        sup_prob_set[0] = kernel_sup.compute_probabilities(set) 

        inf_probs_set.append(inf_prob_set)
        sup_probs_set.append(sup_prob_set)

    # Handle shell; TODO: the following can be removed if the function is optimized (initialized twice)
    shell = regions.shell
    vertices_shell = shell.get_vertices()
    vertices_image_shell = f(vertices_shell)
    image_shell = Polytope(vertices_image_shell)   
    
    # Inf from outer shell is always zero, except to outer shell
    inf_prob_regions = torch.zeros(len(regions.lower))
    # Inf from outer shell to outer shell
    closest = image_shell.closest_from_outside(shell.center)
    kernel_inf = Gaussian(closest, covariance)
    inf_prob_regions[-1] = 1 - kernel_inf.compute_probabilities(shell)

    inf_probs_regions.append(inf_prob_regions)
    inf_probs_set.append(torch.zeros(1))

    # Sup from outer shell to regions
    sup_prob_regions = torch.zeros(len(regions.lower))
    for i, (lower_to, upper_to, center_to) in enumerate(zip(regions.lower[:-1], regions.upper[:-1], regions.center[:-1])):
        region_to = HyperRectangle(lower_to, upper_to)
        closest = image_shell.closest_from_outside(center_to)
        kernel_sup = Gaussian(closest, covariance) 
        sup_prob_regions[i] = kernel_sup.compute_probabilities(region_to) 
    # Sup from outer shell to outer shell is 1
    sup_prob_regions[-1] = 1
    sup_probs_regions.append(sup_prob_regions) 

    # Sup from outer shell to unsafe set
    sup_prob_set = torch.zeros(1)
    closest = image_shell.closest_from_outside(set.center)
    kernel_sup = Gaussian(closest, covariance) 
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

def o_maximization(coeffs:torch.Tensor, lower_bounds:torch.Tensor, upper_bounds:torch.Tensor):
    # inspired from https://www.baymler.com/IntervalMDP.jl/dev/algorithms/#Efficient-value-iteration
    order = torch.argsort(-coeffs)
    p = deepcopy(lower_bounds)
    rem = 1 - sum(p)
    gap = upper_bounds - p
    cumgap = torch.cumsum(gap[order], dim=0)
    for idx, o in enumerate(order):
        rem_state = max(rem - cumgap[idx] + gap[o], 0)
        if gap[o] < rem_state:
            p[o] += gap[o]
        else: 
            p[o] += rem_state
            break
    return p