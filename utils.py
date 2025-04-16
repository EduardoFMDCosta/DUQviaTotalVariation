from typing import Union
import torch
import itertools
from copy import deepcopy
from distributions import Gaussian
from dynamics import Dynamics, factory
from optimization import gradient_descent, project_to_closest_face
from probabilities import kernel_probs_given_optimal_means, gaussian_probabilities
from regions import HyperRectangle, Polytope, HyperRectangularPartition
import bound_propagation as bp

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

def bound_transition_kernel(f: Dynamics,
                            partition: HyperRectangularPartition,
                            target: Union[HyperRectangularPartition, HyperRectangle],
                            covariance: torch.Tensor,
                            supremum: bool = True):

    # Get parameters
    shell = partition.shell
    target_is_partition = isinstance(target, HyperRectangularPartition)

    # Replace the unbounded by the shell (computations are made accordingly later)
    from_lower = partition.lower
    from_lower[-1] = shell.lower
    from_upper = partition.upper
    from_upper[-1] = shell.upper

    # If target is partition, we replace the unbounded by the shell (computations are made accordingly later)
    target_lower = target.lower
    target_upper = target.upper
    target_center = target.center
    if target_is_partition:
        target_lower[-1] = shell.lower
        target_upper[-1] = shell.upper
        target_center[-1] = shell.center

    # Create hyper-cubic envelope for f(R_from)
    net = factory.build(f)
    ibp = net.ibp(bp.HyperRectangle(from_lower, from_upper))
    from_lower = ibp.lower
    from_upper = ibp.upper

    # Project center (target_center) to cube (from_set)
    projection = torch.clamp(target_center.unsqueeze(0),
                            min=from_lower.unsqueeze(1),
                            max=from_upper.unsqueeze(1))

    # Get farther point from from_set to shell
    farthest_from_shell = torch.where(torch.abs(from_lower - shell.center) > torch.abs(from_upper - shell.center),
                                      from_lower,
                                      from_upper)

    # Get farther point from from_set to target_center
    dist_lower = torch.abs(target_center[None, :, :] - from_lower[:, None, :])
    dist_upper = torch.abs(target_center[None, :, :] - from_upper[:, None, :])
    farther = torch.where(dist_lower > dist_upper, from_lower[:, None, :], from_upper[:, None, :])

    if supremum:
        optimal_means = projection
        if target_is_partition:
            optimal_means[:, -1, :] = farthest_from_shell

    else: # Compute inf
        optimal_means = farther

    probs = kernel_probs_given_optimal_means(optimal_means, covariance, target_lower, target_upper, target_is_partition)
    probs[-1, :] = 0.0 # R_unbounded as absorbing state

    # Assuming unbounded region to be absorbing state
    if isinstance(target, HyperRectangularPartition):
        probs[-1, -1] = 1.0

    return probs


def transition_kernel(f: Dynamics,
                      locs: torch.tensor,
                      target: Union[HyperRectangularPartition, HyperRectangle],
                      covariance: torch.Tensor):
    kernel_means = f(locs)
    probs = gaussian_probabilities(kernel_means, covariance, target)
    probs[-1, :] = 0.0 # R_unbounded as absorbing state

    if isinstance(target, HyperRectangularPartition):
        probs[-1, -1] = 1.0

    return probs


def both_from_set_to_set_are_unbounded(from_set, to_set):
    return torch.isinf(from_set.lower).any() and torch.isinf(to_set.lower).any()

def only_from_set_is_unbounded(from_set, to_set):
    return torch.isinf(from_set.lower).any() and not torch.isinf(to_set.lower).any()

def only_to_set_is_unbounded(from_set, to_set):
    return torch.isinf(to_set.lower).any() and not torch.isinf(from_set.lower).any()

def optimize_from_set_to_set(f: Dynamics,
                             covariance: torch.Tensor,
                             from_set: HyperRectangle,
                             to_set: HyperRectangle,
                             shell: HyperRectangle = None,
                             minimize: bool = False):

    if minimize:
        if both_from_set_to_set_are_unbounded(from_set, to_set): # inf_{R_inf} T(R_inf)
            return 0.0 #TODO: conservative
        elif only_from_set_is_unbounded(from_set, to_set): # inf_{R_inf} T(R_k) = 0
            return 0.0
        elif only_to_set_is_unbounded(from_set, to_set):
            z_sup = gradient_descent(f, covariance, from_set, shell, minimize=False)
            return 1 - Gaussian(f(z_sup), covariance).compute_probabilities(shell)
        else:
            z_inf = gradient_descent(f, covariance, from_set, to_set, minimize=True)
            return Gaussian(f(z_inf), covariance).compute_probabilities(to_set)

    else:
        if both_from_set_to_set_are_unbounded(from_set, to_set): # sup_{R_inf} T(R_inf) = 1
            return 1.0
        elif only_from_set_is_unbounded(from_set, to_set): # sup_{R_inf} T(R_k)
            return 1.0 #TODO: conservative
        elif only_to_set_is_unbounded(from_set, to_set): # sup_{R_k} T(R_inf)
            z_inf = gradient_descent(f, covariance, from_set, shell, minimize=True)
            return 1 - Gaussian(f(z_inf), covariance).compute_probabilities(shell)
        else:
            z_sup = gradient_descent(f, covariance, from_set, to_set, minimize=False)
            return Gaussian(f(z_sup), covariance).compute_probabilities(to_set)


def get_inf_sup_for_target_set(f, covariance, partition, to_set):
    inf_for_target_set, sup_for_target_set = [], []
    shell = partition.shell
    for (lower, upper) in zip(partition.lower, partition.upper):
        from_set = HyperRectangle(lower, upper)
        inf_for_target_set.append(
            optimize_from_set_to_set(f, covariance, from_set, to_set, shell=shell, minimize=True)
        )
        sup_for_target_set.append(
            optimize_from_set_to_set(f, covariance, from_set, to_set, shell=shell, minimize=False)
        )

    return torch.tensor(inf_for_target_set), torch.tensor(sup_for_target_set)


def get_inf_sup_for_partition(f, covariance, partition):
    inf_for_partition_all, sup_for_partition_all = [], []
    shell = partition.shell
    for (lower, upper) in zip(partition.lower, partition.upper):
        from_set = HyperRectangle(lower, upper)
        inf_for_partition, sup_for_partition = [], []
        for (lower_target, upper_target) in zip(partition.lower, partition.upper):
            to_set = HyperRectangle(lower_target, upper_target)
            inf_for_partition.append(
                optimize_from_set_to_set(f, covariance, from_set, to_set, shell=shell, minimize=True)
            )
            sup_for_partition.append(
                optimize_from_set_to_set(f, covariance, from_set, to_set, shell=shell, minimize=False)
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

    probs_regions = torch.stack(probs_regions)
    probs_regions[-1, :] = 0.0
    probs_regions[-1, -1] = 1.0

    probs_set = torch.tensor(probs_set)

    return probs_regions, probs_set

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