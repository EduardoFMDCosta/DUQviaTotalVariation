from typing import Union
import torch
import torch.distributions as dist
from distributions.utils import gaussian_probabilities
from dynamics.dynamics import Dynamics, factory
from grid.regions import HyperRectangularPartition, HyperRectangle
import bound_propagation as bp

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

def kernel_probs_given_optimal_means(means_for_target: torch.Tensor,
                                     covariance: torch.Tensor,
                                     target_lower: torch.Tensor,
                                     target_upper: torch.Tensor,
                                     target_is_partition: bool):

    normal = dist.Normal(0, 1)
    std_devs = torch.sqrt(torch.diag(covariance))

    lower_std = (target_lower - means_for_target) / std_devs
    upper_std = (target_upper - means_for_target) / std_devs

    lower_std[lower_std.isnan()] = - float('inf') # Q? Should always be negative?
    upper_std[upper_std.isnan()] = float('inf')

    # Compute Gaussian probs for each pair (\barR_from, R_target)
    lower_cdf = normal.cdf(lower_std)
    upper_cdf = normal.cdf(upper_std)

    probs_per_dimension = upper_cdf - lower_cdf
    probs = probs_per_dimension.prod(dim=-1)

    if target_is_partition:
        probs[:, -1] = 1 - probs[:, -1] # As we had computed P(shell)

    return probs


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

    # TODO: handle the NaNs here

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