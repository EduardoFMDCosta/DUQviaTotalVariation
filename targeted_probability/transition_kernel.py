from typing import Union
import torch
from distributions.utils import gaussian_probabilities
from dynamics.dynamics import Dynamics, factory
from grid.regions import HyperRectangularPartition, HyperRectangle
import bound_propagation as bp
import torch.distributions as dist

def transition_kernel(f: Dynamics,
                      locs: torch.tensor,
                      covariance: torch.Tensor,
                      target: Union[HyperRectangularPartition, HyperRectangle]):

    kernel_means = f(locs)
    probs = gaussian_probabilities(kernel_means, covariance, target)
    probs[-1, :] = 0.0 # R_unbounded as absorbing state

    if isinstance(target, HyperRectangularPartition):
        probs[-1, -1] = 1.0 # R_unbounded as absorbing state

    return probs

def kernel_matrix(optimal_z: torch.Tensor,
                  covariance: torch.Tensor,
                  target: Union[HyperRectangularPartition, HyperRectangle]):

    normal = dist.Normal(0, 1)
    std_devs = torch.sqrt(torch.diag(covariance))

    lower_std = (target.lower - optimal_z) / std_devs
    upper_std = (target.upper - optimal_z) / std_devs

    # Compute Gaussian probs for each pair (\barR_from, R_target)
    lower_cdf = normal.cdf(lower_std)
    upper_cdf = normal.cdf(upper_std)

    probs_per_dimension = upper_cdf - lower_cdf
    probs = probs_per_dimension.prod(dim=-1)

    return probs

def bound_transition_kernel(f: Dynamics,
                            covariance: torch.Tensor,
                            partition: HyperRectangularPartition,
                            target: Union[HyperRectangularPartition, HyperRectangle],
                            supremum: bool):

    net = factory.build(f)

    n, m = partition.lower.shape[0], target.lower.shape[0]

    from_unbounded = torch.zeros(n).bool()
    from_unbounded[-1] = True
    mask_from_unbounded = from_unbounded[:, None].expand(-1, m)

    # Mask checking whether target_j is unbounded (complement of the shell)
    target_unbounded = torch.zeros(m).bool()
    if isinstance(target, HyperRectangularPartition):
        target_unbounded[-1] = True
    mask_target_unbounded = target_unbounded[None, :].expand(n, -1)

    # Initialize kernel bounds (n, m)
    kernel_bounds = torch.zeros(n, m)

    compact_from = bp.HyperRectangle(lower=partition.lower[~from_unbounded], upper=partition.upper[~from_unbounded])
    compact_to = bp.HyperRectangle(lower=target.lower[~target_unbounded], upper=target.upper[~target_unbounded])

    ibp_from = net.ibp(bp.HyperRectangle(compact_from.lower, compact_from.upper))

    # From non-absorbing (thus compact) states, compute sup or inf for compact targets
    if supremum:
        optimal_means = torch.clamp(compact_to.center.unsqueeze(0),
                                 min=ibp_from.lower.unsqueeze(1),
                                 max=ibp_from.upper.unsqueeze(1))
    else:
        dist_lower = torch.abs(compact_to.center[None, :, :] - ibp_from.lower[:, None, :])
        dist_upper = torch.abs(compact_to.center[None, :, :] - ibp_from.upper[:, None, :])
        optimal_means = torch.where(dist_lower > dist_upper, ibp_from.lower[:, None, :], ibp_from.upper[:, None, :])

    kernel_bounds[~mask_from_unbounded & ~mask_target_unbounded] = kernel_matrix(optimal_z=optimal_means,
                                                                                 covariance=covariance,
                                                                                 target=compact_to).reshape(-1)

    # From non-absorbing (thus compact) states, compute sup or inf for unbounded targets
    if mask_target_unbounded.any():
        if supremum:
            dist_lower = torch.abs(target.shell.center[None, :] - ibp_from.lower[:, None, :])
            dist_upper = torch.abs(target.shell.center[None, :] - ibp_from.upper[:, None, :])
            optimal_means = torch.where(dist_lower > dist_upper, ibp_from.lower[:, None, :], ibp_from.upper[:, None, :])
        else:
            optimal_means = torch.clamp(target.shell.center.unsqueeze(0),
                                        min=ibp_from.lower.unsqueeze(1),
                                        max=ibp_from.upper.unsqueeze(1))

        kernel_bounds[~mask_from_unbounded & mask_target_unbounded] = 1 - kernel_matrix(optimal_z=optimal_means,
                                                                                        covariance=covariance,
                                                                                        target=target.shell).reshape(-1)

    if supremum:
        kernel_bounds[mask_from_unbounded] = 1.0
    else:
        kernel_bounds[mask_from_unbounded] = 0.0

    return kernel_bounds