import torch
from typing import Union
from distributions.distributions import GaussianMixture, Gaussian
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangle, HyperRectangularPartition
from imdp.transition_kernel import bound_transition_kernel, transition_kernel
from imdp.utils import o_maximization

PRECISION = torch.finfo(torch.float32).eps

class TargetedBounds:
    def __init__(self,
                 lb_delta: torch.Tensor,
                 ub_delta: torch.Tensor):

        self.lb_delta = lb_delta
        self.ub_delta = ub_delta

def compute_targeted_bound(f: Dynamics,
                           partition_probs: torch.Tensor,
                           noise_distribution: Gaussian,
                           partition: HyperRectangularPartition,
                           bounds_partition: TargetedBounds,
                           target: Union[HyperRectangle, HyperRectangularPartition]):

    num_target_sets = target.lower.shape[0]
    cov_noise = noise_distribution.covariance_matrix

    # Compute alpha and beta for target
    inf_kernel_target = bound_transition_kernel(f, cov_noise, partition, target, supremum=False)
    sup_kernel_target = bound_transition_kernel(f, cov_noise, partition, target, supremum=True)
    kernel_at_locs_target = transition_kernel(f, partition.locs, cov_noise, target)

    # Sanity check
    assert (kernel_at_locs_target <= sup_kernel_target + 2*PRECISION).all(), 'Kernel supremum is not greater than kernel at center locations'
    assert (kernel_at_locs_target >= inf_kernel_target - PRECISION).all(), 'Kernel infimum is not smaller than kernel at center locations'

    # Compute bounds for target
    p_min = o_maximization(- inf_kernel_target, partition_probs + bounds_partition.lb_delta, partition_probs + bounds_partition.ub_delta)
    p_max = o_maximization(sup_kernel_target, partition_probs + bounds_partition.lb_delta, partition_probs + bounds_partition.ub_delta)

    lb_delta_components = inf_kernel_target * p_min - kernel_at_locs_target * partition_probs.unsqueeze(1).expand(-1, num_target_sets)
    ub_delta_components = sup_kernel_target * p_max - kernel_at_locs_target * partition_probs.unsqueeze(1).expand(-1, num_target_sets)

    lb_delta = lb_delta_components.sum(dim=0)
    ub_delta = ub_delta_components.sum(dim=0)

    assert (lb_delta <= 10*PRECISION).all(), 'Lower bound cannot be positive'
    assert (ub_delta >= -10*PRECISION).all(), 'Upper bound cannot be negative'

    contributions = ub_delta_components.sum(dim=1)
    bounds = TargetedBounds(lb_delta, ub_delta)

    return contributions, bounds

def get_objective_targeted(mixture: Union[Gaussian, GaussianMixture]):
    def objective_targeted_bound(partition: HyperRectangularPartition):
        return mixture.compute_probabilities(partition), None
    return objective_targeted_bound

