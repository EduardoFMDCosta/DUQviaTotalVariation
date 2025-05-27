import torch
from typing import Union
from distributions.distributions import GaussianMixture, Gaussian
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangle, HyperRectangularPartition
from targeted_probability.transition_kernel import bound_transition_kernel, transition_kernel
from targeted_probability.utils import o_maximization

class TargetedBounds:
    def __init__(self,
                 lb_delta: torch.Tensor,
                 ub_delta: torch.Tensor):

        self.lb_delta = lb_delta
        self.ub_delta = ub_delta

def compute_targeted_bound(f: Dynamics,
                           mixture: Union[Gaussian, GaussianMixture],
                           noise_distribution: Gaussian,
                           partition: HyperRectangularPartition,
                           bounds_partition: TargetedBounds,
                           target: Union[HyperRectangle, HyperRectangularPartition]):

    num_target_sets = target.lower.shape[0]
    cov_noise = noise_distribution.covariance
    mixture_probs_partition = mixture.compute_probabilities(partition)
    mixture_probs_target = mixture.compute_probabilities(target)

    # Compute alpha and beta for target
    inf_kernel_target = bound_transition_kernel(f, partition, target, cov_noise, supremum=False)
    sup_kernel_target = bound_transition_kernel(f, partition, target, cov_noise, supremum=True)
    kernel_at_locs_target = transition_kernel(f, partition.locs, cov_noise, target)

    # Compute bounds for target
    p_min = o_maximization(- inf_kernel_target, mixture_probs_partition + bounds_partition.lb_delta, mixture_probs_partition + bounds_partition.ub_delta)
    p_max = o_maximization(sup_kernel_target, mixture_probs_partition + bounds_partition.lb_delta, mixture_probs_partition + bounds_partition.ub_delta)

    lb_delta_components = inf_kernel_target * p_min - kernel_at_locs_target * mixture_probs_partition.unsqueeze(1).expand(-1, num_target_sets)
    ub_delta_components = sup_kernel_target * p_max - kernel_at_locs_target * mixture_probs_partition.unsqueeze(1).expand(-1, num_target_sets)

    lb_delta = lb_delta_components.sum(dim=0)
    ub_delta = ub_delta_components.sum(dim=0)

    # Clamp as bounds need to lead to valid probs
    lb_delta.clamp_(min = -mixture_probs_target)
    ub_delta.clamp_(max = 1 - mixture_probs_target)

    contributions = ub_delta_components.sum(dim=1)
    bounds = TargetedBounds(lb_delta, ub_delta)

    return contributions, bounds

def get_objective_targeted(f: Dynamics,
                           noise_distribution: Gaussian,
                           target: HyperRectangle):
    def objective_targeted_bound(partition: HyperRectangularPartition):
        return bound_transition_kernel(f, partition, target, noise_distribution.covariance, supremum=True).squeeze(), 0.0
    return objective_targeted_bound

