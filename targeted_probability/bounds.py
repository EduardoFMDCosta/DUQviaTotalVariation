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

    cov_noise = noise_distribution.covariance
    mixture_probs_partition = mixture.compute_probabilities(partition)

    # Compute alpha and beta for target
    inf_kernel_target = bound_transition_kernel(f, partition, target, cov_noise, supremum=False).squeeze()
    sup_kernel_target = bound_transition_kernel(f, partition, target, cov_noise, supremum=True).squeeze()
    kernel_at_locs_target = transition_kernel(f, partition.locs, cov_noise, target).squeeze()

    # Compute bounds for target
    p_min = o_maximization(- inf_kernel_target, mixture_probs_partition + bounds_partition.lb_delta, mixture_probs_partition + bounds_partition.ub_delta)
    p_max = o_maximization(sup_kernel_target, mixture_probs_partition + bounds_partition.lb_delta, mixture_probs_partition + bounds_partition.ub_delta)

    lb_delta = torch.dot(inf_kernel_target, p_min) - torch.dot(kernel_at_locs_target, mixture_probs_partition)
    ub_delta = torch.dot(sup_kernel_target, p_max) - torch.dot(kernel_at_locs_target, mixture_probs_partition)

    bounds = TargetedBounds(lb_delta, ub_delta)

    return bounds

