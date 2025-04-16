import torch
from copy import deepcopy
from dynamics import LinearDynamics, SinusoidalDynamics, DubinsDynamics
from distributions import Gaussian, GaussianMixture
from plotting import plot_interval
from regions import HyperRectangle, HyperRectangularPartition
from utils import get_shell_loc, uniform_grid, compute_kernel_at_locs, o_maximization, bound_transition_kernel, transition_kernel

if __name__ == '__main__':
    torch.manual_seed(0)

    f = DubinsDynamics()

    # Initial distribution
    mean_initial = torch.Tensor([0, 0, 0])
    cov_initial = torch.diag(torch.Tensor([0.005, 0.005, 0.001]))
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0, 0])
    cov_noise = torch.diag(torch.Tensor([0.06, 0.06, 0.01]))
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Define partition
    shell = torch.tensor([
        [-4., -4., -1.],
        [10., 10., 10.]
    ])
    loc_shell = get_shell_loc(shell)

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([3.0, 1.0, -torch.inf]).unsqueeze(0), torch.tensor([5.0, 4.0, torch.inf]).unsqueeze(0))

    # Define initial partition
    inner_locs = uniform_grid(shell[0], shell[1], 3)
    partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

    # Compute initial distribution
    approx_distribution = deepcopy(initial_distribution)
    approx_probs = approx_distribution.compute_probabilities(partition)

    # Set initial alphas and betas to zero
    alphas_regions, betas_regions = torch.zeros(len(partition.locs)), torch.zeros(len(partition.locs))
    alpha_unsafe_set, beta_unsafe_set = torch.zeros(1), torch.zeros(1)

    # For plotting
    lbs, ts, aps, ubs = [], [], [], []

    for t in range(10):

        # Compute alpha and beta for unsafe set
        inf_kernel_unsafe_set = bound_transition_kernel(f, partition, unsafe_set, cov_noise, supremum=False).squeeze()
        sup_kernel_unsafe_set = bound_transition_kernel(f, partition, unsafe_set, cov_noise, supremum=True).squeeze()
        kernel_at_locs_unsafe_set = transition_kernel(f, partition.locs, unsafe_set, cov_noise).squeeze()

        # Compute approx unsafe
        approx_unsafe = approx_distribution.compute_probabilities(unsafe_set).squeeze()

        # Compute bounds for unsafe set
        p_min = o_maximization(- inf_kernel_unsafe_set, approx_probs + alphas_regions, approx_probs + betas_regions)
        p_max = o_maximization(sup_kernel_unsafe_set, approx_probs + alphas_regions, approx_probs + betas_regions)       

        alpha_unsafe_set = torch.dot(inf_kernel_unsafe_set, p_min) - torch.dot(kernel_at_locs_unsafe_set, approx_probs)
        alpha_unsafe_set = torch.clamp(alpha_unsafe_set, - approx_unsafe, 1 - approx_unsafe)
        beta_unsafe_set = torch.dot(sup_kernel_unsafe_set, p_max) - torch.dot(kernel_at_locs_unsafe_set, approx_probs)
        beta_unsafe_set = torch.clamp(beta_unsafe_set, - approx_unsafe, 1 - approx_unsafe)

        # Define new partition
        new_inner_locs = uniform_grid(shell[0], shell[1], 3)
        new_partition = HyperRectangularPartition(new_inner_locs, loc_shell, shell)
        refined_partition = deepcopy(new_partition)

        # print(f'Locs before refinement for t={t}: {new_partition.locs}')

        for refinement in range(3):
            new_partition = refined_partition
            new_approx_probs = approx_distribution.compute_probabilities(new_partition)

            inf_kernel_regions = bound_transition_kernel(f, partition, new_partition, cov_noise, supremum=False)
            sup_kernel_regions = bound_transition_kernel(f, partition, new_partition, cov_noise, supremum=True)
            kernel_at_locs_regions = transition_kernel(f, partition.locs, new_partition, cov_noise)

            next_alphas_regions, next_betas_regions = torch.zeros(len(new_partition.locs)), torch.zeros(len(new_partition.locs))

            for i, (inf_kernel_region, sup_kernel_region) in enumerate(zip(inf_kernel_regions.T, sup_kernel_regions.T)):
                p_min = o_maximization(- inf_kernel_region, approx_probs + alphas_regions, approx_probs + betas_regions)
                p_max = o_maximization(sup_kernel_region, approx_probs + alphas_regions, approx_probs + betas_regions)
                next_alphas_regions[i] += torch.dot(inf_kernel_region, p_min)
                next_betas_regions[i] += torch.dot(sup_kernel_region, p_max)

            next_alphas_regions -= kernel_at_locs_regions.T @ approx_probs
            next_alphas_regions = torch.clamp(next_alphas_regions, - new_approx_probs, 1 - new_approx_probs)

            next_betas_regions -= kernel_at_locs_regions.T @ approx_probs
            next_betas_regions = torch.clamp(next_betas_regions, - new_approx_probs, 1 - new_approx_probs)

            refined_partition = new_partition.refine(next_betas_regions, threshold=1e-3, pareto=1.0)

        # print(f'Locs after refinement for t={t}: {new_partition.locs}')
        print(f'Size of locs before refinement for t={t}: {new_partition.locs.shape}')

        alphas_regions = next_alphas_regions
        betas_regions = next_betas_regions

        lbs.append(alpha_unsafe_set)
        aps.append(approx_unsafe)
        ubs.append(beta_unsafe_set)

        # Update approximation
        partition = new_partition
        approx_probs = new_approx_probs

        approx_distribution = GaussianMixture(f(partition.locs), cov_noise, approx_probs)

    lbs, aps, ubs = torch.tensor(lbs), torch.tensor(aps), torch.tensor(ubs)
    print(f'alphas for unsafe set: {lbs}')
    print(f'betas for unsafe set: {ubs}')
    plot_interval(aps, lbs, ubs)
