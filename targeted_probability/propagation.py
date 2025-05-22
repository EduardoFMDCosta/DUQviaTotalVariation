import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import LinearDynamics, factory
from grid.regions import HyperRectangle, HyperRectangularPartition
from grid.utils import get_shell, get_shell_loc, uniform_grid
from plotting.plotting import plot_confidence_interval
from targeted_probability.transition_kernel import bound_transition_kernel, transition_kernel
from targeted_probability.utils import o_maximization

SCATTER = True

if __name__ == '__main__':
    torch.manual_seed(0)

    # System parameters
    A = torch.Tensor(
        [
            [0.54, 0.10],
            [0.05, 0.42]
        ])
    f = LinearDynamics(A)

    # Initial distribution
    mean_initial = torch.Tensor([1., 1.])
    cov_initial = torch.diag(torch.Tensor([0.002, 0.002]))
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = torch.diag(torch.Tensor([1.0, 1.0]))
    noise_distribution = Gaussian(mean_noise, cov_noise)

    n_samples = 5000

    # Define partition
    samples = initial_distribution(n_samples)
    shell = get_shell(samples)
    loc_shell = get_shell_loc(shell)

    if SCATTER:
        fig, ax = plt.subplots()
        rect = Rectangle(
            (shell[0, 0], shell[0, 1]),
            shell[1, 0] - shell[0, 0],
            shell[1, 1] - shell[0, 1],
            fill = False
        )
        ax.add_patch(rect)
        samples_cp = samples
        ax.scatter(samples_cp[:, 0], samples_cp[:, 1])

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([2.0, 2.0]).unsqueeze(0), torch.tensor([2.5, 2.5]).unsqueeze(0))

    # Define initial partition
    inner_locs = uniform_grid(shell[0], shell[1], 16)
    partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

    # Compute initial distribution
    approx_distribution = deepcopy(initial_distribution)
    approx_probs = approx_distribution.compute_probabilities(partition)
    approx_unsafe = approx_distribution.compute_probabilities(unsafe_set)

    # Set initial alphas and betas to zero
    alphas_regions, betas_regions = torch.zeros(len(partition.locs)), torch.zeros(len(partition.locs))
    alpha_unsafe_set, beta_unsafe_set = torch.zeros(1), torch.zeros(1)

    # For plotting
    lbs, ts, aps, ubs = [], [], [], []
    lbs.append(alpha_unsafe_set)
    aps.append(approx_unsafe)
    ubs.append(beta_unsafe_set)

    for t in range(5):
        # Compute alpha and beta for unsafe set
        inf_kernel_unsafe_set = bound_transition_kernel(f, partition, unsafe_set, cov_noise, supremum=False).squeeze()
        sup_kernel_unsafe_set = bound_transition_kernel(f, partition, unsafe_set, cov_noise, supremum=True).squeeze()
        kernel_at_locs_unsafe_set = transition_kernel(f, partition.locs, unsafe_set, cov_noise).squeeze()

        # Compute bounds for unsafe set
        p_min = o_maximization(- inf_kernel_unsafe_set, approx_probs + alphas_regions, approx_probs + betas_regions)
        p_max = o_maximization(sup_kernel_unsafe_set, approx_probs + alphas_regions, approx_probs + betas_regions)

        alpha_unsafe_set = torch.dot(inf_kernel_unsafe_set, p_min) - torch.dot(kernel_at_locs_unsafe_set, approx_probs)
        beta_unsafe_set = torch.dot(sup_kernel_unsafe_set, p_max) - torch.dot(kernel_at_locs_unsafe_set, approx_probs)

        # Define new partition
        samples = approx_distribution(n_samples)
        shell = get_shell(samples)
        new_inner_locs = uniform_grid(shell[0], shell[1], 16)
        new_partition = HyperRectangularPartition(new_inner_locs, loc_shell, shell)
        refined_partition = deepcopy(new_partition)

        if SCATTER:
            rect = Rectangle(
                (shell[0, 0], shell[0, 1]),
                shell[1, 0] - shell[0, 0],
                shell[1, 1] - shell[0, 1],
                fill = False
            )
            ax.add_patch(rect)
            samples_cp = f(samples) + noise_distribution(n_samples)
            ax.scatter(samples_cp[:, 0], samples_cp[:, 1])


        # print(f'Locs before refinement for t={t}: {new_partition.locs}')

        for refinement in range(1):
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

        # Update approximation
        partition = new_partition
        approx_probs = new_approx_probs

        approx_distribution = GaussianMixture(f(partition.locs), cov_noise, approx_probs)

        # Compute approx unsafe and rectify alphas, betas
        approx_unsafe = approx_distribution.compute_probabilities(unsafe_set).squeeze()
        alpha_unsafe_set = torch.clamp(alpha_unsafe_set, - approx_unsafe, 1 - approx_unsafe)
        beta_unsafe_set = torch.clamp(beta_unsafe_set, - approx_unsafe, 1 - approx_unsafe)

        lbs.append(alpha_unsafe_set)
        aps.append(approx_unsafe)
        ubs.append(beta_unsafe_set)

    if SCATTER:
        ax.grid(True)
        plt.show()

    lbs, aps, ubs = torch.tensor(lbs), torch.tensor(aps), torch.tensor(ubs)
    print(f'alphas for unsafe set: {lbs}')
    print(f'betas for unsafe set: {ubs}')
    plot_confidence_interval(aps, lbs, ubs)