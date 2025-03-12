import torch
from copy import deepcopy

from dynamics import LinearDynamics
from distributions import Gaussian, GaussianMixture
from experiments import approximation_scheme_tv
from regions import HyperRectangle, HyperRectangularPartition
from utils import compute_inf_sup_kernel, get_shell, get_shell_loc, uniform_grid, compute_kernel_at_locs, o_maximization

import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)

    # System parameters
    A = torch.Tensor(
        [
            [0.84, 0.10],
            [0.05, 0.72]
        ])
    f = LinearDynamics(A)

    # Initial distribution
    mean_initial = torch.Tensor([4, 4])
    cov_initial = torch.eye(2)
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = 0.5 * torch.eye(2)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Define partition
    # n_samples = 100
    # samples = initial_distribution(n_samples)
    # shell = get_shell(samples)
    shell = torch.tensor([
        [-6.,   -6.], 
        [6.,    6. ]
    ])
    loc_shell = get_shell_loc(shell)
    inner_locs = uniform_grid(shell[0], shell[1], 10)

    partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5]))
    
    # Compute sup/inf_{z \in Ri} T(Rj | z) and sup/inf_{z \in Ri} T(U | z)
    inf_kernel_regions, sup_kernel_regions, inf_kernel_unsafe_set, sup_kernel_unsafe_set = compute_inf_sup_kernel(
        f, 
        cov_noise, 
        partition, 
        unsafe_set
    )
    
    locs = partition.locs
    kernel_at_locs_regions, kernel_at_locs_unsafe_set = compute_kernel_at_locs(f, locs, cov_noise, partition, unsafe_set)

    # Initialize approximation distribution
    approx_distribution = deepcopy(initial_distribution)
    alphas_regions, betas_regions = torch.zeros(len(locs)), torch.zeros(len(locs))
    alpha_unsafe_set, beta_unsafe_set = torch.zeros(1), torch.zeros(1)
    
    # For plotting
    lbs, ts, aps, ubs = [], [], [], []

    #Run simulation
    mean_k = deepcopy(mean_initial)
    cov_k = deepcopy(cov_initial)
    means_gmm = f(locs)
    for t in range(30):
        # Compute the true distribution for comparaison
        mean_k = torch.matmul(A, mean_k) + mean_noise
        cov_k = torch.matmul(A, cov_k)
        cov_k = torch.matmul(cov_k, torch.t(A)) + cov_noise
        dis_k = Gaussian(mean_k, cov_k)

        # Update approximation
        approx_probs = approx_distribution.compute_probabilities(partition)
        covs_noise = cov_noise.unsqueeze(0).expand(means_gmm.size(0), -1, -1)
        approx_distribution = GaussianMixture(means_gmm, covs_noise, approx_probs)

        # Compute true error
        true_unsafe = dis_k.compute_probabilities(unsafe_set)
        approx_unsafe = approx_distribution.compute_probabilities(unsafe_set)
        diff_unsafe = true_unsafe - approx_unsafe

        # Compute bounds
        p_min = o_maximization(- inf_kernel_unsafe_set, approx_probs + alphas_regions, approx_probs + betas_regions)
        p_max = o_maximization(sup_kernel_unsafe_set, approx_probs + alphas_regions, approx_probs + betas_regions)

        alpha_unsafe_set = torch.dot(inf_kernel_unsafe_set, p_min) - torch.dot(kernel_at_locs_unsafe_set, approx_probs)
        beta_unsafe_set = torch.dot(sup_kernel_unsafe_set, p_max) - torch.dot(kernel_at_locs_unsafe_set, approx_probs)

        next_alphas_regions, next_betas_regions = torch.zeros(len(locs)), torch.zeros(len(locs))
        for i, (inf_kernel_region, sup_kernel_region) in enumerate(zip(inf_kernel_regions.T, sup_kernel_regions.T)):
            p_min = o_maximization(- inf_kernel_region, approx_probs + alphas_regions, approx_probs + betas_regions)
            p_max = o_maximization(sup_kernel_region, approx_probs + alphas_regions, approx_probs + betas_regions)
            next_alphas_regions[i] += torch.dot(inf_kernel_region, p_min)
            next_betas_regions[i] += torch.dot(sup_kernel_region, p_max)
        next_alphas_regions -= kernel_at_locs_regions.T @ approx_probs
        next_betas_regions -= kernel_at_locs_regions.T @ approx_probs
        alphas_regions = next_alphas_regions
        betas_regions = next_betas_regions

        lbs.append(approx_unsafe + alpha_unsafe_set)
        ts.append(true_unsafe)
        aps.append(approx_unsafe)
        ubs.append(approx_unsafe + beta_unsafe_set)

        print("(t = {}) alpha = {}, true = {}, beta = {}".format(t, alpha_unsafe_set, diff_unsafe[0], beta_unsafe_set))

lbs, ts, aps, ubs = torch.tensor(lbs), torch.tensor(ts), torch.tensor(aps), torch.tensor(ubs)

plt.ylim(0 - 0.005, torch.max(ubs) + 0.05)
plt.fill_between(range(30), lbs, ubs, color="lightgrey", label = "Our bounds")
plt.plot(range(30), ts, label = "True unsafe probability", color = "green")
plt.plot(range(30), aps, label = "Approximated unsafe probability", color = "red")
plt.legend(loc = "upper right")
plt.show()