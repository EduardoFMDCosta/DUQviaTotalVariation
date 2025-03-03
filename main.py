import torch
from dynamics import LinearDynamics
from distributions import GaussianMixture, Gaussian
from probability_mass_computation import gaussian_proba_mass_inside_hypercubes
from regions import HyperRectangle, HyperRectangularVoronoiPartition
from copy import deepcopy
from utils import create_uniform_grid, compute_kernel_at_locs, compute_sup_inf_kernel

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
    cov_initial = 0.05 * torch.eye(2)
    distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = 3. * torch.eye(2)

    # Prepare fixed grid
    n = 3
    locs = create_uniform_grid(torch.tensor([0, 0]), torch.tensor([10, 10]), n)
    regions = HyperRectangularVoronoiPartition(locs) # Csreate a Voronoi partition w.r.t. locs

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([3, 2]), torch.tensor([5, 4]))

    # Kernel related quantities
    kernel_at_locs_regions, kernel_at_locs_unsafe_set = compute_kernel_at_locs(f, locs, cov_noise, regions, unsafe_set)
    sup_kernel_regions, inf_kernel_regions, sup_kernel_unsafe_set, inf_kernel_unsafe_set = compute_sup_inf_kernel(f, cov_noise, regions, unsafe_set)

    # Initialize approximation distribution
    approx_distribution = deepcopy(distribution)

    alphas_regions, betas_regions = torch.zeros(len(locs)), torch.zeros(len(locs))
    alpha_unsafe_set, beta_unsafe_set = torch.zeros(1), torch.zeros(1)

    #Run simulation
    means_gmm = f(locs)
    for t in range(30):
        # Update approximation
        approx_probs = approx_distribution.compute_probabilities(regions)
        covs_noise = cov_noise.unsqueeze(0).expand(means_gmm.size(0), -1, -1)
        approx_distribution = GaussianMixture(means_gmm, covs_noise, approx_probs)

        # Compute bounds
        alpha_unsafe_set = torch.dot(inf_kernel_unsafe_set - kernel_at_locs_unsafe_set, approx_probs) + torch.dot(inf_kernel_unsafe_set, alphas_regions)
        beta_unsafe_set = torch.dot(sup_kernel_unsafe_set - kernel_at_locs_unsafe_set, approx_probs) + torch.dot(sup_kernel_unsafe_set, betas_regions)

        # TODO: This below is only needed because there is an issue with the sup/inf computation. To be checked
        inf_diff = (inf_kernel_regions - kernel_at_locs_regions)
        sup_diff = (sup_kernel_regions - kernel_at_locs_regions)

        alphas_regions = inf_diff.T @ approx_probs + inf_kernel_regions.T @ alphas_regions
        betas_regions = sup_diff.T @ approx_probs + sup_kernel_regions.T @ betas_regions
        
        print("(t = {}) alpha = {}, beta = {}".format(t, alpha_unsafe_set, beta_unsafe_set))






