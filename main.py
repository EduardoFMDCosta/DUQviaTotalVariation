import torch
from dynamics import LinearDynamics
from distributions import Gaussian, GaussianMixture
from experiments import approximation_scheme_tv
from regions import HyperRectangle, HyperRectangularPartition
from utils import compute_sup_inf_kernel, get_shell, get_shell_loc, uniform_grid

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
    n_samples = 100
    samples = initial_distribution(n_samples)
    shell = get_shell(samples)
    loc_shell = get_shell_loc(shell)
    inner_locs = uniform_grid(shell[0], shell[1], 2)
    partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([3, 2]), torch.tensor([5, 4]))
    
    # Compute sup/inf_{z \in Ri} T(Rj | z) and sup/inf_{z \in Ri} T(U | z)
    sup_probs_regions, inf_probs_regions, sup_probs_set, inf_probs_set = compute_sup_inf_kernel(f, cov_noise, partition, set)



    # # Get GMMs and TV bounds
    # mixtures, tv_bounds = approximation_scheme_tv(f,
    #                                               initial_distribution,
    #                                               noise_distribution,
    #                                               initial_grid_size = 10,
    #                                               prediction_horizon = 4,
    #                                               n_samples = 1000)

    # print(f'TV bounds: {tv_bounds}')
