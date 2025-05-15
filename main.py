import torch
from dynamics.dynamics import LinearDynamics
from distributions import Gaussian
from experiments import approximation_scheme_tv
from grid.regions import HyperRectangle, HyperRectangularPartition
from utils import get_shell_loc, uniform_grid

if __name__ == '__main__':
    torch.manual_seed(0)

    # System parameters
    A = torch.Tensor(
        [
            [0.9]
        ])
    f = LinearDynamics(A)

    # Initial distribution
    mean_initial = torch.Tensor([2])
    cov_initial = 0.5 * torch.eye(1)
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0])
    cov_noise = 0.1 * torch.eye(1)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Define partition
    shell = torch.tensor([
        [-8.],
        [8.]
    ])
    loc_shell = get_shell_loc(shell)
    inner_locs = uniform_grid(shell[0], shell[1], 100)

    partition = HyperRectangularPartition(inner_locs, loc_shell, shell)

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([3.0]).unsqueeze(0), torch.tensor([5.0]).unsqueeze(0))

    mixtures, tv_bounds = approximation_scheme_tv(f=f,
                                                  initial_distribution=initial_distribution,
                                                  noise_distribution=noise_distribution,
                                                  initial_grid_size=10,
                                                  prediction_horizon=2,
                                                  n_samples=1000)

    print(tv_bounds)

    # mixtures_hitting_probs = monte_carlo.mixture_approximation_monte_carlo(mixtures, barrier, parameters.n_samples)
    # print(f"Mixtures hitting probs: {mixtures_hitting_probs}")
