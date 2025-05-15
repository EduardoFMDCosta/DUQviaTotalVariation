import torch
from dynamics.dynamics import LinearDynamics
from distributions.distributions import Gaussian
from plotting.plotting import plot_samples
from total_variation.experiments import monte_carlo, sample_from_gmm, propagate_mixture_tv_bounds
from grid.regions import HyperRectangle, HyperRectangularPartition
from grid.utils import get_shell_loc, uniform_grid

if __name__ == '__main__':
    torch.manual_seed(0)

    # System parameters
    A = torch.Tensor(
        [
            [0.9, 0.01],
            [0.02, 0.5]
        ])
    f = LinearDynamics(A)

    # Initial distribution
    mean_initial = torch.Tensor([2, 3])
    cov_initial = 0.02 * torch.eye(2)
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = 0.1 * torch.eye(2)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([3.0, 4]).unsqueeze(0), torch.tensor([5.0, 6]).unsqueeze(0))

    mixtures, tv_bounds = propagate_mixture_tv_bounds(f=f,
                                                  initial_distribution=initial_distribution,
                                                  noise_distribution=noise_distribution,
                                                  initial_grid_size=10,
                                                  prediction_horizon=10,
                                                  n_samples=1000)

    print(tv_bounds)

    monte_carlo_samples = monte_carlo(f=f,
                initial_distribution=initial_distribution,
                noise_distribution=noise_distribution,
                prediction_horizon=10,
                n_samples=1000)

    gmm_samples = sample_from_gmm(mixtures=mixtures,
                    n_samples=1000)

    plot_samples(monte_carlo_samples, gmm_samples)
