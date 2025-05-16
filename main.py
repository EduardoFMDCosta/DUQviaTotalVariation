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
            [0.84, 0.10],
            [0.05, 0.72]
        ])
    f = LinearDynamics(A)

    # Initial distribution
    mean_initial = torch.Tensor([10, 7])
    cov_initial = 0.005 * torch.eye(2)
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = 0.03 * torch.eye(2)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Define unsafe set
    unsafe_set = HyperRectangle(torch.tensor([4.0, 2.0]).unsqueeze(0), torch.tensor([5.0, 3.0]).unsqueeze(0))

    # Set parameters
    horizon = 10
    num_samples = 1000

    mixtures, tv_bounds = propagate_mixture_tv_bounds(f=f,
                                                  initial_distribution=initial_distribution,
                                                  noise_distribution=noise_distribution,
                                                  initial_grid_size=100,
                                                  prediction_horizon=horizon,
                                                  num_samples=num_samples)

    print(tv_bounds)

    monte_carlo_samples = monte_carlo(f=f,
                initial_distribution=initial_distribution,
                noise_distribution=noise_distribution,
                prediction_horizon=horizon,
                num_samples=num_samples)

    gmm_samples = sample_from_gmm(mixtures=mixtures,
                    num_samples=num_samples)

    plot_samples(monte_carlo_samples, gmm_samples)
