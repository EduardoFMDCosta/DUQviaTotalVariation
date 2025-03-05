import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import parameters
import monte_carlo
import experiments as algorithm

from dynamics import LinearDynamics
from distributions import GaussianMixture, Gaussian


# Dynamics parameters
A = torch.Tensor(
        [
            [0.84, 0.10],
            [0.05, 0.72]
        ])

# Initial distribution parameters
initial_weights = torch.Tensor([0.5, 0.5])
initial_means = torch.Tensor([[6, 10], [8, 10]])
sigma = 0.005
cov = sigma * torch.eye(2)
initial_covariances = torch.stack((cov, cov), dim=0)

# Noise
mean_noise = torch.Tensor([0, 0])
sigma_noise = 0.03
cov_noise = sigma_noise * torch.eye(2)  # Assumes uncorrelation (this could be relaxed in further upgrades)

# Barrier (unsafe set)
barrier = torch.Tensor([[3.5, 2.0], [4.5, 3.0]])

if __name__ == "__main__":

    torch.manual_seed(0) # for reproducibility

    f = LinearDynamics(A)

    initial_distribution = GaussianMixture(initial_means, initial_covariances, initial_weights)
    noise_distribution = Gaussian(mean_noise, cov_noise)

    monte_carlo.monte_carlo_simulation(f, initial_distribution, noise_distribution, barrier, parameters.n_steps_ahead, parameters.n_samples)

    tv_bounds_time_step, mixtures = algorithm.tv_bound_algorithm(f, initial_distribution, noise_distribution, parameters.grid_type)
    tv_bounds = torch.cumsum(tv_bounds_time_step, dim=0)
    tv_bounds = torch.minimum(tv_bounds, torch.tensor(1))
    print(f"Final TV bounds: {tv_bounds}")

    mixtures_hitting_probs = monte_carlo.mixture_approximation_monte_carlo(mixtures, barrier, parameters.n_samples)
    print(f"Mixtures hitting probs: {mixtures_hitting_probs}")




