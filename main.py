import torch
from dynamics import LinearDynamics
from distributions import Gaussian, GaussianMixture
from experiments import approximation_scheme_tv

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

    # Get GMMs and TV bounds
    mixtures, tv_bounds = approximation_scheme_tv(f,
                                                  initial_distribution,
                                                  noise_distribution,
                                                  initial_grid_size = 50,
                                                  prediction_horizon = 2,
                                                  n_samples = 1000)

    print(f'TV bounds: {tv_bounds}')
