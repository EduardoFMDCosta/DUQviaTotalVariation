import torch
from common.monte_carlo import simulate_monte_carlo, simulate_mixtures
from distributions.distributions import Gaussian
from dynamics.dynamics import LinearDynamics, DubinsCarDynamics
from grid.obstacles import compute_hitting_prob
from grid.regions import AvoidHyperRectangle, ReachHyperRectangle
from plotting.plotting import plot_confidence_interval, plot_samples
from targeted_probability.propagation import propagate_mixture_targeted_bounds

if __name__ == '__main__':
    torch.manual_seed(0)

    # System parameters
    A = torch.Tensor(
        [
            [0.84, 0.10],
            [0.05, 0.72]
        ])
    f = LinearDynamics(A)
    #f = DubinsCarDynamics()

    # Initial distribution
    mean_initial = torch.Tensor([5., 5.])
    cov_initial = torch.diag(torch.Tensor([0.002, 0.002]))
    initial_distribution = Gaussian(mean_initial, cov_initial)

    # Noise distribution
    mean_noise = torch.Tensor([0, 0])
    cov_noise = torch.diag(torch.Tensor([0.01, 0.01]))
    noise_distribution = Gaussian(mean_noise, cov_noise)

    # Shell
    shell = torch.tensor([[-3, -3], [8, 8]])

    num_samples = 10000
    horizon = 30
    grid_size = 1600

    # Define unsafe set
    avoid_sets = AvoidHyperRectangle(torch.tensor([[4.5, 2], [0.0, 2.0]]), torch.tensor([[5.5, 3.5], [1.0, 2.5]]))
    reach_sets = ReachHyperRectangle(torch.tensor([[0.2, 0.2]]), torch.tensor([[1.2, 1.2]]))

    mixtures, aps_avoid, lbs_avoid, ubs_avoid, aps_reach, lbs_reach, ubs_reach = propagate_mixture_targeted_bounds(f=f,
                                                                initial_distribution=initial_distribution,
                                                                noise_distribution=noise_distribution,
                                                                shell=shell,
                                                                avoid_sets=avoid_sets,
                                                                reach_sets=reach_sets,
                                                                initial_grid_size=grid_size,
                                                                prediction_horizon=horizon,
                                                                num_samples=num_samples,
                                                                plot= True)

    monte_carlo_samples = simulate_monte_carlo(f=f,
                                               initial_distribution=initial_distribution,
                                               noise_distribution=noise_distribution,
                                               prediction_horizon=horizon,
                                               num_samples=num_samples)

    hitting_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                            avoid_sets=avoid_sets)
    plot_confidence_interval(aps_avoid, lbs_avoid, ubs_avoid, hitting_probs_mc)

    reach_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                          avoid_sets=reach_sets)
    plot_confidence_interval(aps_reach, lbs_reach, ubs_reach, reach_probs_mc)

    gmm_samples = simulate_mixtures(mixtures=mixtures,
                                    num_samples=num_samples)

    check = compute_hitting_prob(gmm_samples, avoid_sets)

    plot_samples(monte_carlo_samples=monte_carlo_samples,
                 gmm_samples=gmm_samples,
                 avoid_sets=avoid_sets)
