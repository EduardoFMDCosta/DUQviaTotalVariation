import torch
from common.enums import BoundType
from common.propagate import propagate
from dynamics.dynamics import get_dynamics
from grid.obstacles import compute_hitting_prob
from plotting.plotting import plot_confidence_interval, plot_samples
from common.monte_carlo import simulate_monte_carlo, simulate_mixtures
from configs.handlers import parse_arguments, load_params
from configs.construct import get_initial_dist, get_noise_dist, get_shell, get_avoid_sets, get_reach_sets

if __name__ == '__main__':
    torch.manual_seed(0)

    bound_type = BoundType.IMDP

    args = parse_arguments(
        dynamics_type="LinearDynamics",
        num_dims=2,
        dynamics_setting=0,
        prediction_horizon=30,
        num_samples=5000,
        plot=True
    )
    params = load_params(args)

    dynamics = get_dynamics(**params)
    initial_distribution = get_initial_dist(**params)
    noise_distribution = get_noise_dist(**params)
    shell = get_shell(**params)

    # Define unsafe set
    avoid_sets = get_avoid_sets(**params)
    reach_sets = get_reach_sets(**params)

    mixtures, avoid_bounds, reach_bounds = propagate(
        f=dynamics,
        initial_distribution=initial_distribution,
        noise_distribution=noise_distribution,
        shell=shell,
        avoid_sets=avoid_sets,
        reach_sets=reach_sets,
        bound_type=bound_type,
        **params)

    monte_carlo_samples = simulate_monte_carlo(f=dynamics,
                                               initial_distribution=initial_distribution,
                                               noise_distribution=noise_distribution,
                                               **params)

    hitting_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                            avoid_sets=avoid_sets)
    plot_confidence_interval(avoid_bounds, hitting_probs_mc)

    reach_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                          avoid_sets=reach_sets)
    plot_confidence_interval(reach_bounds, reach_probs_mc)

    gmm_samples = simulate_mixtures(mixtures=mixtures,
                                    **params)

    check = compute_hitting_prob(gmm_samples, avoid_sets)

    plot_samples(monte_carlo_samples=monte_carlo_samples,
                 gmm_samples=gmm_samples,
                 avoid_sets=avoid_sets,
                 reach_sets=reach_sets)