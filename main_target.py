import torch
from common.monte_carlo import simulate_monte_carlo, simulate_mixtures
from configs.construct import get_initial_dist, get_noise_dist, get_shell, get_avoid_sets, get_reach_sets
from configs.handlers import parse_arguments, load_params
from dynamics.dynamics import get_dynamics
from grid.obstacles import compute_hitting_prob
from plotting.plotting import plot_confidence_interval, plot_samples
from targeted_probability.propagation import propagate_mixture_targeted_bounds

if __name__ == '__main__':
    torch.manual_seed(0)
    
    # args = parse_arguments(
    #     dynamics_type="DubinsCarDynamics",
    #     num_dims=3,
    #     dynamics_setting=0,
    #     prediction_horizon=15,
    #     num_samples=5000, 
    #     plot=False
    # )
    args = parse_arguments(
        dynamics_type="LinearDynamics",
        num_dims=2,
        dynamics_setting=0,
        prediction_horizon=20,
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

    mixtures, aps_avoid, lbs_avoid, ubs_avoid, aps_reach, lbs_reach, ubs_reach = propagate_mixture_targeted_bounds(f=dynamics,
                                      initial_distribution=initial_distribution,
                                      noise_distribution=noise_distribution,
                                      shell=shell,
                                      avoid_sets=avoid_sets,
                                      reach_sets=reach_sets,
                                      **params)

    monte_carlo_samples = simulate_monte_carlo(f=dynamics,
                                               initial_distribution=initial_distribution,
                                               noise_distribution=noise_distribution,
                                               **params)

    hitting_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                            avoid_sets=avoid_sets)
    plot_confidence_interval(aps_avoid, lbs_avoid, ubs_avoid, hitting_probs_mc)

    reach_probs_mc = compute_hitting_prob(samples=monte_carlo_samples,
                                          avoid_sets=reach_sets)
    plot_confidence_interval(aps_reach, lbs_reach, ubs_reach, reach_probs_mc)

    gmm_samples = simulate_mixtures(mixtures=mixtures,
                                    **params)

    check = compute_hitting_prob(gmm_samples, avoid_sets)

    plot_samples(monte_carlo_samples=monte_carlo_samples,
                 gmm_samples=gmm_samples,
                 avoid_sets=avoid_sets, 
                 reach_sets=reach_sets)
