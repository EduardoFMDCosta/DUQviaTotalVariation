import torch
import parameters
import grid_generation as grid
import bounds as tv

from distributions import GaussianMixture, Gaussian, UniformMixture, Uniform


def tv_bound_algorithm(dynamics, initial_distribution, noise_distribution, grid_type):

    tv_bounds = [0.0]
    mixtures = []

    for t in range(parameters.n_steps_ahead + 1):

        if t == 0:
            hat_mixture = initial_distribution
        else:
            if isinstance(noise_distribution, Gaussian):
                means_gmm = dynamics(signatures)
                covs_noise = noise_distribution.covariance.unsqueeze(0).expand(means_gmm.size(0), -1, -1)
                hat_mixture = GaussianMixture(means_gmm, covs_noise, double_hat_probs)

            elif isinstance(noise_distribution, Uniform):
                centers_gmm = dynamics(signatures)
                lows_noise = noise_distribution.low.unsqueeze(0).expand(centers_gmm.size(0), -1)
                highs_noise = noise_distribution.high.unsqueeze(0).expand(centers_gmm.size(0), -1)
                hat_mixture = UniformMixture(centers=centers_gmm, lows=lows_noise, highs=highs_noise, weights=double_hat_probs)

        mixtures.append(hat_mixture)
        samples = hat_mixture(parameters.n_samples)

        if t < parameters.n_steps_ahead:
            high_prob_region = grid.identify_high_prob_region(samples)
            outer_signature = grid.outer_point(high_prob_region)

            regions = grid.create_regions(high_prob_region, samples, parameters.min_proportion, parameters.min_size, parameters.max_depth, 0, grid_type)
            signatures = grid.place_signatures(regions)

            double_hat_probs = hat_mixture.compute_regions_probabilities(regions) #TODO: Generalize for GMMs with different covariances

            regions, signatures = grid.add_unbounded_representations(regions, signatures, outer_signature)

            tv_bound, contributions = tv.compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, double_hat_probs, regions)


            for r in range(parameters.n_refinements):

                regions = regions[:-1]
                signatures = signatures[:-1]
                regions, signatures = grid.refine_regions(regions, signatures, contributions, parameters.threshold)

                double_hat_probs = hat_mixture.compute_regions_probabilities(regions) #TODO: Generalize for GMMs with different covariances
                regions, signatures = grid.add_unbounded_representations(regions, signatures, outer_signature)

                tv_bound, contributions = tv.compute_upper_bound_for_TV(dynamics, noise_distribution, signatures, double_hat_probs, regions)


            tv_bounds.append(tv_bound.item())

    return torch.Tensor(tv_bounds), mixtures