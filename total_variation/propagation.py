import torch
from typing import Union
from itertools import accumulate
from common.propagate_mixture import propagate
from total_variation.bounds import compute_bound_tv, get_objective_tv_bound
from distributions.distributions import Gaussian, GaussianMixture
from dynamics.dynamics import Dynamics
from grid.regions import HyperRectangularPartition
from grid.utils import get_shell, get_shell_loc, uniform_grid
from plotting.plotting import plot_partition

def propagate_mixture_tv_bounds(f: Dynamics,
                                initial_distribution: Union[Gaussian, GaussianMixture],
                                noise_distribution: Gaussian,
                                initial_grid_size: int = 10,
                                prediction_horizon: int = 2,
                                num_samples: int = 1000,
                                plot_grid: bool = True):

    tv_bounds = [0.0]
    mixtures = []
    probs = None
    partition = None

    for t in range(prediction_horizon + 1):

        if t == 0:
            mixture_distribution = initial_distribution
        else:
            mixture_distribution = propagate(f, probs, partition.locs, noise_distribution)

        mixtures.append(mixture_distribution)
        samples = mixture_distribution(num_samples)

        if t < prediction_horizon:
            shell = get_shell(samples)
            loc_shell = get_shell_loc(shell)

            inner_partition = uniform_grid(shell[0], shell[1], initial_grid_size)
            partition = HyperRectangularPartition(inner_partition, loc_shell, shell)
            if plot_grid:
                plot_partition(partition)

            contributions, tv_bound = compute_bound_tv(f=f,
                                                       mixture=mixture_distribution,
                                                       noise_distribution=noise_distribution,
                                                       partition=partition)

            # Refinement
            objective = get_objective_tv_bound(f=f,
                                               mixture=mixture_distribution,
                                               noise_distribution=noise_distribution)

            partition = partition.refine(objective=objective,
                                         contributions=contributions,
                                         target=0.02,
                                         pareto=0.3,
                                         max_regions=5000)
            if plot_grid:
                plot_partition(partition)

            probs = mixture_distribution.compute_probabilities(partition)
            contributions, tv_bound = compute_bound_tv(f=f,
                                                       mixture=mixture_distribution,
                                                       noise_distribution=noise_distribution,
                                                       partition=partition)

            tv_bounds.append(tv_bound.item())

        print(f'End of computing for t={t}')

    # TV_{t+1} = min(TV_t + bound, 1)
    tv_bounds = [min(1, tv) for tv in accumulate(tv_bounds)]

    return mixtures, tv_bounds