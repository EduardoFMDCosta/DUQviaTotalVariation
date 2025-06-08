import torch
from typing import Union
from common.enums import BoundType
from dynamics.dynamics import Dynamics
from imdp.propagation import propagate_imdp
from total_variation.propagation import propagate_tv
from distributions.distributions import Gaussian, GaussianMixture
from grid.regions import AvoidHyperRectangle, ReachHyperRectangle


def propagate(f: Dynamics,
              initial_distribution: Union[Gaussian, GaussianMixture],
              noise_distribution: Gaussian,
              shell: torch.Tensor,
              avoid_sets: AvoidHyperRectangle,
              reach_sets: ReachHyperRectangle,
              bound_type: BoundType,
              grid_size: int = 10,
              prediction_horizon: int = 2,
              num_samples: int = 1000,
              plot: bool = True,
              **kwargs):

    if bound_type == BoundType.TV:
        mixtures, avoid_bounds, reach_bounds = propagate_tv(f=f,
                                                            initial_distribution=initial_distribution,
                                                            noise_distribution=noise_distribution,
                                                            shell=shell,
                                                            avoid_sets=avoid_sets,
                                                            reach_sets=reach_sets,
                                                            grid_size=grid_size,
                                                            prediction_horizon=prediction_horizon,
                                                            num_samples=num_samples,
                                                            plot=plot,
                                                            **kwargs)

    elif bound_type == BoundType.IMDP:
        mixtures, avoid_bounds, reach_bounds = propagate_imdp(f=f,
                                                              initial_distribution=initial_distribution,
                                                              noise_distribution=noise_distribution,
                                                              shell=shell,
                                                              avoid_sets=avoid_sets,
                                                              reach_sets=reach_sets,
                                                              grid_size=grid_size,
                                                              prediction_horizon=prediction_horizon,
                                                              num_samples=num_samples,
                                                              plot=plot,
                                                              **kwargs)

    elif bound_type == BoundType.WASSERSTEIN:
        raise NotImplementedError('Wasserstein bounds not implemented.')
    else:
        raise ValueError('Bound type not supported.')

    return mixtures, avoid_bounds, reach_bounds