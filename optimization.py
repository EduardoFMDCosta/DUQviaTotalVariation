from typing import Union
import torch
from mpmath import isinf

from distributions import Gaussian
from dynamics import Dynamics
from regions import HyperRectangularPartition, HyperRectangle


def objective(z, f, covariance, target_set):
    f_z = f(z)
    gaussian = Gaussian(f_z, covariance)

    return gaussian.compute_probabilities(target_set)


def gradient_descent(f: Dynamics,
                     covariance: torch.Tensor,
                     target_set: HyperRectangle,
                     region: HyperRectangle,
                     maximize: bool = False,
                     lr: float = 0.01,
                     max_iter: int = 10000,
                     tolerance=1e-8):

    d = target_set.lower.shape[-1]
    z = torch.randn(d, requires_grad=True)

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([z], lr=lr)

    # Store losses for tracking the optimization progress
    loss_history = []

    # Indicator for minimization or maximization
    factor = 1 if not maximize else -1

    for iteration in range(max_iter):

        optimizer.zero_grad()
        loss = factor * objective(z, f, covariance, target_set)
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())

        with torch.no_grad():
            if torch.isinf(region.lower).any():
                z.clamp_(region.lower, region.upper) #TODO: CHANGE
            z.clamp_(region.lower, region.upper)

        # Check for convergence (early stopping)
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tolerance:
            #print("Converged after {} iterations.".format(iteration))
            break

    return z