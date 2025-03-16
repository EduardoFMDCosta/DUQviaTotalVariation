from typing import Union
import torch
from mpmath import isinf

from distributions import Gaussian
from dynamics import Dynamics
from regions import HyperRectangularPartition, HyperRectangle


def objective(z, f, covariance, to_set):
    f_z = f(z)
    gaussian = Gaussian(f_z, covariance)

    return gaussian.compute_probabilities(to_set)


def gradient_descent(f: Dynamics,
                     covariance: torch.Tensor,
                     from_set: HyperRectangle,
                     to_set: HyperRectangle,
                     minimize: bool = False,
                     lr: float = 0.01,
                     max_iter: int = 10000,
                     tolerance=1e-8):

    d = from_set.lower.shape[-1]
    z = (from_set.lower + torch.rand(d) * (from_set.upper - from_set.lower)).requires_grad_()

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([z], lr=lr)

    # Store losses for tracking the optimization progress
    loss_history = []

    # Indicator for minimization or maximization
    factor = 1 if minimize else -1

    for iteration in range(max_iter):

        optimizer.zero_grad()
        loss = factor * objective(z, f, covariance, to_set)
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())

        with torch.no_grad():
            z.clamp_(from_set.lower, from_set.upper)

        # Check for convergence (early stopping)
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tolerance:
            #print("Converged after {} iterations.".format(iteration))
            break

    return z

def project_to_closest_face(point, lower, upper):
    # Compute distances to the lower and upper boundaries
    dist_to_lower = point - lower
    dist_to_upper = upper - point

    # Find the dimension with the closest boundary
    min_dist = torch.minimum(dist_to_lower, dist_to_upper)
    closest_dim = torch.argmin(min_dist)  # Index of the closest face

    # Create the projected point
    projected_point = point.clone()
    projected_point[closest_dim] = lower[closest_dim] if dist_to_lower[closest_dim] < dist_to_upper[closest_dim] else upper[closest_dim]

    return projected_point