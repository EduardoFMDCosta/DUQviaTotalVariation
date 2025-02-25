import torch
import math
from torch.special import erf
import torch.linalg as linalg
import grid_generation as grid
from abc import ABC, abstractmethod

from distributions import Gaussian
import probability_mass_computation as proba


class _Dynamics(ABC):

    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_hypercube_envelopes(self, regions: torch.Tensor):
        pass

    def compute_envelopes_transform(self, distribution, signatures: torch.Tensor, envelopes: torch.Tensor):

        cov = distribution.covariance
        inv_cov = linalg.inv(cov)
        L = linalg.cholesky(inv_cov)

        propagated_signatures = self(signatures)
        centered_envelopes = grid.get_centered_region(envelopes, propagated_signatures)

        return torch.matmul(centered_envelopes, L)

    def compute_h(self, envelopes: torch.Tensor):

        vertices = grid.get_vertices(envelopes)
        norms = torch.norm(vertices, dim=2)  # Calculate norms across the last dimension
        max_values = torch.max(norms, dim=1).values  # Get the maximum value across the vertices

        return max_values / (2 * math.sqrt(2))

    def compute_max_s(self, noise_distribution, regions, signatures):

        if isinstance(noise_distribution, Gaussian):
            envelopes = self.compute_hypercube_envelopes(regions)
            transformed_envelopes = self.compute_envelopes_transform(noise_distribution, signatures, envelopes)
            h = self.compute_h(transformed_envelopes)
            max_values = erf(h)
            max_values = torch.cat((max_values, torch.ones(1)), dim=0)

            return max_values

        elif isinstance(noise_distribution, Uniform):
            propagated_signatures = self(signatures)
            lows_extremities = propagated_signatures + noise_distribution.low
            highs_extremities = propagated_signatures + noise_distribution.high


            vertices = grid.get_vertices(regions)
            #TODO: This only works for the linear dynamics
            #TODO: In general, we should consider the point that makes f(x) as far as possible from f(signature)
            selected_vertices = vertices[:, 0, :] #using symmetry of bounded regions
            propagated_vertices = self(selected_vertices)
            lows_extremities_vertices = propagated_vertices + noise_distribution.low
            highs_extremities_vertices = propagated_vertices + noise_distribution.high


            min_intersection = torch.max(lows_extremities, lows_extremities_vertices)
            max_intersection = torch.min(highs_extremities, highs_extremities_vertices)
            valid_intersection = (min_intersection <= max_intersection).all(dim=-1)
            intersection_lengths = torch.clamp(max_intersection - min_intersection, min=0)
            intersection_volume = intersection_lengths.prod(dim=-1)
            intersection_volume = intersection_volume * valid_intersection.float()

            support_size = highs_extremities - lows_extremities
            area = support_size.prod(dim=-1)
            height = 1 / area

            max_values = 0.5 * (2 - 2 * intersection_volume * height)
            max_values = torch.cat((max_values, torch.ones(1)), dim=0)

            return max_values





class LinearDynamics(_Dynamics):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def __call__(self, x: torch.Tensor):
        return torch.matmul(x, self.A.T)

    def compute_hypercube_envelopes(self, regions):

        vertices = grid.get_vertices(regions)

        #TODO: Is there a better way to compute this below?
        n, d = vertices.shape[0], vertices.shape[-1]
        propagated_vertices = torch.stack([self(vert) for vert in vertices.reshape(-1, d)]).reshape(n, 2**d, d)

        min_vals, _ = torch.min(propagated_vertices, dim=1)
        max_vals, _ = torch.max(propagated_vertices, dim=1)

        envelopes = torch.stack([min_vals, max_vals], dim=1)
        return envelopes


class PolynomialDynamics(_Dynamics):
    def __init__(self, h: float):
        self.h = h

    def __call__(self, x: torch.Tensor):

        #TODO: Check
        if x.dim() == 1:
            x = x.unsqueeze(0)

        component_1 = x[:, 0] + 1.25 * self.h * x[:, 1]
        component_2 = 1.4 * x[:, 1] + self.h * 0.3 * (0.25 * x[:, 0] ** 2 - 0.4 * x[:, 0] * x[:, 1] + 0.25 * x[:, 1] ** 2)

        return torch.stack((component_1, component_2), dim=1)

    def compute_hypercube_envelopes(self, regions):

        vertices = grid.get_vertices(regions)

        # TODO: Is there a better way to compute this below?
        n, d = vertices.shape[0], vertices.shape[-1]
        propagated_vertices = torch.stack([self(vert) for vert in vertices.reshape(-1, d)]).reshape(n, 2 ** d, d)

        min_vals, _ = torch.min(propagated_vertices, dim=1)
        max_vals, _ = torch.max(propagated_vertices, dim=1)

        envelopes = torch.stack([min_vals, max_vals], dim=1)

        #Check if region contains [0, 0] and update envelope accordingly
        points = torch.Tensor([[0, 0]])
        points = points.expand(regions.size(0), -1)

        lower_extremities = regions[:, 0, :]
        upper_extremities = regions[:, 1, :]

        # Check if the point is inside the regions
        inside = (points >= lower_extremities) & (points <= upper_extremities)
        inside_all = inside.all(dim=1)  # Check across all dimensions

        # Update the envelopes based on the result
        envelopes[inside_all, 0, 1] = 0

        return envelopes


class DubinsDynamics(_Dynamics):
    def __init__(self, h: float, v: float, u: float):
        self.h = h
        self.v = v
        self.u = u

    def __call__(self, x: torch.Tensor):

        # TODO: Check
        if x.dim() == 1:
            x = x.unsqueeze(0)

        component_1 = x[:, 0] + self.h * self.v * torch.sin(x[:, 2])
        component_2 = x[:, 1] + self.h * self.v * torch.cos(x[:, 2])
        component_3 = x[:, 2] + self.h * self.u

        return torch.stack((component_1, component_2, component_3), dim=1)

    def compute_extrema(self, angles_low, angles_high):

        cos_low = torch.cos(angles_low)
        cos_high = torch.cos(angles_high)
        sin_low = torch.sin(angles_low)
        sin_high = torch.sin(angles_high)

        # Find critical points in the interval where cos(x) and sin(x) may have extrema
        critical_cos = (torch.floor(angles_low / math.pi) * math.pi).clamp(min=angles_low, max=angles_high)
        critical_sin = (torch.floor((angles_low + math.pi/2) / math.pi) * math.pi - math.pi/2).clamp(min=angles_low, max=angles_high)

        # Compute cos and sin at critical points
        cos_critical = torch.cos(critical_cos)
        sin_critical = torch.sin(critical_sin)

        # Compute min/max cos and sin over the intervals
        min_cos = torch.min(torch.stack([cos_low, cos_high, cos_critical]), dim=0)[0]
        max_cos = torch.max(torch.stack([cos_low, cos_high, cos_critical]), dim=0)[0]
        min_sin = torch.min(torch.stack([sin_low, sin_high, sin_critical]), dim=0)[0]
        max_sin = torch.max(torch.stack([sin_low, sin_high, sin_critical]), dim=0)[0]

        return min_cos, max_cos, min_sin, max_sin


    def compute_hypercube_envelopes(self, regions):

        angles_low = regions[:, 0, 2]
        angles_high = regions[:, 1, 2]
        min_cos, max_cos, min_sin, max_sin = self.compute_extrema(angles_low, angles_high)

        x_low = regions[:, 0, 0]
        x_high = regions[:, 1, 0]

        y_low = regions[:, 0, 1]
        y_high = regions[:, 1, 1]

        min_first_component = x_low + self.h * self.v * min_sin
        max_first_component = x_high + self.h * self.v * max_sin

        min_second_component = y_low + self.h * self.v * min_cos
        max_second_component = y_high + self.h * self.v * max_cos

        min_third_component = angles_low + self.h * self.u
        max_third_component = angles_high + self.h * self.u

        min_vals = torch.stack([min_first_component, min_second_component, min_third_component], dim=1)
        max_vals = torch.stack([max_first_component, max_second_component, max_third_component], dim=1)

        envelopes = torch.stack([min_vals, max_vals], dim=1)
        return envelopes