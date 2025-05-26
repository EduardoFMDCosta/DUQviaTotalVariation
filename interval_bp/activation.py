import torch
import bound_propagation as bp
from functools import wraps

__all__ = ['BoundIdentity', 'BoxedIdentity', 'BoundBoxedIdentity']

def assert_bound_order(func, position=0, keyword='preactivation'):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > position:
            bounds = args[position]
        else:
            bounds = kwargs[keyword]

        if torch.isnan(bounds.lower).any() or torch.isnan(bounds.upper).any() or \
            not torch.all(bounds.lower <= bounds.upper + 1e-6):
            raise ValueError(f"Bounds are not ordered: {bounds.lower} <= {bounds.upper}")

        return func(self, *args, **kwargs)

    return wrapper


class BoundIdentity(bp.activation.BoundIdentity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        bounds = self.ibp_forward(bounds, save_relaxation, save_input_bounds)
        intersection = self.module(intersection)
        return bounds, intersection


class BoxedIdentity(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, x, all=True):
        if all:
            mask = ((x >= self.min) & (x <= self.max)).all(dim=-1).unsqueeze(-1)
        else:
            mask = (x >= self.min) & (x <= self.max)
        return x * mask


class BoundBoxedIdentity(bp.BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    def clear_relaxation(self):
        super().clear_relaxation()

    @assert_bound_order
    def strict_ibp_forward(self, bounds, intersection, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.strict_alpha_beta(preactivation=bounds, intersection=intersection)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        act_lower, act_upper =  self.module(bounds.lower, all=False),self.module(bounds.upper, all=False)
        mask = act_lower >= act_upper
        act_lower[mask] = 0.
        act_upper[mask] = 0.

        bounds = bp.IntervalBounds(bounds.region, act_lower, act_upper)

        intersection = self.module(intersection, all=False)

        return bounds, intersection

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        raise NotImplementedError

    def alpha_beta(self, preactivation):
        raise NotImplementedError

    @assert_bound_order
    def strict_alpha_beta(self, preactivation, intersection):
        """
        Adaptive is similar to :BoundReLU: with the adaptivity being applied to both bends

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper

        at_lower = torch.isclose(intersection, lower, atol=1e-5)
        at_upper = torch.isclose(intersection, upper, atol=1e-5)
        if not torch.logical_or(at_lower, at_upper).all():
            raise Exception

        zero_width, flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range = bp.saturation.regimes(
            lower, upper, self.module.min, self.module.max)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        act_lower, act_upper = self.module(lower, all=False), self.module(upper, all=False)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        if zero_width.any():
            self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, act_lower[zero_width]
            self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, act_upper[zero_width]

        min, max = self.module.min, self.module.max
        if min is not None and torch.is_tensor(min):
            min = min.view(*[1 for _ in range(self.alpha_lower.dim() - 1)], -1).expand_as(self.alpha_lower)

        if max is not None and torch.is_tensor(max):
            max = max.view(*[1 for _ in range(self.alpha_lower.dim() - 1)], -1).expand_as(self.alpha_lower)

        # Flat lower
        self.alpha_lower[flat_lower] = 0
        self.alpha_upper[flat_lower] = 0

        # Flat upper
        self.alpha_lower[flat_upper] = 0
        self.alpha_upper[flat_upper] = 0

        # Slope
        self.alpha_lower[slope] = 1
        self.alpha_upper[slope] = 1

        z = (self.module(upper, all=False) - self.module(lower, all=False)) / (upper - lower)

        # Lower bend
        if min is not None:
            # intersect at lower (left corner)
            lower_bend_left = lower_bend & at_lower

            lower_bend_left_min = min[lower_bend_left] if torch.is_tensor(min) else torch.as_tensor(min)
            self.alpha_lower[lower_bend_left] = lower_bend_left_min.clip(max=0.) / (lower_bend_left_min - lower[lower_bend_left])
            self.alpha_upper[lower_bend_left] = lower_bend_left_min.clip(min=0.) / (lower_bend_left_min - lower[lower_bend_left])

            # intersect at upper (right corner)
            lower_bend_right = lower_bend & at_upper

            lower_bend_right_min = min[lower_bend_right] if torch.is_tensor(min) else torch.as_tensor(min)
            self.alpha_lower[lower_bend_right] = (act_upper[lower_bend_right] - lower_bend_right_min.clip(max=0.)) / (upper[lower_bend_right] - lower_bend_right_min)
            self.alpha_upper[lower_bend_right] = upper[lower_bend_right] / (upper[lower_bend_right] - lower_bend_right_min)

            # Correct for special cases:
            lower_bend_neg_pos = lower_bend & (lower <= 0.) & (upper >= 0.)
            self.alpha_upper[lower_bend_neg_pos] = z[lower_bend_neg_pos]

            lower_bend_right_pos_pos = lower_bend_right & (lower >= 0.) & (upper >= 0.)
            self.alpha_upper[lower_bend_right_pos_pos] = 1.

        # Upper bend
        if max is not None:
            # intersect at lower
            upper_bend_left = upper_bend & at_lower

            upper_bend_left_max = max[upper_bend_left] if torch.is_tensor(max) else torch.as_tensor(max)
            self.alpha_lower[upper_bend_left] = (upper_bend_left_max.clip(max=0.) - lower[upper_bend_left]) / (upper_bend_left_max - lower[upper_bend_left])
            self.alpha_upper[upper_bend_left] = (upper_bend_left_max.clip(min=0.) - act_lower[upper_bend_left]) / (upper_bend_left_max - lower[upper_bend_left])

            # intersect at upper
            upper_bend_right = upper_bend & at_upper

            upper_bend_right_max = max[upper_bend_right] if torch.is_tensor(max) else torch.as_tensor(max)
            self.alpha_lower[upper_bend_right] = - upper_bend_right_max.clip(max=0.) / (upper[upper_bend_right] - upper_bend_right_max)
            self.alpha_upper[upper_bend_right] = - upper_bend_right_max.clip(min=0) / (upper[upper_bend_right] - upper_bend_right_max)

            # correct for special cases
            upper_bend_pos_neg = upper_bend & (lower <= 0.) & (upper >= 0.)
            self.alpha_lower[upper_bend_pos_neg] = z[upper_bend_pos_neg]

        # Full range
        if self.module.min is not None and self.module.max is not None:
            # intersect at lower
            full_range_left = full_range & at_lower

            full_range_left_min = min[full_range_left] if torch.is_tensor(min) else torch.as_tensor(min)
            full_range_left_max = max[full_range_left] if torch.is_tensor(max) else torch.as_tensor(max)

            self.alpha_lower[full_range_left] = full_range_left_min.clip(max=0.) / (full_range_left_min - lower[full_range_left])
            self.alpha_upper[full_range_left] = full_range_left_max.clip(min=0.) / (full_range_left_max - lower[full_range_left])

            # correction
            full_range_left_pos = full_range_left & (lower > 0.)
            full_range_left_pos_min = min[full_range_left_pos] if torch.is_tensor(min) else torch.as_tensor(min)
            self.alpha_upper[full_range_left_pos] = full_range_left_pos_min / (full_range_left_pos_min - lower[full_range_left_pos])

            # intersect at upper
            full_range_right = full_range & at_upper

            full_range_right_min = min[full_range_right] if torch.is_tensor(min) else torch.as_tensor(min)
            full_range_right_max = max[full_range_right] if torch.is_tensor(max) else torch.as_tensor(max)

            self.alpha_lower[full_range_right] = -full_range_right_min.clip(max=0) / (upper[full_range_right] - full_range_right_min)
            self.alpha_upper[full_range_right] = -full_range_right_max.clip(min=0) / (upper[full_range_right] - full_range_right_max)

            # correction
            full_range_right_neg = full_range_right & (upper <= 0.)
            full_range_right_neg_max = max[full_range_right_neg] if torch.is_tensor(max) else torch.as_tensor(max)
            self.alpha_lower[full_range_right_neg] = -full_range_right_neg_max / (upper[full_range_right_neg] - full_range_right_neg_max)

        self.beta_lower[at_lower] = act_lower[at_lower] - lower[at_lower] * self.alpha_lower[at_lower]
        self.beta_upper[at_lower] = act_lower[at_lower] - lower[at_lower] * self.alpha_upper[at_lower]

        self.beta_lower[at_upper] = act_upper[at_upper] - upper[at_upper] * self.alpha_lower[at_upper]
        self.beta_upper[at_upper] = act_upper[at_upper] - upper[at_upper] * self.alpha_upper[at_upper]