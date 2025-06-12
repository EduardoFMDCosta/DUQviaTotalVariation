import torch
from copy import deepcopy

def o_maximization_state(coeffs: torch.Tensor,
                   lower_bounds: torch.Tensor,
                   upper_bounds: torch.Tensor):

    # Inspired from https://www.baymler.com/IntervalMDP.jl/dev/algorithms/#Efficient-value-iteration
    order = torch.argsort(-coeffs)
    p = deepcopy(lower_bounds)
    rem = 1 - sum(p)
    gap = upper_bounds - p
    cumgap = torch.cumsum(gap[order], dim=0)
    for idx, o in enumerate(order):
        rem_state = max(rem - cumgap[idx] + gap[o], 0)
        if gap[o] < rem_state:
            p[o] += gap[o]
        else:
            p[o] += rem_state
            break

    return p

def o_maximization(coeffs: torch.Tensor,
                   lower_bounds: torch.Tensor,
                   upper_bounds: torch.Tensor):

    n, m = lower_bounds.shape
    p = torch.zeros_like(lower_bounds)

    for k in range(m):
        p[:, k] = o_maximization_state(coeffs, lower_bounds[:, k], upper_bounds[:, k])

    return p