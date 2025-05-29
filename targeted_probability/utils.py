import torch
from copy import deepcopy

def o_maximization(coeffs: torch.Tensor,
                   lower_bounds: torch.Tensor,
                   upper_bounds: torch.Tensor):

    # TODO: Recheck this code

    n, m = coeffs.shape

    # Sort coefficients descending per column
    order = torch.argsort(-coeffs, dim=0)  # (n, m)

    # Get gap and initial p
    p = lower_bounds[:, None].repeat(1, m)  # (n, m)
    gap = upper_bounds[:, None] - p  # (n, m)
    rem = 1 - torch.sum(p, dim=0)  # (m,)

    # Rearranged gap based on order
    gap_sorted = torch.gather(gap, 0, order)  # (n, m)

    # Cumulative gap along sorted order
    cumgap = torch.cumsum(gap_sorted, dim=0)  # (n, m)

    # Calculate rem_state for each position
    # rem_state = max(rem - cumgap + gap_sorted, 0)
    rem_expanded = rem.unsqueeze(0).expand(n, m)  # (n, m)
    rem_state = torch.clamp(rem_expanded - cumgap + gap_sorted, min=0.0)  # (n, m)

    # Make mask of whether full gap can be added or just rem_state
    full_add_mask = gap_sorted <= rem_state  # (n, m)

    # Allocate result matrix to fill
    p_out = p.clone()  # (n, m)

    # For each column, find first index where partial addition occurs
    first_partial = (~full_add_mask).float().cumsum(dim=0) == 1  # (n, m)

    # Add full where possible
    full_add = full_add_mask * gap_sorted  # (n, m)

    # Add partial only at the first partial location
    partial_add = first_partial * rem_state  # (n, m)

    # Combine both additions
    total_add = full_add + partial_add  # (n, m)

    # Scatter additions back to original indices
    p_out.scatter_add_(0, order, total_add)

    return p_out