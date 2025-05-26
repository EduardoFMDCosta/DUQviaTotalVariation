import torch

def o_maximization(coeffs: torch.Tensor,
                   lower_bounds: torch.Tensor,
                   upper_bounds: torch.Tensor):

    n, m = coeffs.shape

    sorted_coeffs, order = torch.sort(coeffs, dim=0, descending=True) # sort in descending order

    lb = lower_bounds[:, None].expand(n, m)
    ub = upper_bounds[:, None].expand(n, m)

    p = lb.clone() # set initial p to lower bounds
    remainder = 1 - torch.sum(lb, dim=0)

    gap = ub - lb # compute gap

    # Gather gap and index map in sorted order
    batch_indices = torch.arange(m)[None, :].expand(n, m)
    ordered_gap = gap.gather(0, order)

    cumgap = torch.cumsum(ordered_gap, dim=0)

    # rem_state = max(rem - cumgap[idx] + gap[o], 0)
    # We'll find the first index where rem <= cumgap[idx] - gap[o] for each column

    # Precompute rem - cumgap + gap[o]
    # For gap[o], we need to reverse the gather to find gap[order]
    ordered_o = order  # (n, m)
    gap_at_o = gap.gather(0, ordered_o)  # (n, m)

    rem_expand = remainder[None, :].expand(n, m)
    rem_state = (rem_expand - cumgap + gap_at_o).clamp(min=0.0)  # (n, m)

    # Determine mask: where gap[o] < rem_state → take full gap[o], else partial then break
    full_mask = gap_at_o < rem_state  # (n, m)

    # Compute cumulative full_mask to find stopping index per column
    full_mask_cumsum = torch.cumsum(full_mask.int(), dim=0)  # (n, m)
    stop_idx = (full_mask_cumsum == full_mask_cumsum[-1:]).int().argmax(dim=0)  # (m,)

    # Build final p
    # First, add gap[o] for all rows before stop_idx
    add_full = full_mask * gap_at_o  # (n, m)

    # Now add rem_state only at stop_idx
    idx_mask = torch.arange(n)[:, None] == stop_idx[None, :]  # (n, m)
    add_partial = idx_mask * (~full_mask) * rem_state  # add rem_state only at stop_idx

    # Total addition
    total_add = add_full + add_partial  # (n, m)

    # Scatter the additions back to original positions
    p.scatter_add_(0, order, total_add)

    return p