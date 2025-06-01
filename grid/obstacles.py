import torch
from grid.regions import HyperRectangle

def compute_hitting_prob(samples: torch.Tensor,
                         avoid_sets: HyperRectangle):

    b, n, d = samples.shape
    m = avoid_sets.lower.shape[0]  # number of obstacles

    # Check if samples are within each unsafe region
    hits = torch.logical_and(
        (samples.unsqueeze(2) >= avoid_sets.lower.unsqueeze(0).unsqueeze(0)),
        (samples.unsqueeze(2) <= avoid_sets.upper.unsqueeze(0).unsqueeze(0))
    ).all(dim=-1)

    # Compute hitting probabilities
    hitting_probabilities = hits.float().sum(dim=1) / n

    return hitting_probabilities.sum(dim=1)

