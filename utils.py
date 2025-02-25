import torch
import itertools

from distributions import Gaussian


def create_uniform_grid(lower: torch.Tensor, upper: torch.Tensor, n: int):

    d = lower.shape[0]
    equidistant_spaces = [torch.linspace(lower[i], upper[i], steps=n) for i in range(d)]
    mesh = torch.meshgrid(*equidistant_spaces, indexing="ij")
    grid = torch.stack(mesh, dim=-1).reshape(-1, d)

    return grid


def generate_hypercube_vertices(lower_point, upper_point):
    d = lower_point.shape[0]  # Number of dimensions
    vertices = torch.tensor(list(itertools.product(*zip(lower_point, upper_point))))
    return vertices

def compute_sup_inf_kernel(f, covariance, regions, set):

    sup_probs_regions, inf_probs_regions = [], []
    sup_probs_set, inf_probs_set = [], []

    for lower_point, upper_point in zip(regions.lower, regions.upper):

        vertices = generate_hypercube_vertices(lower_point, upper_point)

        sup_prob_regions, sup_prob_set = torch.zeros(len(regions.lower)), torch.zeros(1)
        inf_prob_regions, inf_prob_set = torch.ones(len(regions.lower)), torch.ones(1)

        for vertice in vertices:

            if torch.isinf(f(vertice)).any() or torch.isnan(f(vertice)).any():
                prob_regions = torch.zeros(len(regions.lower))
                prob_set = torch.zeros(1)
            else:
                kernel_at_vertice = Gaussian(f(vertice), covariance)
                prob_regions = kernel_at_vertice.compute_probabilities(regions)
                prob_set = kernel_at_vertice.compute_probabilities(set)

            sup_prob_regions = torch.max(sup_prob_regions, prob_regions)
            inf_prob_regions = torch.min(inf_prob_regions, prob_regions)

            sup_prob_set = torch.max(sup_prob_set, prob_set)
            inf_prob_set = torch.min(inf_prob_set, prob_set)


        sup_probs_regions.append(sup_prob_regions)
        inf_probs_regions.append(inf_prob_regions)

        sup_probs_set.append(sup_prob_set)
        inf_probs_set.append(inf_prob_set)

    sup_probs_regions = torch.stack(sup_probs_regions).clamp_min(0.0)
    inf_probs_regions = torch.stack(inf_probs_regions).clamp_max(0.0)

    sup_probs_set = torch.tensor(sup_probs_set).clamp_min(0.0)
    inf_probs_set = torch.tensor(inf_probs_set).clamp_max(0.0)

    return sup_probs_regions, inf_probs_regions, sup_probs_set, inf_probs_set

def compute_kernel_at_locs(f, locs, covariance, regions, set):

    probs_regions, probs_set = [], []

    for loc in locs:
        kernel = Gaussian(f(loc), covariance)

        prob_regions = kernel.compute_probabilities(regions)
        probs_regions.append(prob_regions)

        prob_set = kernel.compute_probabilities(set)
        probs_set.append(prob_set)

    return torch.stack(probs_regions), torch.tensor(probs_set)