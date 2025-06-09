import torch
from distributions.distributions import Gaussian
from grid.regions import AvoidHyperRectangle, ReachHyperRectangle


def get_initial_dist(loc_initial_dist, variance_initial_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_initial_dist, variance_initial_dist)


def get_noise_dist(loc_noise_dist, variance_noise_dist, **kwargs):
    return construct_diag_gaussian_dist(loc_noise_dist, variance_noise_dist)


def construct_diag_gaussian_dist(loc_dist, variance_dist):
    loc_dist = torch.as_tensor(loc_dist)
    covariance_dist = torch.diag(torch.as_tensor(variance_dist))
    return Gaussian(loc_dist, covariance_dist)


def get_shell(safe_shell, **kwargs):
    shell = torch.as_tensor(safe_shell)
    return shell

def get_avoid_sets(avoid, **kwargs):
    return AvoidHyperRectangle(torch.tensor(avoid[0]), torch.tensor(avoid[1]))

def get_reach_sets(reach, **kwargs):
    return ReachHyperRectangle(torch.tensor(reach[0]), torch.tensor(reach[1]))