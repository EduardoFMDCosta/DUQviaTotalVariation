import torch

class Bounds:
    def __init__(self,
                 lb_delta: torch.Tensor,
                 ub_delta: torch.Tensor):

        self.lb_delta = lb_delta
        self.ub_delta = ub_delta

class ConfidenceInterval:
    def __init__(self,
                 reference: torch.Tensor,
                 bounds: Bounds):

        self.reference = reference
        self.lb = reference + bounds.lb_delta
        self.ub = reference + bounds.ub_delta