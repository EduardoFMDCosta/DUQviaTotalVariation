from typing import Union, Optional

import torch
import math
from torch.special import erf
import torch.linalg as linalg
import grid_generation as grid

import bound_propagation as bp
from bound_propagation import BoundModelFactory

from regions import HyperRectangularPartition

factory = bp.BoundModelFactory()

class Dynamics(torch.nn.Sequential):
    num_dims = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def compute_ibp(self,
                    partition: HyperRectangularPartition) -> bp.IntervalBounds:

        net = factory.build(self)
        return net.ibp(bp.HyperRectangle(partition.lower, partition.upper))


class LinearDynamics(Dynamics):
    def __init__(self,
                 weight: Union[torch.Tensor, list],
                 bias: Optional[Union[torch.Tensor, list]] = None,
                 **kwargs):
        if isinstance(weight, list):
            weight = torch.tensor(weight)
        if isinstance(bias, list):
            bias = torch.tensor(bias)

        self.num_dims = weight.size(-1)
        self._global_lipschitz = torch.linalg.svd(weight).S[0]

        super(LinearDynamics, self).__init__(bp.FixedLinear(weight, bias))