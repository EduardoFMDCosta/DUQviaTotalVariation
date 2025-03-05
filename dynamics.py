import torch
import bound_propagation as bp
from typing import Union, Optional
from regions import HyperRectangularPartition, HyperRectangle

factory = bp.BoundModelFactory()

class Dynamics(torch.nn.Sequential):
    num_dims = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def compute_centralized_ibp(self,
                                partition: HyperRectangularPartition) -> HyperRectangle:

        net = factory.build(self)
        ibp =  net.ibp(bp.HyperRectangle(partition.lower, partition.upper))

        locs = partition.locs
        f_locs = self(locs)

        centralized_lower = ibp.lower - f_locs
        centralized_upper = ibp.upper - f_locs

        return HyperRectangle(centralized_lower, centralized_upper)

    @torch.no_grad()
    def compute_local_maximum_distance(self,
                                       partition: HyperRectangularPartition):

        centralized_hypercubes = self.compute_centralized_ibp(partition)
        max_abs_values = torch.maximum(centralized_hypercubes.lower.abs(), centralized_hypercubes.upper.abs())

        return max_abs_values.norm(p=2, dim=1)


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