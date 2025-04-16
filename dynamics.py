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

        centralized_lower[centralized_lower.isnan()] = float('inf')
        centralized_upper[centralized_upper.isnan()] = float('inf')

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


class SinusoidalDynamics(Dynamics):
    def __init__(self, num_dims, **kwargs):
        self.num_dims = num_dims
        super(SinusoidalDynamics, self).__init__(bp.Sin())

    @property
    def global_lipschitz(self):
        return 1.0

class DubinsDynamics(Dynamics):
    def __init__(self, velocity: float = 5.0, u: float = 2.0, h: float = 0.3, **kwargs):
        self.num_dims = 3
        self.velocity = velocity
        self.u = u
        self.h = h

        linear_part = bp.FixedLinear(
            torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]),
            torch.tensor([0.0, 0.0, h * u])
        )

        trig_part = torch.nn.Sequential(
            bp.FixedLinear(
                torch.tensor([
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([torch.pi / 2, 0.0, 0.0])
                ),
            bp.Sin(),
            bp.FixedLinear(
                torch.tensor([
                    [h * velocity, 0.0, 0.0],
                    [0.0, h * velocity, 0.0],
                    [0.0, 0.0, 0.0]
                ]),
                torch.tensor([0.0, 0.0, 0.0])
            ),
        )
 
        super(DubinsDynamics, self).__init__(
            bp.Parallel(linear_part, trig_part),
            bp.VectorAdd(),
        )
