import torch
import math
import bound_propagation as bp
from typing import Union, Optional

class Dynamics(torch.nn.Sequential):
    num_dims = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  


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

        super(LinearDynamics, self).__init__(bp.FixedLinear(weight, bias))

class DubinsCarDynamics(Dynamics):
    def __init__(self, velocity: float = 1.5, u: float = 1.0, h: float = 0.3, **kwargs):
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
 
        super(DubinsCarDynamics, self).__init__(
            bp.Parallel(linear_part, trig_part),
            bp.VectorAdd(),
        )

class PiecewiseAffineBlock(Dynamics):
    def __init__(self, min, max, dynamics):
        min = torch.as_tensor(min) if not torch.is_tensor(min) else min
        max = torch.as_tensor(max) if not torch.is_tensor(max) else max
        super().__init__(
            interval_bp.BoxedIdentity(min=min, max=max),
            dynamics
        )


def rot_mat(theta, rho, delta):
    theta = theta if torch.is_tensor(theta) else torch.as_tensor(theta)
    rho = rho if torch.is_tensor(rho) else torch.as_tensor(rho)
    delta = delta if torch.is_tensor(delta) else torch.as_tensor(delta)
    return rho * torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]) + delta


class DoubleSpiral2dDynamics(Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        region = [[-2., -0.75], [2., 1.25]]

        weight_left = rot_mat(theta=math.pi / 8., rho=0.8, delta=0.)
        weight_right = rot_mat(theta=-math.pi / 8., rho=0.8, delta=0.)
        bias_left = (torch.eye(2) - weight_left) @ torch.tensor([-1.25, -1.0])
        bias_right = (torch.eye(2) - weight_right) @ torch.tensor([1.25, -1.0])

        mode_left = PiecewiseAffineBlock(min=torch.tensor(region[0]), max=torch.tensor([0., region[1][1]]),
                                         dynamics=LinearDynamics(weight=weight_left, bias=bias_left))
        mode_right = PiecewiseAffineBlock(min=torch.tensor([0., region[0][1]]), max=torch.tensor(region[1]),
                                          dynamics=LinearDynamics(weight=weight_right, bias=bias_right))

        super().__init__(
            bp.Clamp(min=torch.tensor(region[0]), max=torch.tensor(region[1])),
            bp.Parallel(mode_left, mode_right),
            bp.VectorAdd()
        )

class SwitchedLinearDynamics(Dynamics):
    num_dims = 2
    def __init__(self, **kwargs):
        region = [[-2., -2.], [2., 2.]]

        mat1 = [[0.79, 0.035], [0., 0.825]]
        mat2 = [[0.79, 0.175], [0., 0.825]]
        mat3 = [[0.79, 0.], [0.175, 0.825]]
        mat4 = [[1., 0.2], [-0.2, 1.]]
        mat5 = [[1., -0.2], [0.2, 1.]]
        redun_mat = torch.eye(2)

        mid_region = [[-0.8, -0.8],[0.8, 0.8]]
        mid_block = PiecewiseAffineBlock(min=mid_region[0], max=mid_region[1], dynamics=LinearDynamics(weight=redun_mat))

        obs_right = PiecewiseAffineBlock(min=[1., 1.], max=[2., 2.], dynamics=LinearDynamics(weight=redun_mat))
        mode2_right = PiecewiseAffineBlock(min=[mid_region[1][0], 0.25], max=[2., 1.], dynamics=LinearDynamics(weight=mat2))
        mode5_right = PiecewiseAffineBlock(min=[mid_region[1][0], -1.], max=[2., 0.25], dynamics=LinearDynamics(weight=mat5))
        mode1_bottom = PiecewiseAffineBlock(min=[0., -2.], max=[2., -1.8], dynamics=LinearDynamics(weight=mat1))
        mode4_bottom = PiecewiseAffineBlock(min=[0., -1.8], max=[2., -1.], dynamics=LinearDynamics(weight=mat4))
        mode3 = PiecewiseAffineBlock(min=[0.3, mid_region[1][1]], max=[1., 2.], dynamics=LinearDynamics(weight=mat1)) # \todo crux
        mode2_bottom = PiecewiseAffineBlock(min=[-0.6, -2.], max=[0., mid_region[0][1]], dynamics=LinearDynamics(weight=mat2))
        mode1_bottom_left = PiecewiseAffineBlock(min=[-1., -2.], max=[-0.6, mid_region[0][1]], dynamics=LinearDynamics(weight=mat1))

        mode4_top = PiecewiseAffineBlock(min=[-1.8, 1.], max=[0.3, 1.8], dynamics=LinearDynamics(weight=mat4))
        mode2_top = PiecewiseAffineBlock(min=[-2, 1.8], max=[0.3, 2.], dynamics=LinearDynamics(weight=mat2))
        mode1_left = PiecewiseAffineBlock(min=[-2., 0.], max=[-1.8, 1.8], dynamics=LinearDynamics(weight=mat1))
        mode5_left = PiecewiseAffineBlock(min=[-1.8, 0.], max=[mid_region[0][0], 1.], dynamics=LinearDynamics(weight=mat5))
        mode2_left = PiecewiseAffineBlock(min=[-2., -1.], max=[mid_region[0][0], 0.], dynamics=LinearDynamics(weight=mat2))
        obs_left = PiecewiseAffineBlock(min=[-2., -2.], max=[-1., -1.], dynamics=LinearDynamics(weight=redun_mat))

        redun_mode = PiecewiseAffineBlock(min=region[0], max=region[1], dynamics=LinearDynamics(weight=torch.zeros((2,2))))

        super().__init__(
            bp.Clamp(min=torch.as_tensor(region[0]), max=torch.as_tensor(region[1])),
            bp.Parallel(
                obs_right, mode2_right, mode5_right, mode1_bottom,
                mode4_bottom, mode3,
                mode2_bottom, mode1_bottom_left, mode4_top, mode2_top,
                mid_block,
                mode1_left, mode5_left, mode2_left, obs_left,
                redun_mode
            ),
            bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd(), bp.VectorAdd()
        )

def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'LinearDynamics':
        return LinearDynamics(**kwargs)
    elif dynamics_type == 'DubinsCarDynamics':
        return DubinsCarDynamics(**kwargs)
    elif dynamics_type == 'DoubleSpiral2dDynamics':
        return DoubleSpiral2dDynamics(**kwargs)
    elif dynamics_type == 'SwitchedLinearDynamics':
        return SwitchedLinearDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")