import bound_propagation as bp
import torch

from .activation import BoundIdentity, BoxedIdentity, BoundBoxedIdentity

factory = bp.BoundModelFactory()

factory.register(torch.nn.Identity, BoundIdentity)
factory.register(BoxedIdentity, BoundBoxedIdentity)

__all__ = [
    'factory',
    'BoxedIdentity'
]