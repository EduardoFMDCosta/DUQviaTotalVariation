import torch
import itertools
from typing import Callable
from itertools import product

class HyperRectangle:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def __len__(self):
        return self.lower.size(0)

    def size(self, dim=None):
        if dim is None:
            return self.lower.size()

        return self.lower.size(dim)

    def get_vertices(self):
        return torch.tensor(list(itertools.product(*zip(self.lower, self.upper))))

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)


class HyperRectangularPartition:
    def __init__(self,
                 inner_partition: HyperRectangle,
                 loc_shell: torch.Tensor,
                 shell: torch.Tensor):

        self._loc_shell = loc_shell
        self._shell = shell
        self._inner_partition = inner_partition
        self._locs_inner = inner_partition.center
        self._num_dims = loc_shell.size(-1)
        self._lower = self._get_lower()
        self._upper = self._get_upper()

        assert (self._lower <= self._upper).all()

    @property
    def num_locs(self):
        return self.locs.size(0)

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def locs(self):
        return torch.cat((self._locs_inner, self._loc_shell.unsqueeze(-2)), dim=-2)

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def inner_partition(self):
        return self._inner_partition

    @property
    def shell(self):
        return HyperRectangle(self._shell[0], self._shell[1])

    def _get_upper(self):
        upper_inner = self._inner_partition.upper
        upper_shell = torch.zeros(self._num_dims).fill_(torch.inf)

        return torch.cat((upper_inner, upper_shell.unsqueeze(-2)), dim=-2)

    def _get_lower(self):
        lower_inner = self._inner_partition.lower
        lower_shell = torch.zeros(self._num_dims).fill_(-torch.inf)

        return torch.cat((lower_inner, lower_shell.unsqueeze(-2)), dim=-2)

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def width(self):
        return self.upper - self.lower

    @staticmethod
    def generate_grid(locs: torch.Tensor):

        n, d = locs.shape
        unique_vals = [torch.unique(locs[:, i]) for i in range(d)]
        mesh = torch.meshgrid(*unique_vals, indexing="ij")
        grid = torch.stack([m.flatten() for m in mesh], dim=-1)

        return grid

    @staticmethod
    def split(inner_partition: HyperRectangle,
              condition_mask: torch.Tensor):

        lower = inner_partition.lower
        upper = inner_partition.upper
        locs = inner_partition.center

        d = lower.shape[-1]
        combinations = torch.tensor(list(product([0, 1], repeat=d)), dtype=torch.bool)[None, :, :]  # (1, n_sub, d)

        # Select rectangles to split (k, 1, d)
        selected_lower = lower[condition_mask][:, None, :]
        selected_upper = upper[condition_mask][:, None, :]
        selected_locs = locs[condition_mask][:, None, :]

        low_mask = ~combinations
        high_mask = combinations

        sub_lowers = torch.where(low_mask, selected_lower, selected_locs).reshape(-1, d)
        sub_uppers = torch.where(high_mask, selected_upper, selected_locs).reshape(-1, d)

        # Keep the rectangles not being split
        keep_mask = ~condition_mask
        kept_lower = lower[keep_mask]
        kept_upper = upper[keep_mask]

        # Concatenate
        refined_lower = torch.cat([kept_lower, sub_lowers], dim=0)
        refined_upper = torch.cat([kept_upper, sub_uppers], dim=0)

        # Get partition
        refined_inner_grid = HyperRectangle(lower=refined_lower, upper=refined_upper)

        return refined_inner_grid

    def refine(self,
               objective: Callable,
               contributions: torch.Tensor,
               target: float = 0.05,
               pareto: float = 0.2,
               max_regions: int = 1000):

        refined_grid = HyperRectangularPartition(self._inner_partition, self._loc_shell, self._shell)

        while contributions.sum() > target and contributions.size(0) < max_regions:
            # Compute the threshold for the top pareto%
            top_k = max(1, int(pareto * contributions.numel()))
            pareto_threshold = torch.topk(contributions, top_k).values.min()

            mask = contributions[:-1] >= pareto_threshold

            refined_inner = self.split(refined_grid._inner_partition, mask)
            refined_grid = HyperRectangularPartition(refined_inner, self._loc_shell, self._shell)

            contributions, _ = objective(refined_grid)

        return refined_grid



