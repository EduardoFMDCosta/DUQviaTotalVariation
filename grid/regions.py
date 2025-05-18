import torch
import itertools
from typing import Callable

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
    def __init__(self, locs_inner: torch.Tensor, loc_shell: torch.Tensor, shell: torch.Tensor):
        if not self._are_locs_in_grid(locs_inner):
            raise ValueError("locs must be in a grid")
        self._loc_shell = loc_shell
        self._shell = shell
        self._locs_inner = locs_inner
        self._num_dims = locs_inner.size(-1)
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
    def shell(self):
        return HyperRectangle(self._shell[0], self._shell[1])

    @staticmethod
    def _are_locs_in_grid(locs: torch.Tensor):
        unique_elements_per_n = torch.tensor([torch.unique(locs[:, i]).numel() for i in range(locs.size(1))])
        return unique_elements_per_n.prod() == torch.unique(locs,dim=0).size(0)

    def _get_upper(self):
        pos_diff = (self._locs_inner.unsqueeze(-3) - self._locs_inner.unsqueeze(-2)).clip(0, torch.inf)
        mask = pos_diff == 0.
        pos_diff[mask] = torch.inf

        upper_inner = self._locs_inner + 0.5 * pos_diff.min(dim=-2).values
        upper_inner = upper_inner.clamp(min=self._shell[0], max=self._shell[1])
        upper_shell = torch.zeros(self._num_dims).fill_(torch.inf)

        return torch.cat((upper_inner, upper_shell.unsqueeze(-2)), dim=-2)

    def _get_lower(self):
        neg_diff = (self._locs_inner.unsqueeze(-3) - self._locs_inner.unsqueeze(-2)).clip(-torch.inf, 0)
        mask = neg_diff == 0.
        neg_diff[mask] = -torch.inf

        lower_inner = self._locs_inner + 0.5 * neg_diff.max(dim=-2).values
        lower_inner = lower_inner.clamp(min=self._shell[0], max=self._shell[1])
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

    def refine(self,
               objective: Callable,
               contributions: torch.Tensor,
               target: float = 0.05,
               pareto: float = 0.2,
               max_regions: int = 1000):

        locs_inner = self._locs_inner
        lower_inner = self.lower[:-1]
        upper_inner = self.upper[:-1]

        refined_grid = HyperRectangularPartition(locs_inner, self._loc_shell, self._shell)

        while contributions.sum() > target and contributions.size(0) < max_regions:
            # Compute the threshold for the top pareto%
            top_k = max(1, int(pareto * contributions.numel()))
            pareto_threshold = torch.topk(contributions, top_k).values.min()

            mask = contributions[:-1] > pareto_threshold

            # Add mid-points in diagonal if contribution condition is met
            mid_low = (locs_inner[mask] + lower_inner[mask]) / 2
            mid_high = (locs_inner[mask] + upper_inner[mask]) / 2
            locs_expanded = torch.cat((locs_inner, mid_low, mid_high), dim=0)

            # Generate grid
            grid = self.generate_grid(locs_expanded)
            refined_grid = HyperRectangularPartition(grid, self._loc_shell, self._shell)

            contributions, _ = objective(refined_grid)

            locs_inner = refined_grid._locs_inner
            lower_inner = refined_grid.lower[:-1]
            upper_inner = refined_grid.upper[:-1]

        return refined_grid



