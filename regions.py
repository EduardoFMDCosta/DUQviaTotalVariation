import itertools
import torch
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

class Polytope:
    def __init__(self, M:torch.Tensor, b:torch.Tensor):
        self._vertices = self.from_halfspaces(M, b)
        self._M = M
        self._b = b

    def __init__(self, vertices:torch.Tensor):
        self._M, self._b = self.from_vertices(vertices)
        self._vertices = vertices

    @property
    def vertices(self):
        return self._vertices

    @property
    def M(self):
        return self._M
    
    @property
    def b(self):
        return self._b
    
    def includes(self, x:torch.Tensor):
        # x must be a batch of points
        return ((self._M @ x.T).T <= self._b).all(dim=1)
    
    def projections(self, x:torch.Tensor):
        normals = self._M / torch.linalg.norm(self._M, dim=1, keepdim=True) 
        distances = ((self._M @ x - self._b) / torch.linalg.norm(self._M, dim=1)**2).unsqueeze(1) 
        projs = x - distances * normals 
        return projs
    
    def furthest_and_closest_from(self, x:torch.Tensor):
        projs = self.projections(x)        
        mask = self.includes(projs)
        projs = projs[mask]
        vertices = self._vertices
        if self.includes(x.unsqueeze(0)):
            candidates = torch.vstack((projs, vertices, x.unsqueeze(0)))
        else:
            candidates = torch.vstack((projs, vertices))
        distances = torch.linalg.norm(candidates - x.repeat(len(candidates), 1), dim = 1)
        imax, imin = torch.argmax(distances), torch.argmin(distances)
        return candidates[imax], candidates[imin]
    
    def closest_from_outside(self, x:torch.Tensor):
        # TODO: not sure if it's the cleanest as an object method

        # if center is in the outer: it is the closest
        if not self.includes(x.unsqueeze(0)):
            return x
        
        # else: look at the other candidates
        projs = self.projections(x)        
        mask = self.includes(projs)
        projs = projs[mask]
        vertices = self._vertices
        candidates = torch.vstack((projs, vertices))
        distances = torch.linalg.norm(candidates - x.repeat(len(candidates), 1), dim = 1)
        imin = torch.argmin(distances)
        return candidates[imin]
        
    @classmethod
    def from_halfspaces(cls, M:torch.tensor, b:torch.Tensor):
        # TODO: test
        halfspaces = torch.hstack([M, -b.reshape(-1, 1)])
        res = linprog(torch.zeros(M.shape[1]), A_ub=M, b_ub=b, method='highs')
        hs = HalfspaceIntersection(halfspaces, res.x)
        return hs.intersections 

    @classmethod
    def from_vertices(cls, vertices:torch.Tensor):
        ch = ConvexHull(vertices)
        M = torch.Tensor(ch.equations[:, :-1])
        b = torch.Tensor(-ch.equations[:, -1])
        return M, b
        


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
        return self._shell

    @staticmethod
    def _are_locs_in_grid(locs: torch.Tensor):
        unique_elements_per_n = torch.tensor([torch.unique(locs[:, i]).numel() for i in range(locs.size(1))])
        return unique_elements_per_n.prod() == torch.unique(locs,dim=0).size(0)

    def _get_upper(self):
        pos_diff = (self._locs_inner.unsqueeze(-3) - self._locs_inner.unsqueeze(-2)).clip(0, torch.inf)
        mask = pos_diff == 0.
        pos_diff[mask] = torch.inf

        upper_inner = self._locs_inner + 0.5 * pos_diff.min(dim=-2).values
        upper_inner = upper_inner.clamp(min=self.shell[0], max=self.shell[1])
        upper_shell = torch.zeros(self._num_dims).fill_(torch.inf)

        return torch.cat((upper_inner, upper_shell.unsqueeze(-2)), dim=-2)

    def _get_lower(self):
        neg_diff = (self._locs_inner.unsqueeze(-3) - self._locs_inner.unsqueeze(-2)).clip(-torch.inf, 0)
        mask = neg_diff == 0.
        neg_diff[mask] = -torch.inf

        lower_inner = self._locs_inner + 0.5 * neg_diff.max(dim=-2).values
        lower_inner = lower_inner.clamp(min=self.shell[0], max=self.shell[1])
        lower_shell = torch.zeros(self._num_dims).fill_(-torch.inf)

        return torch.cat((lower_inner, lower_shell.unsqueeze(-2)), dim=-2)

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def width(self):
        return self.upper - self.lower