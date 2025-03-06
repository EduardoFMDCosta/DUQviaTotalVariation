# import torch
#
# def print_size(func):
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         print(f"GMM Size: {result.size(0)}")
#         return result
#     return wrapper
#
# def print_size_refinement(func):
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         print(f"GMM Size: {result[0].size(0)}")
#         return result
#     return wrapper
#
# # ----------------------------------------------------------------------------------------- #
# # ---------------------------------- Grid generation -------------------------------------- #
# # ----------------------------------------------------------------------------------------- #
# def place_signatures(bounded_regions: torch.Tensor):
#
#     r_min = bounded_regions[:, 0, :]
#     r_max = bounded_regions[:, 1, :]
#
#     return r_min + (r_max - r_min) / 2
#
#
# def outer_point(macro_region: torch.Tensor):
#     dimensions = macro_region.size(1)
#
#     # Calculate max and min coordinates along each dimension
#     max_coords, _ = torch.max(macro_region, dim=0)
#     min_coords, _ = torch.min(macro_region, dim=0)
#
#     # Choose a point close to an arbitrary face
#     outer_point = torch.where(torch.arange(dimensions) != dimensions - 1, (min_coords + max_coords) / 2, max_coords + 1e-1)
#     return outer_point
#
#
# def add_unbounded_representations(regions, signatures, outer_signature):
#
#     r, d = regions.shape[1], regions.shape[-1]
#     unbounded_region = torch.full((1, r, d), torch.inf)
#
#     regions = torch.cat((regions, unbounded_region))
#     signatures = torch.cat((signatures, outer_signature.unsqueeze(0)))
#
#     return regions, signatures
#
#
#
# # ----------------------------------------------------------------------------------------- #
# # --------------------------------- Vertices of the grid ---------------------------------- #
# # ----------------------------------------------------------------------------------------- #
#
# def get_vertices(hypercubes: torch.Tensor):
#
#     if hypercubes.dim() == 2:  #If hypercubes contains only one hypercube
#         hypercubes = hypercubes.unsqueeze(0)
#
#     n, _, dimensions = hypercubes.size()
#
#     # Generate all possible combinations of 0s and 1s
#     combinations = torch.cartesian_prod(*[torch.tensor([0, 1])] * dimensions).float()
#
#     combinations = combinations.unsqueeze(0).expand(n, -1, -1)
#
#     # Determine which dimensions are min or max
#     min_values = hypercubes[:, 0, :]  # (n, d)
#     max_values = hypercubes[:, 1, :]  # (n, d)
#
#     # Compute vertices
#     vertices = combinations * max_values.unsqueeze(1) + (1 - combinations) * min_values.unsqueeze(1)  # (n, 2^d, d)
#
#     return vertices
#
# def get_centered_region(regions, points):
#
#     points = points.unsqueeze(1)
#
#     return torch.sub(regions, points)
#
#
# def identify_high_prob_region(samples: torch.Tensor):
#
#     min_point = torch.min(samples, dim=0).values
#     max_point = torch.max(samples, dim=0).values
#
#     return torch.stack((min_point, max_point), dim=0)
#
#
#
# # ----------------------------------------------------------------------------------------- #
# # ------------------------------------ Recursive grid ------------------------------------- #
# # ----------------------------------------------------------------------------------------- #
# def split_region(region):
#     min_corner, max_corner = region[0], region[1]
#     mid_point = (min_corner + max_corner) / 2
#     subregions = []
#
#     for i in range(2 ** region.shape[1]):
#         binary_representation = torch.tensor([int(x) for x in bin(i)[2:].zfill(region.shape[1])])
#         new_min_corner = torch.where(binary_representation == 0, min_corner, mid_point)
#         new_max_corner = torch.where(binary_representation == 0, mid_point, max_corner)
#         subregions.append(torch.stack([new_min_corner, new_max_corner], dim=0))
#
#     return torch.stack(subregions, dim=0)
#
#
# def recursive_partition(region, condition_fn, samples, min_proportion, min_size, max_depth, current_depth, grid_type):
#     if condition_fn(region, samples, min_proportion, min_size, max_depth, current_depth, grid_type):
#         # Split the region into 2^d smaller regions
#         subregions = split_region(region)
#         result_regions = []
#         for subregion in subregions:
#             result_regions.append(recursive_partition(subregion, condition_fn, samples, min_proportion, min_size, max_depth, current_depth+1, grid_type))
#         return torch.cat(result_regions, dim=0)
#     else:
#         # Base case: condition is not met, return the region itself
#         return region.unsqueeze(0)
#
#
# def generate_regions(macro_region, condition_fn, samples, min_proportion, min_size, max_depth, current_depth, grid_type):
#     return recursive_partition(macro_region, condition_fn, samples, min_proportion, min_size, max_depth, current_depth, grid_type)
#
# def condition(region, samples, min_proportion, min_size, max_depth, current_depth, grid_type):
#     min_bound, max_bound = region
#     inside = (samples >= min_bound).all(dim=1) & (samples <= max_bound).all(dim=1)
#     proportion = torch.mean(inside.float())
#
#     if grid_type == "uniform_grid":
#         return current_depth <= max_depth
#     else:
#         return proportion > min_proportion
#
#
# @print_size
# def create_regions(high_prob_region, samples, min_proportion, min_size, max_depth, current_depth, grid_type):
#
#     regions = generate_regions(high_prob_region, condition, samples, min_proportion, min_size, max_depth, current_depth, grid_type)
#
#     return regions
#
# @print_size_refinement
# def refine_regions(regions, signatures, contributions, threshold):
# #TODO: Vectorize this method
#
#     refined_regions = []
#     contributions = contributions[:-1] #Remove contribution from unbounded region
#
#     for i, contribution in enumerate(contributions):
#
#         if contribution > threshold:
#
#             region_to_transform = regions[i]
#             replacement_regions = split_region(region_to_transform)
#
#             refined_regions.append(replacement_regions)
#
#         else:
#             refined_regions.append(regions[i])
#
#     refined_regions = [t.unsqueeze(0) if t.ndim == 2 else t for t in refined_regions]
#     refined_regions = torch.cat(refined_regions, dim=0)
#     refined_signatures = place_signatures(refined_regions)
#
#     return refined_regions, refined_signatures