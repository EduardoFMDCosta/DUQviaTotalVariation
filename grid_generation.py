import numpy as np
import math
from scipy.stats import norm

import propagation_methods as propag
import bounds_linear_system_2d as bounds_linear

def print_array_size(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Number of signatures: {len(result)}")
        return result
    return wrapper

# ----------------------------------------------------------------------------------------- #
# ---------------------------------- Grid generation -------------------------------------- #
# ----------------------------------------------------------------------------------------- #

def generate_regions(*lists):
    if len(lists) == 0:
        return []

    # Get the lengths of each list
    lengths = [len(lst) for lst in lists]

    # Create slices for each list
    slices = [slice(None)] * len(lists)
    regions = []

    # Recursive function to generate regions
    def generate_region(slice_indices):
        if len(slice_indices) == len(lists):
            region = np.array([[lst[i-1], lst[i]] for lst, i in zip(lists, slice_indices)])
            regions.append(region.T)
        else:
            for i in range(1, lengths[len(slice_indices)]):
                new_slice_indices = slice_indices + [i]
                generate_region(new_slice_indices)

    generate_region([])

    return np.array(regions)


def computeOuterPoint(region):
    dimensions = len(region[0])  # Get the number of dimensions
    max_coords = [max(region[i][d] for i in range(len(region))) for d in range(dimensions)]
    min_coords = [min(region[i][d] for i in range(len(region))) for d in range(dimensions)]
    
    # Choose a point close to the upper face
    outer_point = [(min_coords[d] + max_coords[d]) / 2 if d != dimensions - 1 else max_coords[d] + 1e-1 for d in range(dimensions)]
    
    return np.array(outer_point)


def computePosition(minimum, maximum):
    return minimum + (maximum - minimum) / 2


def placeSignatures(bounded_regions):
    
    # Extract dimensions of regions
    num_regions, _, num_dimensions = bounded_regions.shape

    # Compute signature points for all bounded regions
    signature_points = np.empty((num_regions, num_dimensions))
    for dim in range(num_dimensions):
        signature_points[:, dim] = computePosition(bounded_regions[:, 0, dim], bounded_regions[:, 1, dim])

    return signature_points


def addUnboundedRepresentations(regions, unbounded_region, signatures, outer_signature):

    regions = np.concatenate((regions, [unbounded_region]))
    signatures = np.concatenate((signatures, [outer_signature]))

    return regions, signatures



# ----------------------------------------------------------------------------------------- #
# --------------------------------- Vertices of the grid ---------------------------------- #
# ----------------------------------------------------------------------------------------- #

def getVertices(cube):
    
    dimensions = len(cube[0])  # Get the dimension of the cube
    vertices = []

    for i in range(2 ** dimensions):
        vertex = []
        for j in range(dimensions):
            if (i >> j) & 1:
                vertex.append(cube[1][j])  # Use max value for this dimension
            else:
                vertex.append(cube[0][j])  # Use min value for this dimension
        vertices.append(vertex)

    return np.array(vertices)


def findMinMaxPoints(samples):

    min_point = np.min(samples, axis=0)
    max_point = np.max(samples, axis=0)

    return min_point, max_point


# ----------------------------------------------------------------------------------------- #
# ------------------------------------ Recursive grid ------------------------------------- #
# ----------------------------------------------------------------------------------------- #
def euclideanDistance(point1, point2):
    return np.sqrt(np.sum(point1 - point2) ** 2)

def regionSize(region):
    
    vertices = getVertices(region)
    distance = euclideanDistance(vertices[0], vertices[-1])

    return distance

def pointInsideRegion(point, cube):    
    min_bound, max_bound = cube
    if np.any(point < min_bound) or np.any(point > max_bound):
        return False
    return True

def checkProportionInsideRegion(points, cube):
    min_bound, max_bound = cube
    return np.mean((np.all(points >= min_bound, axis=1)) & (np.all(points <= max_bound, axis=1)))


def check_condition(region, samples, min_proportion, min_size):
    condition_proportion = checkProportionInsideRegion(samples, region) > min_proportion
    condition_size = regionSize(region) > min_size

    return condition_proportion & condition_size


def subdivideRegion(region, samples, min_proportion, min_size):
    subregions = []
    if check_condition(region, samples, min_proportion, min_size):
        # If condition is true, subdivide the region in half
        x_min, y_min = region[0]
        x_max, y_max = region[1]

        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        
        # Define subregion boundaries
        subregion1 = np.array([[x_min, y_min], [mid_x, mid_y]])
        subregion2 = np.array([[mid_x, y_min], [x_max, mid_y]])
        subregion3 = np.array([[x_min, mid_y], [mid_x, y_max]])
        subregion4 = np.array([[mid_x, mid_y], [x_max, y_max]])

        # Recursively check each subregion and collect subregions
        subregions.extend(subdivideRegion(subregion1, samples, min_proportion, min_size))
        subregions.extend(subdivideRegion(subregion2, samples, min_proportion, min_size))
        subregions.extend(subdivideRegion(subregion3, samples, min_proportion, min_size))
        subregions.extend(subdivideRegion(subregion4, samples, min_proportion, min_size))

    else:
        # If condition is false, append the region to the list of subregions
        subregions.append(region)
    return subregions


def refineRegions(regions, signatures, contributions, threshold):
    new_regions = []
    new_signatures = []
    
    for i, contribution in enumerate(contributions):
        if contribution > threshold and not np.isinf(regions[i][0][0]):
            x_min, y_min = regions[i][0]
            x_max, y_max = regions[i][1]
            
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            
            # Subdivide the region into four equal parts
            new_regions.append([[x_min, y_min], [x_mid, y_mid]])  # Region 1
            new_signatures.append([(x_min + x_mid)/2, (y_min + y_mid)/2])

            new_regions.append([[x_mid, y_min], [x_max, y_mid]])  # Region 2
            new_signatures.append([(x_mid + x_max)/2, (y_min + y_mid)/2])

            new_regions.append([[x_min, y_mid], [x_mid, y_max]])  # Region 3
            new_signatures.append([(x_min + x_mid)/2, (y_mid + y_max)/2])

            new_regions.append([[x_mid, y_mid], [x_max, y_max]])  # Region 4
            new_signatures.append([(x_mid + x_max)/2, (y_mid + y_max)/2])
            
        else:
            new_regions.append(regions[i])
            new_signatures.append(signatures[i])
    
    return np.array(new_regions), np.array(new_signatures)