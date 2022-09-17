from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyvista import PolyData
from scipy.sparse import csr_matrix, diags, issparse, lil_matrix, spmatrix
from scipy.spatial import ConvexHull, Delaunay, cKDTree


def rescaling(mat: Union[np.ndarray, spmatrix], new_shape: Union[List, Tuple]) -> Union[np.ndarray, spmatrix]:
    """This function rescale the resolution of the input matrix that represents a spatial domain. For example, if you
    want to decrease the resolution of a matrix by a factor of 2, the new_shape will be `mat.shape / 2`.

    Args:
        mat: The input matrix of the spatial domain (or an image).
        new_shape: The rescaled shape of the spatial domain, each dimension must be an factorial of the original
                    dimension.

    Returns:
        res: the spatial resolution rescaled matrix.
    """
    shape = (new_shape[0], mat.shape[0] // mat[0], new_shape[1], mat.shape[1] // mat[1])

    res = mat.reshape(shape).sum(-1).sum(1)
    return res


def get_mapper(smoothed=True):
    mapper = {
        "X_spliced": "M_s" if smoothed else "X_spliced",
        "X_unspliced": "M_u" if smoothed else "X_unspliced",
        "X_new": "M_n" if smoothed else "X_new",
        "X_old": "M_o" if smoothed else "X_old",
        "X_total": "M_t" if smoothed else "X_total",
        # "X_uu": "M_uu" if smoothed else "X_uu",
        # "X_ul": "M_ul" if smoothed else "X_ul",
        # "X_su": "M_su" if smoothed else "X_su",
        # "X_sl": "M_sl" if smoothed else "X_sl",
        # "X_protein": "M_p" if smoothed else "X_protein",
        "X": "X" if smoothed else "X",
    }
    return mapper


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())

    return dict1


def flatten(arr):
    if type(arr) == pd.core.series.Series:
        ret = arr.values.flatten()
    elif sp.issparse(arr):
        ret = arr.A.flatten()
    else:
        ret = arr.flatten()
    return ret


def calc_1nd_moment(X, W, normalize_W=True):
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W
        return W @ X, W
    else:
        return W @ X


def affine_transform(X, A, b):
    X = np.array(X)
    A = np.array(A)
    b = np.array(b)
    return (A @ X.T).T + b


def gen_rotation_2d(degree: float):
    from math import cos, radians, sin

    rad = radians(degree)
    R = [
        [cos(rad), -sin(rad)],
        [sin(rad), cos(rad)],
    ]
    return np.array(R)


def compute_smallest_distance(
    coords: np.ndarray, leaf_size: int = 40, sample_num=None, use_unique_coords=True
) -> float:
    """Compute and return smallest distance. A wrapper for sklearn API
    Parameters
    ----------
        coords:
            NxM matrix. N is the number of data points and M is the dimension of each point's feature.
        leaf_size : int, optional
            Leaf size parameter for building Kd-tree, by default 40.
        sample_num:
            The number of cells to be sampled.
        use_unique_coords:
            Whether to remove duplicate coordinates
    Returns
    -------
        min_dist: float
            the minimum distance between points
    """
    if len(coords.shape) != 2:
        raise ValueError("Coordinates should be a NxM array.")
    if use_unique_coords:
        # main_info("using unique coordinates for computing smallest distance")
        coords = [tuple(coord) for coord in coords]
        coords = np.array(list(set(coords)))
    # use cKDTree which is implmented in C++ and is much faster than KDTree
    kd_tree = cKDTree(coords, leafsize=leaf_size)
    if sample_num is None:
        sample_num = len(coords)
    N, _ = min(len(coords), sample_num), coords.shape[1]
    selected_estimation_indices = np.random.choice(len(coords), size=N, replace=False)

    # Note k=2 here because the nearest query is always a point itself.
    distances, _ = kd_tree.query(coords[selected_estimation_indices, :], k=2)
    min_dist = min(distances[:, 1])

    return min_dist


def polyhull(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> PolyData:
    """Create a PolyData object from the convex hull constructed with the input data points.

    scipy's ConvexHull to be 500X faster than using vtkDelaunay3D and vtkDataSetSurfaceFilter because you skip the
    expensive 3D tesselation of the volume.

    Args:
        x: x coordinates of the data points.
        y: y coordinates of the data points.
        z: z coordinates of the data points.

    Returns:
        poly: a PolyData object generated with the convex hull constructed based on the input data points.
    """
    hull = ConvexHull(np.column_stack((x, y, z)))
    faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
    poly = PolyData(hull.points, faces)
    return hull, poly


def in_hull(p: np.ndarray, hull: Tuple[Delaunay, np.ndarray]) -> np.ndarray:
    """Test if points in `p` are in `hull`

    Args:
        p: a `N x K` coordinates of `N` points in `K` dimensions
        hull: either a scipy.spatial.Delaunay object or the `MxK` array of the coordinates of `M` points in `K`
        dimensions for which Delaunay triangulation will be computed.

    Returns:
        res: A numpy array with boolean values indicating whether the input points is in the convex hull.
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    res = hull.find_simplex(p) >= 0
    return res
