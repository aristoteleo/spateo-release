# code adapted from https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/tool/utils.py
import numpy as np
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import issparse, csr_matrix, lil_matrix, diags
from scipy.spatial import cKDTree


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
    from math import cos, sin, radians

    rad = radians(degree)
    R = [
        [cos(rad), -sin(rad)],
        [sin(rad), cos(rad)],
    ]
    return np.array(R)


def compute_smallest_distance(coords: list, leaf_size: int = 40, sample_num=None, use_unique_coords=True) -> float:
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
