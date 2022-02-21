"""Functions to segment regions of a slice by UMI density.
"""
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import csr_matrix, issparse, lil_matrix, spmatrix
from sklearn import cluster
from tqdm import tqdm

from . import utils
from ...configuration import SKM
from ...io.utils import bin_matrix
from ...warnings import PreprocessingWarning


def _create_spatial_adjacency(shape: Tuple[int, int]) -> csr_matrix:
    """Create a sparse adjacency matrix for a 2D grid graph of specified shape.
    https://stackoverflow.com/a/16342639

    Args:
        shape: Shape of grid

    Returns:
        A sparse adjacency matrix
    """
    n_rows, n_cols = shape
    n_nodes = n_rows * n_cols
    adjacency = lil_matrix((n_nodes, n_nodes))
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            # Two inner diagonals
            if c > 0:
                adjacency[i - 1, i] = adjacency[i, i - 1] = 1
            # Two outer diagonals
            if r > 0:
                adjacency[i - n_cols, i] = adjacency[i, i - n_cols] = 1
    return adjacency.tocsr()


def _schc(X: np.ndarray, distance_threshold: Optional[float] = None) -> np.ndarray:
    """Spatially-constrained hierarchical clustering.

    Perform hierarchical clustering with Ward linkage on an array
    containing UMI counts per pixel. Spatial constraints are
    imposed by limiting the neighbors of each node to immediate 4
    pixel neighbors.

    This function runs in two steps. First, it computes a Ward linkage tree
    by calling :func:`sklearn.cluster.ward_tree`, with `return_distance=True`,
    which yields distances between clusters. then, if `distance_threshold` is not
    provided, a dynamic threshold is calculated by finding the inflection (knee)
    of the distance (x) vs number of clusters (y) line using the top 1000
    distances, making the assumption that for the vast majority of cases, there
    will be less than 1000 density clusters.

    Args:
        X: UMI counts per pixel
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.

    Returns:
        Clustering result as a Numpy array of same shape, where clusters are
        indicated by integers.
    """
    adjacency = _create_spatial_adjacency(X.shape)
    X_flattened = X.flatten()
    children, _, n_leaves, _, distances = cluster.ward_tree(X_flattened, connectivity=adjacency, return_distance=True)

    # Find distance threshold if not provided
    if not distance_threshold:
        x = np.sort(np.unique(distances))[-1000:]
        # NOTE: number of clusters needs a +1, as also done in sklearn
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_agglomerative.py#L1017
        y = np.array([(distances >= val).sum() + 1 for val in x])
        kl = KneeLocator(
            x, y, curve="convex", direction="decreasing", online=True, interp_method="polynomial", polynomial_degree=10
        )
        distance_threshold = kl.knee

    n_clusters = (distances >= distance_threshold).sum() + 1
    assignments = cluster._agglomerative._hc_cut(n_clusters, children, n_leaves)

    return assignments.reshape(X.shape)


def _segment_densities(
    X: Union[spmatrix, np.ndarray], k: int, distance_threshold: Optional[float] = None, dk: int = 3
) -> np.ndarray:
    """Segment a matrix containing UMI counts into regions by UMI density.

    Args:
        X: UMI counts per pixel
        k: Kernel size for Gaussian blur
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.
        dk: Kernel size for final dilation

    Returns:
        Clustering result as a Numpy array of same shape, where clusters are
        indicated by positive integers.
    """
    # Warn on too large array
    if X.size > 5e5:
        warnings.warn(
            f"Array has {X.size} elements. This may take a while and a lot of memory. "
            "Please consider condensing the array by increasing the binsize."
        )

    # Make dense and normalize.
    if issparse(X):
        X = X.A
    X = X / X.max()

    X = utils.conv2d(X, k, mode="gauss")

    # Add 1 because 0 should indicate background!
    bins = _schc(X, distance_threshold=distance_threshold) + 1

    dilated = np.zeros_like(bins)
    labels = np.unique(bins)
    for label in sorted(labels, key=lambda label: X[np.where(bins == label)].mean()):
        mask = bins == label
        dilate = cv2.dilate(mask.astype(np.uint8), utils.circle(dk))
        where = np.where(utils.mclose_mopen(dilate, dk) > 0)
        dilated[where] = label
    return dilated


def segment_densities(
    adata: AnnData,
    layer: str,
    binsize: int,
    k: int = 11,
    distance_threshold: Optional[float] = None,
    dk: int = 5,
    out_layer: Optional[str] = None,
):
    """Segment into regions by UMI density.

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to segment based on.
        binsize: Size of bins to use. For density segmentation, pixels are binned
            to reduce runtime. 10 is usually a good starting point. Note that this
            value is relative to the original binsize used to read in the
            AnnData.
        k: Kernel size for Gaussian blur
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.
        dk: Kernel size for final dilation
        out_layer: Lyaer to put resulting bins. Defaults to `{layer}_bins`.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=binsize == 1)
    if binsize > 1:
        X = bin_matrix(X, binsize)
        if issparse(X):
            X = X.A
    bins = _segment_densities(X, k, distance_threshold, dk)
    if binsize > 1:
        # Expand back
        bins = cv2.resize(bins, adata.shape[::-1], interpolation=cv2.INTER_NEAREST)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.BINS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, bins)
