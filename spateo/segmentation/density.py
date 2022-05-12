"""Functions to segment regions of a slice by UMI density.
"""
from collections import Counter
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import csr_matrix, issparse, lil_matrix, spmatrix
from sklearn import cluster
from typing_extensions import Literal

from ..configuration import SKM
from ..io.utils import bin_matrix
from ..logging import logger_manager as lm
from . import utils
from .label import _replace_labels


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
    lm.main_debug("Constructing spatial adjacency matrix.")
    adjacency = _create_spatial_adjacency(X.shape)
    X_flattened = X.flatten()
    lm.main_debug("Computing Ward tree.")
    children, _, n_leaves, _, distances = cluster.ward_tree(X_flattened, connectivity=adjacency, return_distance=True)

    # Find distance threshold if not provided
    if not distance_threshold:
        lm.main_debug("Finding dynamic distance threshold by using knee of the top 1000 distances.")
        x = np.sort(np.unique(distances))[-1000:]
        # NOTE: number of clusters needs a +1, as also done in sklearn
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_agglomerative.py#L1017
        y = np.array([(distances >= val).sum() + 1 for val in x])
        kl = KneeLocator(
            x, y, curve="convex", direction="decreasing", online=True, interp_method="polynomial", polynomial_degree=10
        )
        distance_threshold = kl.knee

    n_clusters = (distances >= distance_threshold).sum() + 1
    lm.main_debug("Finding {n_clusters} assignments.")
    assignments = cluster._agglomerative._hc_cut(n_clusters, children, n_leaves)

    return assignments.reshape(X.shape)


def _segment_densities(
    X: Union[spmatrix, np.ndarray], k: int, dk: int, distance_threshold: Optional[float] = None
) -> np.ndarray:
    """Segment a matrix containing UMI counts into regions by UMI density.

    Args:
        X: UMI counts per pixel
        k: Kernel size for Gaussian blur
        dk: Kernel size for final dilation
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.

    Returns:
        Clustering result as a Numpy array of same shape, where clusters are
        indicated by positive integers.
    """
    # Warn on too large array
    if X.size > 5e5:
        lm.main_warning(
            f"Array has {X.size} elements. This may take a while and a lot of memory. "
            "Please consider condensing the array by increasing the binsize."
        )

    # Make dense and normalize.
    if issparse(X):
        lm.main_debug("Converting to dense matrix.")
        X = X.A
    lm.main_debug("Normalizing matrix.")
    X = X / X.max()

    lm.main_debug(f"Applying Gaussian blur with k={k}.")
    X = utils.conv2d(X, k, mode="gauss")

    # Add 1 because 0 should indicate background!
    bins = _schc(X, distance_threshold=distance_threshold) + 1

    lm.main_debug("Dilating labels in ascending mean density order.")
    dilated = np.zeros_like(bins)
    labels = np.unique(bins)
    for label in sorted(labels, key=lambda label: X[bins == label].mean()):
        mask = bins == label
        dilate = cv2.dilate(mask.astype(np.uint8), utils.circle(dk))
        dilated[utils.mclose_mopen(dilate, dk) > 0] = label
    return dilated


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def segment_densities(
    adata: AnnData,
    layer: str,
    binsize: int,
    k: int,
    dk: int,
    distance_threshold: Optional[float] = None,
    background: Optional[Union[Tuple[int, int], Literal[False]]] = None,
    out_layer: Optional[str] = None,
):
    """Segment into regions by UMI density.

    The tissue is segmented into UMI density bins according to the following
    procedure.
    1. The UMI matrix is binned according to `binsize` (recommended >= 20).
    2. The binned UMI matrix (from the previous step) is Gaussian blurred with
        kernel size `k`. Note that `k` is in terms of bins, not pixels.
    3. The elements of the blurred, binned UMI matrix is hierarchically clustered
        with Ward linkage, distance threshold `distance_threshold`, and spatial
        constraints (immediate neighbors). This yields pixel density bins
        (a.k.a. labels) the same shape as the binned matrix.
    4. Each density bin is diluted with kernel size `dk`, starting from the
        bin with the smallest mean UMI (a.k.a. least dense) and going to
        the bin with the largest mean UMI (a.k.a. most dense). This is done in
        an effort to mitigate RNA diffusion and "choppy" borders in subsequent
        steps.
    5. If `background` is not provided, the density bin that is most common in the
        perimeter of the matrix is selected to be background, and thus its label
        is changed to take a value of 0. A pixel can be manually selected to be
        background by providing a `(x, y)` tuple instead. This feature can be
        turned off by providing `False`.
    6. The density bin matrix is resized to be the same size as the original UMI
        matrix.

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to segment based on.
        binsize: Size of bins to use. For density segmentation, pixels are binned
            to reduce runtime. 20 is usually a good starting point. Note that this
            value is relative to the original binsize used to read in the
            AnnData.
        k: Kernel size for Gaussian blur, in bins
        dk: Kernel size for final dilation, in bins
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.
        background: Pixel that should be categorized as background. By
            default, the bin that is most assigned to the outermost pixels are
            categorized as background. Set to False to turn off background detection.
        out_layer: Layer to put resulting bins. Defaults to `{layer}_bins`.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=binsize == 1)
    if binsize > 1:
        lm.main_debug(f"Binning matrix with binsize={binsize}.")
        X = bin_matrix(X, binsize)
        if issparse(X):
            lm.main_debug("Converting to dense matrix.")
            X = X.A
    lm.main_info("Finding density bins.")
    bins = _segment_densities(X, k, dk, distance_threshold)
    if background is not False:
        lm.main_info("Setting background pixels.")
        if background is not None:
            x, y = background
            background_label = bins[x, y]
        else:
            counts = Counter(bins[0]) + Counter(bins[-1]) + Counter(bins[:, 0]) + Counter(bins[:, -1])
            background_label = counts.most_common(1)[0][0]
        bins[bins == background_label] = 0
        bins[bins > background_label] -= 1
    if binsize > 1:
        # Expand back
        bins = cv2.resize(bins, adata.shape[::-1], interpolation=cv2.INTER_NEAREST)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.BINS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, bins)


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def merge_densities(
    adata: AnnData,
    layer: str,
    mapping: Optional[Dict[int, int]] = None,
    out_layer: Optional[str] = None,
):
    """Merge density bins either using an explicit mapping or in a semi-supervised
    way.

    Args:
        adata: Input Anndata
        layer: Layer that was used to generate density bins. Defaults to
            using `{layer}_bins`. If not present, will be taken as a literal.
        mapping: Mapping to use to transform bins
        out_layer: Layer to store results. Defaults to same layer as input.
    """
    # TODO: implement semi-supervised way of merging density bins
    _layer = SKM.gen_new_layer_key(layer, SKM.BINS_SUFFIX)
    if _layer not in adata.layers:
        _layer = layer
    bins = SKM.select_layer_data(adata, _layer)
    lm.main_info(f"Merging densities with mapping {mapping}.")
    replaced = _replace_labels(bins, mapping)
    SKM.set_layer_data(adata, out_layer or _layer, replaced)
