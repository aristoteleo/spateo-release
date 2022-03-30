"""Functions for use when labeling individual nuclei/cells, after obtaining a
mask.
"""
from typing import Dict, Optional, Union

import cv2
import numpy as np
from anndata import AnnData
from numba import njit
from scipy.sparse import issparse, spmatrix
from skimage import filters, segmentation

from ...configuration import SKM
from ...errors import PreprocessingError
from . import utils


def _replace_labels(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Replace labels according to mapping.

    Args:
        labels: Numpy array containing integer labels.
        mapping: Dictionary mapping from labels to labels.

    Returns:
        Replaced labels
    """
    replacement = np.full(labels.max() + 1, -1, dtype=int)
    for from_label, to_label in mapping.items():
        replacement[from_label] = to_label

    new_labels = labels.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            new_label = replacement[labels[i, j]]
            if new_label >= 0:
                new_labels[i, j] = new_label
    return new_labels


def replace_labels(adata: AnnData, layer: str, mapping: Dict[int, int], out_layer: Optional[str] = None):
    """Replace labels according to mapping.

    Args:
        adata: Input Anndata
        layer: Layer containing labels to replace
        mapping: Dictionary mapping that defines label replacement.
        out_layer: Layer to save results. By default, the input layer is
            overridden.
    """
    labels = SKM.select_layer_data(adata, layer)
    new_labels = _replace_labels(labels, mapping)
    SKM.set_layer_data(adata, out_layer or layer, new_labels)


def _watershed(
    X: np.ndarray,
    mask: np.ndarray,
    markers: np.ndarray,
    k: int,
) -> np.ndarray:
    """Assign individual nuclei/cells using the Watershed algorithm.

    Args:
        X: Data array. This array will be Gaussian blurred and used as the
            input values to Watershed.
        mask: Nucleus/cell mask.
        markers: Numpy array indicating where the Watershed markers are. May
            either be a boolean or integer array. If this is a boolean array,
            the markers are identified by calling `cv2.connectedComponents`.
        k: Size of the kernel to use for Gaussian blur.

    Returns:
        Watershed labels.
    """
    blur = utils.conv2d(X, k, mode="gauss")
    if markers.dtype == np.dtype(bool):
        markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
    watershed = segmentation.watershed(-blur, markers, mask=mask)
    return watershed


def watershed_markers(
    adata: AnnData,
    layer: str = SKM.STAIN_LAYER_KEY,
    k: int = 3,
    square: bool = False,
    min_area: int = 80,
    n_iter: int = -1,
    float_k: int = 5,
    float_threshold: Optional[float] = None,
    out_layer: Optional[str] = None,
):
    """Find markers for used in Watershed.

    Args:
        adata: Input Anndata
        layer: Layer that was used to create scores or masks. If `{layer}_scores`
            is present, that is used. Otherwise if `{layer}_mask` is present,
            that is used. Otherwise, the layer is taken as a literal.
        k: Erosion kernel size
        square: Whether to use a square kernel
        min_area: Minimum area
        n_iter: Number of erosions to perform.
        float_k: Morphological close and open kernel size when `X` is a
            float array.
        float_threshold: Threshold to use to determine connected components
            when `X` is a float array. By default, a threshold is automatically
            determined by using Otsu method.
        out_layer: Layer to save results. By default, this will be `{layer}_markers`.
    """
    _layer1 = SKM.gen_new_layer_key(layer, SKM.SCORES_SUFFIX)
    _layer2 = SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    if _layer1 not in adata.layers and _layer2 not in adata.layers and layer not in adata.layers:
        raise PreprocessingError(
            f'Neither "{_layer1}", "{_layer2}", nor "{layer}" are present in AnnData. '
            "Please run either `st.pp.segmentation.icell.mask_nuclei_from_stain` "
            "or `st.pp.segmentation.score_and_mask_pixels` first."
        )
    _layer = layer
    if _layer1 in adata.layers:
        _layer = _layer1
    elif _layer2 in adata.layers:
        _layer = _layer2
    X = SKM.select_layer_data(adata, _layer, make_dense=True)
    if np.issubdtype(X.dtype, np.floating) and not float_threshold:
        float_threshold = filters.threshold_otsu(X)
    markers = utils.safe_erode(X, k, square, min_area, n_iter, float_k, float_threshold)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MARKERS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, markers)


def watershed(
    adata: AnnData,
    layer: str = SKM.STAIN_LAYER_KEY,
    k: int = 3,
    mask_layer: Optional[str] = None,
    markers_layer: Optional[str] = None,
    out_layer: Optional[str] = None,
):
    """Assign individual nuclei/cells using the Watershed algorithm.

    Args:
        adata: Input AnnData
        layer: Original data layer from which segmentation will derive from.
        k: Size of the kernel to use for Gaussian blur.
        mask_layer: Layer containing mask. This will default to `{layer}_mask`.
        markers_layer: Layer containing Watershed markers. This will default to
            `{layer}_markers`. May either be a boolean or integer array.
            If this is a boolean array, the markers are identified by calling
            `cv2.connectedComponents`.
        out_layer: Layer to save results. Defaults to `{layer}_labels`.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    mask_layer = mask_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    mask = SKM.select_layer_data(adata, mask_layer)
    markers_layer = markers_layer or SKM.gen_new_layer_key(layer, SKM.MARKERS_SUFFIX)
    markers = SKM.select_layer_data(adata, markers_layer)
    # Markers should always be included in the mask.
    labels = _watershed(X, mask | (markers > 0), markers, k)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)


def _expand_labels(labels: np.ndarray, distance: int, max_area: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Expand labels up to a certain distance, while ignoring labels that are
    above a certain size.

    Args:
        labels: Numpy array containing integer labels.
        distance: Distance to expand. Internally, this is used as the number
            of iterations of distance 1 dilations.
        max_area: Maximum area of each label.
        mask: Only expand within the provided mask.

    Returns:
        New label array with expanded labels.
    """

    @njit
    def _expand(X, areas, kernel, max_area, n_iter, mask):
        pad = kernel.shape[0] // 2
        expanded = np.zeros((X.shape[0] + 2 * pad, X.shape[1] + 2 * pad), dtype=X.dtype)
        expanded[pad:-pad, pad:-pad] = X
        for _ in range(n_iter):
            new_areas = np.zeros_like(areas)
            _expanded = np.zeros_like(expanded)
            for _i in range(X.shape[0]):
                i = _i + pad
                for _j in range(X.shape[1]):
                    j = _j + pad
                    if expanded[i, j] > 0:
                        _expanded[i, j] = expanded[i, j]
                        continue
                    if not mask[_i, _j]:
                        continue

                    neighbors = expanded[i - pad : i + pad + 1, j - pad : j + pad + 1]
                    unique = np.unique(neighbors * kernel)
                    unique_labels = unique[unique > 0]
                    if len(unique_labels) == 1:
                        label = unique_labels[0]
                        if areas[label] < max_area:
                            _expanded[i, j] = label
                            new_areas[label] += 1
            expanded = _expanded
            areas += new_areas
        return expanded[pad:-pad, pad:-pad]

    return _expand(
        labels,
        np.bincount(labels.flatten()),
        utils.circle(3),
        max_area,
        distance,
        np.ones(labels.shape, dtype=bool) if mask is None else mask,
    )


def expand_labels(
    adata: AnnData,
    layer: str,
    distance: int = 5,
    max_area: int = 400,
    mask_layer: Optional[str] = None,
    out_layer: Optional[str] = None,
):
    """Expand labels up to a certain distance.

    Args:
        adata: Input Anndata
        layer: Layer from which the labels were derived. Then, `{layer}_labels`
            is used as the labels. If not present, it is taken as a literal.
        distance: Distance to expand. Internally, this is used as the number
            of iterations of distance 1 dilations.
        max_area: Maximum area of each label.
        out_layer: Layer to save results. By default, uses `{layer}_labels_expanded`.
    """
    label_layer = SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    if label_layer not in adata.layers:
        label_layer = layer
    labels = SKM.select_layer_data(adata, label_layer)
    expanded = _expand_labels(labels, distance, max_area)
    out_layer = out_layer or SKM.gen_new_layer_key(label_layer, SKM.EXPANDED_SUFFIX)
    SKM.set_layer_data(adata, out_layer, expanded)


def _label_connected_components(
    X: np.ndarray,
    area_threshold: int = 500,
    k: int = 3,
    min_area: int = 100,
    n_iter: int = -1,
    distance: int = 8,
    max_area: int = 400,
    seed_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Label connected components while splitting components that are too large.

    Args:
        X: Boolean mask to compute connected components from.
        area_threshold: Connected components with area greater than this value
            will be split into smaller portions by first eroding and then
            expanding.
        k: Kernel size for erosion.
        min_area: Don't erode labels smaller than this area.
        n_iter: Number of erosion operations. -1 means continue eroding until
            every label is less than `min_area`.
        distance: Distance to expand eroded labels.
        max_area: Maximum area when expanding labels.
        seed_labels: Seed labels.

    Returns:
        New label array
    """
    components = cv2.connectedComponentsWithStats(X.astype(np.uint8))
    areas = components[2][:, cv2.CC_STAT_AREA]
    to_erode = np.zeros(X.shape, dtype=bool)
    saved = np.zeros(X.shape, dtype=int)
    saved_i = (seed_labels.max() + 1) if seed_labels is not None else 1

    for label, area in enumerate(areas):
        if label > 0:
            stats = components[2][label]
            left, top, width, height = (
                stats[cv2.CC_STAT_LEFT],
                stats[cv2.CC_STAT_TOP],
                stats[cv2.CC_STAT_WIDTH],
                stats[cv2.CC_STAT_HEIGHT],
            )
            label_mask = components[1][top : top + height, left : left + width] == label
            if seed_labels is not None:
                if (seed_labels[top : top + height, left : left + width][label_mask] > 0).any():
                    continue

            if area <= area_threshold:
                saved[top : top + height, left : left + width] += label_mask * saved_i
                saved_i += 1
            else:
                to_erode[top : top + height, left : left + width] += label_mask
    eroded = utils.safe_erode(to_erode, k=k, min_area=min_area, n_iter=n_iter)
    labels = cv2.connectedComponents(eroded.astype(np.uint8))[1]
    labels[labels > 0] += saved_i - 1
    if seed_labels is not None:
        labels += seed_labels
    expanded = _expand_labels(labels, distance=distance, max_area=max_area, mask=X > 0)
    return saved + expanded


def label_connected_components(
    adata: AnnData,
    layer: str,
    seed_layer: Optional[str] = None,
    area_threshold: int = 500,
    k: int = 3,
    min_area: int = 100,
    n_iter: int = -1,
    distance: int = 8,
    max_area: int = 400,
    out_layer: Optional[str] = None,
):
    """Label connected components while splitting components that are too large.

    Args:
        adata: Input Anndata
        layer: Data layer that was used to generate the mask. First, will look
            for `{layer}_mask`. Otherwise, this will be use as a literal.
        seed_layer: Layer containing seed labels. These are labels that should be
            used whenever possible in labeling connected components.
        area_threshold: Connected components with area greater than this value
            will be split into smaller portions by first eroding and then
            expanding.
        k: Kernel size for erosion.
        min_area: Don't erode labels smaller than this area.
        n_iter: Number of erosion operations. -1 means continue eroding until
            every label is less than `min_area`.
        distance: Distance to expand eroded labels.
        max_area: Maximum area when expanding labels.
        out_layer: Layer to save results. Defaults to `{layer}_labels`.

    Returns:
        New label array
    """
    mask_layer = SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    if mask_layer not in adata.layers:
        mask_layer = layer
    mask = SKM.select_layer_data(adata, mask_layer)
    seed_labels = SKM.select_layer_data(adata, seed_layer) if seed_layer else None
    labels = _label_connected_components(mask, area_threshold, k, min_area, n_iter, distance, max_area, seed_labels)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)
