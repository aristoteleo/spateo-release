"""Functions for use when labeling individual nuclei/cells, after obtaining a
mask.
"""
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from anndata import AnnData
from joblib import Parallel, delayed
from numba import njit
from skimage import feature, filters, measure, segmentation
from sympy import Segment
from tqdm import tqdm

from ..configuration import SKM, config
from ..errors import SegmentationError
from ..logging import logger_manager as lm
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


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
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
    lm.main_info(f"Replacing labels with mapping {mapping}")
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
        lm.main_debug("Finding connected components.")
        markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
    lm.main_debug("Running Watershed algorithm.")
    watershed = segmentation.watershed(-blur, markers, mask=mask)
    return watershed


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def find_peaks_with_erosion(
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
    """Find peaks for use in Watershed via iterative erosion.

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
        raise SegmentationError(
            f'Neither "{_layer1}", "{_layer2}", nor "{layer}" are present in AnnData. '
            "Please run either `st.cs.mask_nuclei_from_stain` or `st.cs.score_and_mask_pixels` first."
        )
    _layer = layer
    if _layer1 in adata.layers:
        _layer = _layer1
    elif _layer2 in adata.layers:
        _layer = _layer2
    X = SKM.select_layer_data(adata, _layer, make_dense=True)
    if np.issubdtype(X.dtype, np.floating) and not float_threshold:
        lm.main_debug("Finding threshold with Multi-otsu.")
        float_threshold = filters.threshold_otsu(X)
    lm.main_info("Finding Watershed markers with iterative erosion.")
    markers = utils.safe_erode(X, k, square, min_area, n_iter, float_k, float_threshold)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MARKERS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, markers)


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
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
    lm.main_info("Running Watershed.")
    # Markers should always be included in the mask.
    labels = _watershed(X, mask | (markers > 0), markers, k)
    areas = np.bincount(labels.flatten())
    if (areas[1:] > 10000).any():
        lm.main_warning(
            "Some labels have area greater than 10000. If you are segmenting based on RNA, consider "
            "using `st.cs.label_connected_components` instead."
        )
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)


def _expand_labels(
    labels: np.ndarray,
    distance: int,
    max_area: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
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
    masked_labels = labels[mask] if mask is not None else labels
    if (masked_labels > 0).all() or (masked_labels == 0).all():
        return labels

    @njit
    def _expand(X, areas, max_area, mask, start_i, end_i):
        expanded = X[start_i:end_i].copy()
        new_areas = np.zeros_like(areas)
        n_neighbors = 0
        neighbors = np.zeros(4, dtype=X.dtype)
        for i in range(start_i, end_i):
            for j in range(X.shape[1]):
                if X[i, j] > 0 or not mask[i, j]:
                    continue

                if i - 1 >= 0:
                    neighbors[n_neighbors] = X[i - 1, j]
                    n_neighbors += 1
                if i + 1 < X.shape[0]:
                    neighbors[n_neighbors] = X[i + 1, j]
                    n_neighbors += 1
                if j - 1 >= 0:
                    neighbors[n_neighbors] = X[i, j - 1]
                    n_neighbors += 1
                if j + 1 < X.shape[1]:
                    neighbors[n_neighbors] = X[i, j + 1]
                    n_neighbors += 1
                unique = np.unique(neighbors[:n_neighbors])
                unique_labels = unique[unique > 0]
                if len(unique_labels) == 1:
                    label = unique_labels[0]
                    if areas[label] < max_area:
                        expanded[i - start_i, j] = label
                        new_areas[label] += 1
                n_neighbors = 0
        return expanded, new_areas

    areas = np.bincount(labels.flatten())
    mask = np.ones(labels.shape, dtype=bool) if mask is None else mask
    step = math.ceil(labels.shape[0] / config.n_threads)
    expanded = labels.copy()
    with Parallel(n_jobs=config.n_threads) as parallel:
        for _ in tqdm(range(distance), desc="Expanding"):
            new_areas = np.zeros_like(areas)
            subis = range(0, labels.shape[0], step)
            sublabels = []
            submasks = []
            for i in subis:
                sl = slice(max(0, i - 1), min(labels.shape[0], i + step + 1))
                sublabels.append(expanded[sl])
                submasks.append(mask[sl])
            for i, (_expanded, _new_areas) in zip(
                subis,
                parallel(
                    delayed(_expand)(
                        sl, areas, max_area, sm, int(i - 1 >= 0), sl.shape[0] - int(i + step + 1 < labels.shape[0])
                    )
                    for i, sl, sm in zip(subis, sublabels, submasks)
                ),
            ):
                expanded[i : i + step] = _expanded
                new_areas += _new_areas
            areas += new_areas

    return expanded


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
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
        mask_layer: Layer containing mask to restrict expansion to within.
        out_layer: Layer to save results. By default, uses `{layer}_labels_expanded`.
    """
    label_layer = SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    if label_layer not in adata.layers:
        label_layer = layer
    labels = SKM.select_layer_data(adata, label_layer)
    mask = SKM.select_layer_data(adata, mask_layer) if mask_layer else None
    lm.main_info("Expanding labels.")
    expanded = _expand_labels(labels, distance, max_area, mask=mask)
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
            if seed_labels is not None and (seed_labels[top : top + height, left : left + width][label_mask] > 0).any():
                continue

            if area <= area_threshold:
                saved[top : top + height, left : left + width] += label_mask * saved_i
                saved_i += 1
            else:
                to_erode[top : top + height, left : left + width] += label_mask
    erode = (to_erode > 0).any()
    if erode:
        eroded = utils.safe_erode(to_erode, k=k, min_area=min_area, n_iter=n_iter)
        labels = cv2.connectedComponents(eroded.astype(np.uint8))[1]
        labels[labels > 0] += saved_i - 1
    elif seed_labels is None:
        return saved
    else:
        labels = np.zeros_like(saved)
    if seed_labels is not None:
        labels += seed_labels
    expanded = _expand_labels(labels, distance=distance, max_area=max_area, mask=X > 0)
    return saved + expanded


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
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


def _find_peaks(X: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks from an arbitrary image.

    This function is a wrapper around :func:`feature.peak_local_max`.

    Args:
        X: Array to find peaks from
        **kwargs: Keyword arguments to pass to :func:`feature.peak_local_max`.

    Returns:
        Numpy array of the same size as `X` where each peak is labeled with a unique positive
        integer.
    """
    _kwargs = dict(p_norm=2)
    _kwargs.update(kwargs)
    peak_idx = feature.peak_local_max(X, **_kwargs)
    peaks = np.zeros(X.shape, dtype=int)
    for label, (i, j) in enumerate(peak_idx):
        peaks[i, j] = label + 1
    return peaks


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def find_peaks(
    adata: AnnData,
    layer: str,
    k: int,
    min_distance: int,
    mask_layer: Optional[str] = None,
    out_layer: Optional[str] = None,
):
    """Find peaks from an array.

    Args:
        adata: Input AnnData
        layer: Layer to use as values to find peaks from.
        k: Apply a Gaussian blur with this kernel size prior to peak detection.
        min_distance: Minimum distance, in pixels, between peaks.
        mask_layer: Find peaks only in regions specified by the mask.
        out_layer: Layer to save identified peaks as markers. By default, uses
            `{layer}_markers`.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    if X.dtype == np.dtype(bool):
        raise SegmentationError(
            f"Layer {layer} contains a boolean array. Please use `st.cs.find_peaks_from_mask` instead."
        )

    X = utils.conv2d(X, k, mode="gauss")
    peaks = _find_peaks(X, min_distance=min_distance)
    if mask_layer:
        peaks *= SKM.select_layer_data(adata, mask_layer)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MARKERS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, peaks)


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def find_peaks_from_mask(
    adata: AnnData,
    layer: str,
    min_distance: int,
    distances_layer: Optional[str] = None,
    markers_layer: Optional[str] = None,
):
    """Find peaks from a boolean mask. Used to obatin Watershed markers.

    Args:
        adata: Input AnnData
        layer: Layer containing boolean mask. This will default to `{layer}_mask`.
            If not present in the provided AnnData, this argument used as a literal.
        min_distance: Minimum distance, in pixels, between peaks.
        distances_layer: Layer to save distance from each pixel to the nearest zero (False)
            pixel (a.k.a. distance transform). By default, uses `{layer}_distances`.
        markers_layer: Layer to save identified peaks as markers. By default, uses
            `{layer}_markers`.
    """
    mask_layer = SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    if mask_layer not in adata.layers:
        mask_layer = layer
    mask = SKM.select_layer_data(adata, mask_layer)
    if mask.dtype != np.dtype(bool):
        raise SegmentationError(f"Only boolean masks are supported for this function, but got {mask.dtype} instead.")
    lm.main_info(f"Finding peaks with minimum distance {min_distance}.")
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    peaks = _find_peaks(distances, min_distance=min_distance)

    distances_layer = distances_layer or SKM.gen_new_layer_key(layer, SKM.DISTANCES_SUFFIX)
    SKM.set_layer_data(adata, distances_layer, distances)
    markers_layer = markers_layer or SKM.gen_new_layer_key(layer, SKM.MARKERS_SUFFIX)
    SKM.set_layer_data(adata, markers_layer, peaks)


def _augment_labels(source_labels: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
    """Augment the labels in one label array using the labels in another.

    This function modifies the labels in `target_labels` in the following way.
    Note that this function operates on a copy of `target_labels`. It does NOT
    modify in-place.
    * Any labels that are in `source_labels` that have no overlap with
        any labels in `target_labels` is copied over to `target_labels`.
    * Any labels that are in `target_labels` that have no overlap with any labels
        in `source_labels` is removed.

    Args:
        source_labels: Numpy array containing labels to (possibly) transfer.
        target_labels: Numpy array containing labels to augment.

    Returns:
        New Numpy array containing augmented labels.
    """
    augmented = np.zeros_like(target_labels)
    label = 1

    lm.main_debug("Removing non-overlapping labels.")
    target_props = measure.regionprops(target_labels)  # lazy evaluation
    for props in target_props:
        _label = props.label
        min_row, min_col, max_row, max_col = props.bbox
        target_mask = target_labels[min_row:max_row, min_col:max_col] == _label
        if source_labels[min_row:max_row, min_col:max_col][target_mask].sum() > 0:
            augmented[min_row:max_row, min_col:max_col][target_mask] = label
            label += 1

    lm.main_debug("Copying over non-overlapping labels.")
    source_props = measure.regionprops(source_labels)  # lazy evaluation
    for props in source_props:
        _label = props.label
        min_row, min_col, max_row, max_col = props.bbox
        source_mask = source_labels[min_row:max_row, min_col:max_col] == _label
        if target_labels[min_row:max_row, min_col:max_col][source_mask].sum() == 0:
            augmented[min_row:max_row, min_col:max_col][source_mask] = label
            label += 1
    return augmented


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def augment_labels(
    adata: AnnData,
    source_layer: str,
    target_layer: str,
    out_layer: Optional[str] = None,
):
    """Augment the labels in one label array using the labels in another.

    Args:
        adata: Input Anndata
        source_layer: Layer containing source labels to (possibly) take labels
            from.
        target_layer: Layer containing target labels to augment.
        out_layer: Layer to save results. Defaults to `{target_layer}_augmented`.
    """
    source_labels = SKM.select_layer_data(adata, source_layer)
    target_labels = SKM.select_layer_data(adata, target_layer)
    augmented = _augment_labels(source_labels, target_labels)
    out_layer = out_layer or SKM.gen_new_layer_key(target_layer, SKM.AUGMENTED_SUFFIX)
    SKM.set_layer_data(adata, out_layer, augmented)
