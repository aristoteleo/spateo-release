"""Identify cells from RNA signal. Functions in this file are used to
generate a cell mask, NOT to identify individual cells.

Original author @HailinPan, refactored by @Lioscro.
"""
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse, spmatrix
from skimage import filters
from typing_extensions import Literal

from . import bp, em, utils
from ...configuration import SKM
from ...errors import PreprocessingError
from ...warnings import PreprocessingWarning


def _mask_cells_from_stain(X: np.ndarray, otsu_classes: int = 4, otsu_index: int = 0, mk: int = 7) -> np.ndarray:
    """Create a boolean mask indicating cells from stained image."""
    thresholds = filters.threshold_multiotsu(X, otsu_classes)
    return utils.mclose_mopen(X >= thresholds[otsu_index], mk)


def _mask_nuclei_from_stain(
    X: np.ndarray,
    otsu_classes: int = 3,
    otsu_index: int = 0,
    local_k: int = 45,
    offset_factor: float = 1.1,
    mk: int = 5,
) -> np.ndarray:
    """Create a boolean mask indicating nuclei from stained nuclei image.
    See :func:`mask_nuclei_from_stain` for arguments.
    """
    thresholds = filters.threshold_multiotsu(X, otsu_classes)
    background_mask = X < thresholds[otsu_index]
    # local_mask = X >= filters.rank.otsu(X, utils.circle(local_k))
    local_mask = X > (filters.threshold_local(X, local_k) * offset_factor)
    nuclei_mask = utils.mclose_mopen((~background_mask) & local_mask, mk)
    return nuclei_mask


def mask_cells_from_stain(
    adata: AnnData,
    otsu_classes: int = 4,
    otsu_index: int = 0,
    mk: int = 7,
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
):
    """Create a boolean mask indicating cells from stained image.

    Args:
        adata: Input Anndata
        otsu_classes: Number of classes to assign pixels to for cell detection
        otsu_index: Which threshold index should be used for classifying
            cells. All pixel intensities >= the value at this index will be
            classified as cell.
        mk: Size of the kernel used for morphological close and open operations
            applied at the very end.
        out_layer: Layer to put resulting nuclei mask. Defaults to `{layer}_mask`.
    """
    if layer not in adata.layers:
        raise PreprocessingError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    mask = _mask_cells_from_stain(X, otsu_classes, otsu_index, mk)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, out_layer, mask)


def mask_nuclei_from_stain(
    adata: AnnData,
    otsu_classes: int = 3,
    otsu_index: int = 0,
    local_k: int = 45,
    offset_factor: float = 1.1,
    mk: int = 7,
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
):
    """Create a boolean mask indicating nuclei from stained nuclei image, and
    save this mask in the AnnData as an additional layer.

    Args:
        adata: Input Anndata
        otsu_classes: Number of classes to assign pixels to for background
            detection.
        otsu_index: Which threshold index should be used for background.
            All pixel intensities less than the value at this index will be
            classified as background.
        local_k: The size of the local neighborhood of each pixel to use for
            local (adaptive) thresholding to identify the foreground (i.e.
            nuclei).
        offset_factor: Factor to multiply the local thresholding values before
            applying the threshold. Values > 1 lead to more "strict" thresholding,
            and therefore may be helpful in distinguishing nuclei in dense regions.
        mk: Size of the kernel used for morphological close and open operations
            applied at the very end.
        layer: Layer containing nuclei staining
        out_layer: Layer to put resulting nuclei mask. Defaults to `{layer}_mask`.
    """
    if layer not in adata.layers:
        raise PreprocessingError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    mask = _mask_nuclei_from_stain(X, otsu_classes, otsu_index, local_k, offset_factor, mk)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, out_layer, mask)


def _score_pixels(
    X: Union[spmatrix, np.ndarray],
    k: int,
    method: Literal["gauss", "EM", "EM+gauss", "EM+BP"],
    em_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
    certain_mask: Optional[np.ndarray] = None,
    bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score each pixel by how likely it is a cell. Values returned are in
    [0, 1].

    Args:
        X: UMI counts per pixel as either a sparse or dense array.
        k: Kernel size for convolution.
        method: Method to use. Valid methods are:
            gauss: Gaussian blur
            EM: EM algorithm to estimate cell and background expression
                parameters.
            EM+gauss: EM algorithm followed by Gaussian blur.
            EM+BP: EM algorithm followed by belief propagation to estimate the
                marginal probabilities of cell and background.
        em_kwargs: Keyword arguments to the :func:`em.run_em` function.
        bp_kwargs: Keyword arguments to the :func:`bp.run_bp` function.
        certain_mask: A boolean Numpy array indicating which pixels are certain
            to be occupied, a-priori. For example, if nuclei staining is available,
            this would be the nuclei segmentation mask.
        bins: Pixel bins to segment separately. Only takes effect when the EM
            algorithm is run.

    Returns:
        [0, 1] score of each pixel being a cell.

    Raises:
        PreprocessingError: If `bins` and/or `certain_mask` was provided but
            their sizes do not match `X`
    """
    if method.lower() not in ("gauss", "em", "em+gauss", "em+bp"):
        raise PreprocessingError(f"Unknown method `{method}`")
    if certain_mask is not None and X.shape != certain_mask.shape:
        raise PreprocessingError("`certain_mask` does not have the same shape as `X`")
    if bins is not None and X.shape != bins.shape:
        raise PreprocessingError("`bins` does not have the same shape as `X`")

    method = method.lower()
    em_kwargs = em_kwargs or {}
    bp_kwargs = bp_kwargs or {}

    if em_kwargs and "em" not in method:
        warnings.warn(f"`em_kwargs` will be ignored", PreprocessingWarning)
    if bp_kwargs and "bp" not in method:
        warnings.warn(f"`bp_kwargs` will be ignored", PreprocessingWarning)

    # Convert X to dense array
    if issparse(X):
        X = X.A

    # All methods require some kind of 2D convolution to start off
    res = utils.conv2d(X, k, mode="gauss" if method == "gauss" else "circle", bins=bins)

    # All methods other than gauss requires EM
    if method != "gauss":
        em_results = em.run_em(res, bins=bins, **em_kwargs)

        if "bp" in method:
            background_cond, cell_cond = em.conditionals(res, em_results=em_results, bins=bins)
            res = bp.run_bp(res, background_cond, cell_cond, certain_mask=certain_mask, **bp_kwargs)
        else:
            res = em.confidence(res, em_results=em_results, bins=bins)
            if certain_mask is not None:
                res = np.clip(res + certain_mask, 0, 1)

        if "gauss" in method:
            res = utils.conv2d(res, k, mode="gauss", bins=bins)
    else:
        # For just "gauss" method, we should rescale to [0, 1] because all the
        # other methods eventually produce an array of [0, 1] values.
        res = utils.scale_to_01(res)
        if certain_mask is not None:
            res = np.clip(res + certain_mask, 0, 1)
    return res


def score_and_mask_pixels(
    adata: AnnData,
    layer: str,
    k: int,
    method: Literal["gauss", "EM", "EM+gauss", "EM+BP"],
    em_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
    threshold: Optional[float] = None,
    mk: int = 11,
    bins_layer: Optional[Union[Literal[False], str]] = None,
    certain_layer: Optional[str] = None,
    scores_layer: Optional[str] = None,
    mask_layer: Optional[str] = None,
):
    """Score and mask pixels by how likely it is occupied.

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to use
        k: Kernel size for convolution
        method: Method to use. Valid methods are:
            gauss: Gaussian blur
            EM: EM algorithm to estimate cell and background expression
                parameters.
            EM+gauss: EM algorithm followed by Gaussian blur.
            EM+BP: EM algorithm followed by belief propagation to estimate the
                marginal probabilities of cell and background.
        em_kwargs: Keyword arguments to the :func:`em.run_em` function.
        bp_kwargs: Keyword arguments to the :func:`bp.run_bp` function.
        threshold: Score cutoff, above which pixels are considered occupied.
            By default, a threshold is automatically determined by using
            the first value of the 3-class Multiotsu method.
        mk: Kernel size of morphological open and close operations to reduce
            noise in the mask.
        bins_layer: Layer containing assignment of pixels into bins. Each bin
            is considered separately. Defaults to `{layer}_bins`. This can be
            set to `False` to disable binning, even if the layer exists.
        certain_layer: Layer containing a boolean mask indicating which pixels are
            certain to be occupied. If the array is not a boolean array, it is
            casted to boolean.
        scores_layer: Layer to save pixel scores before thresholding. Defaults
            to `{layer}_scores`.
        mask_layer: Layer to save the final mask. Defaults to `{layer}_mask`.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    certain_mask = None
    if certain_layer:
        certain_mask = SKM.select_layer_data(adata, certain_layer).astype(bool)
    bins = None
    if bins_layer is not False:
        bins_layer = bins_layer or SKM.gen_new_layer_key(layer, SKM.BINS_SUFFIX)
        if bins_layer in adata.layers:
            bins = SKM.select_layer_data(adata, bins_layer)
    scores = _score_pixels(X, k, method, em_kwargs, bp_kwargs, certain_mask, bins)
    scores_layer = scores_layer or SKM.gen_new_layer_key(layer, SKM.SCORES_SUFFIX)
    SKM.set_layer_data(adata, scores_layer, scores)

    if not threshold:
        threshold = filters.threshold_otsu(scores)

    mask = utils.apply_threshold(scores, mk, threshold)
    if certain_layer:
        mask += certain_mask
    mask_layer = mask_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, mask_layer, mask)
