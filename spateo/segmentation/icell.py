"""Identify cells from RNA signal. Functions in this file are used to
generate a cell mask, NOT to identify individual cells.

Original author @HailinPan, refactored by @Lioscro.
"""
from functools import partial
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse, spmatrix
from skimage import filters
from typing_extensions import Literal

from ..configuration import SKM
from ..errors import SegmentationError
from ..logging import logger_manager as lm
from . import bp, em, moran, utils, vi


def _mask_cells_from_stain(X: np.ndarray, otsu_classes: int = 3, otsu_index: int = 0, mk: int = 7) -> np.ndarray:
    """Create a boolean mask indicating cells from stained image."""
    lm.main_debug("Filtering with Multi-otsu.")
    thresholds = filters.threshold_multiotsu(X, otsu_classes)
    lm.main_debug("Applying morphological close and open.")
    return utils.mclose_mopen(X >= thresholds[otsu_index], mk)


def _mask_nuclei_from_stain(
    X: np.ndarray,
    otsu_classes: int = 3,
    otsu_index: int = 0,
    local_k: int = 55,
    offset: int = 5,
    mk: int = 5,
) -> np.ndarray:
    """Create a boolean mask indicating nuclei from stained nuclei image.
    See :func:`mask_nuclei_from_stain` for arguments.
    """
    lm.main_debug("Filtering with Multi-otsu.")
    thresholds = filters.threshold_multiotsu(X, otsu_classes)
    background_mask = X < thresholds[otsu_index]
    lm.main_debug("Filtering adaptive threshold.")
    if X.dtype != np.uint8:
        lm.main_warning(
            f"Adaptive thresholding using OpenCV requires {np.uint8} dtype, but array has {X.dtype} dtype. "
            "The slower skimage implementation will be used instead."
        )
        local_mask = X > filters.threshold_local(X, block_size=local_k, method="gaussian", offset=offset)
    else:
        local_mask = cv2.adaptiveThreshold(
            X, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, local_k, offset
        ).astype(bool)
    lm.main_debug("Applying morphological close and open.")
    nuclei_mask = utils.mclose_mopen((~background_mask) & local_mask, mk)
    return nuclei_mask


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def mask_cells_from_stain(
    adata: AnnData,
    otsu_classes: int = 3,
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
        layer: Layer that contains staining image.
        out_layer: Layer to put resulting nuclei mask. Defaults to `{layer}_mask`.
    """
    if layer not in adata.layers:
        raise SegmentationError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    lm.main_info("Constructing cell mask from staining image.")
    mask = _mask_cells_from_stain(X, otsu_classes, otsu_index, mk)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, out_layer, mask)


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def mask_nuclei_from_stain(
    adata: AnnData,
    otsu_classes: int = 3,
    otsu_index: int = 0,
    local_k: int = 55,
    offset: int = 5,
    mk: int = 5,
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
        offset: Offset to local thresholding values such that values > 0 lead to
            more "strict" thresholding, and therefore may be helpful in distinguishing
            nuclei in dense regions.
        mk: Size of the kernel used for morphological close and open operations
            applied at the very end.
        layer: Layer containing nuclei staining
        out_layer: Layer to put resulting nuclei mask. Defaults to `{layer}_mask`.
    """
    if layer not in adata.layers:
        raise SegmentationError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    lm.main_info("Constructing nuclei mask from staining image.")
    mask = _mask_nuclei_from_stain(X, otsu_classes, otsu_index, local_k, -offset, mk)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, out_layer, mask)


def _initial_nb_params(
    X: np.ndarray, bins: Optional[np.ndarray] = None
) -> Union[Dict[str, Tuple[float, float]], Dict[int, Dict[str, Tuple[float, float]]]]:
    """Calculate initial estimates for the negative binomial mixture.

    Args:
        X: UMI counts
        bins: Density bins

    Returns:
        Dictionary containing initial `w`, `mu`, `var` parameters. If `bins` is also
        provided, the dictionary is nested with the outer dictionary containing each
        bin label as the keys. If `zero_inflated=True`, then the dictionary also contains
        a `z` key.
    """
    samples = {}
    if bins is not None:
        for label in np.unique(bins):
            if label > 0:
                samples[label] = X[bins == label]
    else:
        samples[0] = X.flatten()

    params = {}
    for label, _samples in samples.items():
        # Background must have at least 2 different values to prevent the mean from being zero.
        threshold = max(filters.threshold_otsu(_samples), 1)
        mask = _samples > threshold
        background_values = _samples[~mask]
        foreground_values = _samples[mask]
        w = np.array([_samples.size - mask.sum(), mask.sum()]) / _samples.size
        mu = np.array([background_values.mean(), foreground_values.mean()])
        var = np.array([background_values.var(), foreground_values.var()])

        # Negative binomial distribution requires variance > mean
        if var[0] <= mu[0]:
            lm.main_warning(
                f"Bin {label} estimated variance of background ({var[0]:.2e}) is less than the mean ({mu[0]:.2e}). "
                "Initial variance will be arbitrarily set to 1.1x of the mean. "
                "This is usually due to extreme sparsity. Please consider increasing `k` or using "
                "the zero-inflated distribution."
            )
            var[0] = mu[0] * 1.1
        if var[1] <= mu[1]:
            lm.main_warning(
                f"Bin {label} estimated variance of foreground ({var[1]:.2e}) is less than the mean ({mu[1]:.2e}). "
                "Initial variance will be arbitrarily set to 1.1x of the mean. "
                "This is usually due to extreme sparsity. Please consider increasing `k` or using "
                "the zero-inflated distribution."
            )
            var[1] = mu[1] * 1.1
        params[label] = dict(w=tuple(w), mu=tuple(mu), var=tuple(var))
    return params[0] if bins is None else params


def _score_pixels(
    X: Union[spmatrix, np.ndarray],
    k: int,
    method: Literal["gauss", "moran", "EM", "EM+gauss", "EM+BP", "VI+gauss", "VI+BP"],
    moran_kwargs: Optional[dict] = None,
    em_kwargs: Optional[dict] = None,
    vi_kwargs: Optional[dict] = None,
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
            moran: Moran's I based method
            EM: EM algorithm to estimate cell and background expression
                parameters.
            EM+gauss: Negative binomial EM algorithm followed by Gaussian blur.
            EM+BP: EM algorithm followed by belief propagation to estimate the
                marginal probabilities of cell and background.
            VI+gauss: Negative binomial VI algorithm followed by Gaussian blur.
                Note that VI also supports the zero-inflated negative binomial (ZINB)
                by providing `zero_inflated=True`.
            VI+BP: VI algorithm followed by belief propagation. Note that VI also
                supports the zero-inflated negative binomial (ZINB) by providing
                `zero_inflated=True`.
        moran_kwargs: Keyword arguments to the :func:`moran.run_moran` function.
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
        SegmentationError: If `bins` and/or `certain_mask` was provided but
            their sizes do not match `X`
    """
    if method.lower() not in ("gauss", "moran", "em", "em+gauss", "em+bp", "vi+gauss", "vi+bp"):
        raise SegmentationError(f"Unknown method `{method}`")
    if certain_mask is not None and X.shape != certain_mask.shape:
        raise SegmentationError("`certain_mask` does not have the same shape as `X`")
    if bins is not None and X.shape != bins.shape:
        raise SegmentationError("`bins` does not have the same shape as `X`")

    method = method.lower()
    moran_kwargs = moran_kwargs or {}
    em_kwargs = em_kwargs or {}
    vi_kwargs = vi_kwargs or {}
    bp_kwargs = bp_kwargs or {}

    if moran_kwargs and "moran" not in method:
        lm.main_warning(f"`moran_kwargs` will be ignored.")
    if em_kwargs and "em" not in method:
        lm.main_warning(f"`em_kwargs` will be ignored.")
    if vi_kwargs and "vi" not in method:
        lm.main_warning(f"`vi_kwargs` will be ignored.")
    if bp_kwargs and "bp" not in method:
        lm.main_warning(f"`bp_kwargs` will be ignored.")

    # Convert X to dense array
    if issparse(X):
        lm.main_debug("Converting X to dense array.")
        X = X.A

    # All methods require some kind of 2D convolution to start off
    lm.main_debug(f"Computing 2D convolution with k={k}.")
    res = utils.conv2d(X, k, mode="gauss" if method in ("gauss", "moran") else "circle", bins=bins)

    # All methods other than gauss requires EM
    if method == "gauss":
        # For just "gauss" method, we should rescale to [0, 1] because all the
        # other methods eventually produce an array of [0, 1] values.
        res = utils.scale_to_01(res)
    elif method == "moran":
        res = moran.run_moran(res, mask=None if bins is None else bins > 0, **moran_kwargs)
        # Rescale
        res /= res.max()
    else:
        # Obtain initial parameter estimates with Otsu thresholding.
        # These may be overridden by providing the appropriate kwargs.
        nb_kwargs = dict(params=_initial_nb_params(res, bins=bins))
        if "em" in method:
            nb_kwargs.update(em_kwargs)
            lm.main_debug(f"Running EM with kwargs {nb_kwargs}.")
            em_results = em.run_em(res, bins=bins, **nb_kwargs)
            conditional_func = partial(em.conditionals, em_results=em_results, bins=bins)
        else:
            nb_kwargs.update(vi_kwargs)
            lm.main_debug(f"Running VI with kwargs {nb_kwargs}.")
            vi_results = vi.run_vi(res, bins=bins, **nb_kwargs)
            conditional_func = partial(vi.conditionals, vi_results=vi_results, bins=bins)

        if "bp" in method:
            lm.main_debug("Computing conditionals.")
            background_cond, cell_cond = conditional_func(res)
            if certain_mask is not None:
                background_cond[certain_mask] = 1e-2
                cell_cond[certain_mask] = 1 - (1e-2)
            lm.main_debug(f"Running BP with kwargs {bp_kwargs}.")
            res = bp.run_bp(background_cond, cell_cond, **bp_kwargs)
        else:
            lm.main_debug("Computing confidences.")
            res = em.confidence(res, em_results=em_results, bins=bins)
            if certain_mask is not None:
                res = np.clip(res + certain_mask, 0, 1)

        if "gauss" in method:
            lm.main_debug("Computing Gaussian blur.")
            res = utils.conv2d(res, k, mode="gauss", bins=bins)

    return res


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def score_and_mask_pixels(
    adata: AnnData,
    layer: str,
    k: int,
    method: Literal["gauss", "moran", "EM", "EM+gauss", "EM+BP", "VI+gauss", "VI+BP"],
    moran_kwargs: Optional[dict] = None,
    em_kwargs: Optional[dict] = None,
    vi_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
    threshold: Optional[float] = None,
    use_knee: Optional[bool] = False,
    mk: Optional[int] = None,
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
        method: Method to use to obtain per-pixel scores. Valid methods are:
            gauss: Gaussian blur
            moran: Moran's I based method
            EM: EM algorithm to estimate cell and background expression
                parameters.
            EM+gauss: Negative binomial EM algorithm followed by Gaussian blur.
            EM+BP: EM algorithm followed by belief propagation to estimate the
                marginal probabilities of cell and background.
            VI+gauss: Negative binomial VI algorithm followed by Gaussian blur.
                Note that VI also supports the zero-inflated negative binomial (ZINB)
                by providing `zero_inflated=True`.
            VI+BP: VI algorithm followed by belief propagation. Note that VI also
                supports the zero-inflated negative binomial (ZINB) by providing
                `zero_inflated=True`.
        moran_kwargs: Keyword arguments to the :func:`moran.run_moran` function.
        em_kwargs: Keyword arguments to the :func:`em.run_em` function.
        bp_kwargs: Keyword arguments to the :func:`bp.run_bp` function.
        threshold: Score cutoff, above which pixels are considered occupied.
            By default, a threshold is automatically determined by using
            Otsu thresholding.
        use_knee: Whether to use knee point as threshold. By default is False. If
            True, threshold would be ignored.
        mk: Kernel size of morphological open and close operations to reduce
            noise in the mask. Defaults to `k`+2 if EM or VI is run. Otherwise,
            defaults to `k`-2.
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
    method = method.lower()
    lm.main_info(f"Scoring pixels with {method} method.")
    scores = _score_pixels(X, k, method, moran_kwargs, em_kwargs, vi_kwargs, bp_kwargs, certain_mask, bins)
    scores_layer = scores_layer or SKM.gen_new_layer_key(layer, SKM.SCORES_SUFFIX)
    SKM.set_layer_data(adata, scores_layer, scores)

    if not threshold and not use_knee:
        lm.main_debug("Finding Otsu threshold.")
        threshold = filters.threshold_otsu(scores)
        lm.main_info(f"Applying threshold {threshold}.")

    mk = mk or (k + 2 if any(m in method for m in ("em", "vi")) else max(k - 2, 3))
    if use_knee:
        threshold = None
    mask = utils.apply_threshold(scores, mk, threshold)
    if certain_layer:
        mask += certain_mask
    mask_layer = mask_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, mask_layer, mask)
