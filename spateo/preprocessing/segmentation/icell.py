"""Identify cells from RNA signal. Functions in this file are used to
generate a cell mask, NOT to identify individual cells.

Original author @HailinPan, refactored by @Lioscro.
"""
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy.sparse import issparse, spmatrix
from skimage import filters
from typing_extensions import Literal

from . import bp, em, utils
from ...errors import PreprocessingError
from ...warnings import PreprocessingWarning


def mask_nuclei_from_stain(
    X: np.ndarray,
    otsu_classes: int = 3,
    otsu_index: int = 0,
    local_k: int = 45,
    mk: int = 5,
) -> np.ndarray:
    """Create a boolean mask indicating nuclei from stained nuclei image.

    Args:
        X: TIF intensities
        otsu_classes: Number of classes to assign pixels to for background
            detection.
        otsu_index: Which threshold index should be used for background.
            All pixel intensities less than the value at this index will be
            classified as background.
        local_k: The size of the local neighborhood of each pixel to use for
            local (adaptive) thresholding to identify the foreground (i.e.
            nuclei).
        mk: Size of the kernel used for morphological close and open operations
            applied at the very end.

    Returns:
        Boolean mask indicating which pixels are nuclei.
    """
    thresholds = filters.threshold_multiotsu(X, otsu_classes)
    background_mask = X < thresholds[otsu_index]
    local_mask = X > filters.threshold_local(X, local_k)
    nuclei_mask = utils.mclose_mopen((~background_mask) & local_mask, mk)
    return nuclei_mask


def score_pixels(
    X: Union[spmatrix, np.ndarray],
    k: int,
    method: Literal["gauss", "EM", "EM+gauss", "EM+BP"],
    em_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
    certain_mask: Optional[np.ndarray] = None,
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

    Returns:
        [0, 1] score of each pixel being a cell.
    """
    if method.lower() not in ("gauss", "em", "em+gauss", "em+bp"):
        raise PreprocessingError(f"Unknown method `{method}`")
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
    res = utils.conv2d(X, k, mode="gauss" if method == "gauss" else "circle")

    # All methods other than gauss requires EM
    if method != "gauss":
        w, r, p = em.run_em(res, **em_kwargs)

        if "bp" in method:
            res = bp.run_bp(
                res, (r[0], p[0]), (r[1], p[1]), certain_mask=certain_mask, **bp_kwargs
            )
        else:
            res = em.confidence(res, w, r, p)
            if certain_mask is not None:
                res = np.clip(res + certain_mask, 0, 1)

        if "gauss" in method:
            res = utils.conv2d(res, k, mode="gauss")
    else:
        # For just "gauss" method, we should rescale to [0, 1] because all the
        # other methods eventually produce an array of [0, 1] values.
        res = utils.scale_to_01(res)
        if certain_mask is not None:
            res = np.clip(res + certain_mask, 0, 1)
    return res
