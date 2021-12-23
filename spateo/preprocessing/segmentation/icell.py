"""Identify cells from RNA signal. Functions in this file are used to
generate a cell mask, NOT to identify individual cells.
"""
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy import stats
from scipy.sparse import issparse, spmatrix
from typing_extensions import Literal

from . import bp, em, utils
from ...errors import PreprocessingError
from ...warnings import PreprocessingWarning


def mclose_mopen(mask: np.ndarray, k: int) -> np.ndarray:
    """Perform morphological close and open operations on a boolean mask.

    Note:
     The two operations are performed with different kernels. The close operation
     uses a square kernel, while the mopen operation uses a circular kernel.

    Args:
        X: Boolean mask
        k: Kernel size

    Returns:
        New boolean mask with morphological close and open operations performed.
    """
    close_kernel = np.ones((k, k), dtype=np.uint8)
    mclose = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)

    open_kernel = utils.circle(k)
    mopen = cv2.morphologyEx(mclose, cv2.MORPH_OPEN, open_kernel)

    return mopen.astype(bool)


def run_em(
    X: np.ndarray,
    use_peaks: bool = False,
    downsample: int = 1e6,
    w: Tuple[float, float] = (0.99, 0.01),
    mu: Tuple[float, float] = (10.0, 300.0),
    var: Tuple[float, float] = (20.0, 400.0),
    max_iter: int = 2000,
    precision: float = 1e-6,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """EM

    Args:
        X: UMI counts per pixel.
        use_peaks: Whether to use peaks of convolved image as samples for the
            EM algorithm.
        downsample: Use at most this many samples. If `use_peaks` is False,
            samples are chosen uniformly at random to at most this many samples.
            Otherwise, peaks are chosen uniformly at random.
        w:
        mu:
        var:
        max_iter: Maximum number of EM iterations.
        precision: Stop EM algorithm once desired precision has been reached.

    Returns:
        Tuple of parameters estimated by the EM algorithm.
    """
    if use_peaks:
        # TODO implement
        pass
    else:
        samples = X.flatten()
    if samples.size > downsample:
        samples = np.random.choice(samples, downsample, replace=False)

    w, r, p = em.nbn_em(
        samples, w=w, mu=mu, var=var, max_iter=max_iter, precision=precision
    )
    return tuple(w), tuple(r), tuple(p)


def run_bp(
    X: np.ndarray,
    background_params: Tuple[float, float],
    cell_params: Tuple[float, float],
    k: int = 3,
    square: bool = False,
    p: float = 0.7,
    q: float = 0.3,
    precision: float = 1e-6,
    max_iter: int = 100,
    n_threads: int = 1,
) -> np.ndarray:
    """Compute the marginal probability of each pixel being a cell, using
    belief propagation.

    Args:
        X: UMI counts per pixel.
        background_params: Parameters estimated (with EM) for background.
        cell_params: Parameters estimated (with EM) for cell.
        k: Neighborhood size
        square: Whether the neighborhood of each node is a square around it.
            If false, the neighborhood is a circle.

    Returns:
        Numpy array of marginal probabilities.
    """
    background_probs = em.nb_pmf(X, background_params[0], background_params[1])
    cell_probs = em.nb_pmf(X, cell_params[0], cell_params[1])
    neighborhood = np.ones((k, k)) if square else utils.circle(k)
    marginals = bp.cell_marginals(
        background_probs,
        cell_probs,
        neighborhood=neighborhood,
        p=p,
        q=q,
        precision=precision,
        max_iter=max_iter,
        n_threads=n_threads,
    )
    return marginals


def score_pixels(
    X: Union[spmatrix, np.ndarray],
    k: int,
    method: Literal["gauss", "EM", "EM+gauss", "EM+BP"],
    em_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Score each pixel by how likely it is a cell. Values returned are in
    [0, 1].

    Args:
        X: UMI counts per pixel as either a sparse or dense array.
        k: Kernel size for convolution.
        method: Method to use. Valid methods are:
            gauss:
            EM:
            EM+gauss:
            EM+BP:
        em_kwargs: Keyword arguments to the :func:`run_em` function.
        bp_kwargs: Keyword arguments to the :func:`run_bp` function.

    Returns:
        Boolean mask indicating cells.
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
    res, kernel = utils.conv2d(X, k, mode="gauss" if method == "gauss" else "circle")

    # All methods other than gauss requires EM
    if method != "gauss":
        w, r, p = run_em(res, **em_kwargs)

        if "bp" in method:
            # NOTE: bp requires per pixel parameters
            kernel_size = kernel.sum()
            res = run_bp(
                X, (r[0] / kernel_size, p[0]), (r[1] / kernel_size, p[1]), **bp_kwargs
            )
        else:
            res = em.confidence(res, w, r, p)

        if "gauss" in method:
            res = utils.conv2d(res, k, mode="gauss")
    else:
        # For just "gauss" method, we should rescale to [0, 1] because all the
        # other methods eventually produce an array of [0, 1] values.
        res = utils.scale_to_01(res)
    return res


def apply_cutoff(X: np.ndarray, k: int, cutoff: Optional[float] = None) -> np.ndarray:
    # Apply cutoff and mclose,mopen
    cutoff = cutoff or utils.knee(X)
    mask = mclose_mopen(X >= cutoff, k)
    return mask
