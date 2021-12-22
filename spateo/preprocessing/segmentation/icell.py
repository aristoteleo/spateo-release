"""Identify cells from RNA signal. Functions in this file are used to
generate a cell mask, NOT to identify individual cells.
"""
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from scipy import stats
from scipy.sparse import spmatrix
from typing_extensions import Literal

from . import utils


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


def gaussian_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur

    Args:
        X: UMI counts per pixel.
        k: Radius of gaussian blur.

    Returns:
        Blurred array
    """
    return cv2.GaussianBlur(src=X.astype(float), ksize=(k, k), sigmaX=0, sigmaY=0)


def em(
    X: np.ndarray,
    use_peaks: bool = False,
    downsample: int = 1e6,
    w: Tuple[float, float] = (0.99, 0.01),
    mu: Tuple[float, float] = (10.0, 300.0),
    var: Tuple[float, float] = (20.0, 400.0),
    max_iter: int = 2000,
    precision: float = 1e-3,
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
    pass


def em_confidence(
    X: np.ndarray,
    w: Tuple[float, float],
    n: Tuple[float, float],
    p: Tuple[float, float],
) -> np.ndarray:
    """Compute confidence of each pixel being a cell, using the parameters
    estimated by the EM algorithm.

    Args:
        X: UMI counts per pixel.
        w:
        n:
        p:

    Returns:
        Numpy array of confidence scores within the range [0, 1].
    """
    background_probs = stats.nbinom(n=n[0], p=p[0]).pmf(X)
    cell_probs = stats.nbinom(n=n[1], p=p[1]).pmf(X)
    tau = []
    tau.append(w[0] * background_probs)
    tau.append(w[1] * cell_probs)
    tau = np.array(tau)
    return tau[1] / tau.sum(axis=0)


def bp(
    X: np.ndarray,
    background_params: Tuple[float, float],
    cell_params: Tuple[float, float],
    k: int = 1,
    square: bool = False,
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
    pass


def mask_cells(
    X: Union[spmatrix, np.ndarray],
    method: Literal["gblur", "EM", "EM+gblur", "EM+BP"],
    cutoff: Optional[float] = None,
) -> np.ndarray:
    """Identify cells using RNA signal."""
    pass
