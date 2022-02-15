"""Utility functions for cell segmentation.
"""
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from kneed import KneeLocator
from scipy import signal
from typing_extensions import Literal

from ...errors import PreprocessingError


def circle(k: int) -> np.ndarray:
    """Draw a circle of diameter k.

    Args:
        k: Diameter

    Returns:
        8-bit unsigned integer Numpy array with 1s and 0s

    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    r = (k - 1) // 2
    return cv2.circle(np.zeros((k, k), dtype=np.uint8), (r, r), r, 1, -1)


def knee_threshold(X: np.ndarray, n_bins: int = 256) -> float:
    """Find the knee thresholding point of an arbitrary array.

    Note:
        This function does not find the actual knee of X. It computes a
        value to be used to threshold the elements of X by finding the knee of
        the cumulative counts.

    Args:
        X: Numpy array of values
        n_bins: Number of bins to use if `X` is a float array.

    Returns:
        Knee
    """
    # Check if array only contains integers.
    _X = X.astype(int)
    if np.array_equal(X, _X):
        x = np.sort(np.unique(_X))
    else:
        x = np.linspace(X.min(), X.max(), n_bins)
    y = np.array([(X <= val).sum() for val in x]) / X.size

    kl = KneeLocator(x, y, curve="concave")
    return kl.knee


def gaussian_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur

    This function is not designed to be called directly. Use :func:`conv2d`
    with `mode="gauss"` instead.

    Args:
        X: UMI counts per pixel.
        k: Radius of gaussian blur.

    Returns:
        Blurred array
    """
    return cv2.GaussianBlur(src=X.astype(float), ksize=(k, k), sigmaX=0, sigmaY=0)


def conv2d(
    X: np.ndarray, k: int, mode: Literal["gauss", "circle", "square"], bins: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convolve an array with the specified kernel size and mode.

    Args:
        X: The array to convolve.
        k: Kernel size. Must be odd.
        mode: Convolution mode. Supported modes are:
            gauss:
            circle:
            square:
        bins: Convolve per bin. Zeros are ignored.

    Returns:
        The convolved array

    Raises:
        ValueError: if `k` is even or less than 1, or if `mode` is not a
            valid mode, or if `bins` does not have the same shape as `X`
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")
    if mode not in ("gauss", "circle", "square"):
        raise ValueError(f'`mode` must be one of "gauss", "circle", "square"')
    if bins is not None and X.shape != bins.shape:
        raise ValueError("`bins` must have the same shape as `X`")

    def _conv(_X):
        if mode == "gauss":
            return gaussian_blur(_X, k)
        kernel = np.ones((k, k), dtype=np.uint8) if mode == "square" else circle(k)
        return signal.convolve2d(_X, kernel, boundary="symm", mode="same")

    if bins is not None:
        conv = np.zeros(X.shape)
        for label in np.unique(bins):
            if label > 0:
                mask = bins == label
                indices = np.where(mask)
                conv[indices] = _conv(X * mask)[indices]
        return conv
    return _conv(X)


def scale_to_01(X: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1].

    Args:
        X: Array to scale

    Returns:
        Scaled array
    """
    return (X - X.min()) / (X.max() - X.min())


def scale_to_255(X: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 255].

    Args:
        X: Array to scale

    Returns:
        Scaled array
    """
    return scale_to_01(X) * 255


def mclose_mopen(mask: np.ndarray, k: int, square: bool = False) -> np.ndarray:
    """Perform morphological close and open operations on a boolean mask.

    Args:
        X: Boolean mask
        k: Kernel size
        square: Whether or not the kernel should be square

    Returns:
        New boolean mask with morphological close and open operations performed.

    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    kernel = np.ones((k, k), dtype=np.uint8) if square else circle(k)
    mclose = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mopen = cv2.morphologyEx(mclose, cv2.MORPH_OPEN, kernel)

    return mopen.astype(bool)


def apply_threshold(X: np.ndarray, k: int, threshold: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """Apply a threshold value to the given array and perform morphological close
    and open operations.

    Args:
        X: The array to threshold
        k: Kernel size of the morphological close and open operations.
        threshold: Threshold to apply. By default, the knee is used.

    Returns:
        A boolean mask.
    """
    # Apply threshold and mclose,mopen
    threshold = threshold if threshold is not None else knee_threshold(X)
    mask = mclose_mopen(X >= threshold, k)
    return mask


def safe_erode(
    X: np.ndarray,
    k: int,
    square: bool = False,
    min_area: int = 1,
    n_iter: int = 1,
    float_k: Optional[int] = None,
    float_threshold: Optional[float] = None,
) -> np.ndarray:
    """Perform morphological erosion, but don't erode connected regions that
    have less than the provided area.

    Note:
        It is possible for this function to miss some small regions due to how
        erosion works. For instance, a region may have area > `min_area` which
        may be eroded in its entirety in one iteration. In this case, this
        region will not be saved.

    Args:
        X: Array to erode.
        k: Erosion kernel size
        square: Whether to use a square kernel
        min_area: Minimum area
        n_iter: Number of erosions to perform.
        float_k: Morphological close and open kernel size when `X` is a
            float array.
        float_threshold: Threshold to use to determine connected components
            when `X` is a float array.

    Returns:
        Eroded array as a boolean mask

    Raises:
        ValueError: If `X` has floating point dtype but `float_threshold` is
            not provided
    """
    if X.dtype == np.dtype(bool):
        X = X.astype(np.uint8)
    if np.issubdtype(X.dtype, np.floating) and float_k is None:
        raise ValueError("`float_k` must be provided for floating point arrays.")
    saved = np.zeros_like(X, dtype=bool)
    kernel = np.ones((k, k), dtype=np.uint8) if square else circle(k)

    for _ in range(n_iter):
        # Find connected components and save if area <= min_area
        components = cv2.connectedComponentsWithStats(
            apply_threshold(X, float_k, float_threshold).astype(np.uint8) if float_threshold is not None else X
        )
        areas = components[2][:, cv2.CC_STAT_AREA]
        for label in np.where(areas <= min_area)[0]:
            saved += components[1] == label

        X = cv2.erode(X, kernel)

    mask = X >= float_threshold if float_threshold is not None else X > 0
    return (mask + saved).astype(bool)
