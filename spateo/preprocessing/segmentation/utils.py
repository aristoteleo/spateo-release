"""Utility functions for cell segmentation.
"""
from typing import Tuple

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
    """
    r = (k - 1) // 2
    return cv2.circle(np.zeros((k, k), dtype=np.uint8), (r, r), r, 1, -1)


def knee(X: np.ndarray, n_bins: int = 256) -> float:
    """Find the knee point of an arbitrary array.

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

    Args:
        X: UMI counts per pixel.
        k: Radius of gaussian blur.

    Returns:
        Blurred array
    """
    return cv2.GaussianBlur(src=X.astype(float), ksize=(k, k), sigmaX=0, sigmaY=0)


def conv2d(
    X: np.ndarray, k: int, mode: Literal["gauss", "circle", "square"]
) -> np.ndarray:
    """Convolve an array with the specified kernel size and mode.

    Args:
        X: The array to convolve.
        k: Kernel size. Must be odd.
        mode: Convolution mode. Supported modes are:
            gauss:
            circle:
            square:

    Returns:
        The convolved array

    Raises:
        PreprocessingError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise PreprocessingError(f"`k` must be odd and greater than 0.")

    if mode == "gauss":
        return gaussian_blur(X, k)

    kernel = np.ones((k, k), dtype=np.uint8) if mode == "square" else circle(k)
    return signal.convolve2d(X, kernel, boundary="symm", mode="same")


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

    open_kernel = circle(k)
    mopen = cv2.morphologyEx(mclose, cv2.MORPH_OPEN, open_kernel)

    return mopen.astype(bool)
