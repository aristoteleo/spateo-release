"""Cell masking using Moran's I metric.

Adapted from code written by @HailinPan.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import signal, stats

from . import utils


def moranI(
    X: np.ndarray, kernel: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Moran's I for cell masking.

    Args:
        X: Numpy array containing (possibly smoothed) UMI counts or binarized
            values.
        kernel: 2D kernel containing widgets
        mask: If provided, only consider pixels within the mask

    Returns:
        A 4-element tuple containing (z, c, i, pvalue).
    """
    masked_X = X
    n = X.size
    if mask is not None:
        masked_X = X[mask]
        n = mask.sum()
    x_bar = masked_X.sum() / n

    z = X - x_bar
    z_masked = z if mask is None else z[mask]

    m2 = (z_masked**2).sum() / n
    c = signal.convolve2d(z, kernel, boundary="symm", mode="same")
    i = z / m2 * c
    ei = -kernel.sum() / (n - 1)
    wi2 = (kernel**2).sum()
    m4 = (z_masked**4).sum() / n
    b2 = m4 / (m2**2)
    tow_wikh = (kernel.reshape(-1, 1) * kernel.reshape(1, -1)).sum()
    vari = wi2 * (n - b2) / (n - 1) + tow_wikh * (2 * b2 - n) / ((n - 1) * (n - 2)) - kernel.sum() ** 2 / (n - 1) ** 2
    zscore = (i - ei) / vari**0.5
    pvalue = stats.norm.sf(abs(zscore)) * 2
    return z, c, i, pvalue


def run_moran(X: np.ndarray, k: int = 7, p_threshold: float = 0.05, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute scores using Moran's I method.

    Args:
        X: Numpy array containing (possibly smoothed) UMI counts or binarized
            values.
        k: Kernel size
        p_threshold: P-value threshold.
        mask: If provided, only consider pixels within the mask

    Returns:
        A 2D Numpy array indicating pixel scores
    """
    # Create Gaussian kernel
    kx = cv2.getGaussianKernel(k, 0)
    ky = cv2.getGaussianKernel(k, 0)
    kernel = (ky * kx.T) * utils.circle(k)
    kernel[(k - 1) // 2, (k - 1) // 2] = 0

    z, c, i, pvalue = moranI(X, kernel, mask=mask)
    # Set pixels whose p values are < p_threshold to zero, which indicate
    # no spatial correlation.
    c[pvalue >= p_threshold] = 0
    return c
