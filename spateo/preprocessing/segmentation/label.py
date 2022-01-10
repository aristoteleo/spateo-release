"""Functions for use when labeling individual nuclei/cells, after obtaining a
mask.
"""
import cv2
import numpy as np
from skimage import segmentation

from . import utils


def watershed(
    X: np.ndarray,
    mask: np.ndarray,
    marker_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    """Assign individual nuclei/cells using the Watershed algorithm.

    Args:
        X: Data array. This array will be Gaussian blurred and used as the
            input values to Watershed.
        mask: Nucleus/cell mask.
        marker_mask: Boolean Numpy array mask indicating where the Watershed
            markers are.
        k: Size of the kernel to use for Gaussian blur.

    Returns:
        Watershed labels.
    """
    blur = utils.conv2d(X, k, mode="gauss")
    markers = cv2.connectedComponents(marker_mask.astype(np.uint8))[1]
    watershed = segmentation.watershed(-blur, markers, mask=mask)
    return watershed
