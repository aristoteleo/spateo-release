"""Functions for use when labeling individual nuclei/cells, after obtaining a
mask.
"""
from typing import Union

import cv2
import numpy as np
from scipy.sparse import issparse, spmatrix
from skimage import segmentation

from . import utils


def watershed(
    X: Union[spmatrix, np.ndarray],
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
    if issparse(X):
        X = X.A

    blur = utils.conv2d(X, k, mode="gauss")
    if markers.dtype == np.dtype(bool):
        markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
    watershed = segmentation.watershed(-blur, markers, mask=mask)
    return watershed


def expand_labels(labels: np.ndarray, distance: int, max_area: int) -> np.ndarray:
    """Expand labels up to a certain distance, while ignoring labels that are
    above a certain size.

    Args:
        labels: Numpy array containing integer labels.
        distance: Distance to expand. Internally, this is used as the number
            of iterations of distance 1 dilations.
        max_area: Maximum area of each label.

    Returns:
        New label array with expanded labels.
    """
    expanded = labels.copy()
    saved = {}
    for _ in range(distance):
        for label in (np.bincount(expanded.flatten()) >= max_area).nonzero()[0]:
            if label > 0:
                where = np.where(expanded == label)
                saved[label] = where
                # Remove labels that reached max area
                expanded[where] = 0

        # Expand
        expanded = segmentation.expand_labels(expanded, distance=1)

    # Replace with saved labels
    for label, where in saved.items():
        expanded[where] = label

    return expanded
