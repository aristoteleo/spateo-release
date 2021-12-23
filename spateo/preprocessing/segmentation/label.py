"""Functions for use when labeling individual cells, after obtaining a cell mask.

Original author @HailinPan, refactored by @Lioscro.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, segmentation

from . import utils


def watershed(X: np.ndarray, mask: np.ndarray, k: int, min_distance: int) -> np.ndarray:
    """Label cells using the Watershed algorithm.

    Args:
        X: UMI counts
        mask: Cell mask
        k: Kernel size for gaussian blur
        min_distance: Minimum distance between center of cells

    Returns:
        Numpy array of the same shape as `X` containing integer cell labels.
    """
    blur = utils.conv2d(X, k, mode="gauss")
    blur = utils.scale_to_255(blur)
    blur[~mask] = 0
    peak_idx = feature.peak_local_max(
        image=blur, min_distance=min_distance, labels=mask
    )
    local_maxi = np.zeros_like(blur, dtype=bool)
    local_maxi[tuple(peak_idx.T)] = True
    markers = ndi.label(local_maxi)[0]
    labels = segmentation.watershed(-blur, markers, mask=mask)
    return labels


def label_cells(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Assign individual cell labels using total UMIs and a cell mask."""
    pass


def enlarge(labels: np.ndarray, n_iter: int = 5, max_area: int = 400) -> np.ndarray:
    """?"""
    enlarged = labels.copy()
    cell_labels = [l for l in np.unique(labels) if l > 0]
    for _ in range(n_iter):
        aexp = np.zeros([enlarged.shape[0] + 2, enlarged.shape[1] + 2], dtype=np.int32)
        aexp[1:-1, 1:-1] = enlarged
        top = aexp[0:-2, 1:-1]
        left = aexp[1:-1, 0:-2]
        right = aexp[1:-1, 2:]
        bottom = aexp[2:, 1:-1]

        cellArea = np.bincount(enlarged.flatten())
        cWithmaxA = np.argwhere(cellArea >= max_area)
        cWithmaxA = cWithmaxA[cWithmaxA > 0]

        top[np.isin(top, cWithmaxA)] = 0
        left[np.isin(left, cWithmaxA)] = 0
        right[np.isin(right, cWithmaxA)] = 0
        bottom[np.isin(bottom, cWithmaxA)] = 0

        enlarged[(enlarged == 0) & (top > 0)] = top[(enlarged == 0) & (top > 0)]
        enlarged[(enlarged == 0) & (left > 0)] = left[(enlarged == 0) & (left > 0)]
        enlarged[(enlarged == 0) & (right > 0)] = right[(enlarged == 0) & (right > 0)]
        enlarged[(enlarged == 0) & (bottom > 0)] = bottom[
            (enlarged == 0) & (bottom > 0)
        ]
    return enlarged
