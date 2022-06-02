"""Utility functions for cell segmentation.
"""
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from kneed import KneeLocator
from scipy import signal, sparse
from tqdm import tqdm
from typing_extensions import Literal

from ..configuration import SKM
from ..errors import SegmentationError
from ..logging import logger_manager as lm


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


def median_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Median blur

    This function is not designed to be called directly. Use :func:`conv2d`
    with `mode="median"` instead.

    Args:
        X: UMI counts per pixel.
        k: Radius of median blur.

    Returns:
        Blurred array
    """
    return cv2.medianBlur(src=X.astype(np.uint8), ksize=k)


def conv2d(
    X: np.ndarray, k: int, mode: Literal["gauss", "median", "circle", "square"], bins: Optional[np.ndarray] = None
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
    if mode not in ("median", "gauss", "circle", "square"):
        raise ValueError(f'`mode` must be one of "median", "gauss", "circle", "square"')
    if bins is not None and X.shape != bins.shape:
        raise ValueError("`bins` must have the same shape as `X`")
    if k == 1:
        return X

    def _conv(_X):
        if mode == "gauss":
            return gaussian_blur(_X, k)
        if mode == "median":
            return median_blur(_X, k)
        kernel = np.ones((k, k), dtype=np.uint8) if mode == "square" else circle(k)
        return signal.convolve2d(_X, kernel, boundary="symm", mode="same")

    if bins is not None:
        conv = np.zeros(X.shape)
        for label in np.unique(bins):
            if label > 0:
                mask = bins == label
                conv[mask] = _conv(X * mask)[mask]
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
    n_iter: int = -1,
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
        n_iter: Number of erosions to perform. If -1, then erosion is continued
            until every connected component is <= `min_area`.
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
    is_float = np.issubdtype(X.dtype, np.floating)
    if is_float and (float_k is None or float_threshold is None):
        raise ValueError("`float_k` and `float_threshold` must be provided for floating point arrays.")
    saved = np.zeros_like(X, dtype=bool)
    kernel = np.ones((k, k), dtype=np.uint8) if square else circle(k)

    i = 0
    with tqdm(desc="Eroding") as pbar:
        while True:
            # Find connected components and save if area <= min_area
            components = cv2.connectedComponentsWithStats(
                apply_threshold(X, float_k, float_threshold).astype(np.uint8) if is_float else X
            )

            areas = components[2][:, cv2.CC_STAT_AREA]
            for label in np.where(areas <= min_area)[0]:
                if label > 0:
                    stats = components[2][label]
                    left, top, width, height = (
                        stats[cv2.CC_STAT_LEFT],
                        stats[cv2.CC_STAT_TOP],
                        stats[cv2.CC_STAT_WIDTH],
                        stats[cv2.CC_STAT_HEIGHT],
                    )
                    saved[top : top + height, left : left + width] += (
                        components[1][top : top + height, left : left + width] == label
                    )

            X = cv2.erode(X, kernel)

            i += 1
            pbar.update(1)
            if (areas > min_area).sum() == 1 or (n_iter > 0 and n_iter == i):
                break

    mask = (X >= float_threshold) if is_float else (X > 0)
    return (mask + saved).astype(bool)


def label_overlap(X: np.ndarray, Y: np.ndarray) -> sparse.csr_matrix:
    """Compuate the overlaps between two label arrays.

    The integer labels in `X` and `Y` are used as the row and column indices
    of the resulting array.

    Note:
        The overlap array contains background overlap (index 0) as well.

    Args:
        X: First label array. Labels in this array are the rows of the resulting
            array.
        Y: Second label array. Labels in this array are the columns of the resulting
            array.

    Returns:
        A `(max(X)+1, max(Y)+1)` shape sparse array containing how many pixels for
            each label are overlapping.
    """

    def _label_overlap(X, Y):
        overlap = sparse.dok_matrix((X.max() + 1, Y.max() + 1), dtype=np.uint)
        for i in range(X.size):
            overlap[X[i], Y[i]] += 1
        return overlap

    if X.shape != Y.shape:
        raise SegmentationError(
            f"Both arrays must have the same shape, but one is {X.shape} and the other is {Y.shape}."
        )
    return _label_overlap(X.flatten(), Y.flatten()).tocsr()


def clahe(X: np.ndarray, clip_limit: float = 1.0, tile_grid: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization (CLAHE).

    Args:
        X: Image to equalize
        clip_limit: Contrast clipping. Lower values retain more of homogeneous
            regions.
        tile_grid: Apply histogram equalization to tiles of this size.

    Returns:
        Equalized image
    """
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(X)


def cal_cell_area(cell_labels: np.ndarray):
    """Calculate spot numbers for each cell.

    Args:
        cell_labels: cell labels.
    Returns:
        dict
    """
    areas = {}
    for row in cell_labels:
        for item in row:
            if item == 0:
                continue
            if item not in areas:
                areas[item] = 0
            areas[item] += 1
    return areas


def filter_cell_labels_by_area(adata: AnnData, layer: str, area_cutoff: int = 7):
    """Filter out cells with area less than `area_cutoff`

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to use
        area_cutoff: cells with area less than this cutoff would be dicarded.
    """
    X = SKM.select_layer_data(adata, layer, make_dense=True)
    cells = np.unique(X)
    cells = [i for i in cells if i > 0]
    lm.main_info(f"Cell number before filtering is {len(cells)}")

    areas = cal_cell_area(X)
    filtered_cells = [k for k, v in areas.items() if v < area_cutoff]
    X = np.where(np.isin(X, filtered_cells), 0, X)
    SKM.set_layer_data(adata, layer, X)
    cells = np.unique(X)
    cells = [i for i in cells if i > 0]
    lm.main_info(f"Cell number after filtering is {len(cells)}")
