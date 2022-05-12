"""Functions to generate simulated RNA data.

Adapted from code written by @HailinPan.
"""
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from anndata import AnnData
from scipy import stats

from ..configuration import SKM
from ..errors import SegmentationError


def _create_labels(
    shape: Tuple[int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    axes1: np.ndarray,
    axes2: np.ndarray,
    angles: np.ndarray,
    shift: int = 3,
) -> np.ndarray:
    """Given simulated cell variables, generate a label array.

    Args:
        shape: The size of the X and Y axes, in pixels.
        xs, ys: Center X and Y coordinates for each cell.
        axes1, axes2: Axes lengths for each cell.
        angles: Angle for each cell.
        shift: Attempt to minimize overlaps between cells by shifting overlapping
            cells by this amount (in reality, this is just a scaling factor, so
            larger values mean to shift overlapping cells by more). Set to zero
            or negative to disable.

    Returns:
        New Numpy array containing cell labels.
    """
    n = xs.size
    if n != xs.size or n != axes1.size or n != axes2.size or n != angles.size:
        raise SegmentationError(f"All input arrays must have size {n}")
    indices_to_add = deque(range(xs.size))  # +1 to get actual label to add!
    labels = np.zeros(shape, dtype=np.int32)
    i = 0
    while indices_to_add:
        if i >= n * 100:
            raise SegmentationError(
                f"Reached iteration {i}. Try reducing the number of cells or turn off shifting by setting `shift=0`."
            )

        idx = indices_to_add.popleft()
        label = idx + 1
        x, y, axis1, axis2, angle = xs[idx], ys[idx], axes1[idx], axes2[idx], angles[idx]
        prev_labels = labels.copy()
        cv2.ellipse(labels, (x, y), (axis1, axis2), angle, 0, 360, label, -1)

        if shift > 1:
            # Find & remove overlapping labels.
            overlapping = np.unique(prev_labels[(labels == label) & (prev_labels > 0)])
            labels[np.isin(labels, overlapping)] = 0

            # Compute shifted locations for each of the overlapping labels.
            for ov_label in overlapping:
                ov_idx = ov_label - 1
                ov_x, ov_y = xs[ov_idx], ys[ov_idx]
                diff_x = ov_x - x
                diff_y = ov_y - y
                distance = np.sqrt(diff_x**2 + diff_y**2) + 1e-5
                # Shift
                xs[ov_idx] = min(max(0, round(ov_x + ((diff_x + 1e-5) / distance * shift))), shape[0])
                ys[ov_idx] = min(max(0, round(ov_y + (diff_y + 1e-5) / distance * shift)), shape[1])
        i += 1

    return labels


def simulate_cells(
    shape: Tuple[int, int],
    n: int,
    axis1_range: Tuple[int, int] = (7, 15),
    axis2_range: Tuple[int, int] = (5, 14),
    shift: int = 3,
    foreground_params: Tuple[int, int, int] = (0.512, 1.96, 11.4),
    background_params: Tuple[int, int, int] = (0.921, 1.08, 1.74),
    seed: Optional[int] = None,
) -> AnnData:
    """Create a new AnnData object containing simulated cell labels and UMI
    counts.

    Cells are simulated as ellipses with the two axes lengths sampled from a
    log-uniform distribution. The angle at which the cell is placed is sampled
    uniformly at random.

    Args:
        shape: The size of the X and Y axes, in pixels.
        n: Number of cells to simulate.
        axis1_range: Range of the first axes.
        axis2_range: Range of the second axes.
        shift: Attempt to minimize overlaps between cells by shifting overlapping
            cells by this amount (in reality, this is just a scaling factor, so
            larger values mean to shift overlapping cells by more). Set to zero or
            negative to disable.
        foreground_params: Parameters for foreground expression, as a 3-element
            tuple of (dropout rate, mean, variance).
        background_params: Parameters for background expression, as a 3-element
            tuple of (dropout rate, mean, variance).
        seed: Random seed

    Returns:
        An Anndata object where X contains the simulated UMI counts and
            `.layers['labels']` contains the simulated labels.
    """

    def muvar_to_np(mu, var):
        n = mu**2 / (var - mu)
        p = mu / var
        return n, p

    f_do, f_mu, f_var = foreground_params
    b_do, b_mu, b_var = background_params
    if f_var < f_mu or b_var < b_mu:
        raise SegmentationError("Variance must be less than mean.")
    f_n, f_p = muvar_to_np(f_mu, f_var)
    b_n, b_p = muvar_to_np(b_mu, b_var)

    rng = np.random.default_rng(seed)
    xs = rng.integers(0, shape[0], n)
    ys = rng.integers(0, shape[1], n)
    axes1 = stats.loguniform.rvs(axis1_range[0], axis1_range[1], size=n, random_state=rng).astype(np.int32)
    axes2 = stats.loguniform.rvs(axis2_range[0], axis2_range[1], size=n, random_state=rng).astype(np.int32)
    angles = rng.uniform(0, 360, n)

    labels = _create_labels(shape, xs, ys, axes1, axes2, angles, shift=shift)

    f_X = stats.nbinom.rvs(f_n, f_p, size=shape, random_state=rng)
    b_X = stats.nbinom.rvs(b_n, b_p, size=shape, random_state=rng)

    # Dropout
    f_X[rng.random(shape) < f_do] = 0
    b_X[rng.random(shape) < b_do] = 0
    X = np.where(labels > 0, f_X, b_X)
    adata = AnnData(X=X, layers={"labels": labels})
    SKM.init_adata_type(adata, SKM.ADATA_AGG_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, None)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, 1)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, None)
    return adata
