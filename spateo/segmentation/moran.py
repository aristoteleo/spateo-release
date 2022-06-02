"""Cell masking using Moran's I metric.

Adapted from code written by @HailinPan.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from anndata import AnnData
from scipy import signal, stats
from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed

from ..configuration import SKM
from ..logging import logger_manager as lm
from . import utils


def moranI(
    X: np.ndarray, kernel: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Moran's I for cell masking.

    Args:
        X: Numpy array containing (possibly smoothed) UMI counts or binarized
            values.
        kernel: 2D kernel containing weights
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
        p_threshold: P-value threshold. Test. Test Test
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


def run_moran_and_mask_pixels(
    adata: AnnData,
    layer: str,
    k: int = 7,
    method: str = "edge-watershed",
    mk: int = 3,
    mask: Optional[np.ndarray] = None,
    mask_layer: Optional[str] = None,
) -> np.ndarray:
    """Compute scores using Moran's I method.

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to use
        k: Kernel size
        method: Method used for generating cell mask based on p value of Moran's I. 'edge-watershed' or 'otsu'
        mk: Kernel size of morphological open and close operations to reduce
            noise in the mask.
        mask: If provided, only consider pixels within the mask
        mask_layer: Layer to save the final mask. Defaults to `{layer}_mask`.

    Returns:
        A boolean mask.
    """
    # Create Gaussian kernel
    kx = cv2.getGaussianKernel(k, 0)
    ky = cv2.getGaussianKernel(k, 0)
    kernel = (ky * kx.T) * utils.circle(k)
    kernel[(k - 1) // 2, (k - 1) // 2] = 0

    X = SKM.select_layer_data(adata, layer, make_dense=True)
    lm.main_info(f"run Moran’s I.")
    z, c, i, pvalue = moranI(X, kernel, mask=mask)

    if mask is not None:
        m = binary_morani_result(c, pvalue, method=method, tissue_mask=mask)
    else:
        m = binary_morani_result(c, pvalue, method=method)

    m = utils.mclose_mopen(m, mk)

    mask_layer = mask_layer or SKM.gen_new_layer_key(layer, SKM.MASK_SUFFIX)
    SKM.set_layer_data(adata, mask_layer, m)


def binary_morani_result(
    c: np.ndarray,
    p: np.ndarray,
    pvalue_cutoff: float = None,
    method: str = "edge-watershed",  # edge-detection and watershed  'edge-watershed' or 'otsu'
    c_cutoff: float = None,
    tissue_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate cell mask based on Moran's I."""

    if pvalue_cutoff == None:
        if method == "otsu":
            p = (p * 255).astype(np.uint8)
            if isinstance(tissue_mask, np.ndarray):
                p2 = p[tissue_mask > 0]
            else:
                p2 = p.flatten()
            pvalue_cutoff = threshold_otsu(hist=(np.bincount(p2), np.arange(256)))
            # print(f'pvalue_cutoff: {pvalue_cutoff}')
            p_cell_mask = np.where(p <= pvalue_cutoff, 255, 0).astype(np.uint8)
        if method == "edge-watershed":
            edges = sobel(p)
            markers = np.zeros_like(p, np.int8)
            foreground, background = 1, 2
            markers[p > 0.95] = background
            markers[p < 1e-5] = foreground
            ws = watershed(edges, markers)  # np.int32
            p_cell_mask = np.where(ws == 1, 255, 0).astype(np.uint8)
            # cv2.imwrite("p_cell_mask.tif", p_cell_mask)
    else:  # pvalue_cutoff = 0.05
        p_cell_mask = np.where(p <= pvalue_cutoff, 255, 0).astype(np.uint8)

    if c_cutoff == None:
        c = ((c - np.min(c)) / (np.max(c) - np.min(c)) * 255).astype(np.uint8)
        if isinstance(tissue_mask, np.ndarray):
            c2 = c[(p_cell_mask == 255) & (tissue_mask > 0)]
        else:
            c2 = c[p_cell_mask == 255]
        counts = np.bincount(c2)
        if counts[0] == 0:
            counts[0] = 1
        if counts[-1] == 0:
            counts[-1] = 1
        c_cutoff = threshold_otsu(hist=(counts, np.arange(256)))
        # for i in counts:
        #    print(i)
        # print(f'c_cutoff after adjust to 0-255: {c_cutoff}')

        # cv2.imwrite("c_255.tif", c)

    # out
    if isinstance(tissue_mask, np.ndarray):
        cell_mask = np.where((p_cell_mask == 255) & (c >= c_cutoff) & (tissue_mask > 0), 255, 0).astype(np.uint8)
    else:
        cell_mask = np.where((p_cell_mask == 255) & (c >= c_cutoff), 255, 0).astype(np.uint8)

    return cell_mask.astype(np.bool)
