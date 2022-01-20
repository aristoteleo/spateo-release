"""Belief propagation to compute the marginal probabilities of a spot being
occupied by a cell.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from fbgbp import FastBinaryGridBeliefPropagation
from scipy import stats

from . import utils


def create_neighbor_offsets(neighborhood: np.ndarray) -> np.ndarray:
    """Helper function to convert a neighborhood mask to coordinate offsets.

    Args:
        neighborhood: Boolean mask (kernel) indicating the neighborhood to
            consider.

    Returns:
        A Numpy array containing offsets for each dimension. Each row is
        the offsets required to reach a single neighbor.

    Raises:
        ValueError: If `neighborhood` does not have odd size dimensions.
    """
    for s in neighborhood.shape:
        if s % 2 == 0:
            raise ValueError("`neighborhood` must have odd dimension sizes")

    center = tuple(((np.array(neighborhood.shape) - 1) / 2).astype(int))
    neighborhood[center] = False
    max_neighbors = neighborhood.sum()
    neighbor_offsets = np.zeros((max_neighbors, neighborhood.ndim), dtype=np.int16)
    max_neighbors = neighborhood.sum()
    neighbor_offsets = np.zeros((max_neighbors, neighborhood.ndim), dtype=np.int16)
    for i, dim in enumerate(np.where(neighborhood)):
        neighbor_offsets[:, i] = dim - center[i]
    return neighbor_offsets


def cell_marginals(
    background_probs: np.ndarray,
    cell_probs: np.ndarray,
    neighborhood: Optional[np.ndarray] = None,
    p: float = 0.7,
    q: float = 0.3,
    precision: float = 1e-5,
    max_iter: int = 100,
    n_threads: int = 1,
) -> np.ndarray:
    """Compute the marginal probablity of each pixel being a cell, as opposed
    to background. This function calls a fast belief propagation library
    optimized for grid Markov Random Fields of arbitray dimension.

    Args:
        background_probs: The probability of each pixel being background (for
            instance, computed by taking the PDF of the parameters estmiated by EM).
        cell_probs: The probability of each pixel being a cell (for instance,
            computed by taking the PDF of the parameters estmiated by EM).
        neighborhood: A mask (kernel) indicating the neighborhood of each node
            to consider. The node is at the center of this array. Defaults to
            immediate neighbors (no diagonals).
        p: The potential indicating how likely two adjacent cells will be the
            same state. Does not necessarily have to be a probability.
        p: The potential indicating how likely two adjacent cells will be
            different states. Does not necessarily have to be a probability.
        precision: Stop iterations when desired precision is reached, as computed
            by the L2-norm of the messages from two consecutive iterations.
        max_iter: Maximum number of iterations.
        n_threads: Number of threads to use.

    Returns:
        The marginal probability, at each pixel, of the pixel being a cell.
    """
    if cell_probs.shape != background_probs.shape:
        raise ValueError("`cell_probs` and `background_probs` must have the same shape")
    neighborhood = neighborhood > 0 if neighborhood is not None else utils.circle(3).astype(bool)
    if cell_probs.ndim != neighborhood.ndim:
        raise ValueError("`neighborhood` and `cell_probs` must have the same number of dimensions")
    neighbor_offsets = create_neighbor_offsets(neighborhood)
    shape = np.array(cell_probs.shape, dtype=np.uint32)
    potentials0 = background_probs.flatten().astype(np.double)
    potentials1 = cell_probs.flatten().astype(np.double)
    bp = FastBinaryGridBeliefPropagation(shape, neighbor_offsets, potentials0, potentials1, p, q)
    bp.run(precision=precision, max_iter=max_iter, n_threads=n_threads)
    return bp.marginals()


def run_bp(
    X: np.ndarray,
    background_params: Tuple[float, float],
    cell_params: Tuple[float, float],
    k: int = 3,
    square: bool = False,
    p: float = 0.7,
    q: float = 0.3,
    precision: float = 1e-6,
    max_iter: int = 100,
    n_threads: int = 1,
    certain_mask: Optional[np.ndarray] = None,
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
        p: The potential indicating how likely two adjacent cells will be the
            same state. Does not necessarily have to be a probability.
        p: The potential indicating how likely two adjacent cells will be
            different states. Does not necessarily have to be a probability.
        precision: Stop iterations when desired precision is reached, as computed
            by the L2-norm of the messages from two consecutive iterations.
        max_iter: Maximum number of iterations.
        n_threads: Number of threads to use.
        certain_mask: A boolean Numpy array indicating which pixels are certain
            to be occupied, a-priori. For example, if nuclei staining is available,
            this would be the nuclei segmentation mask.

    Returns:
        Numpy array of marginal probabilities.
    """
    background_probs = stats.nbinom(n=background_params[0], p=background_params[1]).pmf(X)
    cell_probs = stats.nbinom(n=cell_params[0], p=cell_params[1]).pmf(X)
    if certain_mask is not None:
        background_probs = np.clip(background_probs - certain_mask, 0, 1)
        cell_probs = np.clip(cell_probs + certain_mask, 0, 1)

    neighborhood = np.ones((k, k)) if square else utils.circle(k)
    marginals = cell_marginals(
        background_probs,
        cell_probs,
        neighborhood=neighborhood,
        p=p,
        q=q,
        precision=precision,
        max_iter=max_iter,
        n_threads=n_threads,
    )
    return marginals
