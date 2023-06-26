"""Belief propagation to compute the marginal probabilities of a spot being
occupied by a cell.
"""
from typing import Optional

import cv2
import numpy as np

from ..configuration import config
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
    p: float = 0.6,
    q: float = 0.4,
    precision: float = 1e-5,
    max_iter: int = 100,
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

    Returns:
        The marginal probability, at each pixel, of the pixel being a cell.
    """
    try:
        from fbgbp import FastBinaryGridBeliefPropagation
    except ImportError:
        raise ImportError(
            "You need to install the package `fbgbp`." "\nInstall fbgbp via `pip install --upgrade fbgbp`"
        )

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
    bp.run(precision=precision, max_iter=max_iter, n_threads=config.n_threads)
    return bp.marginals()


def run_bp(
    background_cond: np.ndarray,
    cell_cond: np.ndarray,
    k: int = 3,
    square: bool = False,
    p: float = 0.6,
    q: float = 0.4,
    precision: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute the marginal probability of each pixel being a cell, using
    belief propagation.

    Args:
        background_cond: Probability of observing UMIs conditioned on background.
        cell_cond: Probability of observing UMIs conditioned on cell.
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

    Returns:
        Numpy array of marginal probabilities.
    """
    neighborhood = np.ones((k, k)) if square else utils.circle(k)
    marginals = cell_marginals(
        background_cond,
        cell_cond,
        neighborhood=neighborhood,
        p=p,
        q=q,
        precision=precision,
        max_iter=max_iter,
    )
    return marginals
