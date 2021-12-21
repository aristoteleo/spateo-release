import numpy as np
from fbgbp import FastBinaryGridBeliefPropagation

def cell_marginals(
    cell_probs: np.ndarray,
    background_probs: np.ndarray,
    p: float = 0.7,
    q: float = 0.3,
    precision: float = 1e-5,
    max_iter: int = 100,
    n_threads: int = 1
) -> np.ndarray:
    """Compute the marginal probablity of each pixel being a cell, as opposed
    to background. This function calls a fast belief propagation library
    optimized for grid Markov Random Fields of arbitray dimension.

    Parameters
    ----------
    cell_probs : :class:`~numpy.ndarray`
        The probability of each pixel being a cell (for instance, computed by
        taking the PDF of the parameters estmiated by EM).
    background_probs : :class:`~numpy.ndarray`
        The probability of each pixel being background (for instance, computed by
        taking the PDF of the parameters estmiated by EM).
    p : float
        The potential indicating how likely two adjacent cells will be the
        same state. Does not necessarily have to be a probability.
    p : float
        The potential indicating how likely two adjacent cells will be
        different states. Does not necessarily have to be a probability.
    precision : float
        Stop iterations when desired precision is reached, as computed by
        the L2-norm of the messages from two consecutive iterations.
    max_iter : int
        Maximum number of iterations.
    n_threads : int
        Number of threads to use.

    Returns
    -------
    marginals : :class:`~numpy.ndarray`
        The marginal probability, at each pixel, of the pixel being a cell.
    """
    if cell_probs.shape != background_probs.shape:
        raise ValueError('`cell_probs` and `background_probs` must have the same shape')

    shape = np.array(cell_probs.shape, dtype=np.uint32)
    potentials0 = background_probs.flatten().astype(np.double)
    potentials1 = cell_probs.flatten().astype(np.double)
    bp = FastBinaryGridBeliefPropagation(
        shape, potentials0, potentials1, p, q
    )
    bp.run(precision=precision, max_iter=max_iter, n_threads=n_threads)
    return bp.marginals()
