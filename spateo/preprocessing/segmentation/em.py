import numpy as np
from numba import njit, vectorize


@vectorize
def nb_pmf(k: int, r: int, p: float):
    """Vectorized fast Negative Binomial PMF.

    Args:
        k:
        r:
        p:

    Returns:

    """
    n = k + r - 1
    coef = np.prod(np.arange(k)[::-1] + (n - k + 1)) / (np.prod(np.arange(k) + 1))
    return coef * ((1 - p) ** r) * (p ** k)


@njit
def lamtheta_to_r(lam: float, theta: float) -> float:
    return -lam / np.log(theta)


@njit
def muvar_to_lamtheta(mu: float, var: float) -> Tuple[float, float]:
    r = mu ** 2 / (var - mu)
    theta = mu / var
    lam = -r * np.log(theta)
    return lam, theta


@njit
def lamtheta_to_muvar(lam: float, theta: float) -> Tuple[float, float]:
    r = lamtheta_to_r(lam, theta)
    mu = r / theta - r
    var = mu + mu ** 2 / r
    return mu, var


@njit
def nbn_em(
    X: np.ndarray,
    w: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    max_iter: int = 2000,
    precision: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass
