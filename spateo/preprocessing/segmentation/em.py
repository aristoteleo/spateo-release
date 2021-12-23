"""Implementation of EM algorithm to identify parameter estimates for a
Negative Binomial mixture model.
https://iopscience.iop.org/article/10.1088/1742-6596/1324/1/012093/meta

Written by @HailinPan, optimized by @Lioscro.
"""

from typing import Tuple

import numpy as np
from numba import njit, vectorize

from ._digamma import digamma


@vectorize("float64(float64, float64, float64)", nopython=True)
def nb_pmf(k: float, r: float, p: float):
    """Vectorized fast Negative Binomial PMF. Calculations are performed in
    the log domain to prevent over/underflow.

    Args:
        k:
        r:
        p:

    Returns:

    """
    n = k + r - 1
    coef = (
        np.log(np.arange(k)[::-1] + (n - k + 1)).sum() - np.log(np.arange(k) + 1).sum()
    )
    return np.exp(coef + k * np.log(1 - p) + r * np.log(p))


@njit
def lamtheta_to_r(lam: float, theta: float) -> float:
    """Convert lambda and theta to r."""
    return -lam / np.log(theta)


@njit
def muvar_to_lamtheta(mu: float, var: float) -> Tuple[float, float]:
    """Convert the mean and variance to lambda and theta."""
    r = mu ** 2 / (var - mu)
    theta = mu / var
    lam = -r * np.log(theta)
    return lam, theta


@njit
def lamtheta_to_muvar(lam: float, theta: float) -> Tuple[float, float]:
    """Convert the lambda and theta to mean and variance."""
    r = lamtheta_to_r(lam, theta)
    mu = r / theta - r
    var = mu + mu ** 2 / r
    return mu, var


@njit
def nbn_em(
    X: np.ndarray,
    w: Tuple[float, float] = (0.99, 0.01),
    mu: Tuple[float, float] = (10.0, 100.0),
    var: Tuple[float, float] = (20.0, 200.0),
    max_iter: int = 2000,
    precision: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the EM algorithm to estimate the parameters for background and cell
    UMIs.

    Args:
        X:
        w:
        mu:
        var:
        max_iter:
        precision:
    Returns:
        Tuple of (w, r, p)
    """
    w = np.array(w)
    mu = np.array(mu)
    var = np.array(var)
    lam, theta = muvar_to_lamtheta(mu, var)
    tau = np.zeros((2,) + X.shape)

    prev_w = w.copy()
    prev_lam = lam.copy()
    prev_theta = theta.copy()

    for _ in range(max_iter):
        # E step
        r = lamtheta_to_r(lam, theta)
        bp = nb_pmf(X, r[0], theta[0])
        cp = nb_pmf(X, r[1], theta[1])
        tau[0] = w[0] * bp
        tau[1] = w[1] * cp
        mu = lamtheta_to_muvar(lam, theta)[0]

        # NOTE: tau changes with each line
        tau[0][(tau.sum(axis=0) <= 1e-9) & (X < mu[0] * 2)] = 1
        tau[1][(tau.sum(axis=0) <= 1e-9) & (X >= mu[0] * 2)] = 1
        tau /= tau.sum(axis=0)

        beta = 1 - 1 / (1 - theta) - 1 / np.log(theta)

        r = r.reshape(-1, 1)
        delta = r * (digamma(r + X) - digamma(r))

        tau_sum = tau.sum(axis=1)
        w = tau_sum / tau_sum.sum()
        lam = (tau * delta).sum(axis=1) / tau_sum
        theta = (
            beta
            * (tau * delta).sum(axis=1)
            / (tau * (X - (1 - beta).reshape(-1, 1) * delta)).sum(axis=1)
        )

        isnan = np.any(np.isnan(w) | np.isnan(lam) | np.isnan(theta))
        if (
            max(
                np.abs(w - prev_w).max(),
                np.abs(lam - prev_lam).max(),
                np.abs(theta - prev_theta).max(),
            )
            < precision
        ) or isnan:
            break

        prev_w = w.copy()
        prev_lam = lam.copy()
        prev_theta = theta.copy()

    return (
        (prev_w, lamtheta_to_r(prev_lam, prev_theta), prev_theta)
        if isnan
        else (w, lamtheta_to_r(lam, theta), theta)
    )


@njit
def confidence(
    X: np.ndarray,
    w: Tuple[float, float],
    r: Tuple[float, float],
    p: Tuple[float, float],
) -> np.ndarray:
    """Compute confidence of each pixel being a cell, using the parameters
    estimated by the EM algorithm.

    Args:
        X: UMI counts per pixel.
        w:
        r:
        p:

    Returns:
        Numpy array of confidence scores within the range [0, 1].
    """
    tau = np.zeros((2,) + X.shape)
    bp = nb_pmf(X, r[0], p[0])
    cp = nb_pmf(X, r[1], p[1])
    tau[0] = w[0] * bp
    tau[1] = w[1] * cp
    return tau[1] / tau.sum(axis=0)
