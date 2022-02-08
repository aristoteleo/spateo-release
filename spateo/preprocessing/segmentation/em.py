"""Implementation of EM algorithm to identify parameter estimates for a
Negative Binomial mixture model.
https://iopscience.iop.org/article/10.1088/1742-6596/1324/1/012093/meta

Written by @HailinPan, optimized by @Lioscro.
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import special, stats
from skimage import feature


def lamtheta_to_r(lam: float, theta: float) -> float:
    """Convert lambda and theta to r."""
    return -lam / np.log(theta)


def muvar_to_lamtheta(mu: float, var: float) -> Tuple[float, float]:
    """Convert the mean and variance to lambda and theta."""
    r = mu**2 / (var - mu)
    theta = mu / var
    lam = -r * np.log(theta)
    return lam, theta


def lamtheta_to_muvar(lam: float, theta: float) -> Tuple[float, float]:
    """Convert the lambda and theta to mean and variance."""
    r = lamtheta_to_r(lam, theta)
    mu = r / theta - r
    var = mu + mu**2 / r
    return mu, var


def nbn_em(
    X: np.ndarray,
    w: Tuple[float, float] = (0.99, 0.01),
    mu: Tuple[float, float] = (10.0, 300.0),
    var: Tuple[float, float] = (20.0, 400.0),
    max_iter: int = 2000,
    precision: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the EM algorithm to estimate the parameters for background and cell
    UMIs.

    Args:
        X: Numpy array containing mixture counts
        w: Initial proportions of cell and background as a tuple.
        mu: Initial means of cell and background negative binomial distributions.
        var: Initial variances of cell and background negative binomial
            distributions.
        max_iter: Maximum number of iterations.
        precision: Desired precision. Algorithm will stop once this is reached.

    Returns:
        Estimated `w`, `r`, `p`.
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
        bp = stats.nbinom(n=r[0], p=theta[0]).pmf(X)
        cp = stats.nbinom(n=r[1], p=theta[1]).pmf(X)
        tau[0] = w[0] * bp
        tau[1] = w[1] * cp
        mu = lamtheta_to_muvar(lam, theta)[0]

        # NOTE: tau changes with each line
        tau[0][(tau.sum(axis=0) <= 1e-9) & (X < mu[0] * 2)] = 1
        tau[1][(tau.sum(axis=0) <= 1e-9) & (X >= mu[0] * 2)] = 1
        tau /= tau.sum(axis=0)

        beta = 1 - 1 / (1 - theta) - 1 / np.log(theta)

        r = r.reshape(-1, 1)
        delta = r * (special.digamma(r + X) - special.digamma(r))

        tau_sum = tau.sum(axis=1)
        w = tau_sum / tau_sum.sum()
        lam = (tau * delta).sum(axis=1) / tau_sum
        theta = beta * (tau * delta).sum(axis=1) / (tau * (X - (1 - beta).reshape(-1, 1) * delta)).sum(axis=1)

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

    return (prev_w, lamtheta_to_r(prev_lam, prev_theta), prev_theta) if isnan else (w, lamtheta_to_r(lam, theta), theta)


def confidence(
    X: np.ndarray,
    w: Tuple[float, float],
    r: Tuple[float, float],
    p: Tuple[float, float],
) -> np.ndarray:
    """Compute confidence of each pixel being a cell, using the parameters
    estimated by the EM algorithm.

    Args:
        X: Numpy array containing mixture counts.
        w: Estimated `w` parameters.
        r: Estimated `r` parameters.
        p: Estimated `p` parameters

    Returns:
        Numpy array of confidence scores within the range [0, 1].
    """
    tau = np.zeros((2,) + X.shape)
    bp = stats.nbinom(n=r[0], p=p[0]).pmf(X)
    cp = stats.nbinom(n=r[1], p=p[1]).pmf(X)
    tau[0] = w[0] * bp
    tau[1] = w[1] * cp
    return tau[1] / tau.sum(axis=0)


def run_em(
    X: np.ndarray,
    use_peaks: bool = False,
    min_distance: int = 21,
    downsample: int = 1e6,
    w: Tuple[float, float] = (0.99, 0.01),
    mu: Tuple[float, float] = (10.0, 300.0),
    var: Tuple[float, float] = (20.0, 400.0),
    max_iter: int = 2000,
    precision: float = 1e-6,
    bins: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Union[
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
]:
    """EM

    Args:
        X: UMI counts per pixel.
        use_peaks: Whether to use peaks of convolved image as samples for the
            EM algorithm.
        min_distance: Minimum distance between peaks when `use_peaks=True`
        downsample: Use at most this many samples. If `use_peaks` is False,
            samples are chosen uniformly at random to at most this many samples.
            Otherwise, peaks are chosen uniformly at random. When `bins` is
            provided, this specifies the number of samples to choose per bin.
        w: Initial proportions of cell and background as a tuple.
        mu: Initial means of cell and background negative binomial distributions.
        var: Initial variances of cell and background negative binomial
            distributions.
        max_iter: Maximum number of EM iterations.
        precision: Stop EM algorithm once desired precision has been reached.
        bins: Bins of pixels to estimate separately, such as those obtained by
            density segmentation.
        seed: Random seed.

    Returns:
        Tuple of parameters estimated by the EM algorithm if `bins` is not provided.
        Otherwise, a dictionary of tuple of parameters, with bin labels as keys.
    """
    samples = {}  # key 0 when bins = None
    if use_peaks:
        picks = feature.peak_local_max(X, min_distance=min_distance, labels=bins)
        b = np.zeros(X.shape, dtype=np.uint8)
        b[picks[:, 0], picks[:, 1]] = 1
        n_objects, labels = cv2.connectedComponents(b)

        added = set()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label = labels[i, j]
                if label > 0 and label not in added:
                    samples.setdefault(bins[i, j] if bins is not None else 0, []).append(X[i, j])
                    added.add(label)
    elif bins is not None:
        for label in np.unique(bins):
            samples[label] = X[np.where(bins == label)].flatten()
    else:
        samples[0] = X.flatten()

    downsample = int(downsample)
    rng = np.random.default_rng(seed)
    results = {}
    for label, _samples in samples.items():
        if len(_samples) > downsample:
            _samples = rng.choice(_samples, downsample, replace=False)

        w, r, p = nbn_em(np.array(_samples), w=w, mu=mu, var=var, max_iter=max_iter, precision=precision)
        results[label] = (tuple(w), tuple(r), tuple(p))
    return results if bins is not None else results[0]
