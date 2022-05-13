"""Implementation of EM algorithm to identify parameter estimates for a
Negative Binomial mixture model.
https://iopscience.iop.org/article/10.1088/1742-6596/1324/1/012093/meta

Written by @HailinPan, optimized by @Lioscro.
"""

from typing import Dict, Optional, Tuple, Union

import ngs_tools as ngs
import numpy as np
from joblib import delayed
from scipy import special, stats

from ..configuration import config
from ..errors import SegmentationError


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


def nbn_pmf(n, p, X):
    """Helper function to compute PMF of negative binomial distribution.

    This function is used instead of calling :func:`stats.nbinom` directly because there
    is some weird behavior when float32 is used. This function essentially casts the `n` and
    `p` parameters as floats.
    """
    return stats.nbinom(n=float(n), p=float(p)).pmf(X)


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

    r = lamtheta_to_r(lam, theta)
    for i in range(max_iter):
        # E step
        bp = nbn_pmf(r[0], theta[0], X)
        cp = nbn_pmf(r[1], theta[1], X)
        tau[0] = w[0] * bp
        tau[1] = w[1] * cp
        # mu = lamtheta_to_muvar(lam, theta)[0]

        # NOTE: tau changes with each line
        # tau[0][(tau.sum(axis=0) <= 1e-9) & (X < mu[0])] = 1
        # tau[1][(tau.sum(axis=0) <= 1e-9) & (X > mu[1])] = 1
        tau = np.clip(tau, 1e-10, 1e10)
        tau /= tau.sum(axis=0)

        beta = 1 - 1 / (1 - theta) - 1 / np.log(theta)

        r = r.reshape(-1, 1)
        delta = r * (special.digamma(r + X) - special.digamma(r))

        tau_sum = tau.sum(axis=1)
        w = tau_sum / tau_sum.sum()
        lam = (tau * delta).sum(axis=1) / tau_sum
        theta = beta * (tau * delta).sum(axis=1) / (tau * (X - (1 - beta).reshape(-1, 1) * delta)).sum(axis=1)

        r = lamtheta_to_r(lam, theta)
        isnan = np.any(np.isnan(r) | np.isnan(w) | np.isnan(theta))
        isinf = np.any(np.isinf(r) | np.isinf(w) | np.isinf(theta))
        isinvalid = np.any((r <= 0) | (theta > 1) | (theta < 0) | (w < 0) | (w > 1))
        use_prev = isnan or isinf or isinvalid

        if (
            max(
                np.abs(w - prev_w).max(),
                np.abs(lam - prev_lam).max(),
                np.abs(theta - prev_theta).max(),
            )
            < precision
        ) or use_prev:
            break

        prev_w = w.copy()
        prev_lam = lam.copy()
        prev_theta = theta.copy()

    return (prev_w, lamtheta_to_r(prev_lam, prev_theta), prev_theta) if use_prev else (w, r, theta)


def conditionals(
    X: np.ndarray,
    em_results: Union[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    ],
    bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the conditional probabilities, for each pixel, of observing the
    observed number of UMIs given that the pixel is background/foreground.

    Args:
        X: UMI counts per pixel
        em_results: Return value of :func:`run_em`.
        bins: Pixel bins, as was passed to :func:`run_em`.

    Returns:
        Two Numpy arrays, the first corresponding to the background conditional
        probabilities, and the second to the foreground conditional probabilities

    Raises:
        SegmentationError: If `em_results` is a dictionary but `bins` was not
            provided.
    """
    if isinstance(em_results, dict):
        if bins is None:
            raise SegmentationError("`em_results` indicate binning was used, but `bins` was not provided")
        background_cond = np.ones(X.shape)
        cell_cond = np.zeros(X.shape)
        for label, (_, r, p) in em_results.items():
            mask = bins == label
            samples = X[mask]
            background_cond[mask] = nbn_pmf(r[0], p[0], samples)
            cell_cond[mask] = nbn_pmf(r[1], p[1], samples)
    else:
        _, r, p = em_results
        background_cond = nbn_pmf(r[0], p[0], X)
        cell_cond = nbn_pmf(r[1], p[1], X)

    return background_cond, cell_cond


def confidence(
    X: np.ndarray,
    em_results: Union[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    ],
    bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute confidence of each pixel being a cell, using the parameters
    estimated by the EM algorithm.

    Args:
        X: Numpy array containing mixture counts.
        em_results: Return value of :func:`run_em`.
        bins: Pixel bins, as was passed to :func:`run_em`.

    Returns:
        Numpy array of confidence scores within the range [0, 1].
    """
    bp, cp = conditionals(X, em_results, bins)
    tau0 = np.zeros(X.shape)
    tau1 = np.zeros(X.shape)
    if isinstance(em_results, dict):
        for label, (w, _, _) in em_results.items():
            mask = bins == label
            tau0[mask] = w[0] * bp[mask]
            tau1[mask] = w[1] * cp[mask]
    else:
        w, _, _ = em_results
        tau0 = w[0] * bp
        tau1 = w[1] * cp
    return tau1 / (tau0 + tau1)


def run_em(
    X: np.ndarray,
    downsample: Union[int, float] = 0.001,
    params: Union[Dict[str, Tuple[float, float]], Dict[int, Dict[str, Tuple[float, float]]]] = dict(
        w=(0.5, 0.5), mu=(10.0, 300.0), var=(20.0, 400.0)
    ),
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
            samples are chosen randomly with probability proportional to the
            log UMI counts. When `bins` is provided, the size of each bin is
            used as a scaling factor. If this is a float, then samples are
            downsampled by this fraction.
        params: Initial parameters. This is a dictionary that contains `w`, `mu`,
            `var` as its keys, each corresponding to initial proportions, means and
            variances of background and foreground pixels. The values must be a
            2-element tuple containing the values for background and foreground.
            This may also be a nested dictionary, where the outermost key maps bin
            labels provided in the `bins` argument. In this case, each of the inner
            dictionaries will be used as the initial paramters corresponding to
            each bin.
        max_iter: Maximum number of EM iterations.
        precision: Stop EM algorithm once desired precision has been reached.
        bins: Bins of pixels to estimate separately, such as those obtained by
            density segmentation. Zeros are ignored.
        seed: Random seed.

    Returns:
        Tuple of parameters estimated by the EM algorithm if `bins` is not provided.
        Otherwise, a dictionary of tuple of parameters, with bin labels as keys.
    """
    samples = {}  # key 0 when bins = None
    if bins is not None:
        for label in np.unique(bins):
            if label > 0:
                samples[label] = X[bins == label]
                _params = params.get(label, params)
                if set(_params.keys()) != {"w", "mu", "var"}:
                    raise SegmentationError("`params` must contain exactly the keys `w`, `mu`, `var`.")
    else:
        samples[0] = X.flatten()
        if set(params.keys()) != {"w", "mu", "var"}:
            raise SegmentationError("`params` must contain exactly the keys `w`, `mu`, `var`.")

    downsample_scale = True
    if downsample > 1:
        downsample_scale = False
    rng = np.random.default_rng(seed)
    final_samples = {}
    total = sum(len(_samples) for _samples in samples.values())
    for label, _samples in samples.items():
        _downsample = int(len(_samples) * downsample) if downsample_scale else int(downsample * (len(_samples) / total))
        if len(_samples) > _downsample:
            weights = np.log1p(_samples + 1)
            _samples = rng.choice(_samples, _downsample, replace=False, p=weights / weights.sum())
        final_samples[label] = np.array(_samples)

    # Run in parallel
    results = {}
    for label, (res_w, res_r, res_p) in zip(
        final_samples.keys(),
        ngs.utils.ParallelWithProgress(n_jobs=config.n_threads, total=len(final_samples), desc="Running EM")(
            delayed(nbn_em)(final_samples[label], max_iter=max_iter, precision=precision, **params.get(label, params))
            for label in final_samples
        ),
    ):
        results[label] = (tuple(res_w), tuple(res_r), tuple(res_p))

    return results if bins is not None else results[0]
