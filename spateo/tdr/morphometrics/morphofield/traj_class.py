from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy
from scipy.interpolate import interp1d


class Trajectory:
    """Base class for handling trajectory interpolation, resampling, etc."""

    def __init__(self, X: np.ndarray, t: Union[None, np.ndarray] = None, sort: bool = True) -> None:
        """Initializes a Trajectory object.

        Args:
            X: trajectory positions, shape (n_points, n_dimensions)
            t: trajectory times, shape (n_points,). Defaults to None.
            sort: whether to sort the time stamps. Defaults to True.
        """
        self.X = X
        if t is None:
            self.t = None
        else:
            self.set_time(t, sort=sort)

    def __len__(self) -> int:
        """Returns the number of points in the trajectory.

        Returns:
            number of points in the trajectory
        """
        return self.X.shape[0]

    def set_time(self, t: np.ndarray, sort: bool = True) -> None:
        """Set the time stamps for the trajectory. Sorts the time stamps if requested.

        Args:
            t: trajectory times, shape (n_points,)
            sort: whether to sort the time stamps. Defaults to True.
        """
        if sort:
            I = np.argsort(t)
            self.t = t[I]
            self.X = self.X[I]
        else:
            self.t = t

    def dim(self) -> int:
        """Returns the number of dimensions in the trajectory.

        Returns:
            number of dimensions in the trajectory
        """
        return self.X.shape[1]

    def calc_tangent(self, normalize: bool = True):
        """Calculate the tangent vectors of the trajectory.

        Args:
            normalize: whether to normalize the tangent vectors. Defaults to True.

        Returns:
            tangent vectors of the trajectory, shape (n_points-1, n_dimensions)
        """
        tvec = self.X[1:] - self.X[:-1]
        if normalize:
            tvec = normalize_vectors(tvec)
        return tvec

    def calc_arclength(self) -> float:
        """Calculate the arc length of the trajectory.

        Returns:
            arc length of the trajectory
        """
        tvec = self.calc_tangent(normalize=False)
        norms = np.linalg.norm(tvec, axis=1)
        return np.sum(norms)

    def calc_curvature(self) -> np.ndarray:
        """Calculate the curvature of the trajectory.

        Returns:
            curvature of the trajectory, shape (n_points,)
        """
        tvec = self.calc_tangent(normalize=False)
        kappa = np.zeros(self.X.shape[0])
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf (p. 55)
            kappa[i] = angle(tvec[i - 1], tvec[i]) / (np.linalg.norm(tvec[i - 1] / 2) + np.linalg.norm(tvec[i] / 2))
        return kappa

    def resample(self, n_points: int, tol: float = 1e-4, inplace: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the curve with the specified number of points.

        Args:
            n_points: An integer specifying the number of points in the resampled curve.
            tol: A float specifying the tolerance for removing redundant points. Default is 1e-4.
            inplace: A boolean flag indicating whether to modify the curve object in place. Default is True.

        Returns:
            A tuple containing the resampled curve coordinates and time values (if available).

        Raises:
            ValueError: If the specified number of points is less than 2.

        TODO:
            Decide whether the tol argument should be included or not during the code refactoring and optimization.
        """
        # remove redundant points
        """if tol is not None:
            X, arclen, discard = remove_redundant_points_trajectory(self.X, tol=tol, output_discard=True)
            if self.t is not None:
                t = np.array(self.t[~discard], copy=True)
            else:
                t = None
        else:
            X = np.array(self.X, copy=True)
            t = np.array(self.t, copy=True) if self.t is not None else None
            arclen = self.calc_arclength()"""

        # resample using the arclength sampling
        # ret = arclength_sampling(X, arclen / n_points, t=t)
        ret = arclength_sampling_n(self.X, n_points, t=self.t)
        X = ret[0]
        if self.t is not None:
            t = ret[2]

        if inplace:
            self.X, self.t = X, t

        return X, t

    def archlength_sampling(
        self,
        sol: scipy.integrate._ivp.common.OdeSolution,
        interpolation_num: int,
        integration_direction: str,
    ) -> None:
        """Sample the curve using archlength sampling.

        Args:
            sol: The ODE solution from scipy.integrate.solve_ivp.
            interpolation_num: The number of points to interpolate the curve at.
            integration_direction: The direction to integrate the curve in. Can be "forward", "backward", or "both".
        """
        tau, x = self.t, self.X.T
        idx = dup_osc_idx_iter(x, max_iter=100, tol=x.ptp(0).mean() / 1000)[0]

        # idx = dup_osc_idx_iter(x)
        x = x[:idx]
        _, arclen, _ = remove_redundant_points_trajectory(x, tol=1e-4, output_discard=True)
        cur_Y, alen, self.t = arclength_sampling_n(x, num=interpolation_num + 1, t=tau[:idx])
        self.t = self.t[1:]
        cur_Y = cur_Y[:, 1:]

        if integration_direction == "both":
            neg_t_len = sum(np.array(self.t) < 0)

        self.X = (
            sol(self.t)
            if integration_direction != "both"
            else np.hstack(
                (
                    sol[0](self.t[:neg_t_len]),
                    sol[1](self.t[neg_t_len:]),
                )
            )
        )

    def logspace_sampling(
        self,
        sol: scipy.integrate._ivp.common.OdeSolution,
        interpolation_num: int,
        integration_direction: str,
    ) -> None:
        """Sample the curve using logspace sampling.

        Args:
            sol: The ODE solution from scipy.integrate.solve_ivp.
            interpolation_num: The number of points to interpolate the curve at.
            integration_direction: The direction to integrate the curve in. Can be "forward", "backward", or "both".
        """
        tau, x = self.t, self.X.T
        neg_tau, pos_tau = tau[tau < 0], tau[tau >= 0]

        if len(neg_tau) > 0:
            t_0, t_1 = (
                -(
                    np.logspace(
                        0,
                        np.log10(abs(min(neg_tau)) + 1),
                        interpolation_num,
                    )
                )
                - 1,
                np.logspace(0, np.log10(max(pos_tau) + 1), interpolation_num) - 1,
            )
            self.t = np.hstack((t_0[::-1], t_1))
        else:
            self.t = np.logspace(0, np.log10(max(tau) + 1), interpolation_num) - 1

        if integration_direction == "both":
            neg_t_len = sum(np.array(self.t) < 0)

        self.X = (
            sol(self.t)
            if integration_direction != "both"
            else np.hstack(
                (
                    sol[0](self.t[:neg_t_len]),
                    sol[1](self.t[neg_t_len:]),
                )
            )
        )

    def interpolate(self, t: np.ndarray, **interp_kwargs) -> np.ndarray:
        """Interpolate the curve at new time values.

        Args:
            t: The new time values at which to interpolate the curve.
            **interp_kwargs: Additional arguments to pass to `scipy.interpolate.interp1d`.

        Returns:
            The interpolated values of the curve at the specified time values.

        Raises:
            Exception: If `self.t` is `None`, which is needed for interpolation.
        """
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return interp1d(self.t, self.X, axis=0, **interp_kwargs)(t)

    def interp_t(self, num: int = 100) -> np.ndarray:
        """Interpolates the `t` parameter linearly.

        Args:
            num: Number of interpolation points.

        Returns:
            The array of interpolated `t` values.
        """
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return np.linspace(self.t[0], self.t[-1], num=num)

    def interp_X(self, num: int = 100, **interp_kwargs) -> np.ndarray:
        """Interpolates the curve at `num` equally spaced points in `t`.

        Args:
            num: The number of points to interpolate the curve at.
            **interp_kwargs: Additional keyword arguments to pass to `scipy.interpolate.interp1d`.

        Returns:
            The interpolated curve at `num` equally spaced points in `t`.
        """
        if self.t is None:
            raise Exception("`self.t` is `None`, which is needed for interpolation.")
        return self.interpolate(self.interp_t(num=num), **interp_kwargs)

    def integrate(self, func: Callable) -> np.ndarray:
        """Calculate the integral of a function along the curve.

        Args:
            func: A function to integrate along the curve.

        Returns:
            The integral of func along the discrete curve.
        """
        F = np.zeros(func(self.X[0]).shape)
        tvec = self.calc_tangent(normalize=False)
        for i in range(1, self.X.shape[0] - 1):
            # ref: http://www.cs.jhu.edu/~misha/Fall09/1-curves.pdf P. 47
            F += func(self.X[i]) * (np.linalg.norm(tvec[i - 1]) + np.linalg.norm(tvec[i])) / 2
        return F

    def calc_msd(self, decomp_dim: bool = True, ref: int = 0) -> Union[float, np.ndarray]:
        """Calculate the mean squared displacement (MSD) of the curve with respect to a reference point.

        Args:
            decomp_dim: If True, return the MSD of each dimension separately. If False, return the total MSD.
            ref: Index of the reference point. Default is 0.

        Returns:
            The MSD of the curve with respect to the reference point.
        """
        S = (self.X - self.X[ref]) ** 2
        if decomp_dim:
            S = S.sum(axis=0)
        else:
            S = S.sum()
        S /= len(self)
        return S


def arclength_sampling_n(
    X: np.ndarray,
    num: int,
    t: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """Uniformly sample data points on an arc curve that generated from vector field predictions.

    Args:
        X: The data points to sample from.
        num: The number of points to sample.
        t: The time values for the data points. Defaults to None.

    Returns:
        The sampled data points and the arc length of the curve.
    """
    arclen = np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))
    arclen = np.hstack((0, arclen))

    z = np.linspace(arclen[0], arclen[-1], num)
    X_ = interp1d(arclen, X, axis=0)(z)
    if t is not None:
        t_ = interp1d(arclen, t)(z)
        return X_, arclen[-1], t_
    else:
        return X_, arclen[-1]


def remove_redundant_points_trajectory(
    X: np.ndarray,
    tol: float = 1e-4,
    output_discard: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """Remove consecutive data points that are too close to each other.

    Args:
        X: The data points to remove redundant points from.
        tol: The tolerance for removing redundant points. Defaults to 1e-4.
        output_discard: Whether to output the discarded points. Defaults to False.

    Returns:
        The data points with redundant points removed and the arc length of the curve.
    """
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        for i in range(len(X) - 1):
            dist = np.linalg.norm(X[i + 1] - X[i])
            if dist < tol:
                discard[i + 1] = True
        X = X[~discard]

    arclength = 0

    x0 = X[0]
    for i in range(1, len(X)):
        tangent = X[i] - x0 if i == 1 else X[i] - X[i - 1]
        d = np.linalg.norm(tangent)

        arclength += d

    if output_discard:
        return (X, arclength, discard)
    else:
        return (X, arclength)


def arclength_sampling(X: np.ndarray, step_length: float, n_steps: int, t: Optional[np.ndarray] = None) -> np.ndarray:
    """Uniformly sample data points on an arc curve that generated from vector field predictions.

    Args:
        X: The data points to sample from.
        step_length: The length of each step.
        n_steps: The number of steps to sample.
        t: The time values for the data points. Defaults to None.

    Returns:
        The sampled data points and the arc length of the curve.
    """
    Y = []
    x0 = X[0]
    T = [] if t is not None else None
    t0 = t[0] if t is not None else None
    i = 1
    terminate = False
    arclength = 0

    def _calculate_new_point():
        x = x0 if j == i else X[j - 1]
        cur_y = x + (step_length - L) * tangent / d

        if t is not None:
            cur_tau = t0 if j == i else t[j - 1]
            cur_tau += (step_length - L) / d * (t[j] - cur_tau)
            T.append(cur_tau)
        else:
            cur_tau = None

        Y.append(cur_y)

        return cur_y, cur_tau

    while i < len(X) - 1 and not terminate:
        L = 0
        for j in range(i, len(X)):
            tangent = X[j] - x0 if j == i else X[j] - X[j - 1]
            d = np.linalg.norm(tangent)
            if L + d >= step_length:
                y, tau = _calculate_new_point()
                t0 = tau if t is not None else None
                x0 = y
                i = j
                break
            else:
                L += d
        if j == len(X) - 1:
            i += 1
        arclength += step_length
        if L + d < step_length:
            terminate = True

    if len(Y) < n_steps:
        _, _ = _calculate_new_point()

    if T is not None:
        return np.array(Y), arclength, T
    else:
        return np.array(Y), arclength
    

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python#answer-13849249
# answer from crizCraig
# @njit(cache=True, nogil=True) # causing numba error_write issue
def angle(vector1, vector2):
    """Returns the angle in radians between given vectors"""
    v1_norm, v1_u = unit_vector(vector1)
    v2_norm, v2_u = unit_vector(vector2)

    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    else:
        minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)
    
# @njit(cache=True, nogil=True) # causing numba error_write issue
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return vec_norm, vector
    else:
        return vec_norm, vector / vec_norm


def normalize_vectors(vectors, axis=1, **kwargs):
    """Returns the unit vectors of the vectors."""
    vec = np.array(vectors, copy=True)
    vec = np.atleast_2d(vec)
    vec_norm = np.linalg.norm(vec, axis=axis, **kwargs)

    vec_norm[vec_norm == 0] = 1
    vec = (vec.T / vec_norm).T
    return vec


def dup_osc_idx_iter(x: np.ndarray, max_iter: int = 5, **kwargs) -> Tuple[int, np.ndarray]:
    """
    Find the index of the end of the first division in an array where the oscillatory patterns of two consecutive divisions are similar within a given tolerance, using iterative search.

    Args:
        x: An array-like object containing the data to be analyzed.
        max_iter: An integer specifying the maximum number of iterations to perform. Defaults to 5.

    Returns:
        A tuple containing the index of the end of the first division and an array of differences between the FFTs of consecutive divisions. If the oscillatory patterns of the two divisions are not similar within the given tolerance after the maximum number of iterations, returns the index and array from the final iteration.
    """
    stop = False
    idx = len(x)
    j = 0
    D = []
    while not stop:
        i, d = dup_osc_idx(x[:idx], **kwargs)
        D.append(d)
        if i is None:
            stop = True
        else:
            idx = i
        j += 1
        if j >= max_iter or idx == 0:
            stop = True
    D = np.array(D)
    return idx, D


def calc_fft(x):
    out = np.fft.rfft(x)
    n = len(x)
    xFFT = abs(out) / n * 2
    freq = np.arange(int(n / 2)) / n
    return xFFT[: int(n / 2)], freq

def dup_osc_idx(x: np.ndarray, n_dom: int = 3, tol: float = 0.05):
    """
    Find the index of the end of the first division in an array where the oscillatory patterns of two consecutive divisions are similar within a given tolerance.

    Args:
        x: An array-like object containing the data to be analyzed.
        n_dom: An integer specifying the number of divisions to make in the array. Defaults to 3.
        tol: A float specifying the tolerance for considering the oscillatory patterns of two divisions to be similar. Defaults to 0.05.

    Returns:
        A tuple containing the index of the end of the first division and the difference between the FFTs of the two divisions. If the oscillatory patterns of the two divisions are not similar within the given tolerance, returns (None, None).
    """
    l_int = int(np.floor(len(x) / n_dom))
    ind_a, ind_b = np.arange((n_dom - 2) * l_int, (n_dom - 1) * l_int), np.arange((n_dom - 1) * l_int, n_dom * l_int)
    y1 = x[ind_a]
    y2 = x[ind_b]

    def calc_fft_k(x):
        ret = []
        for k in range(x.shape[1]):
            xFFT, _ = calc_fft(x[:, k])
            ret.append(xFFT[1:])
        return np.hstack(ret)

    try:
        xFFt1 = calc_fft_k(y1)
        xFFt2 = calc_fft_k(y2)
    except ValueError:
        print("calc_fft_k run failed...")
        return None, None

    diff = np.linalg.norm(xFFt1 - xFFt2) / len(xFFt1)
    if diff <= tol:
        idx = (n_dom - 1) * l_int
    else:
        idx = None
    return idx, diff