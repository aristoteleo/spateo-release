import itertools
import multiprocessing as mp
import sys
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, Dict, List, Optional, Tuple, Union

import numdifftools as nd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.optimize import fsolve
from scipy.sparse import issparse
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

# from ..dynamo_logger import LoggerManager, main_info
# from ..tools.utils import form_triu_matrix, index_condensed_matrix, timeit
# from .FixedPoints import FixedPoints

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class NormDict(TypedDict):
    xm: np.ndarray
    ym: np.ndarray
    xscale: float
    yscale: float
    fix_velocity: bool


class VecFldDict(TypedDict):
    X: np.ndarray
    valid_ind: float
    X_ctrl: np.ndarray
    ctrl_idx: float
    Y: np.ndarray
    beta: float
    V: np.ndarray
    C: np.ndarray
    P: np.ndarray
    VFCIndex: np.ndarray
    sigma2: float
    grid: np.ndarray
    grid_V: np.ndarray
    iteration: int
    tecr_traj: np.ndarray
    E_traj: np.ndarray
    norm_dict: NormDict
    
    
# ---------------------------------------------------------------------------------------------------
# Hessian


def Hessian_rkhs_gaussian(x: np.ndarray, vf_dict: VecFldDict) -> np.ndarray:
    """analytical Hessian for RKHS vector field functions with Gaussian kernel.

    Args:
        x: Coordinates where the Hessian is evaluated. Note that x has to be 1D.
        vf_dict: A dictionary containing RKHS vector field control points, Gaussian bandwidth,
            and RKHS coefficients.
            Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        H: Hessian matrix stored as d-by-d-by-d numpy arrays evaluated at x.
            d is the number of dimensions.
    """
    x = np.atleast_2d(x)

    C = vf_dict["C"]
    beta = vf_dict["beta"]
    K, D = con_K(x, vf_dict["X_ctrl"], beta, return_d=True)

    K = K * C.T

    D = D.T
    D = np.eye(x.shape[1]) - 2 * beta * D @ np.transpose(D, axes=(0, 2, 1))

    H = -2 * beta * np.einsum("ij, jlm -> ilm", K, D)

    return H


def hessian_transformation(H: np.ndarray, qi: np.ndarray, Qj: np.ndarray, Qk: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k x k Hessian matrix (:math:`\partial^2 F_i / \partial x_j \partial x_k`)
    back to the d-dimensional gene expression space. The formula used to inverse transform Hessian matrix calculated
    from low dimension (PCs) is:
                                            :math:`h = \sum_i\sum_j\sum_k q_i q_j q_k H_ijk`,
    where `q, H, h` are the PCA loading matrix, low dimensional Hessian matrix and the inverse transformed element from
    the high dimensional Hessian matrix.

    Args:
        H: k x k x k matrix of the Hessian.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector i.
        Qj: The submatrix of the PC loading matrix Q with dimension d x k, corresponding to regulators j.
        Qk: The submatrix of the PC loading matrix Q with dimension d x k, corresponding to co-regulators k.

    Returns:
        h: The calculated Hessian matrix for the effector i w.r.t regulators j and co-regulators k.
    """

    h = np.einsum("ijk, di -> djk", H, qi)
    Qj, Qk = np.atleast_2d(Qj), np.atleast_2d(Qk)
    h = Qj @ h @ Qk.T

    return h


def elementwise_hessian_transformation(H: np.ndarray, qi: np.ndarray, qj: np.ndarray, qk: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k x k Hessian matrix (:math:`\partial^2 F_i / \partial x_j \partial x_k`) back to the
    d-dimensional gene expression space. The formula used to inverse transform Hessian matrix calculated from
    low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

    Args:
        H: k x k x k matrix of the Hessian.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector i.
        qj: The j-th row of the PC loading matrix Q with dimension d x k, corresponding to the regulator j.
        qk: The k-th row of the PC loading matrix Q with dimension d x k, corresponding to the co-regulator k.

    Returns:
        h: The calculated Hessian elements for each cell.
    """

    h = np.einsum("ijk, i -> jk", H, qi)
    h = qj @ h @ qk

    return h


# ---------------------------------------------------------------------------------------------------
def Laplacian(H: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian of the Hessian matrix by summing the diagonal elements of the Hessian matrix (summing the unmixed second partial derivatives)
                                            :math: `\Delta f = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}`
    Args:
        H: Hessian matrix
    """
    # when H has four dimensions, H is calculated across all cells
    if H.ndim == 4:
        L = np.zeros([H.shape[2], H.shape[3]])
        for sample_indx in range(H.shape[3]):
            for out_indx in range(L.shape[0]):
                L[out_indx, sample_indx] = np.diag(H[:, :, out_indx, sample_indx]).sum()
    else:
        # when H has three dimensions, H is calculated only on one single cell
        L = np.zeros([H.shape[2], 1])
        for out_indx in range(L.shape[0]):
            L[out_indx, 0] = np.diag(H[:, :, out_indx]).sum()

    return L


# ---------------------------------------------------------------------------------------------------
# dynamical properties
def _divergence(f: Callable, x: np.ndarray) -> float:
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)



def compute_divergence(
    f_jac: Callable, X: np.ndarray, Js: Optional[np.ndarray] = None, vectorize_size: int = 1000
) -> np.ndarray:
    """Calculate divergence for many samples by taking the trace of a Jacobian matrix.

    vectorize_size is used to control the number of samples computed in each vectorized batch.
        If vectorize_size = 1, there's no vectorization whatsoever.
        If vectorize_size = None, all samples are vectorized.

    Args:
        f_jac: function for calculating Jacobian from cell states
        X: cell states
        Js: Jacobian matrices for each sample, if X is not provided
        vectorize_size: number of Jacobian matrices to process at once in the vectorization

    Returns:
        divergence np.ndarray across Jacobians for many samples
    """
    n = len(X)
    if vectorize_size is None:
        vectorize_size = n

    div = np.zeros(n)
    for i in tqdm(range(0, n, vectorize_size), desc="Calculating divergence"):
        J = f_jac(X[i : i + vectorize_size]) if Js is None else Js[:, :, i : i + vectorize_size]
        div[i : i + vectorize_size] = np.trace(J)
    return div


def acceleration_(v: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Calculate acceleration by dotting the Jacobian and the velocity vector.

    Args:
        v: velocity vector
        J: Jacobian matrix

    Returns:
        Acceleration vector, with one element for the acceleration of each component
    """
    if v.ndim == 1:
        v = v[:, None]
    return J.dot(v)


def curvature_method1(a: np.array, v: np.array) -> float:
    """https://link.springer.com/article/10.1007/s12650-018-0474-6"""
    if v.ndim == 1:
        v = v[:, None]
    kappa = np.linalg.norm(np.outer(v, a)) / np.linalg.norm(v) ** 3

    return kappa


def curvature_method2(a: np.array, v: np.array) -> float:
    """https://dl.acm.org/doi/10.5555/319351.319441"""
    # if v.ndim == 1: v = v[:, None]
    kappa = (np.multiply(a, np.dot(v, v)) - np.multiply(v, np.dot(v, a))) / np.linalg.norm(v) ** 4

    return kappa


def torsion_(v, J, a):
    """only works in 3D"""
    if v.ndim == 1:
        v = v[:, None]
    tau = np.outer(v, a).dot(J.dot(a)) / np.linalg.norm(np.outer(v, a)) ** 2

    return tau



def compute_acceleration(vf, f_jac, X, Js=None, return_all=False):
    """Calculate acceleration for many samples via

    .. math::
    a = J \cdot v.

    """
    n = len(X)
    acce = np.zeros(n)
    acce_mat = np.zeros((n, X.shape[1]))

    v_ = vf(X)
    J_ = f_jac(X) if Js is None else Js
    # temp_logger = LoggerManager.get_temp_timer_logger()
    # for i in LoggerManager.progress_logger(range(n), temp_logger, progress_name="Calculating acceleration"):
    for i in range(n):
        v = v_[i]
        J = J_[:, :, i]
        acce_mat[i] = acceleration_(v, J).flatten()
        acce[i] = np.linalg.norm(acce_mat[i])

    if return_all:
        return v_, J_, acce, acce_mat
    else:
        return acce, acce_mat



def compute_curvature(vf, f_jac, X, Js=None, formula=2):
    """Calculate curvature for many samples via

    Formula 1:
    .. math::
    \kappa = \frac{||\mathbf{v} \times \mathbf{a}||}{||\mathbf{V}||^3}

    Formula 2:
    .. math::
    \kappa = \frac{||\mathbf{Jv} (\mathbf{v} \cdot \mathbf{v}) -  ||\mathbf{v} (\mathbf{v} \cdot \mathbf{Jv})}{||\mathbf{V}||^4}
    """
    n = len(X)

    curv = np.zeros(n)
    v, _, _, a = compute_acceleration(vf, f_jac, X, Js=Js, return_all=True)
    cur_mat = np.zeros((n, X.shape[1])) if formula == 2 else None

    # for i in LoggerManager.progress_logger(range(n), progress_name="Calculating curvature"):
    for i in range(n):
        if formula == 1:
            curv[i] = curvature_method1(a[i], v[i])
        elif formula == 2:
            cur_mat[i] = curvature_method2(a[i], v[i])
            curv[i] = np.linalg.norm(cur_mat[i])

    return curv, cur_mat



def compute_torsion(vf, f_jac, X):
    """Calculate torsion for many samples via

    .. math::
    \tau = \frac{(\mathbf{v} \times \mathbf{a}) \cdot (\mathbf{J} \cdot \mathbf{a})}{||\mathbf{V} \times \mathbf{a}||^2}
    """
    if X.shape[1] != 3:
        raise Exception(f"torsion is only defined in 3 dimension.")

    n = len(X)

    tor = np.zeros((n, X.shape[1], X.shape[1]))
    v, J, a_, a = compute_acceleration(vf, f_jac, X, return_all=True)

    for i in tqdm(range(n), desc="Calculating torsion"):
        tor[i] = torsion_(v[i], J[:, :, i], a[i])

    return tor



def compute_sensitivity(f_jac, X):
    """Calculate sensitivity for many samples via

    .. math::
    S = (I - J)^{-1} D(\frac{1}{{I-J}^{-1}})
    """
    J = f_jac(X)

    n_genes, n_genes_, n_cells = J.shape
    S = np.zeros_like(J)

    I = np.eye(n_genes)
    for i in tqdm(
        np.arange(n_cells),
        desc="Calculating sensitivity matrix with precomputed component-wise Jacobians",
    ):
        s = np.linalg.inv(I - J[:, :, i])  # np.transpose(J)
        S[:, :, i] = s.dot(np.diag(1 / np.diag(s)))
        # tmp = np.transpose(J[:, :, i])
        # s = np.linalg.inv(I - tmp)
        # S[:, :, i] = s * (1 / np.diag(s)[None, :])

    return S


def curl3d(f, x, method="analytical", VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 3D"""
    if jac is None:
        if method == "analytical" and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x, method="analytical", VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 2D"""
    if jac is None:
        if method == "analytical" and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    curl = jac[1, 0] - jac[0, 1]

    return curl



def compute_curl(f_jac, X):
    """Calculate curl for many samples for 2/3 D systems."""
    if X.shape[1] > 3:
        raise Exception(f"curl is only defined in 2/3 dimension.")

    n = len(X)

    if X.shape[1] == 2:
        curl = np.zeros(n)
        f = curl2d
    else:
        curl = np.zeros((n, 3, 3))
        f = curl3d

    for i in tqdm(range(n), desc=f"Calculating {X.shape[1]}-D curl"):
        J = f_jac(X[i])
        curl[i] = f(None, None, method="analytical", VecFld=None, jac=J)

    return curl
