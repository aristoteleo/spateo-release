from typing import Callable, Optional, Tuple

import numpy as np
from anndata import AnnData
from tqdm import tqdm

#########################
# Differential Geometry #
#########################


def compute_acceleration(vf, f_jac, X, Js=None, return_all=False):
    """Calculate acceleration."""

    n = len(X)
    acce = np.zeros(n)
    acce_mat = np.zeros((n, X.shape[1]))

    v_ = vf(X)
    J_ = f_jac(X) if Js is None else Js

    _acceleration = lambda v, J: J.dot(v[:, None]) if v.ndim == 1 else J.dot(v)
    for i in range(n):
        v = v_[i]
        J = J_[:, :, i]
        acce_mat[i] = _acceleration(v, J).flatten()
        acce[i] = np.linalg.norm(acce_mat[i])

    if return_all:
        return v_, J_, acce, acce_mat
    else:
        return acce, acce_mat


def compute_curvature(vf, f_jac, X, Js=None, formula=2):
    """Calculate curvature."""

    n = len(X)
    curv = np.zeros(n)
    v, _, _, a = compute_acceleration(vf, f_jac, X, Js=Js, return_all=True)
    cur_mat = np.zeros((n, X.shape[1])) if formula == 2 else None

    for i in range(n):
        if formula == 1:
            ai, vi = a[i], v[i]
            vi = vi[:, None] if vi.ndim == 1 else vi
            curv[i] = np.linalg.norm(np.outer(vi, ai)) / np.linalg.norm(vi) ** 3
        elif formula == 2:
            ai, vi = a[i], v[i]
            cur_mat[i] = (np.multiply(ai, np.dot(vi, vi)) - np.multiply(vi, np.dot(vi, ai))) / np.linalg.norm(vi) ** 4
            curv[i] = np.linalg.norm(cur_mat[i])
    return curv, cur_mat


def compute_curl(f_jac, X):
    """Calculate curl for 2D or 3D systems."""

    n = len(X)
    if X.shape[1] == 2:
        curl = np.zeros(n)
        for i in tqdm(range(n), desc=f"Calculating {X.shape[1]}-D curl"):
            jac = f_jac(X[i])
            curl[i] = jac[1, 0] - jac[0, 1]
    elif X.shape[1] == 3:
        curl = np.zeros((n, 3, 3))
        for i in tqdm(range(n), desc=f"Calculating {X.shape[1]}-D curl"):
            jac = f_jac(X[i])
            curl[i] = np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])
    else:
        raise ValueError(f"X has incorrect dimensions.")
    return curl


def compute_torsion(vf, f_jac, X):
    """Calculate torsion."""

    def _torsion(v, J, a):
        """only works in 3D"""
        v = v[:, None] if v.ndim == 1 else v
        tau = np.outer(v, a).dot(J.dot(a)) / np.linalg.norm(np.outer(v, a)) ** 2
        return tau

    if X.shape[1] != 3:
        raise Exception(f"torsion is only defined in 3 dimension.")

    n = len(X)

    tor = np.zeros((n, X.shape[1], X.shape[1]))
    v, J, a_, a = compute_acceleration(vf, f_jac, X, return_all=True)

    for i in tqdm(range(n), desc="Calculating torsion"):
        tor[i] = _torsion(v[i], J[:, :, i], a[i])

    return tor


def compute_divergence(f_jac, X: np.ndarray, Js: Optional[np.ndarray] = None, vectorize_size: int = 1000) -> np.ndarray:
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


def compute_sensitivity(f_jac, X):
    """Calculate sensitivity."""

    J = f_jac(X)
    n_genes, n_genes_, n_cells = J.shape
    S = np.zeros_like(J)

    I = np.eye(n_genes)
    for i in tqdm(np.arange(n_cells), desc="Calculating sensitivity matrix with precomputed component-wise Jacobians"):
        s = np.linalg.inv(I - J[:, :, i])
        S[:, :, i] = s.dot(np.diag(1 / np.diag(s)))
    return S


#################
# GPVectorField #
#################


def Jacobian_GP_gaussian_kernel(X: np.ndarray, vf_dict: dict, vectorize: bool = False) -> np.ndarray:
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

    Args:
    x: Coordinates where the Jacobian is evaluated.
    vf_dict: A dictionary containing RKHS vector field control points, Gaussian bandwidth,
        and RKHS coefficients.
        Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
            d is the number of dimensions and n the number of coordinates in x.
    """
    from ..morphofield.gaussian_process import _con_K, _con_K_geodist

    pre_scale = vf_dict["norm_dict"]["scale_fixed"] / vf_dict["norm_dict"]["scale_transformed"]
    x_norm = (X - vf_dict["norm_dict"]["mean_transformed"]) / vf_dict["norm_dict"]["scale_transformed"]
    if x_norm.ndim == 1:
        if vf_dict["kernel_dict"]["dist"] == "cdist":
            K, D = _con_K(x_norm[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        else:
            K, D = _con_K_geodist(x_norm[None, :], vf_dict["kernel_dict"], vf_dict["beta"], return_d=True)
        J = (vf_dict["C"].T * K) @ D[0].T
    elif not vectorize:
        n, d = x_norm.shape
        J = np.zeros((d, d, n))
        for i, xi in enumerate(x_norm):
            if vf_dict["kernel_dict"]["dist"] == "cdist":
                K, D = _con_K(xi[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
            else:
                K, D = _con_K_geodist(xi[None, :], vf_dict["kernel_dict"], vf_dict["beta"], return_d=True)
            J[:, :, i] = (vf_dict["C"].T * K) @ D[0].T
    else:
        if vf_dict["kernel_dict"]["dist"] == "cdist":
            K, D = _con_K(x_norm, vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        else:
            K, D = _con_K_geodist(x_norm, vf_dict["kernel_dict"], vf_dict["beta"], return_d=True)
        if K.ndim == 1:
            K = K[None, :]
        J = np.einsum("nm, mi, njm -> ijn", K, vf_dict["C"], D)

    return -2 * vf_dict["beta"] * J * pre_scale


class GPVectorField:
    def __init__(self):
        self.data = {}

    def from_adata(self, adata: AnnData, vf_key: str = "VecFld"):
        from ..morphofield.gaussian_process import _gp_velocity

        if vf_key in adata.uns.keys():
            vf_dict = adata.uns[vf_key]
        else:
            raise Exception(
                f"The {vf_key} that corresponds to the reconstructed vector field is not in ``anndata.uns``."
                f"Please run ``st.align.morpho_align(adata, vecfld_key_added='{vf_key}')`` before running this function."
            )

        self.vf_dict = vf_dict
        self.func = lambda x: _gp_velocity(x, vf_dict)
        self.data["X"] = vf_dict["X"]
        self.data["V"] = vf_dict["V"]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data["X"], self.data["V"]

    def compute_velocity(self, X: np.ndarray):
        from ..morphofield.gaussian_process import _gp_velocity

        return _gp_velocity(X, self.vf_dict)

    def compute_acceleration(self, X: Optional[np.ndarray] = None, **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(**kwargs)
        return compute_acceleration(vf=self.func, f_jac=f_jac, X=X)

    def compute_curvature(self, X: Optional[np.ndarray] = None, formula: int = 2, **kwargs):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(**kwargs)
        return compute_curvature(vf=self.func, f_jac=f_jac, X=X, formula=formula)

    def compute_curl(
        self, X: Optional[np.ndarray] = None, dim1: int = 0, dim2: int = 1, dim3: int = 2, **kwargs
    ) -> np.ndarray:

        X = self.data["X"] if X is None else X
        if dim3 is None or X.shape[1] == 2:
            X = X[:, [dim1, dim2]]
        else:
            X = X[:, [dim1, dim2, dim3]]
        f_jac = self.get_Jacobian(**kwargs)
        return compute_curl(f_jac=f_jac, X=X)

    def compute_torsion(self, X: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(**kwargs)
        return compute_torsion(vf=self.func, f_jac=f_jac, X=X)

    def compute_divergence(self, X: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(**kwargs)
        return compute_divergence(f_jac=f_jac, X=X)

    def get_Jacobian(self, method: str = "analytical", **kwargs) -> Callable:
        """
        Get the Jacobian of the vector field function.
        The analytical Jacobian will be returned and it always take row vectors as input no
        matter what input_vector_convention is.

        The returned Jacobian is of the following format:
                df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                ...         ...         ...         ...
        """
        if method == "analytical":
            return lambda x: Jacobian_GP_gaussian_kernel(X=x, vf_dict=self.vf_dict)
