from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from scipy.spatial.distance import cdist, pdist


from .utils import (
    compute_divergence,
    compute_acceleration,
    compute_curvature,
    compute_torsion,
    compute_sensitivity,
    compute_curl,
)


def get_vf_dict(
    adata: AnnData, 
    basis: str = "", 
    vf_key: str = "VecFld"
) -> dict:
    """Get vector field dictionary from the `.uns` attribute of the AnnData object.

    Args:
        adata: `AnnData` object
        basis: string indicating the embedding data to use for calculating velocities. Defaults to "".
        vf_key: _description_. Defaults to "VecFld".

    Raises:
        ValueError: if vf_key or vfkey_basis is not included in the adata object.

    Returns:
        vector field dictionary
    """
    if basis is not None:
        if len(basis) > 0:
            vf_key = "%s_%s" % (vf_key, basis)

    if vf_key not in adata.uns.keys():
        raise ValueError(
            f"Vector field function {vf_key} is not included in the adata object! "
            f"Try firstly running dyn.vf.VectorField(adata, basis='{basis}')"
        )

    vf_dict = adata.uns[vf_key]
    return vf_dict

def vecfld_from_adata(
    adata: AnnData, 
    basis: str = "", 
    vf_key: str = "VecFld"
) -> Tuple[dict, Callable]:
    vf_dict = get_vf_dict(adata, basis=basis, vf_key=vf_key)

    method = vf_dict["method"]
    if method.lower() == "sparsevfc":
        func = lambda x: vector_field_function(x, vf_dict)
    elif method.lower() == "dynode":
        func = lambda x: dynode_vector_field_function(x, vf_dict)
    elif method.lower() == "morpho":
        func = lambda x: morpho_vector_field_function(x, vf_dict)
    else:
        raise ValueError(f"current only support three methods, SparseVFC, dynode, and morpho")

    return vf_dict, func

def morpho_vector_field_function(
    x: np.ndarray,
    vf_dict: dict,
    dim: Optional[Union[int, np.ndarray]] = None,
    kernel: str = "full",
    X_ctrl_ind: Optional[List] = None,
    **kernel_kwargs,
) -> np.ndarray:
    
    pre_scale = vf_dict['pre_norm_scale']
    
    norm_x = _normalize(x, vf_dict["norm_dict"])
    if vf_dict['kernel_dict']['dist'] == 'cdist':
        quary_kernel = _con_K(norm_x, vf_dict["X_ctrl"], vf_dict["beta"])
    elif vf_dict['kernel_dict']['dist'] == 'geodist':
        quary_kernel = _con_K_geodist(norm_x, vf_dict['kernel_dict'], vf_dict["beta"])
    else:
        raise ValueError(f"current only support cdist and geodist")
    quary_velocities = np.dot(quary_kernel, vf_dict["C"])
    quary_rigid = np.dot(norm_x, vf_dict["R"].T) + vf_dict["t"]
    
    # quary_norm_x = quary_velocities + quary_rigid
    # quary_x = _denormalize(quary_norm_x, vf_dict["norm_dict"])
    quary_velocities = quary_velocities * vf_dict["norm_dict"]["scale"]
    
    quary_velocities = quary_velocities + (pre_scale - 1) * x  # add a pre computed normalize scale
    return quary_velocities / 10000

def morpho_predict_function(
    x: np.ndarray,
    vf_dict: dict,
    dim: Optional[Union[int, np.ndarray]] = None,
    kernel: str = "full",
    X_ctrl_ind: Optional[List] = None,
    **kernel_kwargs,
) -> np.ndarray: # current not add pre_scale 
    
    norm_x = _normalize(x, vf_dict["norm_dict"])
    if vf_dict['kernel_dict']['dist'] == 'cdist':
        quary_kernel = _con_K(norm_x, vf_dict["X_ctrl"], vf_dict["beta"])
    elif vf_dict['kernel_dict']['dist'] == 'geodist':
        quary_kernel = _con_K_geodist(norm_x, vf_dict['kernel_dict'], vf_dict["beta"])
    else:
        raise ValueError(f"current only support cdist and geodist")
    quary_velocities = np.dot(quary_kernel, vf_dict["C"])
    quary_rigid = np.dot(norm_x, vf_dict["R"].T) + vf_dict["t"]
    
    quary_norm_x = quary_velocities + quary_rigid
    quary_x = _denormalize(quary_norm_x, vf_dict["norm_dict"])
    # quary_velocities = quary_velocities * vf_dict["norm_dict"]["scale"]
    return quary_x

def _normalize(
    x: np.ndarray,
    norm_dict: dict,
):
    norm_x = (x - norm_dict["mean_transformed"]) / norm_dict["scale"]
    return norm_x

def _denormalize(
    x: np.ndarray,
    norm_dict: dict,
):
    denorm_x = x * norm_dict["scale"] + norm_dict["mean_fixed"]
    return denorm_x

def _con_K(
    x: np.ndarray, 
    y: np.ndarray, 
    beta: float = 0.1, 
    method: str = "cdist", 
    return_d: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Args:
        x: Original training data points.
        y: Control points used to build kernel basis functions.
        beta: Paramerter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        return_d: If True the intermediate 3D matrix x - y will be returned for analytical Jacobian.

    Returns:
        Tuple(K: the kernel to represent the vector field function, D:
    """
    if len(x.shape) == 1:
        x = x[None, :]
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
        D = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
        K = np.squeeze(np.sum(D**2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K
    
def _con_K_geodist(
    x: np.ndarray,
    kernel_dict: dict,
    beta: float = 0.1,
    return_d: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    
    # find the nearest neighbor
    if len(x.shape) == 1:
        x = x[None, :]
    d = cdist(x, kernel_dict['X'], "euclidean")
    nearest_idx = np.argmin(d, axis=1)
    # calculate the geodesic distance
    
    # get the first node in the path to inducing points
    nearest_inducing_nodes = kernel_dict['first_node_idx'][nearest_idx]
    # mask that indicates whether the inducing points are in the same connected component
    K_mask = nearest_inducing_nodes < 0
    nearest_inducing_nodes[nearest_inducing_nodes < 0] = 0
    # calculate the distance to that first nodes
    gather_inducing_nodes = kernel_dict['X'][nearest_inducing_nodes]
    to_first_node_dist_D = np.tile(x[:,None,:], [1, gather_inducing_nodes.shape[1], 1]) - gather_inducing_nodes
    to_first_node_dist = np.sqrt(np.sum(to_first_node_dist_D**2, axis=2))
    origin_to_first_node_dist = np.tile(kernel_dict['X'][nearest_idx][:,None,:], [1, gather_inducing_nodes.shape[1], 1]) - gather_inducing_nodes
    origin_to_first_node_dist = np.sqrt(np.sum(origin_to_first_node_dist**2, axis=2))
    D = kernel_dict['kernel_graph_distance'][nearest_idx] + to_first_node_dist - origin_to_first_node_dist
    
    # apply the mask
    D[K_mask] = 10000
    # calculate the kernel
    K = D ** 2
    K = -beta * K
    K = np.squeeze(np.exp(K))
    if return_d:
        to_first_node_dist_D[K_mask,:] = 0
        D = D[:,:,None] * to_first_node_dist_D / to_first_node_dist[:,:,None]
        D = D.transpose([0, 2, 1])
        return K, D
    else:
        return K
    


def Jacobian_GP_gaussian_kernel(
    x: np.ndarray, 
    vf_dict: dict, 
    vectorize: bool = False
) -> np.ndarray:
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
    pre_scale = vf_dict['pre_norm_scale']
    x_norm = _normalize(x, vf_dict["norm_dict"])
    if x_norm.ndim == 1:
        if vf_dict['kernel_dict']['dist'] == 'cdist':
            K, D = _con_K(x_norm[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        else:
            K, D = _con_K_geodist(x_norm[None, :], vf_dict['kernel_dict'], vf_dict["beta"], return_d=True)
        J = (vf_dict["C"].T * K) @ D[0].T
    elif not vectorize:
        n, d = x_norm.shape
        J = np.zeros((d, d, n))
        for i, xi in enumerate(x_norm):
            if vf_dict['kernel_dict']['dist'] == 'cdist':
                K, D = _con_K(xi[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
            else:
                K, D = _con_K_geodist(xi[None, :], vf_dict['kernel_dict'], vf_dict["beta"], return_d=True)
            J[:, :, i] = (vf_dict["C"].T * K) @ D[0].T
    else:
        if vf_dict['kernel_dict']['dist'] == 'cdist':
            K, D = _con_K(x_norm, vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        else:
            K, D = _con_K_geodist(x_norm, vf_dict['kernel_dict'], vf_dict["beta"], return_d=True)
        if K.ndim == 1:
            K = K[None, :]
        J = np.einsum("nm, mi, njm -> ijn", K, vf_dict["C"], D)

    return -2 * vf_dict["beta"] * J * pre_scale


class GPVectorField():
    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
        Grid: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        """Initialize the VectorField class.

        Args:
            X: (dimension: n_obs x n_features), Original data.
            V: (dimension: n_obs x n_features), Velocities of cells in the same order and dimension of X.
            Grid: The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
            M: `int` (default: None)
                The number of basis functions to approximate the vector field. By default it is calculated as
                `min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100))))`. So that any datasets with less
                than  about 900 data points (cells) will use full data for vector field reconstruction while any dataset
                larger than that will at most use 1500 data points.
            a: `float` (default 5)
                Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
                outlier's variation space is a.
            beta: `float` (default: None)
                Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
                If None, a rule-of-thumb bandwidth will be computed automatically.
            ecr: `float` (default: 1e-5)
                The minimum limitation of energy change rate in the iteration process.
            gamma: `float` (default:  0.9)
                Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
                Default value is 0.9.
            lambda_: `float` (default: 3)
                Represents the trade-off between the goodness of data fit and regularization.
            minP: `float` (default: 1e-5)
                The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
                minP.
            MaxIter: `int` (default: 500)
                Maximum iteration times.
            theta: `float` (default 0.75)
                Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
                then it is regarded as an inlier.
            div_cur_free_kernels: `bool` (default: False)
                A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
                vector field.
            sigma: `int`
                Bandwidth parameter.
            eta: `int`
                Combination coefficient for the divergence-free or the curl-free kernels.
            seed : int or 1-d array_like, optional (default: `0`)
                Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
                Default is to be 0 for ensure consistency between different runs.
        """

        if X is not None and V is not None:
            self.parameters = kwargs
            self.parameters = update_n_merge_dict(
                self.parameters,
                {
                    "M": kwargs.pop("M", None) or max(min([50, len(X)]), int(0.05 * len(X)) + 1),
                    # min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100)))),
                    "a": kwargs.pop("a", 5),
                    "beta": kwargs.pop("beta", None),
                    "ecr": kwargs.pop("ecr", 1e-5),
                    "gamma": kwargs.pop("gamma", 0.9),
                    "lambda_": kwargs.pop("lambda_", 3),
                    "minP": kwargs.pop("minP", 1e-5),
                    "MaxIter": kwargs.pop("MaxIter", 500),
                    "theta": kwargs.pop("theta", 0.75),
                    "div_cur_free_kernels": kwargs.pop("div_cur_free_kernels", False),
                    "velocity_based_sampling": kwargs.pop("velocity_based_sampling", True),
                    "sigma": kwargs.pop("sigma", 0.8),
                    "eta": kwargs.pop("eta", 0.5),
                    "seed": kwargs.pop("seed", 0),
                },
            )

        # self.norm_dict = {}
        self.data = {}
        
    def from_adata(
        self, 
        adata: AnnData, 
        basis: str = "", 
        vf_key: str = "VecFld",
    ):
        vf_dict, func = vecfld_from_adata(adata, basis=basis, vf_key=vf_key)
        
        self.vf_dict = vf_dict
        self.func = func
        self.data["X"] = vf_dict["X"]
        if "V" in vf_dict.keys():
            self.data["V"] = vf_dict["V"]
        else:
            self.data["V"] = self.predict(vf_dict["X"])
        
    def get_X(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is None:
            return self.data["X"]
        else:
            return self.data["X"][idx]

    def get_V(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is None:
            return self.data["V"]
        else:
            return self.data["V"][idx]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data["X"], self.data["V"]
        
    def predict(
        self,
        X: np.ndarray,
    ):
        return morpho_predict_function(
            X,
            self.vf_dict,
        )
        
    def compute_velocity(
        self,
        X: np.ndarray,
    ):
        return morpho_vector_field_function(
            X,
            self.vf_dict,
        )
        
    def compute_curvature(
        self, 
        X: Optional[np.ndarray] = None, 
        method: str = "analytical", 
        formula: int = 2, 
        **kwargs,
    ):
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_curvature(self.func, f_jac, X, formula=formula, **kwargs)
    
    def compute_acceleration(
        self, 
        X: Optional[np.ndarray] = None, 
        method: str = "analytical", 
        **kwargs
    ) -> np.ndarray:
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_acceleration(self.func, f_jac, X, **kwargs)
    
    def compute_curl(
        self,
        X: Optional[np.ndarray] = None,
        method: str = "analytical",
        dim1: int = 0,
        dim2: int = 1,
        dim3: int = 2,
        **kwargs,
    ) -> np.ndarray:
        
        X = self.data["X"] if X is None else X
        if dim3 is None or X.shape[1] < 3:
            X = X[:, [dim1, dim2]]
        else:
            X = X[:, [dim1, dim2, dim3]]
        f_jac = self.get_Jacobian(method=method, **kwargs)
        return compute_curl(f_jac, X, **kwargs)
    
    def compute_torsion(
        self, 
        X: Optional[np.ndarray] = None, 
        method: str = "analytical", 
        **kwargs
    ) -> np.ndarray:
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_torsion(self.func, f_jac, X, **kwargs)
    
    def compute_divergence(
        self, 
        X: Optional[np.ndarray] = None, 
        method: str = "analytical", 
        **kwargs
    ) -> np.ndarray:
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_divergence(f_jac, X, **kwargs)
    
    def get_Jacobian(
        self, 
        method: str = "analytical", 
        input_vector_convention: str = "row", 
        **kwargs
    ) -> np.ndarray:
        """
        Get the Jacobian of the vector field function.
        If method is 'analytical':
        The analytical Jacobian will be returned and it always
        take row vectors as input no matter what input_vector_convention is.

        If method is 'numerical':
        If the input_vector_convention is 'row', it means that fjac takes row vectors
        as input, otherwise the input should be an array of column vectors. Note that
        the returned Jacobian would behave exactly the same if the input is an 1d array.

        The column vector convention is slightly faster than the row vector convention.
        So the matrix of row vector convention is converted into column vector convention
        under the hood.

        No matter the method and input vector convention, the returned Jacobian is of the
        following format:
                df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                ...         ...         ...         ...
        """
        if method == "numerical":
            return Jacobian_numerical(self.func, input_vector_convention, **kwargs)
        elif method == "parallel":
            return lambda x: Jacobian_rkhs_gaussian_parallel(x, self.vf_dict, **kwargs)
        elif method == "analytical":
            return lambda x: Jacobian_GP_gaussian_kernel(x, self.vf_dict, **kwargs)
        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Currently only "
                f"supports 'analytical', 'numerical', and 'parallel'."
            )
    