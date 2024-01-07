from typing import List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from spateo.logging import logger_manager as lm
from spateo.tdr.interpolations import get_X_Y_grid


def _con_K(
    x: np.ndarray, y: np.ndarray, beta: float = 0.1, method: str = "cdist", return_d: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    if len(x.shape) == 1:
        x = x[None, :]
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]
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
    d = cdist(x, kernel_dict["X"], "euclidean")
    nearest_idx = np.argmin(d, axis=1)
    # calculate the geodesic distance

    # get the first node in the path to inducing points
    nearest_inducing_nodes = kernel_dict["first_node_idx"][nearest_idx]
    # mask that indicates whether the inducing points are in the same connected component
    K_mask = nearest_inducing_nodes < 0
    nearest_inducing_nodes[nearest_inducing_nodes < 0] = 0
    # calculate the distance to that first nodes
    gather_inducing_nodes = kernel_dict["X"][nearest_inducing_nodes]
    to_first_node_dist_D = np.tile(x[:, None, :], [1, gather_inducing_nodes.shape[1], 1]) - gather_inducing_nodes
    to_first_node_dist = np.sqrt(np.sum(to_first_node_dist_D**2, axis=2))
    origin_to_first_node_dist = (
        np.tile(kernel_dict["X"][nearest_idx][:, None, :], [1, gather_inducing_nodes.shape[1], 1])
        - gather_inducing_nodes
    )
    origin_to_first_node_dist = np.sqrt(np.sum(origin_to_first_node_dist**2, axis=2))
    D = kernel_dict["kernel_graph_distance"][nearest_idx] + to_first_node_dist - origin_to_first_node_dist

    # apply the mask
    D[K_mask] = 10000
    # calculate the kernel
    K = D**2
    K = -beta * K
    K = np.squeeze(np.exp(K))
    if return_d:
        to_first_node_dist_D[K_mask, :] = 0
        D = D[:, :, None] * to_first_node_dist_D / to_first_node_dist[:, :, None]
        D = D.transpose([0, 2, 1])
        return K, D
    else:
        return K


def _gp_velocity(X: np.ndarray, vf_dict: dict) -> np.ndarray:
    pre_scale = vf_dict["pre_norm_scale"]
    norm_x = (X - vf_dict["norm_dict"]["mean_transformed"]) / vf_dict["norm_dict"]["scale"]
    if vf_dict["kernel_dict"]["dist"] == "cdist":
        quary_kernel = _con_K(norm_x, vf_dict["X_ctrl"], vf_dict["beta"])
    elif vf_dict["kernel_dict"]["dist"] == "geodist":
        quary_kernel = _con_K_geodist(norm_x, vf_dict["kernel_dict"], vf_dict["beta"])
    else:
        raise ValueError(f"current only support cdist and geodist")
    quary_velocities = np.dot(quary_kernel, vf_dict["C"])
    quary_velocities = quary_velocities * vf_dict["norm_dict"]["scale"]
    quary_velocities = quary_velocities + (pre_scale - 1) * X
    return quary_velocities / 10000


def morphofield_gp(
    adata: AnnData,
    spatial_key: str = "align_spatial",
    vf_key: str = "VecFld_morpho",
    NX: Optional[np.ndarray] = None,
    grid_num: Optional[List[int]] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculating and predicting the vector field during development by the Gaussian Process method.

    Args:
        adata: AnnData object that contains the cell coordinates of the two states after alignment.
        spatial_key: The key from the ``.obsm`` that corresponds to the spatial coordinates of each cell.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the vector field key in ``.uns``.
        NX: The spatial coordinates of new data point. If NX is None, generate new points based on grid_num.
        grid_num: The number of grids in each dimension for generating the grid velocity. Default is ``[50, 50, 50]``.
        inplace: Whether to copy adata or modify it inplace.

    Returns:

        An ``AnnData`` object is updated/copied with the ``key_added`` dictionary in the ``.uns`` attribute.

        The ``key_added`` dictionary which contains:

            X: Cell coordinates of the current state.
            V: Developmental direction of the X.
            grid: Grid coordinates of current state.
            grid_V: Prediction of developmental direction of the grid.
            method: The method of learning vector field. Here method == 'gaussian_process'.
    """

    adata = adata if inplace else adata.copy()
    if vf_key in adata.uns.keys():
        vf_dict = adata.uns[vf_key]
        vf_dict["X"] = np.asarray(adata.obsm[spatial_key], dtype=float)
        vf_dict["V"] = _gp_velocity(vf_dict["X"], vf_dict=vf_dict)

        if not (NX is None):
            predict_X = NX
        else:
            if grid_num is None:
                grid_num = [50, 50, 50]
                lm.main_warning(f"grid_num and NX are both None, using `grid_num = [50,50,50]`.", indent_level=1)
            _, _, Grid, grid_in_hull = get_X_Y_grid(X=vf_dict["X"].copy(), Y=vf_dict["V"].copy(), grid_num=grid_num)
            predict_X = Grid
        vf_dict["grid"] = predict_X
        vf_dict["grid_V"] = _gp_velocity(predict_X, vf_dict=vf_dict)

        vf_dict["method"] = "gaussian_process"
        lm.main_finish_progress(progress_name="morphofield")
    else:
        raise Exception(
            f"The {vf_key} that corresponds to the reconstructed vector field is not in ``anndata.uns``."
            f"Please run ``st.align.morpho_align(adata, vecfld_key_added='{vf_key}')`` before running this function."
        )

    return None if inplace else adata
