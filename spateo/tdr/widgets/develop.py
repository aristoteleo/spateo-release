from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from pyvista import MultiBlock

from ..models import add_model_labels, collect_model
from .interpolations import get_X_Y_grid


def _develop_vectorfield(
    stage1_X: Optional[np.ndarray] = None,
    stage2_X: Optional[np.ndarray] = None,
    NX: Optional[np.ndarray] = None,
    grid_num: List = [50, 50, 50],
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    **kwargs,
) -> dict:
    """Learn a continuous mapping from stage1 to stage2 with the Kernel method (sparseVFC).

    Args:
        stage1_X: The spatial coordinates of each data point of stage1.
        stage2_X: The spatial coordinates of each data point of stage2.
        NX: The spatial coordinates of new data point. If NX is None, generate new points based on grid_num.
        grid_num: Number of grid to generate. Default is 50 for each dimension. Must be non-negative.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
                 weights on regularization.
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.
        **kwargs: Additional parameters that will be passed to SparseVFC function.

    Returns:
        A dictionary which contains:
            X: Current state.
            valid_ind: The indices of cells that have finite velocity values.
            X_ctrl: Sample control points of current state.
            ctrl_idx: Indices for the sampled control points.
            Y: Velocity estimates in delta t.
            beta: Parameter of the Gaussian Kernel for the kernel matrix (Gram matrix).
            V: Prediction of velocity of X.
            C: Finite set of the coefficients for the
            P: Posterior probability Matrix of inliers.
            VFCIndex: Indexes of inliers found by sparseVFC.
            sigma2: Energy change rate.
            grid: Grid of current state.
            grid_V: Prediction of velocity of the grid.
            iteration: Number of the last iteration.
            tecr_vec: Vector of relative energy changes rate comparing to previous step.
            E_traj: Vector of energy at each iteration.
        Here the most important results are X, V, grid and grid_V.
            X: Cell coordinates of the current state.
            V: Prediction of development direction of the X.
            grid: Grid coordinates of current state.
            grid_V: Prediction of development direction of the grid.
    """

    from dynamo.vectorfield.scVectorField import SparseVFC

    _, _, Grid, grid_in_hull = get_X_Y_grid(X=stage1_X.copy(), Y=stage2_X.copy(), grid_num=grid_num)

    predict_X = Grid if NX is None else NX
    res = SparseVFC(stage1_X, stage2_X, predict_X, lambda_=lambda_, lstsq_method=lstsq_method, **kwargs)
    return res


def develop_vectorfield(
    adata: AnnData,
    mapping_key: str = "model_align",
    key_added: str = "VecFld_develop",
    NX: Optional[np.ndarray] = None,
    grid_num: List = [50, 50, 50],
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    inplace: bool = True,
    **kwargs,
) -> Optional[AnnData]:
    """Learn a continuous mapping from stage1 to stage2 with the Kernel method (sparseVFC).

    Args:
        adata: AnnData object that contains the cell coordinates of the two states after alignment.
        mapping_key: The key from the adata uns that corresponds to the aligned cell coordinates.
        key_added: The key that will be used for the vector field key in adata.uns.
        NX: The spatial coordinates of new data point. If NX is None, generate new points based on grid_num.
        grid_num: Number of grid to generate. Default is 50 for each dimension. Must be non-negative.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
                 weights on regularization.
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.
        inplace: Whether to copy `adata` or modify it inplace.
        **kwargs: Additional parameters that will be passed to SparseVFC function.

    Returns:
        An `annData` object is updated/copied with the `key_added` dictionary in the `uns` attribute.
        The `key_added` dictionary which contains:
            X: Current state.
            valid_ind: The indices of cells that have finite velocity values.
            X_ctrl: Sample control points of current state.
            ctrl_idx: Indices for the sampled control points.
            Y: Velocity estimates in delta t.
            beta: Parameter of the Gaussian Kernel for the kernel matrix (Gram matrix).
            V: Prediction of velocity of X.
            C: Finite set of the coefficients for the
            P: Posterior probability Matrix of inliers.
            VFCIndex: Indexes of inliers found by sparseVFC.
            sigma2: Energy change rate.
            grid: Grid of current state.
            grid_V: Prediction of velocity of the grid.
            iteration: Number of the last iteration.
            tecr_vec: Vector of relative energy changes rate comparing to previous step.
            E_traj: Vector of energy at each iteration.
        Here the most important results are X, V, grid and grid_V.
            X: Cell coordinates of the current state.
            V: Prediction of development direction of the X.
            grid: Grid coordinates of current state.
            grid_V: Prediction of development direction of the grid.
    """

    adata = adata if inplace else adata.copy()

    align_spatial_data = adata.uns[mapping_key].copy()
    stage1_coords = np.asarray(align_spatial_data["mapping_X"], dtype=float)
    stage2_coords = np.asarray(align_spatial_data["mapping_Y"], dtype=float)

    adata.uns[key_added] = _develop_vectorfield(
        stage1_X=stage1_coords,
        stage2_X=stage2_coords,
        NX=NX,
        grid_num=grid_num,
        lambda_=lambda_,
        lstsq_method=lstsq_method,
        **kwargs,
    )

    return None if inplace else adata


def develop_trajectory(
    adata: AnnData,
    vf_key: str = "VecFld_develop",
    key_added: str = "develop",
    layer: str = "X",
    direction: str = "forward",
    interpolation_num: int = 250,
    average: bool = False,
    inverse_transform: bool = False,
    inplace: bool = True,
    cores: int = 1,
    **kwargs,
):
    """ """
    import dynamo as dyn

    adata = adata if inplace else adata.copy()

    tmp_adata = adata.copy()
    tmp_adata.uns[f"VecFld_{key_added}"] = tmp_adata.uns[vf_key]
    tmp_adata.obsm[f"X_{key_added}"] = tmp_adata.uns[f"VecFld_{key_added}"]["X"]

    dyn.pd.fate(
        tmp_adata,
        init_cells=tmp_adata.obs_names.tolist(),
        basis=key_added,
        layer=layer,
        interpolation_num=interpolation_num,
        direction=direction,
        inverse_transform=inverse_transform,
        average=average,
        cores=cores,
        **kwargs,
    )

    adata.uns[f"fate_{key_added}"] = tmp_adata.uns[f"fate_{key_added}"]
    return None if inplace else adata