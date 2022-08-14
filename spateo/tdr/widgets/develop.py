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
    align_key: str = "3d_align_spatial",
    key_added: str = "DevVecFld",
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
        align_key: The key from the adata uns that corresponds to the aligned cell coordinates.
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

    align_spatial_data = adata.uns[align_key].copy()
    stage1_coords = np.asarray(align_spatial_data["map_spatial_coords"], dtype=float)
    stage2_coords = np.asarray(align_spatial_data["align_spatial_coords"], dtype=float)

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
            spatial: Grid coordinates.
            direction: Prediction of development direction of the grid.
    """


def cells_development(
    stages_X: List[np.ndarray],
    n_spacing: int = 100,
    key_added: str = "develop",
    label: str = "cells_develop",
    color: str = "skyblue",
) -> MultiBlock:

    cells_points = []
    for i in range(len(stages_X) - 1):
        stage1_X = stages_X[i].copy()
        stage2_X = stages_X[i + 1].copy()
        spacing = (stage2_X - stage1_X) / n_spacing
        cells_points.extend([stage1_X.copy() + j * spacing for j in range(n_spacing)])
    cells_points.append(stages_X[-1])

    cells_models = []
    for points in cells_points:
        model = pv.PolyData(points)
        add_model_labels(
            model=model,
            key_added=key_added,
            labels=np.asarray([label] * model.n_points),
            where="point_data",
            colormap=color,
            inplace=True,
        )
        cells_models.append(model)

    return collect_model(models=cells_models)
