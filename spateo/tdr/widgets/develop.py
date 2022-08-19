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
            method: The method of learning vector field. Here method == 'sparsevfc'.
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
    res["method"] = "sparsevfc"
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
        An `AnnData` object is updated/copied with the `key_added` dictionary in the `.uns` attribute.
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
            method: The method of learning vector field. Here method == 'sparsevfc'.
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
    cores: int = 1,
    inplace: bool = True,
    **kwargs,
) -> Optional[AnnData]:
    """
    Prediction of cell development trajectory based on reconstructed vector field.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `.uns` attribute.
        vf_key: The key in `.uns` that corresponds to the reconstructed vector field.
        key_added: The key (`fate_{key_added}`) under which to add the dictionary Fate (includes `t` and `prediction` keys).
        layer: Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
        direction: The direction to predict the cell fate. One of the `forward`, `backward` or `both` string.
        interpolation_num:  The number of uniformly interpolated time points.
        average: The method to calculate the average cell state at each time step, can be one of `origin` or
                 `trajectory`. If `origin` used, the average expression state from the init_cells will be calculated and
                 the fate prediction is based on this state. If `trajectory` used, the average expression states of all
                 cells predicted from the vector field function at each time point will be used. If `average` is
                 `False`, no averaging will be applied.
        cores: Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
               multiprocessing will be used to parallel the fate prediction.
        inplace: Whether to copy `adata` or modify it inplace.
        **kwargs: Additional parameters that will be passed into the fate function.

    Returns:
        An `AnnData` object is updated/copied with the `fate_{key_added}` dictionary in the `.uns` attribute.
        The  `fate_{key_added}`  dictionary which contains:
            t: The time at which the cell state are predicted.
            prediction: Predicted cells states at different time points. Row order corresponds to the element order in
                        t. If init_states corresponds to multiple cells, the expression dynamics over time for each cell
                        is concatenated by rows. That is, the final dimension of prediction is (len(t) * n_cells,
                        n_features). n_cells: number of cells; n_features: number of genes or number of low dimensional
                        embeddings. Of note, if the average is set to be True, the average cell state at each time point
                        is calculated for all cells.
    """
    from dynamo.prediction.fate import fate

    adata = adata if inplace else adata.copy()
    if vf_key not in adata.uns_keys():
        raise Exception(
            f"You need to first perform sparseVFC before fate prediction, please run"
            f"st.tdr.develop_vectorfield(adata, key_added='{vf_key}' before running this function."
        )
    if f"VecFld_{key_added}" not in adata.uns_keys():
        adata.uns[f"VecFld_{key_added}"] = adata.uns[vf_key]
    if f"X_{key_added}" not in adata.obsm_keys():
        adata.obsm[f"X_{key_added}"] = adata.uns[f"VecFld_{key_added}"]["X"]

    fate(
        adata,
        init_cells=adata.obs_names.tolist(),
        basis=key_added,
        layer=layer,
        interpolation_num=interpolation_num,
        direction=direction,
        average=average,
        cores=cores,
        **kwargs,
    )

    return None if inplace else adata
