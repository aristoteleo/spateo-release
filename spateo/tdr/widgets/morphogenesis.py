from typing import List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from pyvista import MultiBlock

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ...logging import logger_manager as lm
from ..models import add_model_labels, collect_model
from .interpolations import get_X_Y_grid


def _morphofield(
    X: np.ndarray,
    V: np.ndarray,
    NX: Optional[np.ndarray] = None,
    grid_num: Optional[List[int]] = None,
    M: int = 100,
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    min_vel_corr: float = 0.8,
    restart_num: int = 10,
    restart_seed: Union[List[int], Tuple[int], np.ndarray] = (0, 100, 200, 300, 400),
    **kwargs,
) -> dict:
    """
    Calculating and predicting the vector field during development by the Kernel method (sparseVFC).

    Args:
        X: The spatial coordinates of each data point.
        V: The developmental direction of each data point.
        NX: The spatial coordinates of new data point (grid). If ``NX`` is None, generate grid based on ``grid_num``.
        grid_num: The number of grids in each dimension for generating the grid velocity. Default is ``[50, 50, 50]``.
        M: The number of basis functions to approximate the vector field.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
                 weights on regularization.
        lstsq_method: The name of the linear least square solver, can be either ``'scipy'`` or ``'douin'``.
        min_vel_corr: The minimal threshold for the cosine correlation between input velocities and learned velocities
                      to consider as a successful vector field reconstruction procedure. If the cosine correlation is
                      less than this threshold and ``restart_num`` > 1, ``restart_num`` trials will be attempted with
                      different seeds to reconstruct the vector field function. This can avoid some reconstructions to
                      be trapped in some local optimal.
        restart_num: The number of retrials for vector field reconstructions.
        restart_seed: A list of seeds for each retrial. Must be the same length as ``restart_num`` or None.
        **kwargs: Additional parameters that will be passed to ``SparseVFC`` function.

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
            V: Developmental direction of the X.
            grid: Grid coordinates of current state.
            grid_V: Prediction of developmental direction of the grid.
    """

    from dynamo.vectorfield.scVectorField import SparseVFC

    if not (NX is None):
        predict_X = NX
    else:
        if grid_num is None:
            grid_num = [50, 50, 50]
            lm.main_warning(f"grid_num and NX are both None, using `grid_num = [50,50,50]`.", indent_level=1)
        _, _, Grid, grid_in_hull = get_X_Y_grid(X=X.copy(), Y=V.copy(), grid_num=grid_num)
        predict_X = Grid

    if restart_num > 0:
        restart_seed = np.asarray(restart_seed)
        if len(restart_seed) != restart_num:
            lm.main_warning(
                f"The length of {restart_seed} is different from {restart_num}, " f"using `np.range(restart_num) * 100",
                indent_level=1,
            )
            restart_seed = np.arange(restart_num) * 100

        restart_counter, cur_vf_list, res_list = 0, [], []
        while True:
            cur_vf_dict = SparseVFC(
                X=X,
                Y=V,
                Grid=predict_X,
                M=M,
                lstsq_method=lstsq_method,
                lambda_=lambda_,
                seed=restart_seed[restart_counter],
                **kwargs,
            )

            # consider refactor with .simulation.evaluation.py
            reference, prediction = (
                cur_vf_dict["Y"][cur_vf_dict["valid_ind"]],
                cur_vf_dict["V"][cur_vf_dict["valid_ind"]],
            )
            true_normalized = reference / (np.linalg.norm(reference, axis=1).reshape(-1, 1) + 1e-20)
            predict_normalized = prediction / (np.linalg.norm(prediction, axis=1).reshape(-1, 1) + 1e-20)
            res = np.mean(true_normalized * predict_normalized) * prediction.shape[1]

            cur_vf_list += [cur_vf_dict]
            res_list += [res]
            if res < min_vel_corr:
                restart_counter += 1
                # main_info
                lm.main_info(
                    f"Current cosine correlation ({round(res, 5)}) between input velocities and learned velocities is less than "
                    f"{min_vel_corr}. Make a {restart_counter}-th vector field reconstruction trial.",
                    indent_level=1,
                )
            else:
                vf_dict = cur_vf_dict
                break

            if restart_counter > restart_num - 1:
                # main_warning
                lm.main_warning(
                    f"Cosine correlation between ({round(res, 5)}) input velocities and learned velocities is less than"
                    f" {min_vel_corr} after {restart_num} trials of vector field reconstruction.",
                    indent_level=1,
                )
                vf_dict = cur_vf_list[np.argmax(np.array(res_list))]

                break
    else:
        vf_dict = SparseVFC(X=X, Y=V, Grid=predict_X, M=M, lstsq_method=lstsq_method, lambda_=lambda_, **kwargs)

    vf_dict["method"] = "sparsevfc"
    lm.main_finish_progress(progress_name="morphofield")
    return vf_dict


def morphofield(
    adata: AnnData,
    mapping_key: str = "model_align",
    spatial_key: Literal["mapping", "raw"] = "mapping",
    key_added: str = "VecFld_morpho",
    NX: Optional[np.ndarray] = None,
    grid_num: Optional[List[int]] = None,
    M: int = 100,
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    min_vel_corr: float = 0.8,
    restart_num: int = 10,
    restart_seed: Union[List[int], Tuple[int], np.ndarray] = (0, 100, 200, 300, 400),
    inplace: bool = True,
    **kwargs,
) -> Optional[AnnData]:
    """
    Calculating and predicting the vector field during development by the Kernel method (sparseVFC).

    Args:
        adata: AnnData object that contains the cell coordinates of the two states after alignment.
        mapping_key: The key from the ``.uns`` that corresponds to the cell coordinates.
        spatial_key: The key from the ``.uns[mapping_key]`` that corresponds to the cell coordinates.

                     If ``spatial_key = 'mapping'``, the vector field is constructed using the aligned coordinates.
                     If ``spatial_key = 'raw'``, the vector field is constructed using the raw coordinates.
        key_added: The key that will be used for the vector field key in ``.uns``.
        NX: The spatial coordinates of new data point. If NX is None, generate new points based on grid_num.
        grid_num: The number of grids in each dimension for generating the grid velocity. Default is ``[50, 50, 50]``.
        M: The number of basis functions to approximate the vector field.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
                 weights on regularization.
        lstsq_method: The name of the linear least square solver, can be either ``'scipy'`` or ``'douin'``.
        min_vel_corr: The minimal threshold for the cosine correlation between input velocities and learned velocities
                      to consider as a successful vector field reconstruction procedure. If the cosine correlation is
                      less than this threshold and ``restart_num`` > 1, ``restart_num`` trials will be attempted with
                      different seeds to reconstruct the vector field function. This can avoid some reconstructions to
                      be trapped in some local optimal.
        restart_num: The number of retrials for vector field reconstructions.
        restart_seed: A list of seeds for each retrial. Must be the same length as ``restart_num`` or None.
        **kwargs: Additional parameters that will be passed to ``SparseVFC`` function.

    Returns:

        An ``AnnData`` object is updated/copied with the ``key_added`` dictionary in the ``.uns`` attribute.

        The ``key_added`` dictionary which contains:

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
            V: Developmental direction of the X.
            grid: Grid coordinates of current state.
            grid_V: Prediction of developmental direction of the grid.
    """

    adata = adata if inplace else adata.copy()

    align_spatial_data = adata.uns[mapping_key].copy()
    X = np.asarray(align_spatial_data[f"{spatial_key}_X"], dtype=float)
    V = np.asarray(align_spatial_data[f"{spatial_key}_Y"], dtype=float) - X

    adata.uns[key_added] = _morphofield(
        X=X,
        V=V,
        NX=NX,
        grid_num=grid_num,
        M=M,
        lambda_=lambda_,
        lstsq_method=lstsq_method,
        min_vel_corr=min_vel_corr,
        restart_num=restart_num,
        restart_seed=restart_seed,
        **kwargs,
    )

    return None if inplace else adata


def morphopath(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "fate_morpho",
    layer: str = "X",
    direction: str = "forward",
    interpolation_num: int = 250,
    t_end: Optional[Union[int, float]] = None,
    average: bool = False,
    cores: int = 1,
    inplace: bool = True,
    **kwargs,
) -> Optional[AnnData]:
    """
    Prediction of cell developmental trajectory based on reconstructed vector field.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the ``.uns`` attribute.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key under which to add the dictionary Fate (includes ``t`` and ``prediction`` keys).
        layer: Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
        direction: The direction to predict the cell fate. One of the ``forward``, ``backward`` or ``both`` string.
        interpolation_num:  The number of uniformly interpolated time points.
        t_end: The length of the time period from which to predict cell state forward or backward over time.
        average: The method to calculate the average cell state at each time step, can be one of ``origin`` or
                 ``trajectory``. If ``origin`` used, the average expression state from the init_cells will be calculated and
                 the fate prediction is based on this state. If ``trajectory`` used, the average expression states of all
                 cells predicted from the vector field function at each time point will be used. If ``average`` is
                 ``False``, no averaging will be applied.
        cores: Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
               multiprocessing will be used to parallel the fate prediction.
        inplace: Whether to copy adata or modify it inplace.
        **kwargs: Additional parameters that will be passed into the ``fate`` function.

    Returns:

        An ``AnnData`` object is updated/copied with the ``key_added`` dictionary in the ``.uns`` attribute.

        The  ``key_added``  dictionary which contains:

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
        t_end=t_end,
        direction=direction,
        average=average,
        cores=cores,
        **kwargs,
    )
    adata.uns[key_added] = adata.uns[f"fate_{key_added}"].copy()
    del adata.uns[f"fate_{key_added}"]

    cells_states = adata.uns[key_added]["prediction"]
    cells_times = adata.uns[key_added]["t"]
    adata.uns[key_added]["prediction"] = {i: cell_states.T for i, cell_states in enumerate(cells_states)}
    adata.uns[key_added]["t"] = {i: cell_times for i, cell_times in enumerate(cells_times)}

    return None if inplace else adata
