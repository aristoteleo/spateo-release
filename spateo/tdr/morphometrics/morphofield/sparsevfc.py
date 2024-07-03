from typing import List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from spateo.alignment import get_optimal_mapping_relationship
from spateo.alignment.methods import paste_pairwise_align
from spateo.logging import logger_manager as lm
from spateo.tdr.interpolations import get_X_Y_grid


def cell_directions(
    adataA: AnnData,
    adataB: AnnData,
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "align_spatial",
    key_added: str = "mapping",
    alpha: float = 0.001,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float32",
    device: str = "cpu",
    keep_all: bool = False,
    inplace: bool = True,
    **kwargs,
) -> Tuple[Optional[AnnData], np.ndarray]:
    """
    Obtain the optimal mapping relationship and developmental direction between cells for samples between continuous developmental stages.

    Args:
        adataA: AnnData object of sample A from continuous developmental stages.
        adataB: AnnData object of sample B from continuous developmental stages.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each cell.
        The key that will be used for the vector field key in ``.uns``.
        key_added: The key that will be used in ``.obsm``.

                  * ``X_{key_added}``-The ``X_{key_added}`` that will be used for the coordinates of the cell that maps optimally in the next stage.
                  * ``V_{key_added}``-The ``V_{key_added}`` that will be used for the cell developmental directions.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.
        inplace: Whether to copy adata or modify it inplace.
        **kwargs: Additional parameters that will be passed to ``pairwise_align`` function.

    Returns:
        An ``AnnData`` object of sample A is updated/copied with the ``X_{key_added}`` and ``V_{key_added}`` in the ``.obsm`` attribute.
        A pi metrix.
    """
    # Calculate and returns optimal alignment of two models.
    pi, _ = paste_pairwise_align(
        sampleA=adataA.copy(),
        sampleB=adataB.copy(),
        spatial_key=spatial_key,
        layer=layer,
        genes=genes,
        alpha=alpha,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        dtype=dtype,
        device=device,
        **kwargs,
    )

    max_index, pi_value, _, _ = get_optimal_mapping_relationship(
        X=adataA.obsm[spatial_key].copy(), Y=adataB.obsm[spatial_key].copy(), pi=pi, keep_all=keep_all
    )

    mapping_data = pd.DataFrame(
        np.concatenate([max_index, pi_value], axis=1),
        columns=["index_x", "index_y", "pi_value"],
    ).astype(
        dtype={
            "index_x": np.int32,
            "index_y": np.int32,
            "pi_value": np.float64,
        }
    )
    mapping_data.sort_values(by=["index_x", "pi_value"], ascending=[True, False], inplace=True)
    mapping_data.drop_duplicates(subset=["index_x"], keep="first", inplace=True)

    adataA.obsm[f"X_{key_added}"] = adataB.obsm[spatial_key][mapping_data["index_y"].values]
    adataA.obsm[f"V_{key_added}"] = adataA.obsm[f"X_{key_added}"] - adataA.obsm[spatial_key]

    return None if inplace else adataA, pi


def _morphofield_sparsevfc(
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
        X: The spatial coordinates of each cell.
        V: The developmental direction of each cell.
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


def morphofield_sparsevfc(
    adata: AnnData,
    spatial_key: str = "align_spatial",
    V_key: str = "V_mapping",
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
        spatial_key: The key from the ``.obsm`` that corresponds to the spatial coordinates of each cell.
        V_key: The key from the ``.obsm`` that corresponds to the developmental direction of each cell.
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
        inplace: Whether to copy adata or modify it inplace.
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
    adata.uns[key_added] = _morphofield_sparsevfc(
        X=np.asarray(adata.obsm[spatial_key], dtype=float),
        V=np.asarray(adata.obsm[V_key], dtype=float),
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
