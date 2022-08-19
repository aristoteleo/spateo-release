from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from pyvista import MultiBlock, PolyData

from ....logging import logger_manager as lm
from ..utilities import add_model_labels, collect_model, merge_models
from .arrow_model import construct_arrows
from .line_model import construct_lines


def construct_cells_trajectory(
    stages_X: List[np.ndarray],
    n_spacing: int = 1,
    key_added: str = "develop",
    label: str = "cells_develop",
    color: str = "skyblue",
    alpha: float = 1.0,
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
            alphamap=alpha,
            inplace=True,
        )
        cells_models.append(model)

    return collect_model(models=cells_models)


def construct_trajectory(
    adata: AnnData,
    fate_key: str = "fate_develop",
    n_sampling: Optional[int] = None,
    sampling_method: str = "trn",
    key_added: str = "trajectory",
    label: str = "cell_trajectory",
    tip_factor: Union[int, float] = 10,
    tip_radius: float = 0.2,
    trajectory_color: str = "gainsboro",
    tip_color: str = "orangered",
    alpha: float = 1.0,
) -> PolyData:
    """
    Reconstruction of cell development trajectory model based on cell fate prediction.

    Args:
        adata: AnnData object that contains the fate prediction in the `.uns` attribute.
        fate_key: The key under which are the active fate information.
        n_sampling: n_sampleing is the number of coordinates to keep after sampling. If there are too many coordinates
                    in start_points, the generated arrows model will be too complex   and unsightly, so sampling is
                    used to reduce the number of coordinates.
        sampling_method: The method to sample data points, can be one of ["trn", "kmeans", "random"].
        key_added: The key under which to add the labels.
        label: The label of trajectory model.
        tip_factor: Scale factor applied to scaling the tips.
        tip_radius: Radius of the tips.
        trajectory_color: Color to use for plotting trajectories.
        tip_color: Color to use for plotting tips.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        trajectory_model: 3D cell development trajectory model.
    """

    from dynamo.tools.sampling import sample

    if fate_key not in adata.uns_keys():
        raise Exception(
            f"You need to first perform develop_trajectory prediction before animate the prediction, please run"
            f"st.tdr.develop_trajectory(adata, key_added='{fate_key[5:]}' before running this function"
        )
    init_states = adata.uns[fate_key]["init_states"]
    n_cells = init_states.shape[0]
    index_arr = np.arange(0, n_cells)

    if not (n_sampling is None):
        index_arr = sample(
            arr=index_arr,
            n=n_sampling,
            method=sampling_method,
            X=init_states,
        )
    else:
        if n_cells > 200:
            lm.main_warning(
                f"The number of cells with fate prediction is more than 200. You may want to "
                f"lower the max number of cell trajectories to draw."
            )

    cells_states = adata.uns[fate_key]["prediction"]
    se_ind = 0
    tips_points, tips_vectors = [], []
    trajectories_points, trajectories_edges = [], []
    for ind in index_arr:
        trajectory_points = np.concatenate([init_states[[ind]], cells_states[ind].T], axis=0)
        n_states = len(trajectory_points)
        trajectory_edges = np.concatenate(
            [
                np.arange(se_ind, se_ind + n_states - 1).reshape(-1, 1),
                np.arange(se_ind + 1, se_ind + n_states).reshape(-1, 1),
            ],
            axis=1,
        )
        se_ind += n_states

        trajectories_points.append(trajectory_points)
        trajectories_edges.append(trajectory_edges)
        tips_points.append(trajectory_points[-1])
        tips_vectors.append(trajectory_points[-1] - trajectory_points[-2])

    streamlines = construct_lines(
        points=np.concatenate(trajectories_points, axis=0),
        edges=np.concatenate(trajectories_edges, axis=0),
        key_added=key_added,
        label=label,
        color=trajectory_color,
        alpha=alpha,
    )

    arrows = construct_arrows(
        start_points=np.asarray(tips_points),
        direction=np.asarray(tips_vectors),
        arrows_scale=np.ones(shape=(len(tips_points), 1)),
        factor=tip_factor,
        tip_length=1,
        tip_radius=tip_radius,
        key_added=key_added,
        label=label,
        color=tip_color,
        alpha=alpha,
    )

    trajectory_model = merge_models([streamlines, arrows])
    return trajectory_model
