from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from pyvista import MultiBlock, PolyData

from ....logging import logger_manager as lm
from ..utilities import add_model_labels, collect_models, merge_models
from .arrow_model import construct_arrows
from .line_model import construct_lines


def construct_genesis_X(
    stages_X: List[np.ndarray],
    n_spacing: Optional[int] = None,
    key_added: str = "genesis",
    label: Optional[Union[str, list, np.ndarray]] = None,
    color: Union[str, list, dict] = "skyblue",
    alpha: Union[float, list, dict] = 1.0,
) -> MultiBlock:
    """
    Reconstruction of cell-level cell developmental change model based on the cell fate prediction results. Here we only
    need to enter the three-dimensional coordinates of the cells at different developmental stages.

    Args:
        stages_X: The three-dimensional coordinates of the cells at different developmental stages.
        n_spacing: Subdivided into ``n_spacing`` time points between two periods.
        key_added: The key under which to add the labels.
        label: The label of cell developmental change model. If ``label == None``, the label will be automatically generated.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        A MultiBlock contains cell models for all stages.
    """

    if n_spacing is None:
        cells_points = stages_X
    else:
        cells_points = []
        for i in range(len(stages_X) - 1):
            stage1_X = stages_X[i].copy()
            stage2_X = stages_X[i + 1].copy()
            spacing = (stage2_X - stage1_X) / n_spacing
            cells_points.extend([stage1_X.copy() + j * spacing for j in range(n_spacing)])
        cells_points.append(stages_X[-1])

    # Set label
    if isinstance(label, (str, int, float)) or label is None:
        label = "cell_state" if label is None else label
        labels = [np.asarray([f"{label}_{i}"] * pts.shape[0]) for i, pts in enumerate(cells_points)]
    elif isinstance(label, (list, np.ndarray)) and len(label) == len(cells_points):
        labels = (
            label
            if isinstance(label[0], (list, np.ndarray))
            else [np.asarray([la] * pts.shape[0]) for la, pts in zip(label, cells_points)]
        )
    else:
        raise ValueError("`label` value is wrong.")

    # Set color
    colors = color if isinstance(color, list) else [color] * len(cells_points)

    # Generate point cloud models
    cells_models = []
    for pts, la, co in zip(cells_points, labels, colors):
        model = pv.PolyData(pts)
        add_model_labels(
            model=model,
            key_added=key_added,
            labels=la,
            where="point_data",
            colormap=co,
            alphamap=alpha,
            inplace=True,
        )
        cells_models.append(model)

    return collect_models(models=cells_models)


def construct_genesis(
    adata: AnnData,
    fate_key: str = "fate_morpho",
    n_steps: int = 100,
    logspace: bool = False,
    t_end: Optional[Union[int, float]] = None,
    key_added: str = "genesis",
    label: Optional[Union[str, list, np.ndarray]] = None,
    color: Union[str, list, dict] = "skyblue",
    alpha: Union[float, list, dict] = 1.0,
) -> MultiBlock:
    """
    Reconstruction of cell-level cell developmental change model based on the cell fate prediction results. Here we only
    need to enter the three-dimensional coordinates of the cells at different developmental stages.

    Args:
        adata: AnnData object that contains the fate prediction in the ``.uns`` attribute.
        fate_key: The key under which are the active fate information.
        n_steps: The number of times steps fate prediction will take.
        logspace: Whether or to sample time points linearly on log space. If not, the sorted unique set of all times
                  points from all cell states' fate prediction will be used and then evenly sampled up to `n_steps`
                  time points.
        t_end: The length of the time period from which to predict cell state forward or backward over time.
        key_added: The key under which to add the labels.
        label: The label of cell developmental change model. If ``label == None``, the label will be automatically generated.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        A MultiBlock contains cell models for all stages.
    """

    from dynamo.vectorfield import SvcVectorField
    from scipy.integrate import odeint

    if fate_key not in adata.uns_keys():
        raise Exception(
            f"You need to first perform develop_trajectory prediction before animate the prediction, please run"
            f"st.tdr.develop_trajectory(adata, key_added='{fate_key}' before running this function"
        )

    t_ind = np.asarray(list(adata.uns[fate_key]["t"].keys()), dtype=int)
    t_sort_ind = np.argsort(t_ind)
    t = np.asarray(list(adata.uns["fate_morpho"]["t"].values()))[t_sort_ind]
    flats = np.unique([int(item) for sublist in t for item in sublist])
    flats = np.hstack((0, flats))
    flats = np.sort(flats) if t_end is None else np.sort(flats[flats <= t_end])
    time_vec = (
        np.logspace(0, np.log10(max(flats) + 1), n_steps) - 1
        if logspace
        else flats[(np.linspace(0, len(flats) - 1, n_steps)).astype(int)]
    )

    vf = SvcVectorField()
    vf.from_adata(adata, basis=fate_key[5:])
    f = lambda x, _: vf.func(x)
    displace = lambda x, dt: odeint(f, x, [0, dt])

    init_states = adata.uns[fate_key]["init_states"]
    pts = [i.tolist() for i in init_states]
    stages_X = []
    for i in range(n_steps):
        pts = [displace(cur_pts, time_vec[i])[1].tolist() for cur_pts in pts]
        stages_X.append(np.asarray(pts))

    cells_developmental_model = construct_genesis_X(
        stages_X=stages_X, n_spacing=None, key_added=key_added, label=label, color=color, alpha=alpha
    )

    return cells_developmental_model


def construct_trajectory_X(
    cells_states: Union[np.ndarray, List[np.ndarray]],
    init_states: Optional[np.ndarray] = None,
    n_sampling: Optional[int] = None,
    sampling_method: str = "trn",
    key_added: str = "trajectory",
    label: Optional[Union[str, list, np.ndarray]] = None,
    tip_factor: Union[int, float] = 5,
    tip_radius: float = 0.2,
    trajectory_color: Union[str, list, dict] = "gainsboro",
    tip_color: Union[str, list, dict] = "orangered",
    alpha: Union[float, list, dict] = 1.0,
) -> PolyData:
    """
    Reconstruction of cell developmental trajectory model.

    Args:
        cells_states: Three-dimensional coordinates of all cells at all times points.
        init_states: Three-dimensional coordinates of all cells at the starting time point.
        n_sampling: n_sampling is the number of coordinates to keep after sampling. If there are too many coordinates
                    in start_points, the generated arrows model will be too complex and unsightly, so sampling is
                    used to reduce the number of coordinates.
        sampling_method: The method to sample data points, can be one of ``['trn', 'kmeans', 'random']``.
        key_added: The key under which to add the labels.
        label: The label of trajectory model.
        tip_factor: Scale factor applied to scaling the tips.
        tip_radius: Radius of the tips.
        trajectory_color: Color to use for plotting trajectory model.
        tip_color: Color to use for plotting tips.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        trajectory_model: 3D cell developmental trajectory model.
    """

    from dynamo.tools.sampling import sample

    init_states = np.asarray([cell_states[0] for cell_states in cells_states]) if init_states is None else init_states
    index_arr = np.arange(0, len(init_states))

    if isinstance(label, (str, int, float)) or label is None:
        labels = ["trajectory"] * len(index_arr) if label is None else [label] * len(index_arr)
    elif isinstance(label, (list, np.ndarray)) and len(label) == len(index_arr):
        labels = label
    else:
        raise ValueError("`label` value is wrong.")

    if not (n_sampling is None):
        index_arr = sample(
            arr=index_arr,
            n=n_sampling,
            method=sampling_method,
            X=init_states,
        )
    else:
        if index_arr.shape[0] > 200:
            lm.main_warning(
                f"The number of cells is more than 200. You may want to "
                f"lower the max number of cell trajectories to draw."
            )

    se_ind = 0
    tips_points, tips_vectors, tips_labels = [], [], []
    trajectories_points, trajectories_edges, trajectories_labels = [], [], []
    for ind in index_arr:
        trajectory_points = (
            cells_states[ind]
            if init_states is None
            else np.concatenate([init_states[[ind]], cells_states[ind]], axis=0)
        )
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
        trajectories_labels.extend([labels[ind]] * n_states)

        tips_points.append(trajectory_points[-1])
        tips_vectors.append(trajectory_points[-1] - trajectory_points[-2])
        tips_labels.append(labels[ind])

    streamlines = construct_lines(
        points=np.concatenate(trajectories_points, axis=0),
        edges=np.concatenate(trajectories_edges, axis=0),
        key_added=key_added,
        label=np.asarray(trajectories_labels),
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
        label=np.asarray(tips_labels),
        color=tip_color,
        alpha=alpha,
    )

    trajectory_model = merge_models([streamlines, arrows])
    return trajectory_model


def construct_trajectory(
    adata: AnnData,
    fate_key: str = "fate_develop",
    n_sampling: Optional[int] = None,
    sampling_method: str = "trn",
    key_added: str = "trajectory",
    label: Optional[Union[str, list, np.ndarray]] = None,
    tip_factor: Union[int, float] = 5,
    tip_radius: float = 0.2,
    trajectory_color: Union[str, list, dict] = "gainsboro",
    tip_color: Union[str, list, dict] = "orangered",
    alpha: float = 1.0,
) -> PolyData:
    """
    Reconstruction of cell developmental trajectory model based on cell fate prediction.

    Args:
        adata: AnnData object that contains the fate prediction in the ``.uns`` attribute.
        fate_key: The key under which are the active fate information.
        n_sampling: n_sampling is the number of coordinates to keep after sampling. If there are too many coordinates
                    in start_points, the generated arrows model will be too complex and unsightly, so sampling is
                    used to reduce the number of coordinates.
        sampling_method: The method to sample data points, can be one of ``['trn', 'kmeans', 'random']``.
        key_added: The key under which to add the labels.
        label: The label of trajectory model.
        tip_factor: Scale factor applied to scaling the tips.
        tip_radius: Radius of the tips.
        trajectory_color: Color to use for plotting trajectory model.
        tip_color: Color to use for plotting tips.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        trajectory_model: 3D cell developmental trajectory model.
    """

    if fate_key not in adata.uns_keys():
        raise Exception(
            f"You need to first perform develop_trajectory prediction before animate the prediction, please run"
            f"st.tdr.morphopath(adata, key_added='{fate_key}' before running this function"
        )

    init_states = np.asarray(adata.uns[fate_key]["init_states"])

    cell_states_ind = np.asarray(list(adata.uns[fate_key]["prediction"].keys()), dtype=int)
    cell_states_sort_ind = np.argsort(cell_states_ind)
    cells_states = np.asarray(list(adata.uns[fate_key]["prediction"].values()))[cell_states_sort_ind]

    trajectory_model = construct_trajectory_X(
        cells_states=cells_states,
        init_states=init_states,
        n_sampling=n_sampling,
        sampling_method=sampling_method,
        key_added=key_added,
        label=label,
        tip_factor=tip_factor,
        tip_radius=tip_radius,
        trajectory_color=trajectory_color,
        tip_color=tip_color,
        alpha=alpha,
    )

    return trajectory_model
