from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from pyvista import MultiBlock

from ..models import add_model_labels, collect_model
from .interpolations import get_X_Y_grid


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

    from dynamo.vectorfield.scVectorField import SparseVFC

    _, _, Grid, grid_in_hull = get_X_Y_grid(X=stage1_X.copy(), Y=stage2_X.copy(), grid_num=grid_num)

    predict_X = Grid if NX is None else NX
    res = SparseVFC(stage1_X, stage2_X, predict_X, lambda_=lambda_, lstsq_method=lstsq_method, **kwargs)

    grid_dict = {
        "spatial": res["grid"][grid_in_hull] if NX is None else NX,
        "direction": res["grid_V"][grid_in_hull] if NX is None else res["grid_V"],
    }

    return grid_dict


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
