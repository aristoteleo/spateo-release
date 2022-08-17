from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from pyvista import MultiBlock

from ..utilities import add_model_labels, collect_model


def construct_cells_trajectory(
    stages_X: List[np.ndarray],
    n_spacing: int = 1,
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
