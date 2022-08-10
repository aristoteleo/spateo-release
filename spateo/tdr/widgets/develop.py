from typing import List, Optional

import numpy as np
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
    """Learn a continuous mapping from space to gene expression pattern with the Kernel method (sparseVFC).

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
        "direction": res["grid_V"][grid_in_hull] if NX is None else res["grid_V"]
    }

    return grid_dict
