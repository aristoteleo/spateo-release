from typing import List, Optional, Tuple

import numpy as np
from anndata import AnnData

from ...logging import logger_manager as lm
from ...tools.utils import in_hull, polyhull


def get_X_Y_grid(
    adata: Optional[AnnData] = None,
    genes: Optional[List] = None,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    grid_num: List = [50, 50, 50],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the X (spatial coordinates), Y (gene expression) and grid points for the kernel or deep model.

    Args:
       adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
       genes: Gene list whose interpolate expression across space needs to learned. If Y is provided, genes will only
            be used to retrive the gene annotation info.
       X: The spatial coordinates of each data point.
       Y: The gene expression of the corresponding data point.
       grid_num: Number of grid to generate. Default is 50 for each dimension. Must be non-negative.

    Returns:
        X: spatial coordinates.
        Y: gene expression of the associated spatial coordinates.
        Grid: grid points formed with the input spatial coordinates.
        grid_in_hull: A list of booleans indicates whether the current grid points is within the convex hull formed by
            the input data points.
    """
    lm.main_info("Learn a continuous mapping from space to gene expression pattern")

    X, Y = adata.obsm["spatial"] if X is None else X, adata[:, genes].X if Y is None else Y

    # Generate grid
    lm.main_info("Generate grid...")
    min_vec, max_vec = (
        X.min(0),
        X.max(0),
    )
    min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
    max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)
    Grid_list = np.meshgrid(*[np.linspace(i, j, k) for i, j, k in zip(min_vec, max_vec, grid_num)])
    Grid = np.array([i.flatten() for i in Grid_list]).T

    lm.main_info("Creating a Convex Hull...")
    hull, _ = polyhull(X[:, 0], X[:, 1], X[:, 2])

    lm.main_info("Identify grid points within the Convex Hull...")
    grid_in_hull = in_hull(Grid, hull.points[hull.vertices, :])

    return X, Y, Grid, grid_in_hull
