"""
Todo:
    * @Xiaojieqiu: update with Google style documentation, function typings, tests
"""
from typing import List, Optional, Tuple, Union

from ..logging import logger_manager as lm
from .deep_interpolation import *
from .utils import in_hull, polyhull


def KernelInterpolation(
    adata: AnnData,
    genes: Tuple[None, List] = None,
    grid_num: List = [50, 50, 50],
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    **kwargs,
) -> AnnData:
    """Learn a continuous mapping from space to gene expression pattern with the Kernel method (sparseVFC).

    Args:
        adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        genes: Gene list that needs to interpolate.
        grid_num: Number of grid to generate. Default is 50 for each dimension. Must be non-negative.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more weights
            on regularization.
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.
        **kwargs: Additional parameters that will be passed to SparseVFC function.

    Returns:
        interp_adata: an anndata object that has interpolated expression. The row of the adata object is a grid point
        within the convex hull formed by the input data points while each column corresponds a gene whose expression
        values are interpolated.
    """

    from dynamo.vectorfield.scVectorField import SparseVFC

    lm.info("Learn a continuous mapping from space to gene expression pattern")
    lm.log_time()

    X, V = adata.obsm["spatial"], adata[:, genes].X

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
    hull = polyhull(X[:, 0], X[:, 1], X[:, 2])

    lm.main_info("Identify grid points within the Convex Hull...")
    grid_in_hull = in_hull(Grid, hull.points[hull.vertices, :])

    res = SparseVFC(X, V, Grid, lambda_=lambda_, lstsq_method=lstsq_method, **kwargs)

    lm.main_info("Creating an adata object with the interpolated expression...")
    interp_adata = AnnData(
        X=res["grid_V"][grid_in_hull],
        obsm={"spatial": res["grid"][grid_in_hull]},
        var=adata[:, genes].var,
    )

    lm.finish_progress(progress_name="KernelInterpolation")

    return interp_adata
