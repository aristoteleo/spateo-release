from typing import List, Optional, Tuple

import numpy as np
from anndata import AnnData

from ...configuration import SKM
from ...logging import logger_manager as lm
from ...tools.utils import in_hull, polyhull
from . import interpolation_nn
from .deep_interpolation import DataSampler, DeepInterpolation


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def kernel_interpolation(
    adata: Optional[AnnData] = None,
    genes: Optional[List] = None,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    NX: Optional[np.ndarray] = None,
    grid_num: List = [50, 50, 50],
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    **kwargs,
) -> AnnData:
    """Learn a continuous mapping from space to gene expression pattern with the Kernel method (sparseVFC).

    Args:
        adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        genes: Gene list whose interpolate expression across space needs to learned. If Y is provided, genes will only
            be used to retrive the gene annotation info.
        X: The spatial coordinates of each data point.
        Y: The gene expression of the corresponding data point.
        NX: The spatial coordinates of new data point. If NX is None, generate new points based on grid_num.
        grid_num: Number of grid to generate. Default is 50 for each dimension. Must be non-negative.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.
        **kwargs: Additional parameters that will be passed to SparseVFC function.

    Returns:
        interp_adata: an anndata object that has interpolated expression. The row of the adata object is a grid point
        within the convex hull formed by the input data points while each column corresponds a gene whose expression
        values are interpolated.
    """

    from dynamo.vectorfield.scVectorField import SparseVFC

    X, Y, Grid, grid_in_hull = get_X_Y_grid(adata=adata, X=X, Y=Y, genes=genes, grid_num=grid_num)

    predict_X = Grid if NX is None else NX
    res = SparseVFC(X, Y, predict_X, lambda_=lambda_, lstsq_method=lstsq_method, **kwargs)

    lm.main_info("Creating an adata object with the interpolated expression...")
    interp_adata = AnnData(
        X=res["grid_V"][grid_in_hull] if NX is None else res["grid_V"],
        obsm={"spatial": res["grid"][grid_in_hull] if NX is None else NX},
        var=adata[:, genes].var if genes is not None and Y.shape[1] == len(genes) else None,
    )

    lm.main_finish_progress(progress_name="KernelInterpolation")

    return interp_adata


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def deep_intepretation(
    adata: Optional[AnnData] = None,
    genes: Optional[List] = None,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    NX: Optional[np.ndarray] = None,
    grid_num: List = [50, 50, 50],
    **kwargs,
) -> AnnData:
    """Learn a continuous mapping from space to gene expression pattern with the deep neural net model.

    Args:
        adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        genes: Gene list whose interpolate expression across space needs to learned. If Y is provided, genes will only
            be used to retrive the gene annotation info.
        X: The spatial coordinates of each data point.
        Y: The gene expression of the corresponding data point.
        NX: The spatial coordinates of new data point. If NX is None, generate new points based on grid_num.
        grid_num: Number of grid to generate. Default is 50 for each dimension. Must be non-negative.
        **kwargs: Additional parameters that will be passed to the training step of the deep neural net.

    Returns:
        interp_adata: an anndata object that has interpolated expression. The row of the adata object is a grid point
        within the convex hull formed by the input data points while each column corresponds a gene whose expression
        values are interpolated.
    """
    X, Y, Grid, grid_in_hull = get_X_Y_grid(adata=adata, X=X, Y=Y, genes=genes, grid_num=grid_num)

    data_dict = {"X": X, "Y": Y}

    velocity_data_sampler = DataSampler(data=data_dict, normalize_data=False)

    NN_model = DeepInterpolation(
        model=interpolation_nn,
        data_sampler=velocity_data_sampler,
        enforce_positivity=False,
    )

    NN_model.train(
        max_iter=1000, data_batch_size=5000, autoencoder_batch_size=50, data_lr=1e-4, autoencoder_lr=1e-4, **kwargs
    )

    predict_X = Grid[grid_in_hull] if NX is None else NX
    predict_Y = NN_model.predict(input_x=predict_X)

    interp_adata = AnnData(
        X=predict_Y,
        obsm={"spatial": predict_X},
        var=adata[:, genes].var if genes is not None and Y.shape[1] == len(genes) else None,
    )

    return interp_adata


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
