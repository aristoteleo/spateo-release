from typing import List, Optional, Tuple

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from ...configuration import SKM
from ...logging import logger_manager as lm
from ...tools.utils import in_hull, polyhull
from . import interpolation_nn
from .deep_interpolation import DataSampler, DeepInterpolation




@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def deep_intepretation_2d(
    adata: Optional[AnnData] = None,
    genes: Optional[List] = None,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    NX: Optional[np.ndarray] = None,
    grid_num: List = [50, 50, 50],
    train_on_pos: bool = False,
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
    #X, Y, Grid, grid_in_hull = get_X_Y_grid(adata=adata, X=X, Y=Y, grid_num=grid_num)
    #X, Y, Grid, grid_in_hull = get_X_Y_grid(adata=adata, genes=genes, X=X, Y=Y, grid_num=grid_num)
    #Y = adata[:,genes].X.A if issparse(adata[:,genes].X) else adata[:,genes].X if genes else Y
    Y = Y if not isinstance(Y, type(None)) else adata[:,genes].X.A if issparse(adata[:,genes].X) else adata[:,genes].X
    if train_on_pos:
        Y = Y.flatten()
        X = X[Y>0]
        Y = Y[Y>0]
        Y = Y[:,None]

     

    data_dict = {"X": X, "Y": Y}

    velocity_data_sampler = DataSampler(data=data_dict, normalize_data=False)

    NN_model = DeepInterpolation(
        model=interpolation_nn,
        data_sampler=velocity_data_sampler,
        enforce_positivity=False,
    )

    NN_model.train(
        max_iter=1000, data_batch_size=100, autoencoder_batch_size=50, data_lr=1e-4, autoencoder_lr=1e-4, **kwargs
    )

    predict_X = X if NX is None else NX
    predict_Y = NN_model.predict(input_x=predict_X)

    interp_adata = AnnData(
        X=predict_Y,
        obsm={"spatial": predict_X},
        var=adata[:, genes].var if genes is not None and Y.shape[1] == len(genes) else None,
    )

    return interp_adata


