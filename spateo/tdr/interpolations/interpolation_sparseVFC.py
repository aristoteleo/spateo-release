from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from dynamo.vectorfield.scVectorField import SparseVFC
from numpy import ndarray
from scipy.sparse import issparse

from ...logging import logger_manager as lm


def kernel_interpolation(
    source_adata: AnnData,
    target_points: Optional[ndarray] = None,
    keys: Union[str, list] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    lambda_: float = 0.02,
    lstsq_method: str = "scipy",
    **kwargs,
) -> AnnData:
    """
    Learn a continuous mapping from space to gene expression pattern with Kernel method (sparseVFC).

    Args:
        source_adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        target_points: The spatial coordinates of new data point. If target_coords is None, generate new points based on grid_num.
        keys: Gene list or info list in the `obs` attribute whose interpolate expression across space needs to learned.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.
        **kwargs: Additional parameters that will be passed to SparseVFC function.

    Returns:
        interp_adata: an anndata object that has interpolated expression.
    """

    # Inference
    source_adata = source_adata.copy()
    source_adata.X = source_adata.X if layer == "X" else source_adata.layers[layer]

    source_spatial_data = source_adata.obsm[spatial_key]

    info_data = np.ones(shape=(source_spatial_data.shape[0], 1))
    assert keys != None, "`keys` cannot be None."
    keys = [keys] if isinstance(keys, str) else keys
    obs_keys = [key for key in keys if key in source_adata.obs.keys()]
    if len(obs_keys) != 0:
        obs_data = np.asarray(source_adata.obs[obs_keys].values)
        info_data = np.c_[info_data, obs_data]
    var_keys = [key for key in keys if key in source_adata.var_names.tolist()]
    if len(var_keys) != 0:
        var_data = source_adata[:, var_keys].X
        if issparse(var_data):
            var_data = var_data.toarray()
        info_data = np.c_[info_data, var_data]
    info_data = info_data[:, 1:]

    # Interpolation
    res = SparseVFC(source_spatial_data, info_data, target_points, lambda_=lambda_, lstsq_method=lstsq_method, **kwargs)
    target_info_data = res["grid_V"]

    lm.main_info("Creating an adata object with the interpolated expression...")

    if len(obs_keys) != 0:
        obs_data = target_info_data[:, : len(obs_keys)]
        obs_data = pd.DataFrame(obs_data, columns=obs_keys)

    if len(var_keys) != 0:
        X = target_info_data[:, len(obs_keys) :]
        var_data = pd.DataFrame(index=var_keys)

    interp_adata = AnnData(
        X=X if len(var_keys) != 0 else None,
        obs=obs_data if len(obs_keys) != 0 else None,
        obsm={spatial_key: np.asarray(target_points)},
        var=var_data if len(var_keys) != 0 else None,
    )

    lm.main_finish_progress(progress_name="KernelInterpolation")

    return interp_adata
