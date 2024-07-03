from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy import ndarray
from scipy.sparse import issparse

from ...logging import logger_manager as lm
from .interpolation_deeplearn import (DataSampler, DeepInterpolation,
                                      interpolation_nn)


def deep_intepretation(
    source_adata: AnnData,
    target_points: Optional[ndarray] = None,
    keys: Union[str, list] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    max_iter: int = 1000,
    data_batch_size: int = 2000,
    autoencoder_batch_size: int = 50,
    data_lr: float = 1e-4,
    autoencoder_lr: float = 1e-4,
    **kwargs,
) -> AnnData:
    """Learn a continuous mapping from space to gene expression pattern with the deep neural net model.

    Args:
        source_adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        target_points: The spatial coordinates of new data point. If target_coords is None, generate new points based on grid_num.
        keys: Gene list or info list in the `obs` attribute whose interpolate expression across space needs to learned.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        max_iter: The maximum iteration the network will be trained.
        data_batch_size: The size of the data sample batches to be generated in each iteration.
        autoencoder_batch_size: The size of the auto-encoder training batches to be generated in each iteration.
                                Must be no greater than batch_size. .
        data_lr: The learning rate for network training.
        autoencoder_lr: The learning rate for network training the auto-encoder. Will have no effect if network_dim
                        equal data_dim.
        **kwargs: Additional parameters that will be passed to the training step of the deep neural net.

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
            var_data = var_data.A
        info_data = np.c_[info_data, var_data]
    info_data = info_data[:, 1:]

    data_dict = {"X": source_spatial_data, "Y": info_data}
    velocity_data_sampler = DataSampler(data=data_dict, normalize_data=False)

    NN_model = DeepInterpolation(
        model=interpolation_nn,
        data_sampler=velocity_data_sampler,
        enforce_positivity=False,
    )

    NN_model.train(
        max_iter=max_iter,
        data_batch_size=data_batch_size,
        autoencoder_batch_size=autoencoder_batch_size,
        data_lr=data_lr,
        autoencoder_lr=autoencoder_lr,
        **kwargs,
    )

    # Interpolation
    target_info_data = NN_model.predict(input_x=target_points)

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

    lm.main_finish_progress(progress_name="DeepLearnInterpolation")
    return interp_adata
