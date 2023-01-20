from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy import ndarray

from spateo.configuration import SKM
from spateo.logging import logger_manager as lm

from .methods import generalized_procrustes_analysis, pairwise_align
from .transform import paste_transform


def _iteration(n: int, progress_name: str, verbose: bool = True):
    iteration = lm.progress_logger(range(n), progress_name=progress_name) if verbose else range(n)
    return iteration


def _downsampling(
    models: Union[List[AnnData], AnnData],
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    spatial_key: str = "spatial",
) -> Union[List[AnnData], AnnData]:
    from dynamo.tools.sampling import sample

    models = models if isinstance(models, list) else [models]
    sampling_models = []
    for m in models:
        sampling_model = m.copy()
        sampling = sample(
            arr=np.asarray(sampling_model.obs_names),
            n=n_sampling,
            method=sampling_method,
            X=sampling_model.obsm[spatial_key],
        )
        sampling_model = sampling_model[sampling, :]
        sampling_models.append(sampling_model)
    return sampling_models


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models")
def models_align(
    models: List[AnnData],
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key_added: str = "models_align",
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[AnnData], List[Union[ndarray, ndarray]]]:
    """
    Align spatial coordinates of models.

    Args:
        models: List of models (AnnData Object).
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinates.
        mapping_key_added: `.uns` key under which to add the alignment info.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``pairwise_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        pis: List of pi matrices.
    """
    for m in models:
        m.obsm[key_added] = m.obsm[spatial_key]

    pis = []
    align_models = [model.copy() for model in models]
    for i in _iteration(n=len(align_models) - 1, progress_name="Models alignment", verbose=verbose):
        modelA = align_models[i]
        modelB = align_models[i + 1]

        # Calculate and returns optimal alignment of two models.
        pi, _ = pairwise_align(
            sampleA=modelA.copy(),
            sampleB=modelB.copy(),
            layer=layer,
            genes=genes,
            spatial_key=key_added,
            alpha=alpha,
            numItermax=numItermax,
            numItermaxEmd=numItermaxEmd,
            dtype=dtype,
            device=device,
            verbose=verbose,
            **kwargs,
        )
        pis.append(pi)

        # Calculate new coordinates of two models
        modelA_coords, modelB_coords, mapping_dict = generalized_procrustes_analysis(
            X=modelA.obsm[key_added], Y=modelB.obsm[key_added], pi=pi
        )

        modelA.obsm[key_added] = modelA_coords
        modelB.obsm[key_added] = modelB_coords
        modelB.uns[mapping_key_added] = mapping_dict

    return align_models, pis


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models_ref", optional=True)
def models_align_ref(
    models: List[AnnData],
    models_ref: Optional[List[AnnData]] = None,
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key_added: str = "models_align",
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData], List[Union[ndarray, ndarray]]]:
    """
    Align the spatial coordinates of one model list through the affine transformation matrix obtained from another model list.

    Args:
        models: List of models (AnnData Object).
        models_ref: Another list of models (AnnData Object).
        n_sampling: When ``models_ref`` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ``["trn", "kmeans", "random"]``.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinates.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinates.
        mapping_key_added: `.uns` key under which to add the alignment info.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``models_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        align_models_ref: List of models_ref (AnnData Object) after alignment.
        pis: The list of pi matrices from align_models_ref.
    """

    # Downsampling
    if models_ref is None:
        models_sampling = [model.copy() for model in models]
        models_ref = _downsampling(
            models=models_sampling, n_sampling=n_sampling, sampling_method=sampling_method, spatial_key=spatial_key
        )

    # Align spatial coordinates of slices with a small number of coordinates.
    align_models_ref, pis = models_align(
        models=models_ref,
        layer=layer,
        genes=genes,
        spatial_key=spatial_key,
        key_added=key_added,
        mapping_key_added=mapping_key_added,
        alpha=alpha,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        dtype=dtype,
        device=device,
        verbose=verbose,
        **kwargs,
    )

    align_models = []
    for i, (align_model_ref, model) in enumerate(zip(align_models_ref, models)):
        align_model = model.copy()
        if i != 0:
            align_model = paste_transform(
                adata=align_model,
                adata_ref=align_model_ref,
                spatial_key=spatial_key,
                key_added=key_added,
                mapping_key=mapping_key_added,
            )
            align_model.uns[mapping_key_added] = align_model_ref.uns[mapping_key_added]
        align_models.append(align_model)

    return align_models, align_models_ref, pis


def get_align_labels(
    model: AnnData,
    align_X: np.ndarray,
    key: Union[str, List[str]],
    spatial_key: str = "align_spatial",
) -> pd.DataFrame:
    """Obtain the label information in anndata.obs[key] corresponding to the align_X coordinate."""

    key = [key] if isinstance(key, str) else key

    cols = ["x", "y", "z"] if align_X.shape[1] == 3 else ["x", "y"]
    X_data = pd.DataFrame(model.obsm[spatial_key], columns=cols)
    X_data[key] = model.obs[key].values
    X_data.drop_duplicates(inplace=True, keep="first")

    Y_data = pd.DataFrame(align_X.copy(), columns=cols)
    Y_data["map_index"] = Y_data.index
    merge_data = pd.merge(Y_data, X_data, on=cols, how="inner")

    return merge_data
