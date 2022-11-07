from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ..configuration import SKM
from ..logging import logger_manager as lm
from .paste import (
    center_align,
    generalized_procrustes_analysis,
    mapping_aligned_coords,
    mapping_center_coords,
    pairwise_align,
)


def rigid_transform_2D(
    coords: np.ndarray,
    coords_refA: np.ndarray,
    coords_refB: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal transformation based on the two sets of 2D points and apply the transformation to other points.

    Args:
        coords: 2D coordinate matrix needed to be transformed.
        coords_refA: Referential 2D coordinate matrix before transformation.
        coords_refB: Referential 2D coordinate matrix after transformation.

    Returns:
        The 2D coordinate matrix after transformation
    """

    coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
    coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

    coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]

    new_coords = rigid_transform_3D(coords=coords, coords_refA=coords_refA, coords_refB=coords_refB)

    return np.asarray(new_coords[:, :2])


def rigid_transform_3D(
    coords: np.ndarray,
    coords_refA: np.ndarray,
    coords_refB: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal transformation based on the two sets of 3D points and apply the transformation to other points.

    Args:
        coords: 3D coordinate matrix needed to be transformed.
        coords_refA: Referential 3D coordinate matrix before transformation.
        coords_refB: Referential 3D coordinate matrix after transformation.

    Returns:
        The 3D coordinate matrix after transformation
    """

    # Compute optimal transformation based on the two sets of 3D points.
    coords_refA = coords_refA.T
    coords_refB = coords_refB.T

    centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
    centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

    Am = coords_refA - centroid_A
    Bm = coords_refB - centroid_B
    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # Apply the transformation to other points
    new_coords = (R @ coords.T) + t
    return np.asarray(new_coords.T)


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
    dtype: str = "float32",
    device: str = "cpu",
    keep_all: bool = False,
    **kwargs,
) -> List[AnnData]:
    """
    Align spatial coordinates of models.

    Args:
        models: List of models (AnnData Object).
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        mapping_key_added: `.uns` key under which to add the alignment info.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.
        **kwargs: Additional parameters that will be passed to ``pairwise_align`` function.

    Returns:
        List of models (AnnData Object) after alignment.
    """
    for m in models:
        m.obsm[key_added] = m.obsm[spatial_key]

    align_models = [model.copy() for model in models]
    for i in lm.progress_logger(range(len(align_models) - 1), progress_name="Models alignment"):

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
            **kwargs,
        )

        # Calculate new coordinates of two models
        modelA_coords, modelB_coords, mapping_dict = generalized_procrustes_analysis(
            X=modelA.obsm[key_added], Y=modelB.obsm[key_added], pi=pi
        )

        modelA.obsm[key_added] = modelA_coords
        modelB.obsm[key_added] = modelB_coords

        mapping_A_dict, mapping_B_dict = mapping_aligned_coords(
            X=modelA_coords, Y=modelB_coords, pi=pi, keep_all=keep_all
        )

        modelA.uns[f"latter_{mapping_key_added}"] = mapping_A_dict.copy()
        modelA.uns[f"latter_{mapping_key_added}"]["raw_X"] = modelA.obsm[spatial_key][mapping_A_dict["pi_index"][:, 0]]
        modelA.uns[f"latter_{mapping_key_added}"]["raw_Y"] = modelB.obsm[spatial_key][mapping_A_dict["pi_index"][:, 1]]
        modelA.uns[f"latter_{mapping_key_added}"]["mapping_relations"] = {"t": mapping_dict["tX"], "R": None}
        modelA.uns[f"latter_{mapping_key_added}"]["mapping_relations"] = {"t": mapping_dict["tX"], "R": None}

        modelB.uns[f"former_{mapping_key_added}"] = mapping_B_dict.copy()
        modelB.uns[f"former_{mapping_key_added}"]["raw_X"] = modelA.obsm[spatial_key][
            mapping_B_dict["pi_index"][:, 0].flatten()
        ]
        modelB.uns[f"former_{mapping_key_added}"]["raw_Y"] = modelB.obsm[spatial_key][
            mapping_B_dict["pi_index"][:, 1].flatten()
        ]
        modelB.uns[f"former_{mapping_key_added}"]["mapping_relations"] = {
            "t": mapping_dict["tY"],
            "R": mapping_dict["R"],
        }

    return align_models


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
    dtype: str = "float32",
    device: str = "cpu",
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData]]:
    """
    Align the spatial coordinates of one model list through the affine transformation matrix obtained from another model list.

    Args:
        models: List of models (AnnData Object).
        models_ref: Another list of models (AnnData Object).
        n_sampling: When ``models_ref`` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ``["trn", "kmeans", "random"]``.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        mapping_key_added: `.uns` key under which to add the alignment info.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``
        **kwargs: Additional parameters that will be passed to ``pairwise_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        align_models_ref: List of models_ref (AnnData Object) after alignment.
    """
    from dynamo.tools.sampling import sample

    if models_ref is None:
        models_ref = []
        for m in models:
            model_ref = m.copy()
            sampling = sample(
                arr=np.asarray(model_ref.obs_names), n=n_sampling, method=sampling_method, X=model_ref.obsm[spatial_key]
            )
            slice_ref = model_ref[sampling, :]
            models_ref.append(slice_ref)

    # Align spatial coordinates of slices with a small number of coordinates.
    align_models_ref = models_align(
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
        **kwargs,
    )

    align_models = [model.copy() for model in models]
    for i, (align_model_ref, align_model) in enumerate(zip(align_models_ref, align_models)):
        _mapping_key = f"latter_{mapping_key_added}" if i == 0 else f"former_{mapping_key_added}"
        t = align_model_ref.uns[_mapping_key]["mapping_relations"]["t"]
        R = align_model_ref.uns[_mapping_key]["mapping_relations"]["R"]

        align_model_coords = align_model.obsm[spatial_key].copy() - t
        align_model.obsm[key_added] = align_model_coords if R is None else R.dot(align_model_coords.T).T

        for key in [f"former_{mapping_key_added}", f"latter_{mapping_key_added}"]:
            if key in align_model_ref.uns_keys():
                align_model.uns[key] = align_model_ref.uns[key]

    return align_models, align_models_ref


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "init_center_model")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models")
def models_center_align(
    init_center_model: AnnData,
    models: List[AnnData],
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key_added: str = "models_align",
    lmbda: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dissimilarity: str = "kl",
    norm: bool = False,
    random_seed: Optional[int] = None,
    pis_init: Optional[List[np.ndarray]] = None,
    distributions: Optional[List[np.ndarray]] = None,
    dtype: str = "float32",
    device: str = "cpu",
    keep_all: bool = False,
) -> Tuple[AnnData, List[AnnData]]:
    """
    Align spatial coordinates of a list of models to a center model.

    Args:
        init_center_model: AnnData object to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        models: List of AnnData objects to use in the center alignment.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        mapping_key_added: `.uns` key under which to add the alignment info.
        lmbda: List of probability weights assigned to each slice; If `None`, use uniform weights.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        n_components: Number of components in NMF decomposition.
        threshold: Threshold for convergence of W and H during NMF decomposition.
        max_iter: Maximum number of iterations for our center alignment algorithm.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        norm: If ``norm = True``, scales spatial distances such that neighboring spots are at distance 1.

              Otherwise, spatial distances remain unchanged.
        random_seed: Set random seed for reproducibility.
        pis_init: Initial list of mappings between ``A`` and ``models`` to solver.

                  Otherwise, default will automatically calculate mappings.
        distributions: Distributions of spots for each slice. Otherwise, default is uniform.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.

    Returns:
        new_center_model: The center model.
        align_models: List of models (AnnData Object) after alignment.
    """
    init_center_model.obsm[key_added] = init_center_model.obsm[spatial_key]
    for model in models:
        model.obsm[key_added] = model.obsm[spatial_key]

    center_model, pis = center_align(
        init_center_sample=init_center_model,
        samples=models,
        layer=layer,
        genes=genes,
        spatial_key=spatial_key,
        lmbda=lmbda,
        alpha=alpha,
        n_components=n_components,
        threshold=threshold,
        max_iter=max_iter,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        dissimilarity=dissimilarity,
        norm=norm,
        random_seed=random_seed,
        pis_init=pis_init,
        distributions=distributions,
        dtype=dtype,
        device=device,
    )

    align_models = [model.copy() for model in models]
    for model, pi in zip(align_models, pis):
        center_coords, model_coords, mapping_dict = generalized_procrustes_analysis(
            center_model.obsm[key_added].copy(), model.obsm[key_added].copy(), pi.copy()
        )
        center_model.obsm[key_added] = center_coords
        model.obsm[key_added] = model_coords

        center_model.uns[mapping_key_added] = {"mapping_relations": {"t": mapping_dict["tX"], "R": None}}
        model.uns[f"center_{mapping_key_added}"] = mapping_aligned_coords(
            X=center_coords, Y=model_coords, pi=pi, keep_all=keep_all
        )[1]
        model.uns[f"center_{mapping_key_added}"]["raw_X"] = center_model.obsm[spatial_key][
            model.uns[f"center_{mapping_key_added}"]["pi_index"][:, 0]
        ]
        model.uns[f"center_{mapping_key_added}"]["raw_Y"] = model.obsm[spatial_key][
            model.uns[f"center_{mapping_key_added}"]["pi_index"][:, 1]
        ]
        model.uns[f"center_{mapping_key_added}"]["mapping_relations"] = {
            "t": mapping_dict["tY"],
            "R": mapping_dict["R"],
        }

    for i in range(len(align_models) - 1):
        modelA = align_models[i]
        modelB = align_models[i + 1]

        mapping_coords_dict = mapping_center_coords(modelA, modelB, center_key=f"center_{mapping_key_added}")
        modelA.uns[f"latter_{mapping_key_added}"] = mapping_coords_dict.copy()
        modelB.uns[f"former_{mapping_key_added}"] = mapping_coords_dict.copy()

    new_center_model = init_center_model.copy()
    new_center_model.obsm[key_added] = center_model.obsm[key_added]
    new_center_model.uns[key_added] = center_model.uns[key_added]

    return new_center_model, align_models


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "init_center_model")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models_ref", optional=True)
def models_center_align_ref(
    init_center_model: AnnData,
    models: List[AnnData],
    models_ref: Optional[List[AnnData]] = None,
    n_sampling: Optional[int] = 1000,
    sampling_method: str = "trn",
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key_added: str = "models_align",
    lmbda: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dissimilarity: str = "kl",
    norm: bool = False,
    random_seed: Optional[int] = None,
    pis_init: Optional[List[np.ndarray]] = None,
    distributions: Optional[List[np.ndarray]] = None,
    dtype: str = "float32",
    device: str = "cpu",
) -> Tuple[AnnData, List[AnnData], List[AnnData]]:
    """
    Align the spatial coordinates of one model list to the central model through the affine transformation matrix obtained from another model list.

    Args:
        init_center_model: AnnData object to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        models: List of AnnData objects to use in the center alignment.
        models_ref: List of AnnData objects with a small number of coordinates.
        n_sampling: When `models_ref` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ["trn", "kmeans", "random"].
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        mapping_key_added: `.uns` key under which to add the alignment info.
        lmbda: List of probability weights assigned to each slice; If `None`, use uniform weights.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        n_components: Number of components in NMF decomposition.
        threshold: Threshold for convergence of W and H during NMF decomposition.
        max_iter: Maximum number of iterations for our center alignment algorithm.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        norm: If ``norm = True``, scales spatial distances such that neighboring spots are at distance 1.

              Otherwise, spatial distances remain unchanged.
        random_seed: Set random seed for reproducibility.
        pis_init: Initial list of mappings between ``A`` and ``models`` to solver.

                  Otherwise, default will automatically calculate mappings.
        distributions: Distributions of spots for each slice. Otherwise, default is uniform.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``

    Returns:
        new_center_model: The center model.
        align_models: List of models (AnnData Object) after alignment.
        align_models_ref: List of models_ref (AnnData Object) after alignment.
    """
    from dynamo.tools.sampling import sample

    if models_ref is None:
        center_sampling = sample(
            arr=np.asarray(init_center_model.obs_names),
            n=n_sampling,
            method=sampling_method,
            X=init_center_model.obsm[spatial_key],
        )
        init_center_model = init_center_model[center_sampling, :]

        models_ref = []
        for m in models:
            model_ref = m.copy()
            model_sampling = sample(
                arr=np.asarray(model_ref.obs_names), n=n_sampling, method=sampling_method, X=model_ref.obsm[spatial_key]
            )
            model_ref = model_ref[model_sampling, :]
            models_ref.append(model_ref)

    new_center_model, align_models_ref = models_center_align(
        init_center_model=init_center_model,
        models=models_ref,
        layer=layer,
        genes=genes,
        spatial_key=spatial_key,
        key_added=key_added,
        mapping_key_added=mapping_key_added,
        lmbda=lmbda,
        alpha=alpha,
        n_components=n_components,
        threshold=threshold,
        max_iter=max_iter,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        dissimilarity=dissimilarity,
        norm=norm,
        random_seed=random_seed,
        pis_init=pis_init,
        distributions=distributions,
        dtype=dtype,
        device=device,
    )

    align_models = []
    for i, (align_model_ref, m) in enumerate(zip(align_models_ref, models)):
        align_model = m.copy()

        t = align_model_ref.uns[f"center_{mapping_key_added}"]["mapping_relations"]["t"]
        R = align_model_ref.uns[f"center_{mapping_key_added}"]["mapping_relations"]["R"]

        align_model_coords = align_model.obsm[spatial_key].copy() - t
        align_model.obsm[key_added] = align_model_coords if R is None else R.dot(align_model_coords.T).T
        for key in [f"former_{mapping_key_added}", f"latter_{mapping_key_added}", f"center_{mapping_key_added}"]:
            if key in align_model_ref.uns_keys():
                align_model.uns[key] = align_model_ref.uns[key]
        align_models.append(align_model)

    return new_center_model, align_models, align_models_ref


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
