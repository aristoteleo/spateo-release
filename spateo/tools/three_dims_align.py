from typing import List, Optional, Tuple

import dynamo as dyn
import numpy as np
from anndata import AnnData

from ..configuration import SKM
from ..logging import logger_manager as lm
from .paste import center_align, generalized_procrustes_analysis, pairwise_align


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
    try:
        import nudged
    except:
        raise ImportError("You need to install the package `nudged`." "\nInstall nudged via `pip install nudged`")

    trans = nudged.estimate(coords_refA.tolist(), coords_refB.tolist())
    new_coords = trans.transform(coords.tolist())
    return np.asarray(new_coords)


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


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "slices")
def slices_align(
    slices: List[AnnData],
    layer: str = "X",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: str = "cpu",
    **kwargs,
) -> List[AnnData]:
    """
    Align spatial coordinates of slices.

    Args:
        slices: List of slices (AnnData Object).
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinate.
        key_added: adata.obsm key under which to add the registered spatial coordinate.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        List of slices (AnnData Object) after alignment.
    """
    for s in slices:
        s.obsm[key_added] = s.obsm[spatial_key]

    align_slices = []
    for i in lm.progress_logger(range(len(slices) - 1), progress_name="Slices alignment"):

        sliceA = slices[i].copy() if i == 0 else align_slices[i].copy()
        sliceB = slices[i + 1].copy()

        # Calculate and returns optimal alignment of two slices.
        pi, _ = pairwise_align(
            sampleA=sliceA,
            sampleB=sliceB,
            spatial_key=key_added,
            layer=layer,
            alpha=alpha,
            numItermax=numItermax,
            numItermaxEmd=numItermaxEmd,
            device=device,
            **kwargs,
        )

        # Calculate new coordinates of two slices
        sliceA_coodrs, sliceB_coodrs = generalized_procrustes_analysis(
            X=sliceA.obsm[key_added], Y=sliceB.obsm[key_added], pi=pi
        )
        sliceA.obsm[key_added] = sliceA_coodrs
        sliceB.obsm[key_added] = sliceB_coodrs

        if i == 0:
            align_slices.append(sliceA)
        align_slices.append(sliceB)

    return align_slices


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "slices")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "slices_ref", optional=True)
def slices_align_ref(
    slices: List[AnnData],
    slices_ref: Optional[List[AnnData]],
    n_sampling: Optional[int] = 1000,
    layer: str = "X",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: str = "cpu",
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData]]:
    """
    Align the spatial coordinates of one slice list through the affine transformation matrix obtained from another slice list.
    If there are too many slice coordinates to be aligned, this method can be selected.
    First select the slices with fewer coordinates for alignment, and then calculate the affine transformation matrix.
    Secondly, the required slices are aligned through the calculated affine transformation matrix.

    Args:
        slices: List of slices (AnnData Object).
        slices_ref: List of slices (AnnData Object) with a small number of coordinates.
        n_sampling: When `slices_ref` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinate.
        key_added: adata.obsm key under which to add the registered spatial coordinate.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        align_slices_ref: List of slices_ref (AnnData Object) after alignment.
        align_slices: List of slices (AnnData Object) after alignment.
    """

    if slices_ref is None:
        slices_ref = []
        for s in slices:
            slice_ref = s.copy()
            sampling = dyn.tl.sample(
                arr=np.asarray(slice_ref.obs_names), n=n_sampling, method="trn", X=slice_ref.obsm[spatial_key]
            )
            slice_ref = slice_ref[sampling, :]
            slices_ref.append(slice_ref)

    # Align spatial coordinates of slices with a small number of coordinates.
    align_slices_ref = slices_align(
        slices=slices_ref,
        layer=layer,
        spatial_key=spatial_key,
        key_added=key_added,
        alpha=alpha,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        device=device,
        **kwargs,
    )

    align_slices = []
    for slice_ref, align_slice_ref, s in zip(slices_ref, align_slices_ref, slices):
        align_slice = s.copy()

        align_slice_coords = rigid_transform_2D(
            coords=s.obsm[spatial_key],
            coords_refA=slice_ref.obsm[spatial_key],
            coords_refB=align_slice_ref.obsm[key_added],
        )

        align_slice.obsm[key_added] = align_slice_coords
        align_slices.append(align_slice)

    return align_slices, align_slices_ref


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "init_center_model")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models")
def models_align(
    init_center_model: AnnData,
    models: List[AnnData],
    layer: str = "X",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
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
    device: str = "cpu",
) -> Tuple[AnnData, List[AnnData]]:
    """
    Align spatial coordinates of a list of models to a center model.

    Args:
        init_center_model: Model to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        models: List of models to use in the center alignment.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinate.
        key_added: adata.obsm key under which to add the registered spatial coordinate.
        lmbda: List of probability weights assigned to each slice; If ``None``, use uniform weights.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        n_components: Number of components in NMF decomposition.
        threshold: Threshold for convergence of W and H during NMF decomposition.
        max_iter: Maximum number of iterations for our center alignment algorithm.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        random_seed: Set random seed for reproducibility.
        pis_init: Initial list of mappings between 'A' and 'slices' to solver. Otherwise, default will automatically calculate mappings.
        distributions: Distributions of spots for each slice. Otherwise, default is uniform.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

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
        device=device,
    )

    align_models = []
    for model, pi in zip(models, pis):
        center_coords, model_coords = generalized_procrustes_analysis(
            center_model.obsm[key_added], model.obsm[key_added], pi
        )
        model.obsm[key_added] = model_coords
        align_models.append(model)

    new_center_model = init_center_model.copy()
    new_center_model.obsm[key_added] = center_coords

    return new_center_model, align_models


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "init_center_model")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "models_ref", optional=True)
def models_align_ref(
    init_center_model: AnnData,
    models: List[AnnData],
    models_ref: Optional[List[AnnData]] = None,
    n_sampling: Optional[int] = 1000,
    layer: str = "X",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
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
    device: str = "cpu",
) -> Tuple[AnnData, List[AnnData], List[AnnData]]:
    """
    Align the spatial coordinates of one model list to the central model through the affine transformation matrix obtained from another model list.

    Args:
        init_center_model: Model to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        models: List of models to use in the center alignment.
        models_ref: List of models (AnnData Object) with a small number of coordinates.
        n_sampling: When `models_ref` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinate.
        key_added: adata.obsm key under which to add the registered spatial coordinate.
        lmbda: List of probability weights assigned to each slice; If ``None``, use uniform weights.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        n_components: Number of components in NMF decomposition.
        threshold: Threshold for convergence of W and H during NMF decomposition.
        max_iter: Maximum number of iterations for our center alignment algorithm.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        random_seed: Set random seed for reproducibility.
        pis_init: Initial list of mappings between 'A' and 'slices' to solver. Otherwise, default will automatically calculate mappings.
        distributions: Distributions of spots for each slice. Otherwise, default is uniform.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        new_center_model: The center model.
        align_models_ref: List of models_ref (AnnData Object) after alignment.
        align_models: List of models (AnnData Object) after alignment.

    """

    if models_ref is None:
        center_sampling = dyn.tl.sample(
            arr=np.asarray(init_center_model.obs_names),
            n=n_sampling,
            method="trn",
            X=init_center_model.obsm[spatial_key],
        )
        init_center_model = init_center_model[center_sampling, :]

        models_ref = []
        for m in models:
            model_ref = m.copy()
            model_sampling = dyn.tl.sample(
                arr=np.asarray(model_ref.obs_names), n=n_sampling, method="trn", X=model_ref.obsm[spatial_key]
            )
            model_ref = model_ref[model_sampling, :]
            models_ref.append(model_ref)

    new_center_model, align_models_ref = models_align(
        init_center_model=init_center_model,
        models=models_ref,
        layer=layer,
        spatial_key=spatial_key,
        key_added=key_added,
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
        device=device,
    )

    align_models = []
    for model_ref, align_model_ref, m in zip(models_ref, align_models_ref, models):
        align_model = m.copy()

        align_model_coords = rigid_transform_3D(
            coords=m.obsm[spatial_key],
            coords_refA=model_ref.obsm[spatial_key],
            coords_refB=align_model_ref.obsm[key_added],
        )

        align_model.obsm[key_added] = align_model_coords
        align_models.append(align_model)

    return new_center_model, align_models_ref, align_models
