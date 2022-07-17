from typing import List, Optional, Tuple

import numpy as np
from anndata import AnnData

from ..configuration import SKM
from ..logging import logger_manager as lm
from .paste_bio import center_align, generalized_procrustes_analysis, pairwise_align


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
):
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
    for slice in slices:
        slice.obsm[key_added] = slice.obsm[spatial_key]

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
        sliceA.obsm[key_added] = np.around(sliceA_coodrs, decimals=2)
        sliceB.obsm[key_added] = np.around(sliceB_coodrs, decimals=2)

        if i == 0:
            align_slices.append(sliceA)
        align_slices.append(sliceB)

    return align_slices


def slices_align_ref(
    slices: List[AnnData],
    slices_ref: List[AnnData],
    layer: str = "X",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: str = "cpu",
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData]]:
    """Align spatial coordinates of slices.

    If there are too many slice coordinates to be aligned, this method can be selected.

    First select the slices with fewer coordinates for alignment, and then calculate the affine transformation matrix.
    Secondly, the required slices are aligned through the calculated affine transformation matrix.

    Args:
        slices: List of slices (AnnData Object).
        slices_ref: List of slices (AnnData Object) with a small number of coordinates.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinate.
        key_added: adata.obsm key under which to add the registered spatial coordinate.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        Tuple of two elements. The first contains a list of slices after alignment.
        The second contains a list of slices with a small number of coordinates
        after alignment.
    """

    import nudged

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
    for slice_ref, align_slice_ref, slice in zip(slices_ref, align_slices_ref, slices):
        # Calculate the affine transformation matrix through nudged
        slice_ref_coords = slice_ref.obsm[spatial_key].tolist()
        align_slice_ref_coords = align_slice_ref.obsm[key_added].tolist()
        trans = nudged.estimate(slice_ref_coords, align_slice_ref_coords)
        slice_coords = slice.obsm[spatial_key].tolist()

        #  Align slices through the calculated affine transformation matrix.
        align_slice_coords = np.around(trans.transform(slice_coords), decimals=2)
        align_slice = slice.copy()
        align_slice.obsm[key_added] = np.array(align_slice_coords)
        align_slice.obs["x"] = align_slice.obsm[key_added][:, 0].astype(float)
        align_slice.obs["y"] = align_slice.obsm[key_added][:, 1].astype(float)
        align_slices.append(align_slice)

    return align_slices, align_slices_ref


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
):
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
        List of slices (AnnData Object) after alignment.
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
