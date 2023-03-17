try:
    from typing import Any, List, Literal, Tuple, Union
except ImportError:
    from typing_extensions import Literal

from typing import List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from .methods import BA_align, BA_align_multi, align_preprocess
from .transform import BA_transform, BA_transform_and_assignment
from .utils import _iteration, downsampling


def morpho_align(
    models: List[AnnData],
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: str = "VecFld_morpho",
    mode: Literal["S", "N", "SN"] = "SN",
    dissimilarity: str = "kl",
    max_iter: int = 100,
    dtype: str = "float64",
    device: str = "cpu",
    verbose_level: int = 0,
    label_key: Optional[str] = None,
    **kwargs,
) -> Tuple[List[AnnData], List[np.ndarray], List[np.ndarray]]:
    """
    Continuous alignment of spatial transcriptomic coordinates based on Morpho.

    Args:
        models: List of models (AnnData Object).
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        iter_key_added: ``.uns`` key under which to add the result of each iteration of the iterative process. If ``iter_key_added``  is None, the results are not saved.
        vecfld_key_added: The key that will be used for the vector field key in ``.uns``. If ``vecfld_key_added`` is None, the results are not saved.
        mode: The method of alignment. Available ``mode`` are: ``'S'``, ``'N'`` and ``'SN'``.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        max_iter: Max number of iterations for morpho alignment.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose_level: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``BA_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        pis: List of pi matrices.
        sigma2s: List of sigma2.
    """
    for m in models:
        m.obsm[key_added] = m.obsm[spatial_key]
    for m in models:
        m.obsm["Rigid_3d_align_spatial"] = m.obsm[spatial_key]
    for m in models:
        m.obsm["Coarse_alignment"] = m.obsm[spatial_key]

    pis, sigma2s = [], []
    align_models = [model.copy() for model in models]
    progress_name = f"Models alignment based on morpho, mode: {mode}."
    verbose, sub_verbose = False, False
    if verbose_level > 0:
        verbose = True
    if verbose_level > 1:
        sub_verbose = True
    for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=verbose):
        modelA = align_models[i]
        modelB = align_models[i + 1]
        if label_key is not None:
            # calculate label similarity
            catA = modelA.obs[label_key]
            catB = modelB.obs[label_key]
            UnionCategories = np.union1d(catA.cat.categories, catB.cat.categories)
            catACode, catBCode = np.zeros(catA.shape, dtype=int), np.zeros(catB.shape, dtype=int)
            for code, cat in enumerate(UnionCategories):
                if cat == "unknown":
                    code = -1
                catACode[catA == cat] = code
                catBCode[catB == cat] = code
            LabelSimMat = np.zeros((catA.shape[0], catB.shape[0]))
            for index in range(catB.shape[0]):
                LabelSimMat[:, index] = catACode != catBCode[i]
            LabelSimMat = LabelSimMat.T
        else:
            LabelSimMat = None
        _, P, sigma2 = BA_align(
            sampleA=modelA,
            sampleB=modelB,
            genes=genes,
            spatial_key=key_added,
            key_added=key_added,
            iter_key_added=iter_key_added,
            vecfld_key_added=vecfld_key_added,
            layer=layer,
            mode=mode,
            dissimilarity=dissimilarity,
            max_iter=max_iter,
            dtype=dtype,
            device=device,
            inplace=True,
            verbose=sub_verbose,
            added_similarity=LabelSimMat,
            **kwargs,
        )
        pis.append(P)
        sigma2s.append(sigma2)

    return align_models, pis, sigma2s


def morpho_align_ref(
    models: List[AnnData],
    models_ref: Optional[List[AnnData]] = None,
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: Optional[str] = "VecFld_morpho",
    mode: Literal["S", "N", "SN"] = "SN",
    dissimilarity: str = "kl",
    max_iter: int = 100,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
    verbose_level: int = 1,
    return_full_assignment: bool = True,
    return_similarity: bool = True,
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Continuous alignment of spatial transcriptomic coordinates with the reference models based on Morpho.

    Args:
        models: List of models (AnnData Object).
        models_ref: Another list of models (AnnData Object).
        n_sampling: When ``models_ref`` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ``["trn", "kmeans", "random"]``.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        iter_key_added: ``.uns`` key under which to add the result of each iteration of the iterative process. If ``iter_key_added``  is None, the results are not saved.
        vecfld_key_added: The key that will be used for the vector field key in ``.uns``. If ``vecfld_key_added`` is None, the results are not saved.
        mode: The method of alignment. Available ``mode`` are: ``'S'``, ``'N'`` and ``'SN'``.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        max_iter: Max number of iterations for morpho alignment.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``BA_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        align_models_ref: List of models_ref (AnnData Object) after alignment.
        pis: List of pi matrices for models.
        pis_ref: List of pi matrices for models_ref.
        sigma2s: List of sigma2.
    """

    # Downsampling
    if models_ref is None:
        models_sampling = [model.copy() for model in models]
        models_ref = downsampling(
            models=models_sampling,
            n_sampling=n_sampling,
            sampling_method=sampling_method,
            spatial_key=spatial_key,
        )
    verbose, sub_verbose = False, False
    if verbose_level > 0:
        verbose = True
    if verbose_level > 1:
        sub_verbose = True
    pis, pis_ref, sigma2s = [], [], []
    align_models = [model.copy() for model in models]
    align_models_ref = [model.copy() for model in models_ref]
    for model in align_models_ref:
        model.obsm[key_added] = model.obsm[spatial_key]
    align_models[0].obsm[key_added] = align_models[0].obsm[spatial_key]
    progress_name = f"Models alignment with ref-models based on morpho, mode: {mode}."
    for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=verbose):
        modelA_ref = align_models_ref[i]
        modelB_ref = align_models_ref[i + 1]

        _, P, sigma2 = BA_align(
            sampleA=modelA_ref,
            sampleB=modelB_ref,
            genes=genes,
            spatial_key=key_added,
            key_added=key_added,
            iter_key_added=iter_key_added,
            vecfld_key_added=vecfld_key_added,
            layer=layer,
            mode=mode,
            dissimilarity=dissimilarity,
            max_iter=max_iter,
            dtype=dtype,
            device=device,
            inplace=True,
            verbose=sub_verbose,
            **kwargs,
        )
        align_models_ref[i + 1] = modelB_ref
        pis_ref.append(P)
        sigma2s.append(sigma2)

        modelA, modelB = align_models[i], align_models[i + 1]
        modelB.uns[vecfld_key_added] = modelB_ref.uns[vecfld_key_added]
        if return_full_assignment:
            P, modelB.obsm[key_added] = BA_transform_and_assignment(
                samples=[modelB, modelA],
                vecfld=modelB_ref.uns[vecfld_key_added],
                genes=genes,
                layer=layer,
                small_variance=True,
                spatial_key=spatial_key,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        else:
            if return_similarity:
                (modelB.obsm[key_added], _, modelB.obsm["Rigid_align_spatial"],) = BA_transform(
                    vecfld=modelB_ref.uns[vecfld_key_added],
                    quary_points=modelB.obsm[spatial_key],
                    device=device,
                    dtype=dtype,
                    return_similarity=True,
                )
            else:
                modelB.obsm[key_added], _ = BA_transform(
                    vecfld=modelB_ref.uns[vecfld_key_added],
                    quary_points=modelB.obsm[spatial_key],
                    device=device,
                    dtype=dtype,
                    return_similarity=False,
                )
        pis.append(P)

    return align_models, align_models_ref, pis, pis_ref, sigma2s


def morpho_align_ref_withlabel(
    models: List[AnnData],
    models_ref: Optional[List[AnnData]] = None,
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    label_key: str = "label",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: Optional[str] = "VecFld_morpho",
    mode: Literal["S", "N", "SN"] = "SN",
    dissimilarity: str = "kl",
    small_variance: bool = False,
    max_iter: int = 100,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Continuous alignment of spatial transcriptomic coordinates with the reference models based on Morpho.

    Args:
        models: List of models (AnnData Object).
        models_ref: Another list of models (AnnData Object).
        n_sampling: When ``models_ref`` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ``["trn", "kmeans", "random"]``.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        label_key: The key in ``obs`` that corresponds to the label of each spots.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        iter_key_added: ``.uns`` key under which to add the result of each iteration of the iterative process. If ``iter_key_added``  is None, the results are not saved.
        vecfld_key_added: The key that will be used for the vector field key in ``.uns``. If ``vecfld_key_added`` is None, the results are not saved.
        mode: The method of alignment. Available ``mode`` are: ``'S'``, ``'N'`` and ``'SN'``.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        small_variance: When approximating the assignment matrix, if True, we use small sigma2 (0.001) rather than the infered sigma2
        max_iter: Max number of iterations for morpho alignment.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``BA_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        align_models_ref: List of models_ref (AnnData Object) after alignment.
        pis: List of pi matrices for models.
        pis_ref: List of pi matrices for models_ref.
        sigma2s: List of sigma2.
    """

    # Downsampling
    if models_ref is None:
        models_sampling = [model.copy() for model in models]
        models_ref = downsampling(
            models=models_sampling,
            n_sampling=n_sampling,
            sampling_method=sampling_method,
            spatial_key=spatial_key,
        )

    pis, pis_ref, sigma2s = [], [], []
    align_models = [model.copy() for model in models]
    align_models_ref = [model.copy() for model in models_ref]
    align_models[0].obsm[key_added] = align_models[0].obsm[spatial_key]
    progress_name = f"Models alignment with ref-models based on morpho, mode: {mode}."
    for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=verbose):
        modelA_ref = align_models_ref[i]
        modelB_ref = align_models_ref[i + 1]
        # calculate label similarity
        catA = modelA_ref.obs[label_key]
        catB = modelB_ref.obs[label_key]
        UnionCategories = np.union1d(catA.cat.categories, catB.cat.categories)
        catACode, catBCode = np.zeros(catA.shape, dtype=int), np.zeros(catB.shape, dtype=int)
        for code, cat in enumerate(UnionCategories):
            if cat == "unknown":
                code = -1
            catACode[catA == cat] = code
            catBCode[catB == cat] = code
        LabelSimMat = np.zeros((catA.shape[0], catB.shape[0]))
        for index in range(catB.shape[0]):
            LabelSimMat[:, index] = catACode != catBCode[i]
        _, P, sigma2 = BA_align(
            sampleA=modelA_ref,
            sampleB=modelB_ref,
            genes=genes,
            spatial_key=spatial_key,
            key_added=key_added,
            iter_key_added=iter_key_added,
            vecfld_key_added=vecfld_key_added,
            layer=layer,
            mode=mode,
            dissimilarity=dissimilarity,
            max_iter=max_iter,
            small_variance=small_variance,
            dtype=dtype,
            device=device,
            inplace=True,
            verbose=verbose,
            added_similarity=LabelSimMat,
            **kwargs,
        )
        pis_ref.append(P)
        sigma2s.append(sigma2)

        modelA, modelB = align_models[i], align_models[i + 1]
        modelB.uns[vecfld_key_added] = modelB_ref.uns[vecfld_key_added]

        P, modelB.obsm[key_added] = BA_transform_and_assignment(
            samples=[modelB, modelA],
            vecfld=modelB_ref.uns[vecfld_key_added],
            genes=genes,
            layer=layer,
            small_variance=True,
            spatial_key=spatial_key,
            device=device,
            dtype=dtype,
        )
        pis.append(P)

    return align_models, align_models_ref, pis, pis_ref, sigma2s


def morpho_global_align(
    models: List[AnnData],
    min_sigma_index: int,
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "align_spatial",
    key_added: str = "global_align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    mode: Literal["S", "N", "SN"] = "SN",
    dissimilarity: str = "kl",
    neighbor_size: int = 2,
    max_iter: int = 100,
    max_iter_global: int = 10,
    normalize_c: bool = True,
    normalize_g: bool = True,
    dtype: str = "float64",
    device: str = "cpu",
    verbose_level: int = 2,
    vis_optimiation: bool = False,
    vis_file: str = "./visual/",
    **kwargs,
) -> Tuple[List[AnnData], List[np.ndarray]]:

    n_models = len(models)
    global_align_models = [m.copy() for m in models]
    normalize_g = False if dissimilarity == "kl" else normalize_g
    verbose, sub_verbose = False, False
    if verbose_level > 0:
        verbose = True
    if verbose_level > 1:
        sub_verbose = True
    # data pre-process
    (nx, type_as, new_samples, exp_matrices, spatial_coords, normalize_scale, normalize_mean_list,) = align_preprocess(
        samples=global_align_models,
        layer=layer,
        genes=genes,
        spatial_key=spatial_key,
        normalize_c=normalize_c,
        normalize_g=normalize_g,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )
    spatial_coords = [nx.to_numpy(spatial_coord) for spatial_coord in spatial_coords]
    exp_matrices = [nx.to_numpy(exp_matrix) for exp_matrix in exp_matrices]
    spatial_coords_hat = [spatial_coord.copy() for spatial_coord in spatial_coords]
    sigma2_change = np.ones((n_models, 1))
    sigma2_array = 100 * np.ones((n_models, 1))
    param_array = len(models) * [None]

    if vis_optimiation:
        import os
        import shutil

        if os.path.exists(vis_file):
            shutil.rmtree(vis_file)
        os.mkdir(vis_file)

    # main loop
    pis = None
    progress_name = f"Models global alignment based on morpho, mode: {mode}."
    for iter in _iteration(n=max_iter_global, progress_name=progress_name, verbose=verbose):
        if np.max(sigma2_change) < 1e-3:
            break

        iter_pis = []
        for index in np.random.permutation(range(n_models)):
            if index == min_sigma_index:
                sigma2_change[index] = 0
                continue

            coords_model, genes_model = spatial_coords[index], exp_matrices[index]
            neighbor_coords = [
                spatial_coords_hat[model_index]
                for model_index in range(n_models)
                if ((np.abs(model_index - index) <= neighbor_size) and (model_index != index))
            ]
            neighbor_genes = [
                exp_matrices[model_index]
                for model_index in range(n_models)
                if ((np.abs(model_index - index) <= neighbor_size) and (model_index != index))
            ]
            neighbor_weight = [
                1 / np.abs(model_index - index)
                for model_index in range(n_models)
                if ((np.abs(model_index - index) <= neighbor_size) and (model_index != index))
            ]
            param = BA_align_multi(
                coords_model=coords_model,
                genes_model=genes_model,
                neighbor_coords=neighbor_coords,
                neighbor_genes=neighbor_genes,
                neighbor_weight=neighbor_weight,
                nx=nx,
                type_as=type_as,
                iter_key_added=iter_key_added,
                key_added=key_added,
                mode=mode,
                dissimilarity=dissimilarity,
                max_iter=max_iter,
                verbose=sub_verbose,
                init_Param=param_array[index],
                vis_optimiation=vis_optimiation,
                file_name=vis_file + str(iter) + "_" + str(index) + ".gif",
                **kwargs,
            )
            spatial_coords_hat[index] = param["XnHat"]
            param_array[index] = param
            sigma2_change[index] = np.abs(sigma2_array[index] - param["sigma2"]) / sigma2_array[index]
            sigma2_array[index] = param["sigma2"]
            iter_pis.append(param["P"])
        pis = iter_pis.copy()

    for model, coord in zip(global_align_models, spatial_coords_hat):
        model.obsm[key_added] = coord * nx.to_numpy(normalize_scale) if normalize_c else coord

    return global_align_models, pis
