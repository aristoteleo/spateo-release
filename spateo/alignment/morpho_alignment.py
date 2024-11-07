try:
    from typing import Any, List, Literal, Tuple, Union
except ImportError:
    from typing_extensions import Literal

import functools
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
from anndata import AnnData

from spateo.alignment.methods import Morpho_pairwise, empty_cache
from spateo.alignment.transform import BA_transform
from spateo.alignment.utils import _iteration, downsampling, solve_RT_by_correspondence
from spateo.logging import logger_manager as lm


# TODO: update the args docstring
def morpho_align(
    models: List[AnnData],
    rep_layer: Union[str, List[str]] = "X",
    rep_field: Union[str, List[str]] = "layer",
    genes: Optional[Union[List[str], np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: str = "VecFld_morpho",
    mode: Literal["SN-N", "SN-S"] = "SN-S",
    dissimilarity: Union[str, List[str]] = "kl",
    max_iter: int = 200,
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[AnnData], List[np.ndarray]]:
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
        mode: The method of alignment. Available ``mode`` are: ``'SN-N'``, and ``'SN-S'``.

                * ``'SN-N'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a non-rigid aligned result;
                * ``'SN-S'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a rigid aligned result. The non-rigid is used here to solve the optimal mapping, thus returning a more accurate rigid transformation. The default is ``'SN-S'``.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        max_iter: Max number of iterations for morpho alignment.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``BA_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
        pis: List of pi matrices.
        sigma2s: List of sigma2.
    """

    align_models = [model.copy() for model in models]
    for m in align_models:
        m.obsm[key_added] = m.obsm[spatial_key].copy()
        m.obsm[f"{key_added}_rigid"] = m.obsm[spatial_key].copy()
        m.obsm[f"{key_added}_nonrigid"] = m.obsm[spatial_key].copy()

    pis = []
    progress_name = f"Models alignment based on morpho, mode: {mode}."
    for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=True):
        modelA = align_models[i]
        modelB = align_models[i + 1]

        morpho_model = Morpho_pairwise(
            sampleA=modelB,  # reverse
            sampleB=modelA,  # reverse
            rep_layer=rep_layer,
            rep_field=rep_field,
            dissimilarity=dissimilarity,
            genes=genes,
            spatial_key=key_added,
            key_added=key_added,
            iter_key_added=iter_key_added,
            vecfld_key_added=vecfld_key_added,
            max_iter=max_iter,
            dtype=dtype,
            device=device,
            verbose=verbose,
            **kwargs,
        )
        P = morpho_model.run()
        modelB.obsm[f"{key_added}_rigid"] = morpho_model.optimal_RnA.copy()
        modelB.obsm[f"{key_added}_nonrigid"] = morpho_model.XAHat.copy()
        if mode == "SN-S":
            modelB.obsm[key_added] = modelB.obsm[f"{key_added}_rigid"]
        elif mode == "SN-N":
            modelB.obsm[key_added] = modelB.obsm[f"{key_added}_nonrigid"]

        if iter_key_added is not None:
            modelB.uns[iter_key_added] = morpho_model.iter_added
        if vecfld_key_added is not None:
            modelB.uns[vecfld_key_added] = morpho_model.vecfld
        pis.append(P.T)
        empty_cache(device=device)

    return align_models, pis


def morpho_align_transformation(
    models: List[Union[AnnData, str]],
    models_path: Optional[str] = None,
    save_transformation: bool = False,
    transformation_path: Optional[str] = "./Spateo_transformation",
    resume: bool = False,
    rep_layer: Union[str, List[str]] = "X",
    rep_field: Union[str, List[str]] = "layer",
    genes: Optional[Union[List[str], np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: str = "VecFld_morpho",
    dissimilarity: Union[str, List[str]] = "kl",
    max_iter: int = 200,
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
):
    """
    Continuous alignment of spatial transcriptomic coordinates based on Morpho, and return the transformation matrix.

    Args:
        models (List[AnnData]): _description_

    Returns:
        _type_: _description_
    """

    # check models is a list of anndata or a list of file name
    if models_path is not None:
        assert all(
            [isinstance(model, str) for model in models]
        ), "models should be a list of file name if models_path is given."
    else:
        assert all(
            [isinstance(model, AnnData) for model in models]
        ), "models should be a list of anndata if models_path is not given."

    # check if files exists in models_path
    if models_path is not None:
        assert all(
            [os.path.exists(os.path.join(models_path, model)) for model in models]
        ), "Some files in models_path do not exist."

    iteration = 0
    transformation = []
    if save_transformation:
        Path(transformation_path).mkdir(parents=True, exist_ok=True)
        if resume:
            # find the highest iteration number
            for i in range(len(models) - 1):
                if os.path.exists(os.path.join(transformation_path, f"transformation_{i}.npy")):
                    iteration = i
                    transformation.append(np.load(os.path.join(transformation_path, f"transformation_{i}.npy")))
        else:
            # remove all files in the transformation_path
            remove_all_files_in_directory(transformation_path)

    if resume:
        progress_name = f"Models alignment based on morpho, starting from iteration {iteration}."
    else:
        progress_name = f"Models alignment based on morpho."

    if models_path is not None:
        modelA = ad.read_h5ad(os.path.join(models_path, models[iteration]))
    for i in _iteration(start_n=iteration, n=len(models) - 1, progress_name=progress_name, verbose=True):
        # load models if models is a list of file name
        if models_path is not None:
            modelB = ad.read_h5ad(os.path.join(models_path, models[i + 1]))
        else:
            modelA = models[i]
            modelB = models[i + 1]

        morpho_model = Morpho_pairwise(
            sampleA=modelB,  # reverse
            sampleB=modelA,  # reverse
            rep_layer=rep_layer,
            rep_field=rep_field,
            dissimilarity=dissimilarity,
            genes=genes,
            spatial_key=spatial_key,
            key_added=key_added,
            iter_key_added=iter_key_added,
            vecfld_key_added=vecfld_key_added,
            max_iter=max_iter,
            dtype=dtype,
            device=device,
            verbose=verbose,
            **kwargs,
        )
        _ = morpho_model.run()

        optimal_R, optimal_t = solve_RT_by_correspondence(
            morpho_model.optimal_RnA[:, :2], modelB.obsm[spatial_key][:, :2]
        )
        cur_transformation = {"Rotation": optimal_R, "Translation": optimal_t}
        transformation.append(cur_transformation)
        if save_transformation:
            np.save(os.path.join(transformation_path, f"transformation_{i}.npy"), cur_transformation)

        if models_path is not None:
            modelA = modelB
    return transformation


def morpho_align_apply_transformation(
    models: List[Union[AnnData, str]],
    models_path: Optional[str] = None,
    transformation: List[dict] = None,
    transformation_path: Optional[str] = "./Spateo_transformation",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    save_models_path: Optional[str] = None,
    verbose: bool = True,
):
    """
    Apply the transformation to the models.

    Args:
        models (List[AnnData]): _description_
        transformation (List[dict]): _description_

    Returns:
        _type_: _description_
    """

    # check models is a list of anndata or a list of file name
    if models_path is not None:
        assert all(
            [isinstance(model, str) for model in models]
        ), "models should be a list of file name if models_path is given."
    else:
        assert all(
            [isinstance(model, AnnData) for model in models]
        ), "models should be a list of anndata if models_path is not given."

    # check if files exists in models_path
    if models_path is not None:
        assert all(
            [os.path.exists(os.path.join(models_path, model)) for model in models]
        ), "Some files in models_path do not exist."

    if transformation is None:
        assert os.path.exists(transformation_path), "transformation_path does not exist."
        # check if transformations exists in transformation_path
        transformation = []
        for i in range(len(models) - 1):
            cur_transformation = np.load(
                os.path.join(transformation_path, f"transformation_{i}.npy"), allow_pickle=True
            )
            transformation.append(cur_transformation)
    else:
        assert len(transformation) == len(models) - 1, "The length of transformation should be len(models) - 1."

    if save_models_path is not None:
        Path(save_models_path).mkdir(parents=True, exist_ok=True)

    # initialize rotation and translation
    cur_R = np.diag((1, 1))
    cur_t = np.zeros((2,))

    if models_path is not None and save_models_path is None:
        align_models = []

    if models_path is not None:
        cur_model = ad.read_h5ad(os.path.join(models_path, models[0]))
    else:
        cur_model = models[0]
    cur_model.obsm[key_added] = cur_model.obsm[spatial_key].copy()

    if save_models_path is not None:
        cur_model.write(os.path.join(save_models_path, models[0]))
    else:
        if models_path is not None:
            align_models.append(cur_model)

    progress_name = f"Models alignment based on morpho, applying transformation."
    for i in _iteration(n=len(models) - 1, progress_name=progress_name, verbose=True):
        # load models if models is a list of file name
        if models_path is not None:
            cur_model = ad.read_h5ad(os.path.join(models_path, models[i + 1]))
        else:
            cur_model = models[i + 1]

        cur_t = transformation[i]["Translation"] @ cur_R.T + cur_t
        cur_R = cur_R @ transformation[i]["Rotation"]

        cur_model.obsm[key_added] = cur_model.obsm[spatial_key].copy() @ cur_R.T + cur_t

        if save_models_path is not None:
            cur_model.write(os.path.join(save_models_path, models[i + 1]))
        else:
            if models_path is not None:
                align_models.append(cur_model)

    if models_path is not None:
        return align_models
    else:
        return models


# TODO: add the args docstring
def morpho_align_ref(
    models: List[AnnData],
    models_ref: Optional[List[AnnData]] = None,
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    rep_layer: Union[str, List[str]] = "X",
    rep_field: Union[str, List[str]] = "layer",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: Optional[str] = "VecFld_morpho",
    mode: Literal["SN-N", "SN-S"] = "SN-S",
    dissimilarity: Union[str, List[str]] = "kl",
    max_iter: int = 200,
    # return_full_assignment: bool = False,
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[AnnData], List[AnnData], List[np.ndarray], List[np.ndarray]]:
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
        mode: The method of alignment. Available ``mode`` are: ``'SN-N'``, and ``'SN-S'``.

                * ``'SN-N'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a non-rigid aligned result;
                * ``'SN-S'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a rigid aligned result. The non-rigid is used here to solve the optimal mapping, thus returning a more accurate rigid transformation. The default is ``'SN-S'``.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        max_iter: Max number of iterations for morpho alignment.
        SVI_mode: Whether to use stochastic variational inferential (SVI) optimization strategy.
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
    # TODO: this operation is very reducdant, need to be optimized
    if models_ref is None:
        models_sampling = [model.copy() for model in models]
        models_ref = downsampling(
            models=models_sampling,
            n_sampling=n_sampling,
            sampling_method=sampling_method,
            spatial_key=spatial_key,
        )

    pis, pis_ref = [], []

    align_models = [model.copy() for model in models]
    for model in align_models:
        model.obsm[key_added] = model.obsm[spatial_key].copy()
        model.obsm[f"{key_added}_rigid"] = model.obsm[spatial_key].copy()
        model.obsm[f"{key_added}_nonrigid"] = model.obsm[spatial_key].copy()

    align_models_ref = [model.copy() for model in models_ref]
    for model in align_models_ref:
        model.obsm[key_added] = model.obsm[spatial_key].copy()
        model.obsm[f"{key_added}_rigid"] = model.obsm[spatial_key].copy()
        model.obsm[f"{key_added}_nonrigid"] = model.obsm[spatial_key].copy()

    progress_name = f"Models alignment with ref-models based on morpho, mode: {mode}."
    for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=True):
        modelA_ref = align_models_ref[i]
        modelB_ref = align_models_ref[i + 1]

        morpho_model = Morpho_pairwise(
            sampleA=modelB_ref,  # reverse
            sampleB=modelA_ref,  # reverse
            rep_layer=rep_layer,
            rep_field=rep_field,
            dissimilarity=dissimilarity,
            genes=genes,
            spatial_key=key_added,
            key_added=key_added,
            iter_key_added=iter_key_added,
            vecfld_key_added=vecfld_key_added,
            max_iter=max_iter,
            dtype=dtype,
            device=device,
            verbose=verbose,
            **kwargs,
        )
        P = morpho_model.run()
        modelB_ref.obsm[f"{key_added}_rigid"] = morpho_model.optimal_RnA.copy()
        modelB_ref.obsm[f"{key_added}_nonrigid"] = morpho_model.XAHat.copy()

        if mode == "SN-S":
            modelB_ref.obsm[key_added] = modelB_ref.obsm[f"{key_added}_rigid"]
        elif mode == "SN-N":
            modelB_ref.obsm[key_added] = modelB_ref.obsm[f"{key_added}_nonrigid"]

        align_models_ref[i + 1] = modelB_ref
        pis_ref.append(P)

        # use the reference model to align the original model
        modelB = align_models[i + 1]
        vecfld = morpho_model.vecfld
        if iter_key_added is not None:
            modelB_ref.uns[iter_key_added] = morpho_model.iter_added
            modelB.uns[iter_key_added] = morpho_model.iter_added
        if vecfld_key_added is not None:
            modelB_ref.uns[vecfld_key_added] = morpho_model.vecfld
            modelB.uns[vecfld_key_added] = morpho_model.vecfld
        ## Deprecated
        # if return_full_assignment:
        #     (
        #         modelB.obsm[f"{key_added}_nonrigid"],
        #         _,
        #         modelB.obsm[f"{key_added}_rigid"],
        #         P,
        #     ) = BA_transform_and_assignment(
        #         samples=[modelB, modelA],
        #         vecfld=modelB_ref.uns[vecfld_key_added],
        #         genes=genes,
        #         layer=layer,
        #         spatial_key=spatial_key,
        #         device=device,
        #         dtype=dtype,
        #         **kwargs,
        #     )
        # else:
        modelB.obsm[f"{key_added}_nonrigid"], _, modelB.obsm[f"{key_added}_rigid"] = BA_transform(
            vecfld=vecfld,
            quary_points=modelB.obsm[key_added],
            device=device,
            dtype=dtype,
        )
        if mode == "SN-S":
            modelB.obsm[key_added] = modelB.obsm[f"{key_added}_rigid"]
        elif mode == "SN-N":
            modelB.obsm[key_added] = modelB.obsm[f"{key_added}_nonrigid"]

        pis.append(P)

    return align_models, align_models_ref, pis, pis_ref


## Deprecated
# def morpho_align_sparse(
#     models: List[AnnData],
#     layer: str = "X",
#     genes: Optional[Union[list, np.ndarray]] = None,
#     spatial_key: str = "spatial",
#     key_added: str = "align_spatial",
#     iter_key_added: Optional[str] = "iter_spatial",
#     vecfld_key_added: str = "VecFld_morpho",
#     mode: Literal["SN-N", "SN-S"] = "SN-S",
#     dissimilarity: str = "kl",
#     max_iter: int = 100,
#     SVI_mode: bool = True,
#     use_label_prior: bool = False,
#     label_key: Optional[str] = "cluster",
#     label_transfer_prior: Optional[dict] = None,
#     dtype: str = "float32",
#     device: str = "0",
#     verbose: bool = True,
#     **kwargs,
# ) -> Tuple[List[AnnData], List[np.ndarray], List[np.ndarray]]:
#     """
#     Continuous alignment of spatial transcriptomic coordinates based on Morpho.

#     Args:
#         models: List of models (AnnData Object).
#         layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
#         genes: Genes used for calculation. If None, use all common genes for calculation.
#         spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
#         key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
#         iter_key_added: ``.uns`` key under which to add the result of each iteration of the iterative process. If ``iter_key_added``  is None, the results are not saved.
#         vecfld_key_added: The key that will be used for the vector field key in ``.uns``. If ``vecfld_key_added`` is None, the results are not saved.
#         mode: The method of alignment. Available ``mode`` are: ``'SN-N'``, and ``'SN-S'``.

#                 * ``'SN-N'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a non-rigid aligned result;
#                 * ``'SN-S'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a rigid aligned result. The non-rigid is used here to solve the optimal mapping, thus returning a more accurate rigid transformation. The default is ``'SN-S'``.
#         dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
#         max_iter: Max number of iterations for morpho alignment.
#         SVI_mode: Whether to use stochastic variational inferential (SVI) optimization strategy.
#         dtype: The floating-point number type. Only ``float32`` and ``float64``.
#         device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
#         verbose: If ``True``, print progress updates.
#         **kwargs: Additional parameters that will be passed to ``BA_align_sparse`` function.

#     Returns:
#         align_models: List of models (AnnData Object) after alignment.
#         pis: List of pi matrices.
#         sigma2s: List of sigma2.
#     """
#     import torch

#     from .methods import BA_align_sparse

#     lm.main_warning(message="Currently this function can only be run using GPU.")
#     if not torch.cuda.is_available():
#         raise Exception(f"The CUDA is not available, please check the CUDA version and GPU device.")

#     align_models = [model.copy() for model in models]
#     for m in align_models:
#         m.obsm[key_added] = m.obsm[spatial_key]
#     for m in align_models:
#         m.obsm[key_added + "_rigid"] = m.obsm[spatial_key]
#     for m in align_models:
#         m.obsm[key_added + "_nonrigid"] = m.obsm[spatial_key]

#     pis, sigma2s = [], []
#     progress_name = f"Models alignment based on morpho, mode: {mode}."
#     for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=True):
#         modelA = align_models[i]
#         modelB = align_models[i + 1]
#         _, P, sigma2 = BA_align_sparse(
#             sampleA=modelA,
#             sampleB=modelB,
#             genes=genes,
#             spatial_key=key_added,
#             key_added=key_added,
#             iter_key_added=iter_key_added,
#             vecfld_key_added=vecfld_key_added,
#             layer=layer,
#             dissimilarity=dissimilarity,
#             max_iter=max_iter,
#             dtype=dtype,
#             device=device,
#             inplace=True,
#             verbose=verbose,
#             SVI_mode=SVI_mode,
#             use_label_prior=use_label_prior,
#             label_key=label_key,
#             label_transfer_prior=label_transfer_prior,
#             **kwargs,
#         )
#         if mode == "SN-S":
#             modelB.obsm[key_added] = modelB.obsm[key_added + "_rigid"]
#         elif mode == "SN-N":
#             modelB.obsm[key_added] = modelB.obsm[key_added + "_nonrigid"]
#         modelB.uns[vecfld_key_added]["X"] = modelB.obsm[spatial_key]
#         pis.append(P)
#         sigma2s.append(sigma2)
#         empty_cache(device=device)

#     return align_models, pis, sigma2s

# def morpho_align(
#     models: List[AnnData],
#     layer: str = "X",
#     genes: Optional[Union[list, np.ndarray]] = None,
#     spatial_key: str = "spatial",
#     key_added: str = "align_spatial",
#     iter_key_added: Optional[str] = None,
#     vecfld_key_added: Optional[str] = None,
#     transformation_type: Literal["SN-N", "SN-S", "S"] = "SN-S",
#     dissimilarity: Literal["euc", "kl", "cos"] = "kl",
#     max_iter: int = 100,
#     SVI_mode: bool = True,
#     dtype: str = "float32",
#     device: str = "cpu",
#     verbose: bool = True,
#     **kwargs,
# ) -> Tuple[List[AnnData], List[np.ndarray], List[np.ndarray]]:
#     """
#     Serial alignment of spatial transcriptomic coordinates based on Spateo.

#     Args:
#         models: List of models (AnnData Object).
#         layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
#         genes: Genes used for calculation. If None, use all common genes for calculation.
#         spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
#         key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
#         iter_key_added: ``.uns`` key under which to add the result of each iteration of the iterative process. If ``iter_key_added``  is None, the results are not saved.
#         vecfld_key_added: The key that will be used for the vector field key in ``.uns``. If ``vecfld_key_added`` is None, the results are not saved.
#         mode: The method of alignment. Available ``mode`` are: ``'SN-N'``, and ``'SN-S'``.

#                 * ``'SN-N'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a non-rigid aligned result;
#                 * ``'SN-S'``: use both rigid and non-rigid alignment to keep the overall shape unchanged, while including local non-rigidity, and finally returns a rigid aligned result. The non-rigid is used here to solve the optimal mapping, thus returning a more accurate rigid transformation. The default is ``'SN-S'``.
#         dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
#         max_iter: Max number of iterations for morpho alignment.
#         SVI_mode: Whether to use stochastic variational inferential (SVI) optimization strategy.
#         dtype: The floating-point number type. Only ``float32`` and ``float64``.
#         device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
#         verbose: If ``True``, print progress updates.
#         **kwargs: Additional parameters that will be passed to ``BA_align`` function.

#     Returns:
#         align_models: List of models (AnnData Object) after alignment.
#         pis: List of pi matrices.
#         sigma2s: List of sigma2.
#     """
#     align_models = [model.copy() for model in models]
#     for m in align_models:
#         m.obsm[key_added] = m.obsm[spatial_key]
#     for m in align_models:
#         m.obsm["Rigid_align_spatial"] = m.obsm[spatial_key]
#     for m in align_models:
#         m.obsm["Nonrigid_align_spatial"] = m.obsm[spatial_key]

#     pis, sigma2s = [], []
#     progress_name = f"Models alignment based on morpho, mode: {mode}."
#     for i in _iteration(n=len(align_models) - 1, progress_name=progress_name, verbose=True):
#         modelA = align_models[i]
#         modelB = align_models[i + 1]
#         _, P, sigma2 = BA_align(
#             sampleA=modelA,
#             sampleB=modelB,
#             genes=genes,
#             spatial_key=key_added,
#             key_added=key_added,
#             iter_key_added=iter_key_added,
#             vecfld_key_added=vecfld_key_added,
#             layer=layer,
#             dissimilarity=dissimilarity,
#             max_iter=max_iter,
#             dtype=dtype,
#             device=device,
#             inplace=True,
#             verbose=verbose,
#             SVI_mode=SVI_mode,
#             **kwargs,
#         )
#         if mode == "SN-S":
#             modelB.obsm[key_added] = modelB.obsm["Rigid_align_spatial"]
#         elif mode == "SN-N":
#             modelB.obsm[key_added] = modelB.obsm["Nonrigid_align_spatial"]
#         pis.append(P)
#         sigma2s.append(sigma2)
#         empty_cache(device=device)

#     return align_models, pis, sigma2s


def remove_all_files_in_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The directory {directory_path} does not exist.")
