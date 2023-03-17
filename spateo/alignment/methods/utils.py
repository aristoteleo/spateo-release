import os
from typing import List, Optional, Tuple, Union

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from scipy.linalg import pinv
from scipy.sparse import issparse
from scipy.special import psi

from spateo.logging import logger_manager as lm

# Get the intersection of lists
intersect_lsts = lambda *lsts: list(set(lsts[0]).intersection(*lsts[1:]))

# Covert a sparse matrix into a dense np array
to_dense_matrix = lambda X: X.toarray() if issparse(X) else np.array(X)

# Returns the data matrix or representation
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.layers[rep]


###########################
# Check data and computer #
###########################


def check_backend(device: str = "cpu", dtype: str = "float32", verbose: bool = True):
    """
    Check the proper backend for the device.

    Args:
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        dtype: The floating-point number type. Only float32 and float64.
        verbose: If ``True``, print progress updates.

    Returns:
        backend: The proper backend.
        type_as: The type_as.device is the device used to run the program and the type_as.dtype is the floating-point number type.
    """
    if device == "cpu":
        backend = ot.backend.NumpyBackend()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if torch.cuda.is_available():
            torch.cuda.init()
            backend = ot.backend.TorchBackend()
        else:
            backend = ot.backend.NumpyBackend()
            if verbose:
                lm.main_info(
                    message="GPU is not available, resorting to torch cpu.",
                    indent_level=1,
                )
    if nx_torch(backend):
        type_as = backend.__type_list__[-2] if dtype == "float32" else backend.__type_list__[-1]
    else:
        type_as = backend.__type_list__[0] if dtype == "float32" else backend.__type_list__[1]
    return backend, type_as


def check_spatial_coords(sample: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Check spatial coordinate information.

    Args:
        sample: An anndata object.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.

    Returns:
        The spatial coordinates.
    """
    coordinates = sample.obsm[spatial_key].copy()
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.values

    return np.asarray(coordinates)


def check_exp(sample: AnnData, layer: str = "X") -> np.ndarray:
    """
    Check expression matrix.

    Args:
        sample: An anndata object.
        layer: The key in `.layers` that corresponds to the expression matrix.

    Returns:
        The expression matrix.
    """

    exp_martix = sample.X.copy() if layer == "X" else sample.layers[layer].copy()
    exp_martix = to_dense_matrix(exp_martix)
    return exp_martix


######################
# Data preprocessing #
######################


def filter_common_genes(*genes, verbose: bool = True) -> list:
    """
    Filters for the intersection of genes between all samples.

    Args:
        genes: List of genes.
        verbose: If ``True``, print progress updates.
    """

    common_genes = intersect_lsts(*genes)
    if len(common_genes) == 0:
        raise ValueError("The number of common gene between all samples is 0.")
    else:
        if verbose:
            lm.main_info(
                message=f"Filtered all samples for common genes. There are {(len(common_genes))} common genes.",
                indent_level=1,
            )
        return common_genes


def normalize_coords(
    coords: Union[List[np.ndarray or torch.Tensor], np.ndarray, torch.Tensor],
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend] = ot.backend.NumpyBackend,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], float, List[np.ndarray]]:
    """Normalize the spatial coordinate.

    Args:
        coords: Spatial coordinate of sample.
        nx: The proper backend.
        verbose: If ``True``, print progress updates.
    """
    if type(coords) in [np.ndarray, torch.Tensor]:
        coords = [coords]

    normalize_scale = 0
    normalize_mean_list = []
    for i in range(len(coords)):
        normalize_mean = nx.einsum("ij->j", coords[i]) / coords[i].shape[0]
        normalize_mean_list.append(normalize_mean)
        coords[i] -= normalize_mean
        normalize_scale += nx.sqrt(nx.einsum("ij->", nx.einsum("ij,ij->ij", coords[i], coords[i])) / coords[i].shape[0])

    normalize_scale /= len(coords)
    for i in range(len(coords)):
        coords[i] /= normalize_scale
    if verbose:
        lm.main_info(message=f"Coordinates normalization params:", indent_level=1)
        lm.main_info(message=f"Scale: {normalize_scale}.", indent_level=2)
        # lm.main_info(message=f"Mean:  {normalize_mean_list}", indent_level=2)
    return coords, normalize_scale, normalize_mean_list


def normalize_exps(
    matrices: List[np.ndarray or torch.Tensor],
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend] = ot.backend.NumpyBackend,
    verbose: bool = True,
) -> List[np.ndarray]:
    """Normalize the gene expression.

    Args:
        matrices: Gene expression of sample.
        nx: The proper backend.
        verbose: If ``True``, print progress updates.
    """
    # n_matrix_index = [m.shape[0] for m in matrices]
    # normalize_mean = nx.sum(_data(nx,[nx.sum(m,0) for m in matrices],matrices[0]),0) / sum(n_matrix_index)
    # # print(normalize_mean.shape)
    # # matrices = [m - normalize_mean[None,:] for m in matrices]
    # # normalize_mean = sum([nx.einsum("ij->j", m) for m in matrices]) / (
    # #     sum(n_matrix_index)
    # # )
    # normalize_scale = nx.maximum(
    #         nx.sqrt(
    #         sum([nx.sum(m**2,0) for m in matrices]) / sum(n_matrix_index)
    #     ),
    #     0.0001
    # )

    # # print(normalize_scale)
    # N_matrices = [m / normalize_scale for m in matrices]

    n_matrix_index = [m.shape[0] for m in matrices]
    integrate_matrix = _cat(nx=nx, x=matrices, dim=0)

    normalize_mean = nx.einsum("ij->j", integrate_matrix) / integrate_matrix.shape[0]
    integrate_matrix = integrate_matrix - normalize_mean
    normalize_scale = nx.sqrt(
        nx.einsum("ij->", nx.einsum("ij,ij->ij", integrate_matrix, integrate_matrix)) / integrate_matrix.shape[0]
    )
    N_integrate_matrix = integrate_matrix / normalize_scale

    N_matrices, start_i = [], 0
    for i in n_matrix_index:
        N_matrices.append(N_integrate_matrix[start_i : start_i + i, :])
        start_i = start_i + i

    if verbose:
        lm.main_info(message=f"Gene expression normalization params:", indent_level=1)
        lm.main_info(message=f"Mean: {normalize_mean}.", indent_level=2)
        lm.main_info(message=f"Scale: {normalize_scale}.", indent_level=2)

    return N_matrices


def align_preprocess(
    samples: List[AnnData],
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    normalize_c: bool = False,
    normalize_g: bool = False,
    select_high_exp_genes: Union[bool, float, int] = False,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> Tuple[
    ot.backend.TorchBackend or ot.backend.NumpyBackend,
    torch.Tensor or np.ndarray,
    list,
    list,
    list,
    Optional[float],
    Optional[list],
]:
    """
    Data preprocessing before alignment.

    Args:
        samples: A list of anndata object.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        normalize_c: Whether to normalize spatial coordinates.
        normalize_g: Whether to normalize gene expression.
        select_high_exp_genes: Whether to select genes with high differences in gene expression.
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        verbose: If ``True``, print progress updates.
    """

    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)
    # Subset for common genes
    new_samples = [s.copy() for s in samples]
    all_samples_genes = [s[0].var.index for s in new_samples]
    common_genes = filter_common_genes(*all_samples_genes, verbose=verbose)
    common_genes = common_genes if genes is None else intersect_lsts(common_genes, genes)
    new_samples = [s[:, common_genes] for s in new_samples]

    # Gene expression matrix of all samples
    exp_matrices = [nx.from_numpy(check_exp(sample=s, layer=layer), type_as=type_as) for s in new_samples]
    if not (select_high_exp_genes is False):
        # Select significance genes if select_high_exp_genes is True
        ExpressionData = _cat(nx=nx, x=exp_matrices, dim=0)

        ExpressionVar = _var(nx, ExpressionData, 0)
        exp_threshold = 10 if isinstance(select_high_exp_genes, bool) else select_high_exp_genes
        EvidenceExpression = nx.where(ExpressionVar > exp_threshold)[0]
        exp_matrices = [exp_matrix[:, EvidenceExpression] for exp_matrix in exp_matrices]
        if verbose:
            lm.main_info(message=f"Evidence expression number: {len(EvidenceExpression)}.")

    # Spatial coordinates of all samples
    spatial_coords = [
        nx.from_numpy(check_spatial_coords(sample=s, spatial_key=spatial_key), type_as=type_as) for s in new_samples
    ]
    coords_dims = nx.unique(_data(nx, [c.shape[1] for c in spatial_coords], type_as))
    # coords_dims = np.unique(np.asarray([c.shape[1] for c in spatial_coords]))
    assert len(coords_dims) == 1, "Spatial coordinate dimensions are different, please check again."

    normalize_scale, normalize_mean_list = None, None
    if normalize_c:
        spatial_coords, normalize_scale, normalize_mean_list = normalize_coords(
            coords=spatial_coords, nx=nx, verbose=verbose
        )
    if normalize_g:
        exp_matrices = normalize_exps(matrices=exp_matrices, nx=nx, verbose=verbose)

    return (
        nx,
        type_as,
        new_samples,
        exp_matrices,
        spatial_coords,
        normalize_scale,
        normalize_mean_list,
    )


def shape_align_preprocess(
    coordsA,
    coordsB,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
):
    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)
    coordsA = nx.from_numpy(coordsA, type_as=type_as)
    coordsB = nx.from_numpy(coordsB, type_as=type_as)
    spatial_coordA, normalize_scale_A, normalize_mean_A = normalize_coords(coords=coordsA, nx=nx, verbose=verbose)
    spatial_coordB, normalize_scale_B, normalize_mean_B = normalize_coords(coords=coordsB, nx=nx, verbose=verbose)
    normalize_scale_list = [normalize_scale_A, normalize_scale_B]
    normalize_mean_list = [normalize_mean_A, normalize_mean_B]
    return (
        nx,
        type_as,
        spatial_coordA,
        spatial_coordB,
        normalize_scale_list,
        normalize_mean_list,
    )


######################################
# Calculate expression dissimilarity #
######################################


def kl_divergence_backend(X, Y, probabilistic=True):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    Takes advantage of POT backend to speed up computation.
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)
    if probabilistic:
        X = X / nx.sum(X, axis=1, keepdims=True)
        Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return D


def kl_distance(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
    symmetry: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the KL distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.
        symmetry (bool, optional): Whether to use symmetric KL divergence. Defaults to True.

    Returns:
        Union[np.ndarray, torch.Tensor]: KL distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    # Probabilistic normalization
    X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
    X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)
    while True:
        try:
            if chunk_num == 1:
                if symmetry:
                    DistMat = (kl_divergence_backend(X_A, X_B, False) + kl_divergence_backend(X_B, X_A, False).T) / 2
                else:
                    DistMat = kl_divergence_backend(X_A, X_B, False)
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            if symmetry:
                                arr2.append(
                                    (
                                        kl_divergence_backend(
                                            nx.from_numpy(x_As, type_as=type_as).cuda(),
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            False,
                                        ).cpu()
                                        + kl_divergence_backend(
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            nx.from_numpy(x_As, type_as=type_as).cuda(),
                                            False,
                                        )
                                        .cpu()
                                        .T
                                    )
                                    / 2
                                )
                            else:
                                arr2.append(
                                    kl_divergence_backend(
                                        nx.from_numpy(x_As, type_as=type_as).cuda(),
                                        nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                        False,
                                    ).cpu()
                                )
                        else:
                            if symmetry:
                                arr2.append(
                                    nx.to_numpy(
                                        kl_divergence_backend(
                                            nx.from_numpy(x_As, type_as=type_as),
                                            nx.from_numpy(x_Bs, type_as=type_as),
                                            False,
                                        )
                                        + kl_divergence_backend(
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            nx.from_numpy(x_As, type_as=type_as),
                                            False,
                                        ).T
                                    )
                                    / 2
                                )
                            else:
                                arr2.append(
                                    kl_divergence_backend(
                                        nx.from_numpy(x_As, type_as=type_as),
                                        nx.from_numpy(x_Bs, type_as=type_as),
                                        False,
                                    )
                                )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)
                break
        except:
            chunk_num = chunk_num * 2
            print("kl chunk more")
    if data_on_gpu and chunk_num != 1:
        DistMat = DistMat.cuda()
    return DistMat


def calc_exp_dissimilarity(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    dissimilarity: str = "kl",
    **kwargs,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate expression dissimilarity.
    Args:
        X_A: Gene expression matrix of sample A.
        X_B: Gene expression matrix of sample B.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
    """
    assert dissimilarity in [
        "kl",
        "euclidean",
        "euc",
    ], "``dissimilarity`` value is wrong. Available ``dissimilarity`` are: ``'kl'``, ``'euclidean'`` and ``'euc'``."
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        GeneDistMat = cal_dist(X_A, X_B, **kwargs)
    else:
        s_A = X_A + 0.01
        s_B = X_B + 0.01
        GeneDistMat = kl_distance(s_A, s_B, **kwargs)
    return GeneDistMat


def cal_dist(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.

    Returns:
        Union[np.ndarray, torch.Tensor]: Distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    while True:
        try:
            if chunk_num == 1:
                DistMat = ot.dist(X_A, X_B)
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as).cuda(),
                                    nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                ).cpu()
                            )
                        else:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as),
                                    nx.from_numpy(x_Bs, type_as=type_as),
                                )
                            )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)  # not convert to GPU
                break
        except:
            chunk_num = chunk_num * 2
            print("dist chunk more")
    if data_on_gpu and chunk_num != 1:
        DistMat = DistMat.cuda()
    return DistMat

    # if not use_chunk:
    #     DistMat = ot.dist(X_A,X_B)
    #     return DistMat
    # else:
    #     # convert to numpy to save the GPU memory
    #     X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
    #     # chunk
    #     X_As = np.array_split(X_A, chunk_num, axis=0)
    #     X_Bs = np.array_split(X_B, chunk_num, axis=0)
    #     arr = [] # array for temporary storage of results
    #     for x_As in X_As:
    #         arr2 = [] # array for temporary storage of results
    #         for x_Bs in X_Bs:
    #             if use_gpu:
    #                 arr2.append(ot.dist(nx.from_numpy(x_As,type_as=type_as).cuda(),nx.from_numpy(x_Bs,type_as=type_as).cuda()).cpu())
    #             else:
    #                 arr2.append(ot.dist(nx.from_numpy(x_As,type_as=type_as),nx.from_numpy(x_Bs,type_as=type_as)))
    #         arr.append(nx.concatenate(arr2,axis=1))
    #     DistMat = nx.concatenate(arr,axis=0) # not convert to GPU
    #     return DistMat


def cal_dot(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    use_chunk: bool = False,
    use_gpu: bool = True,
    chunk_num: int = 20,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the matrix multiplication of two matrices

    Args:
        mat1 (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        mat2 (Union[np.ndarray, torch.Tensor]): The second input matrix with shape d x m. We suppose m << n and does not require chunk.
        use_chunk (bool, optional): Whether to use chunk to reduce the GPU memory usage. Note that if set to ``True'' it will slow down the calculation. Defaults to False.
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.

    Returns:
        Union[np.ndarray, torch.Tensor]: Matrix multiplication result with shape n x m
    """
    nx = ot.backend.get_backend(mat1, mat2)
    type_as = mat1[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    if not use_chunk:
        Mat = _dot(nx)(mat1, mat2)
        return Mat
    else:
        # convert to numpy to save the GPU memory
        mat1 = nx.to_numpy(mat1)
        if use_gpu:
            mat2 = mat2.cuda()
        # chunk
        mat1s = np.array_split(mat1, chunk_num, axis=0)
        arr = []  # array for temporary storage of results
        for mat1ss in mat1s:
            if use_gpu:
                arr.append(_dot(nx)(nx.from_numpy(mat1ss, type_as=type_as).cuda(), mat2).cpu())
            else:
                arr.append(_dot(nx)(nx.from_numpy(mat1ss, type_as=type_as), mat2))
        Mat = nx.concatenate(arr, axis=0)
        return Mat


def PCA_reduction(
    data_mat: Union[np.ndarray, torch.Tensor],
    reduced_dim: int = 64,
    center: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor],]:
    """PCA dimensionality reduction using SVD decomposition

    Args:
        data_mat (Union[np.ndarray, torch.Tensor]): Input data matrix with shape n x k, where n is the data point number and k is the feature dimension.
        reduced_dim (int, optional): Size of dimension after dimensionality reduction. Defaults to 64.
        center (bool, optional): if True, center the input data, otherwise, assume that the input is centered. Defaults to True.

    Returns:
        projected_data (Union[np.ndarray, torch.Tensor]): Data matrix after dimensionality reduction with shape n x r.
        V_new_basis (Union[np.ndarray, torch.Tensor]): New basis with shape k x r.
        mean_data_mat (Union[np.ndarray, torch.Tensor]): The mean of the input data matrix.
    """

    nx = ot.backend.get_backend(data_mat)
    mean_data_mat = _unsqueeze(nx)(nx.mean(data_mat, axis=0), 0)
    if center:
        mean_re_data_mat = data_mat - mean_data_mat
    else:
        mean_re_data_mat = data_mat
    # SVD to perform PCA
    _, S, VH = _linalg(nx).svd(mean_re_data_mat)
    S_index = nx.argsort(-S)
    V_new_basis = VH.t()[:, S_index[:reduced_dim]]
    projected_data = nx.einsum("ij,jk->ik", mean_re_data_mat, V_new_basis)
    return projected_data, V_new_basis, mean_data_mat


def PCA_project(
    data_mat: Union[np.ndarray, torch.Tensor],
    V_new_basis: Union[np.ndarray, torch.Tensor],
    mean_data_mat: Optional[Union[np.ndarray, torch.Tensor]] = None,
    center: bool = True,
):
    nx = ot.backend.get_backend(data_mat)
    if not center:
        projected_data = nx.einsum("ij,jk->ik", data_mat, V_new_basis)
    return projected_data


def PCA_recover(
    projected_data: Union[np.ndarray, torch.Tensor],
    V_new_basis: Union[np.ndarray, torch.Tensor],
    mean_data_mat: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:

    nx = ot.backend.get_backend(projected_data)
    return nx.einsum("ij,jk->ik", projected_data, V_new_basis.t()) + mean_data_mat


def coarse_rigid_alignment(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    DistMat: Union[np.ndarray, torch.Tensor],
    nx: ot.backend.TorchBackend or ot.backend.NumpyBackend,
) -> Union[np.ndarray, torch.Tensor]:
    assert (
        coordsA.shape[0] == DistMat.shape[0]
    ), "coordsA and the first dim of DistMat do not have the same number of features."
    assert (
        coordsB.shape[0] == DistMat.shape[1]
    ), "coordsB and the second dim of DistMat do not have the same number of features."
    D = coordsA.shape[1]
    coordsA = nx.to_numpy(coordsA)
    coordsB = nx.to_numpy(coordsB)
    DistMat = nx.to_numpy(DistMat)
    nx = ot.backend.NumpyBackend()

    # construct nearest neighbor set
    minDistMat = np.min(DistMat, axis=1)
    min_threshold = minDistMat[nx.argsort(minDistMat)[max(int(0.1 * minDistMat.shape[0]), 30)]]
    nn = np.vstack((np.arange(DistMat.shape[0]), np.argsort(DistMat, axis=1)[:, 0])).T
    nn = nn[minDistMat < min_threshold, :]
    # _, unique_index = np.unique(nn[:,1],return_index=True)

    unique_values, counts = np.unique(nn[:, 1], return_counts=True)
    indices = np.where(counts < 5)[0]
    unique_index = np.where(np.isin(nn[:, 1], unique_values[indices]))[0]

    nn = nn[unique_index, :]
    minDistMat = np.min(DistMat, axis=0)
    min_threshold = minDistMat[np.argsort(minDistMat)[max(int(0.1 * minDistMat.shape[0]), 30)]]
    nn2 = np.vstack((np.argsort(DistMat, axis=0)[0, :], np.arange(DistMat.shape[1]))).T
    nn2 = nn2[minDistMat < min_threshold, :]
    # _, unique_index = np.unique(nn2[:,1],return_index=True)

    unique_values, counts = np.unique(nn2[:, 1], return_counts=True)
    indices = np.where(counts < 5)[0]
    unique_index = np.where(np.isin(nn2[:, 1], unique_values[indices]))[0]

    nn2 = nn2[unique_index, :]
    correspondence = np.vstack((nn, nn2))
    # calculate rigid transformation based on nn correspondence
    src_points = coordsA[correspondence[:, 0], :]
    dst_points = coordsB[correspondence[:, 1], :]

    # Calculate the center of gravity
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Move all points to the origin
    src_points_centered = src_points - src_centroid
    dst_points_centered = dst_points - dst_centroid

    # Calculate the rotation matrix
    H = np.dot(src_points_centered.T, dst_points_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # If the determinant of the rotation matrix is less than 0, mirroring is required
    if np.linalg.det(R) < 0:
        Vt[D - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate the translation vector
    t = dst_centroid - np.dot(R, src_centroid)

    # Constructingtransformation matrix
    T = np.identity(D + 1)
    T[:D, :D] = R
    T[:D, D] = t

    coordsA_h = np.hstack((coordsA, np.ones((len(coordsA), 1))))
    dst_points_h = np.dot(T, coordsA_h.T).T
    transformed_points = dst_points_h[:, :D]
    return transformed_points


#################################
# Funcs between Numpy and Torch #
#################################

# Empty cache
def empty_cache(device: str = "cpu"):
    if device != "cpu":
        torch.cuda.empty_cache()


# Check if nx is a torch backend
nx_torch = lambda nx: True if isinstance(nx, ot.backend.TorchBackend) else False

# Concatenate expression matrices
_cat = lambda nx, x, dim: torch.cat(x, dim=dim) if nx_torch(nx) else np.concatenate(x, axis=dim)
_unique = lambda nx, x, dim: torch.unique(x, dim=dim) if nx_torch(nx) else np.unique(x, axis=dim)
_var = lambda nx, x, dim: torch.var(x, dim=dim) if nx_torch(nx) else np.var(x, axis=dim)

_data = (
    lambda nx, data, type_as: torch.tensor(data, device=type_as.device, dtype=type_as.dtype)
    if nx_torch(nx)
    else np.asarray(data, dtype=type_as.dtype)
)
_unsqueeze = lambda nx: torch.unsqueeze if nx_torch(nx) else np.expand_dims
_mul = lambda nx: torch.multiply if nx_torch(nx) else np.multiply
_power = lambda nx: torch.pow if nx_torch(nx) else np.power
_psi = lambda nx: torch.special.psi if nx_torch(nx) else psi
_pinv = lambda nx: torch.linalg.pinv if nx_torch(nx) else pinv
_dot = lambda nx: torch.matmul if nx_torch(nx) else np.dot
_identity = (
    lambda nx, N, type_as: torch.eye(N, dtype=type_as.dtype, device=type_as.device)
    if nx_torch(nx)
    else np.identity(N, dtype=type_as.dtype)
)
_linalg = lambda nx: torch.linalg if nx_torch(nx) else np.linalg
_prod = lambda nx: torch.prod if nx_torch(nx) else np.prod
_pi = lambda nx: torch.pi if nx_torch(nx) else np.pi
_chunk = (
    lambda nx, x, chunk_num, dim: torch.chunk(x, chunk_num, dim=dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_num, axis=dim)
)
_unsqueeze = lambda nx: torch.unsqueeze if nx_torch(nx) else np.expand_dims

# _svd = (
#     lambda nx, x, full_matrices=True, compute_uv=True: torch.svd(
#         x, some=full_matrices, compute_uv=compute_uv
#     )
#     if nx_torch(nx)
#     else np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
# )
