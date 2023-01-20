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


######################################
# Calculate expression dissimilarity #
######################################


def kl_divergence_backend(X, Y):
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

    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return D


# def calc_exp_dissimilarity(
#     X_A: Union[np.ndarray, torch.Tensor],
#     X_B: Union[np.ndarray, torch.Tensor],
#     dissimilarity: str = "kl",
# ) -> Union[np.ndarray, torch.Tensor]:
#     """
#     Calculate expression dissimilarity.

#     Args:
#         X_A: Gene expression matrix of sample A.
#         X_B: Gene expression matrix of sample B.
#         dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
#     """
#     assert dissimilarity in [
#         "kl",
#         "euclidean",
#         "euc",
#     ], "``dissimilarity`` value is wrong. Available ``dissimilarity`` are: ``'kl'``, ``'euclidean'`` and ``'euc'``."
#     if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
#         GeneDistMat = ot.dist(X_A, X_B)
#     else:
#         s_A = X_A + 0.01
#         s_B = X_B + 0.01
#         GeneDistMat = (
#             kl_divergence_backend(s_A, s_B) + kl_divergence_backend(s_B, s_A).T
#         ) / 2

#     return GeneDistMat


def calc_exp_dissimilarity(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    dissimilarity: str = "kl",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate expression dissimilarity.

    Args:
        X_A: Gene expression matrix of sample A.
        X_B: Gene expression matrix of sample B.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    assert dissimilarity in [
        "kl",
        "euclidean",
        "euc",
    ], "``dissimilarity`` value is wrong. Available ``dissimilarity`` are: ``'kl'``, ``'euclidean'`` and ``'euc'``."
    NA, NB = X_A.shape[0], X_B.shape[0]
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        GeneDistMat = ot.dist(X_A, X_B)
    else:
        s_A = X_A + 0.01
        s_B = X_B + 0.01
        limit = 1e6
        if NA * NB < limit:
            GeneDistMat = (kl_divergence_backend(s_A, s_B) + kl_divergence_backend(s_B, s_A).T) / 2
        else:
            cur_ind = 0

            batch = int(limit / NB)
            GeneDistMat_A = []
            while cur_ind < NA:
                if cur_ind + batch < NA:
                    GeneDistMat_A.append(kl_divergence_backend(s_A[cur_ind : cur_ind + batch, :], s_B))
                else:
                    GeneDistMat_A.append(kl_divergence_backend(s_A[cur_ind:, :], s_B))
                cur_ind = cur_ind + batch
            GeneDistMat_A = nx.to_numpy(nx.concatenate(GeneDistMat_A, 0))

            GeneDistMat_B = []
            cur_ind = 0
            while cur_ind < NA:
                if cur_ind + batch < NA:
                    GeneDistMat_B.append(kl_divergence_backend(s_B, s_A[cur_ind : cur_ind + batch, :]).T)
                else:
                    GeneDistMat_B.append(kl_divergence_backend(s_B, s_A[cur_ind:, :]).T)
                cur_ind = cur_ind + batch
            GeneDistMat_B = nx.to_numpy(nx.concatenate(GeneDistMat_B, 0))
            GeneDistMat = (GeneDistMat_A + GeneDistMat_B) / 2
            del GeneDistMat_A
            del GeneDistMat_B
    return _data(nx, GeneDistMat, X_A)


#######################################
# Mapping aligned spatial coordinates #
#######################################


def get_optimal_mapping_connections(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    keep_all: bool = False,
):
    from scipy.spatial import cKDTree

    X_max_index = np.argwhere((pi.T == pi.T.max(axis=0)).T)
    Y_max_index = np.argwhere(pi == pi.max(axis=0))
    if not keep_all:

        values, counts = np.unique(X_max_index[:, 0], return_counts=True)
        x_index_unique, x_index_repeat = values[counts == 1], values[counts != 1]
        X_max_index_unique = X_max_index[np.isin(X_max_index[:, 0], x_index_unique)]

        for i in x_index_repeat:
            i_max_index = X_max_index[X_max_index[:, 0] == i]
            i_kdtree = cKDTree(Y[i_max_index[:, 1]])
            _, ii = i_kdtree.query(X[i], k=1)
            X_max_index_unique = np.concatenate([X_max_index_unique, i_max_index[ii].reshape(1, 2)], axis=0)

        values, counts = np.unique(Y_max_index[:, 1], return_counts=True)
        y_index_unique, y_index_repeat = values[counts == 1], values[counts != 1]
        Y_max_index_unique = Y_max_index[np.isin(Y_max_index[:, 1], y_index_unique)]

        for i in y_index_repeat:
            i_max_index = Y_max_index[Y_max_index[:, 1] == i]
            i_kdtree = cKDTree(X[i_max_index[:, 0]])
            _, ii = i_kdtree.query(Y[i], k=1)
            Y_max_index_unique = np.concatenate([Y_max_index_unique, i_max_index[ii].reshape(1, 2)], axis=0)

        X_max_index = X_max_index_unique.copy()
        Y_max_index = Y_max_index_unique.copy()

    X_pi_value = pi[X_max_index[:, 0], X_max_index[:, 1]].reshape(-1, 1)
    Y_pi_value = pi[Y_max_index[:, 0], Y_max_index[:, 1]].reshape(-1, 1)
    return X_max_index, X_pi_value, Y_max_index, Y_pi_value


def mapping_aligned_coords(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    keep_all: bool = False,
) -> Tuple[dict, dict]:
    """
    Optimal mapping coordinates between X and Y.

    Args:
        X: Aligned spatial coordinates.
        Y: Aligned spatial coordinates.
        pi: Mapping between the two layers output by PASTE.
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.

    Returns:
        Two dicts of mapping_X, mapping_Y, pi_index, pi_value.
            mapping_X is X coordinates aligned with Y coordinates.
            mapping_Y is the Y coordinate aligned with X coordinates.
            pi_index is index between optimal mapping points in the pi matrix.
            pi_value is the value of optimal mapping points.
    """

    X = X.copy()
    Y = Y.copy()
    pi = pi.copy()

    # Obtain the optimal mapping between points
    (
        X_max_index,
        X_pi_value,
        Y_max_index,
        Y_pi_value,
    ) = get_optimal_mapping_connections(X=X, Y=Y, pi=pi, keep_all=keep_all)

    mappings = []
    for max_index, pi_value, subset in zip(
        [X_max_index, Y_max_index], [X_pi_value, Y_pi_value], ["index_x", "index_y"]
    ):
        mapping_data = pd.DataFrame(
            np.concatenate([max_index, pi_value], axis=1),
            columns=["index_x", "index_y", "pi_value"],
        ).astype(
            dtype={
                "index_x": np.int32,
                "index_y": np.int32,
                "pi_value": np.float64,
            }
        )
        mapping_data.sort_values(by=[subset, "pi_value"], ascending=[True, False], inplace=True)
        mapping_data.drop_duplicates(subset=[subset], keep="first", inplace=True)
        mappings.append(
            {
                "mapping_X": X[mapping_data["index_x"].values],
                "mapping_Y": Y[mapping_data["index_y"].values],
                "pi_index": mapping_data[["index_x", "index_y"]].values,
                "pi_value": mapping_data["pi_value"].values,
            }
        )

    return mappings[0], mappings[1]


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
