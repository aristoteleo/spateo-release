from typing import Optional, Union

import numpy as np
import ot
import pandas as pd
import torch
from scipy.sparse import coo_array
from torch import sparse_coo_tensor as SparseTensor

from .utils import _data, _identity, _linalg, _unsqueeze


def calc_distance(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    metric: str = "euc",
    # chunk_num: int = 1,
    batch_capacity: int = 1,
    use_sparse: bool = False,
    sparse_method: str = "topk",
    threshold: Union[int, float] = 100,
    return_mask: bool = False,
    save_to_cpu: bool = False,
    **kwargs,
):
    assert metric in [
        "euc",
        "euclidean",
        "square_euc",
        "square_euclidean",
        "kl",
    ], "``metric`` value is wrong. Available ``metric`` are: ``'euc'``, ``'euclidean'``, ``'square_euc'``, ``'square_euclidean'``, and ``'kl'``."

    if use_sparse:
        assert sparse_method in [
            "topk",
            "threshold",
        ], "``sparse_method`` value is wrong. Available ``metric`` are: ``'topk'`` and ``'threshold'``."
        if sparse_method == "topk":
            threshold = int(threshold)

    NA, NB = X_A.shape[0], X_B.shape[0]
    D = X_A.shape[1]
    batch_base = 1e9

    split_size = min(int(batch_capacity * batch_base / (NB * D)), NA)
    split_size = 1 if split_size == 0 else split_size

    nx = ot.backend.get_backend(X_A, X_B)

    if metric.lower() == "kl":
        X_A = X_A + 0.01
        X_B = X_B + 0.01
        X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
        X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)

    # only chunk X_A
    X_A_chunks = _split(nx, X_A, split_size, dim=0)
    if use_sparse:
        rows, cols, vals = [], [], []
        cur_row = 0
        for X_A_chunk in X_A_chunks:
            DistMat = _dist(X_A_chunk, X_B, metric)
            if sparse_method == "topk":
                sorted_DistMat, sorted_idx = nx.sort2(DistMat, axis=1)
                row = _repeat_interleave(nx, nx.arange(X_A_chunk.shape[0], type_as=X_A), threshold, axis=0) + cur_row
                col = sorted_idx[:, :threshold].reshape(-1)
                val = sorted_DistMat[:, :threshold].reshape(-1)
            else:
                row, col = _where(nx, DistMat < threshold)
                val = DistMat[row, col]
                row += cur_row
            rows.append(row)
            cols.append(col)
            vals.append(val)
            cur_row += X_A_chunk.shape[0]
        rows = _cat(nx, rows, dim=0)
        cols = _cat(nx, cols, dim=0)
        vals = _cat(nx, vals, dim=0)
        DistMat = _SparseTensor(nx=nx, row=rows, col=cols, value=vals, sparse_sizes=(NA, NB))
        if return_mask:
            vals = nx.ones((vals.shape[0],), type_as=X_A)
            DistMask = _SparseTensor(nx=nx, row=rows, col=cols, value=vals, sparse_sizes=(NA, NB))
            return DistMat, DistMask
        else:
            return DistMat
    else:
        DistMats = [_dist(X_A_chunk, X_B, metric) for X_A_chunk in X_A_chunks]
        DistMat = nx.concatenate(DistMats, axis=0)
        return DistMat


def calc_P_related(
    XnAHat: Union[np.ndarray, torch.Tensor],
    XnB: Union[np.ndarray, torch.Tensor],
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    sigma2_robust: Union[int, float, np.ndarray, torch.Tensor],
    beta2: Union[int, float, np.ndarray, torch.Tensor],
    spatial_outlier,
    col_mul=None,
    batch_capacity: int = 1,
    dissimilarity: str = "kl",
    labelA: Optional[pd.Series] = None,
    labelB: Optional[pd.Series] = None,
    label_transfer_prior: Optional[dict] = None,
    top_k: int = 1024,
):
    # metric = 'square_euc'
    NA, NB = XnAHat.shape[0], XnB.shape[0]
    D = XnAHat.shape[1]
    batch_base = 1e7
    split_size = min(int(batch_capacity * batch_base / (NA * D)), NB)
    split_size = 1 if split_size == 0 else split_size

    nx = ot.backend.get_backend(XnAHat, XnB)

    # only chunk XnB and X_B (data points)
    XnB_chunks = _split(nx, XnB, split_size, dim=0)
    X_B_chunks = _split(nx, X_B, split_size, dim=0)

    label_mask = _construct_label_mask(labelA, labelB, label_transfer_prior).T

    mask_chunks_arr = _split(nx, nx.arange(NB), split_size, dim=0)

    K_NA_spatial = nx.zeros((NA,), type_as=XnAHat)
    K_NA_sigma2 = nx.zeros((NA,), type_as=XnAHat)
    Ps = []
    sigma2_temp = 0
    cur_row = 0
    for XnB_chunk, X_B_chunk, mask_chunk_arr in zip(XnB_chunks, X_B_chunks, mask_chunks_arr):
        label_mask_chunk = (
            None if labelA is None else _data(nx, label_mask[:, np.array(mask_chunk_arr)], type_as=XnB_chunk)
        )

        # calculate distance matrix (common step)
        SpatialMat = _dist(XnAHat, XnB_chunk, "square_euc")
        # calculate spatial_P and keep K_NA_spatials
        exp_SpatialMat = torch.exp(-SpatialMat / (2 * sigma2_robust))
        spatial_term1 = exp_SpatialMat * col_mul.unsqueeze(-1)
        spatial_term1 = spatial_term1 * label_mask_chunk if label_mask_chunk is not None else spatial_term1
        spatial_term2 = spatial_outlier + spatial_term1.sum(0)
        spatial_P = spatial_term1 / _unsqueeze(nx)(spatial_term2, 0)
        K_NA_spatial += spatial_P.sum(1)
        del spatial_P

        exp_SpatialMat = torch.exp(-SpatialMat / (2 * sigma2))
        spatial_inlier = 1 - spatial_outlier / (spatial_outlier + exp_SpatialMat.sum(0))
        # calculate sigma2_P
        term1 = exp_SpatialMat * col_mul.unsqueeze(-1)
        term1 = term1 * label_mask_chunk if label_mask_chunk is not None else term1
        sigma2_P = term1 / (_unsqueeze(nx)(term1.sum(0), 0) + 1e-8)
        sigma2_P = sigma2_P * spatial_inlier.unsqueeze(0)
        K_NA_sigma2 += sigma2_P.sum(1)
        sigma2_temp += (sigma2_P * SpatialMat).sum()
        del sigma2_P

        # calculate P
        GeneDistMat = _dist(X_A, X_B_chunk, metric=dissimilarity)
        exp_GeneMat = torch.exp(-GeneDistMat / (2 * beta2))
        term1 = exp_GeneMat * term1
        P = term1 / (_unsqueeze(nx)(term1.sum(0), 0) + 1e-8)
        P = P * spatial_inlier.unsqueeze(0)

        P = _dense_to_sparse(
            mat=P,
            sparse_method="topk",
            threshold=top_k,
            axis=0,
            descending=True,
        )

        Ps.append(P)

    P = torch.cat(Ps, axis=1)
    del Ps
    Sp_sigma2 = K_NA_sigma2.sum()
    sigma2_temp = sigma2_temp / (D * Sp_sigma2)
    return K_NA_spatial, K_NA_sigma2, P, sigma2_temp


def get_optimal_R_sparse(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    P: Union[np.ndarray, torch.Tensor, SparseTensor],
    R_init: Union[np.ndarray, torch.Tensor],
):
    """Get the optimal rotation matrix R

    Args:
        coordsA (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        coordsB (Union[np.ndarray, torch.Tensor]): The second input matrix with shape n x d
        P (Union[np.ndarray, torch.Tensor]): The optimal transport matrix with shape n x n

    Returns:
        Union[np.ndarray, torch.Tensor]: The optimal rotation matrix R with shape d x d
    """
    nx = ot.backend.get_backend(coordsA, coordsB, R_init)
    NA, NB, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]
    Sp = P.sum()
    K_NA = P.sum(1).to_dense()
    K_NB = P.sum(0).to_dense()
    VnA = nx.zeros(coordsA.shape, type_as=coordsA[0, 0])
    mu_XnA, mu_VnA, mu_XnB = (
        _dot(nx)(K_NA, coordsA) / Sp,
        _dot(nx)(K_NA, VnA) / Sp,
        _dot(nx)(K_NB, coordsB) / Sp,
    )
    XnABar, VnABar, XnBBar = coordsA - mu_XnA, VnA - mu_VnA, coordsB - mu_XnB
    A = -_dot(nx)(nx.einsum("ij,i->ij", VnABar, K_NA).T - _dot(nx)(P, XnBBar).T, XnABar)

    # get the optimal rotation matrix R
    svdU, svdS, svdV = _linalg(nx).svd(A)
    C = _identity(nx, D, type_as=coordsA[0, 0])
    C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
    R = _dot(nx)(_dot(nx)(svdU, C), svdV)
    t = mu_XnB - mu_VnA - _dot(nx)(mu_XnA, R.T)
    optimal_RnA = _dot(nx)(coordsA, R.T) + t
    return optimal_RnA, R, t


def _init_guess_sigma2(
    XA,
    XB,
    subsample=2000,
):
    NA, NB, D = XA.shape[0], XB.shape[0], XA.shape[1]
    sub_sample_A = np.random.choice(NA, subsample, replace=False) if NA > subsample else np.arange(NA)
    sub_sample_B = np.random.choice(NB, subsample, replace=False) if NB > subsample else np.arange(NB)
    SpatialDistMat = calc_distance(
        X_A=XA[sub_sample_A, :],
        X_B=XB[sub_sample_B, :],
        metric="square_euc",
        use_sparse=False,
    )
    sigma2 = 2 * SpatialDistMat.sum() / (D * sub_sample_A.shape[0] * sub_sample_A.shape[0])  # 2 for 3D
    return sigma2


def _init_guess_beta2(
    nx,
    XA,
    XB,
    dissimilarity="kl",
    partial_robust_level=1,
    beta2=None,
    beta2_end=None,
    subsample=2000,
):
    NA, NB, D = XA.shape[0], XB.shape[0], XA.shape[1]
    sub_sample_A = np.random.choice(NA, subsample, replace=False) if NA > subsample else np.arange(NA)
    sub_sample_B = np.random.choice(NB, subsample, replace=False) if NB > subsample else np.arange(NB)
    GeneDistMat = calc_distance(
        X_A=XA[sub_sample_A, :],
        X_B=XB[sub_sample_B, :],
        metric=dissimilarity,
        use_sparse=False,
    )
    minGeneDistMat = nx.min(GeneDistMat, 1)
    if beta2 is None:
        beta2 = minGeneDistMat[nx.argsort(minGeneDistMat)[int(sub_sample_A.shape[0] * 0.05)]] / 5
    else:
        beta2 = _data(nx, beta2, XA)

    if beta2_end is None:
        beta2_end = nx.max(minGeneDistMat) / nx.sqrt(_data(nx, partial_robust_level, XA))
    else:
        beta2_end = _data(nx, beta2_end, XA)
    beta2 = nx.maximum(beta2, _data(nx, 1e-2, XA))
    print("beta2: {} --> {}".format(beta2, beta2_end))
    return beta2, beta2_end


def _construct_label_mask(labelA, labelB, label_transfer_prior):
    label_mask = np.zeros((labelB.shape[0], labelA.shape[0]))
    for k in label_transfer_prior.keys():
        idx = np.where((labelB == k))[0]
        cur_P = labelA.map(label_transfer_prior[k]).values
        label_mask[idx, :] = cur_P
    return label_mask


## Sparse operation
def _dense_to_sparse(
    mat: Union[np.ndarray, torch.Tensor],
    sparse_method: str = "topk",
    threshold: Union[int, float] = 100,
    axis: int = 0,
    descending=False,
):
    assert sparse_method in [
        "topk",
        "threshold",
    ], "``sparse_method`` value is wrong. Available ``sparse_method`` are: ``'topk'`` and ``'threshold'``."
    threshold = int(threshold) if sparse_method == "topk" else threshold
    nx = ot.backend.get_backend(mat)

    NA, NB = mat.shape[0], mat.shape[1]

    if sparse_method == "topk":
        sorted_mat, sorted_idx = _sort(nx, mat, axis=axis, descending=descending)
        if axis == 0:
            col = _repeat_interleave(nx, nx.arange(NB, type_as=mat), threshold, axis=0)
            row = sorted_idx[:threshold, :].T.reshape(-1)
            val = sorted_mat[:threshold, :].T.reshape(-1)
        elif axis == 1:
            col = sorted_idx[:, :threshold].reshape(-1)
            row = _repeat_interleave(nx, nx.arange(NA, type_as=mat), threshold, axis=0)
            val = sorted_mat[:, :threshold].reshape(-1)
    elif sparse_method == "threshold":
        row, col = _where(nx, DistMat < threshold)
        val = DistMat[row, col]

    results = _SparseTensor(nx=nx, row=row, col=col, value=val, sparse_sizes=(NA, NB))
    return results


def _SparseTensor(nx, row, col, value, sparse_sizes):
    if nx_torch(nx):
        return SparseTensor(indices=torch.vstack((row, col)), values=value, size=sparse_sizes)
    else:
        return coo_array((value, (row, col)), shape=sparse_sizes)


def _dist(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    metric: str = "euc",
) -> Union[np.ndarray, torch.Tensor]:
    assert metric in [
        "euc",
        "euclidean",
        "square_euc",
        "square_euclidean",
        "kl",
    ], "``metric`` value is wrong. Available ``metric`` are: ``'euc'``, ``'euclidean'``, ``'square_euc'``, ``'square_euclidean'``, and ``'kl'``."
    nx = ot.backend.get_backend(mat1, mat2)
    if (
        metric.lower() == "euc"
        or metric.lower() == "euclidean"
        or metric.lower() == "square_euc"
        or metric.lower() == "square_euclidean"
    ):
        distMat = nx.sum(mat1**2, 1)[:, None] + nx.sum(mat2**2, 1)[None, :] - 2 * _dot(nx)(mat1, mat2.T)
        if metric.lower() == "euc" or metric.lower() == "euclidean":
            distMat = nx.sqrt(distMat)
    elif metric.lower() == "kl":
        if mat1.min() == 0:
            mat1 = mat1 + 0.01
            mat2 = mat2 + 0.01
            mat1 = mat1 / nx.sum(mat1, 1)[:, None]
            mat2 = mat2 / nx.sum(mat2, 1)[:, None]
        distMat = (
            nx.sum(mat1 * nx.log(mat1), 1)[:, None]
            + nx.sum(mat2 * nx.log(mat2), 1)[None, :]
            - _dot(nx)(mat1, nx.log(mat2).T)
            - _dot(nx)(mat2, nx.log(mat1).T).T
        ) / 2
    return distMat


# Check if nx is a torch backend
nx_torch = lambda nx: True if isinstance(nx, ot.backend.TorchBackend) else False
_cat = lambda nx, x, dim: torch.cat(x, dim=dim) if nx_torch(nx) else np.concatenate(x, axis=dim)

_dot = lambda nx: torch.matmul if nx_torch(nx) else np.dot
_split = (
    lambda nx, x, chunk_size, dim: torch.split(x, chunk_size, dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_size, axis=dim)
)
_where = lambda nx, condition: torch.where(condition) if nx_torch(nx) else np.where(condition)
_repeat_interleave = (
    lambda nx, x, repeats, axis: torch.repeat_interleave(x, repeats, dim=axis)
    if nx_torch(nx)
    else np.repeat(x, repeats, axis)
)


def _sort(nx, arr, axis=-1, descending=False):
    if not descending:
        sorted_arr, sorted_idx = nx.sort2(arr, axis=axis)
    else:
        sorted_arr, sorted_idx = nx.sort2(-arr, axis=axis)
        sorted_arr = -sorted_arr
    return sorted_arr, sorted_idx
