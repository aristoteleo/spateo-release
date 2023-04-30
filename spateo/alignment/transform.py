from typing import List, Optional, Union

import numpy as np
import torch
from anndata import AnnData

from .methods import cal_dist, cal_dot
from .methods.mropho import con_K
from .methods.utils import (
    _data,
    _dot,
    _mul,
    _pi,
    _power,
    _prod,
    _unsqueeze,
    calc_exp_dissimilarity,
    check_backend,
    check_exp,
    filter_common_genes,
    intersect_lsts,
)


def paste_transform(
    adata: AnnData,
    adata_ref: AnnData,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key: str = "models_align",
) -> AnnData:
    """
    Align the space coordinates of the new model with the transformation matrix obtained from PASTE.

    Args:
        adata: The anndata object that need to be aligned.
        adata_ref: The anndata object that have been aligned by PASTE.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinates.
        mapping_key: The key in `.uns` that corresponds to the alignment info from PASTE.

    Returns:
        adata: The anndata object that have been to be aligned.
    """

    assert mapping_key in adata_ref.uns_keys(), "`mapping_key` value is wrong."
    tX = adata_ref.uns[mapping_key]["tX"]
    tY = adata_ref.uns[mapping_key]["tY"]
    R = adata_ref.uns[mapping_key]["R"]

    adata_coords = adata.obsm[spatial_key].copy() - tY
    adata.obsm[key_added] = R.dot(adata_coords.T).T + tX
    return adata


def BA_transform(
    vecfld,
    quary_points,
    deformation_scale: int = 1,
    dtype: str = "float64",
    device: str = "cpu",
):
    """Apply non-rigid transform to the quary points

    Args:
        vecfld: A dictionary containing information about vector fields
        quary_points:
        deformation_scale: If deformation_scale is greater than 1, increase the degree of deformation.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
    """
    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)
    normalize_scale = _data(nx, vecfld["normalize_scale"], type_as)
    normalize_mean_ref = _data(nx, vecfld["normalize_mean_list"][0], type_as)
    normalize_mean_quary = _data(nx, vecfld["normalize_mean_list"][1], type_as)
    XA = _data(nx, quary_points, type_as)

    # normalize coordinate
    if vecfld["normalize_c"]:
        XA = (XA - normalize_mean_quary) / normalize_scale
    ctrl_pts = _data(nx, vecfld["ctrl_pts"], type_as)
    Coff = _data(nx, vecfld["Coff"], type_as)
    s = _data(nx, vecfld["s"], type_as)
    R = _data(nx, vecfld["R"], type_as)
    t = _data(nx, vecfld["t"], type_as)
    optimal_R = _data(nx, vecfld["optimal_R"], type_as)
    optimal_t = _data(nx, vecfld["optimal_t"], type_as)
    init_R = _data(nx, vecfld["init_R"], type_as)
    init_t = _data(nx, vecfld["init_t"], type_as)
    XA = _dot(nx)(XA, init_R.T) + init_t

    beta = vecfld["beta"]
    quary_kernel = con_K(XA, ctrl_pts, beta)
    quary_velocities = _dot(nx)(quary_kernel, Coff) * deformation_scale
    quary_similarity = s * _dot(nx)(XA, R.T) + t
    quary_optimal_similarity = _dot(nx)(XA, optimal_R.T) + optimal_t
    XAHat = quary_velocities + quary_similarity

    if vecfld["normalize_c"]:
        XAHat = XAHat * normalize_scale + normalize_mean_ref
        quary_velocities = quary_velocities * normalize_scale
        quary_optimal_similarity = quary_optimal_similarity * normalize_scale + normalize_mean_ref

    XAHat = nx.to_numpy(XAHat)
    quary_velocities = nx.to_numpy(quary_velocities)
    quary_optimal_similarity = nx.to_numpy(quary_optimal_similarity)
    return XAHat, quary_velocities, quary_optimal_similarity


def BA_transform_and_assignment(
    samples,
    vecfld,
    layer: str = "X",
    genes: Optional[Union[List, torch.Tensor]] = None,
    spatial_key: str = "spatial",
    small_variance: bool = True,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = False,
):
    # Use the CPU to prevent insufficient GPU memory
    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)
    type_as_cpu = type_as.cpu()
    normalize_scale = vecfld["normalize_scale"]
    normalize_mean_ref = vecfld["normalize_mean_list"][0]
    normalize_mean_quary = vecfld["normalize_mean_list"][1]
    XA = samples[0].obsm[spatial_key]
    XB = samples[1].obsm[spatial_key]

    # normalize coordinate
    if vecfld["normalize_c"]:
        XA = (XA - normalize_mean_quary) / normalize_scale
        XB = (XB - normalize_mean_ref) / normalize_scale
    ctrl_pts = vecfld["ctrl_pts"]
    Coff = vecfld["Coff"]
    s = vecfld["s"]
    R = vecfld["R"]
    t = vecfld["t"]
    beta = vecfld["beta"]
    quary_kernel = con_K(XA, ctrl_pts, beta, True)
    quary_velocities = cal_dot(quary_kernel, Coff, use_chunk=True)
    quary_similarity = s * cal_dot(XA, R.T, use_chunk=True) + t
    XAHat = quary_velocities + quary_similarity
    XAHat = nx.from_numpy(XAHat)

    XB = nx.from_numpy(XB)
    new_samples = [s.copy() for s in samples]
    all_samples_genes = [s[0].var.index for s in new_samples]
    common_genes = filter_common_genes(*all_samples_genes, verbose=verbose)
    common_genes = common_genes if genes is None else intersect_lsts(common_genes, genes)
    new_samples = [s[:, common_genes] for s in new_samples]

    # Gene expression matrix of all samples
    exp_matrices = [nx.from_numpy(check_exp(sample=s, layer=layer), type_as=type_as_cpu) for s in new_samples]
    X_A, X_B = exp_matrices[0], exp_matrices[1]
    GeneDistMat = calc_exp_dissimilarity(
        X_A=X_A,
        X_B=X_B,
        dissimilarity=vecfld["dissimilarity"],
        use_chunk=True,
        chunk_num=20,
    )
    NA = vecfld["NA"]
    D = XA.shape[1]
    beta2 = vecfld["beta2"]
    if small_variance:
        sigma2 = _data(nx, 0.01, type_as_cpu)
    else:
        sigma2 = nx.from_numpy(vecfld["sigma2"])
    outlier_g = nx.from_numpy(vecfld["outlier_g"])
    gamma = nx.from_numpy(vecfld["gamma"])
    alpha = nx.ones((XA.shape[0]), type_as=type_as_cpu)
    Sigma = nx.zeros((XA.shape[0]), type_as=type_as_cpu)
    SpatialDistMat = cal_dist(
        XAHat,
        XB,
        use_chunk=True,
    )
    if not vecfld.__contains__("samples_s"):
        samples_s = nx.maximum(
            _prod(nx)(nx.max(XAHat, axis=0) - nx.min(XAHat, axis=0)),
            _prod(nx)(nx.max(XB, axis=0) - nx.min(XB, axis=0)),
        )
    outlier_s = samples_s * NA
    term1 = nx.einsum(
        "ij,i->ij",
        _mul(nx)(nx.exp(-SpatialDistMat / (2 * sigma2)), nx.exp(-GeneDistMat / (2 * beta2))),
        (_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
    )
    term2 = _power(nx)((2 * _pi(nx) * sigma2), _data(nx, D / 2, type_as_cpu)) * (1 - gamma) / (
        gamma * outlier_s * outlier_g
    ) + nx.einsum("ij->j", term1)
    P = term1 / _unsqueeze(nx)(term2, 0)
    P = nx.to_numpy(P)
    P = P / np.max(P, axis=0)[np.newaxis, :]
    if vecfld["normalize_c"]:
        XAHat = XAHat * normalize_scale + normalize_mean_ref
    XAHat = nx.to_numpy(XAHat)
    return P.T, XAHat
