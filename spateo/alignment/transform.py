from typing import List, Optional, Union

import numpy as np
import ot
import torch
from anndata import AnnData

# from .methods.morpho import con_K
from .methods import (
    _chunk,
    _data,
    _dot,
    _mul,
    _pi,
    _power,
    _prod,
    _unsqueeze,
    cal_dist,
    cal_dot,
    calc_exp_dissimilarity,
    check_backend,
    check_exp,
    con_K,
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
    adata.obsm[key_added] = R.dot(adata_coords.T).T
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
    quary_similarity = _dot(nx)(XA, R.T) + t
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
    small_variance: bool = False,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = False,
):
    # Use the CPU to prevent insufficient GPU memory
    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)
    if device != "cpu":
        type_as_cpu = type_as.cpu()
    else:
        type_as_cpu = type_as
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
    R = vecfld["R"]
    t = vecfld["t"]
    optimal_R = vecfld["optimal_R"]
    optimal_t = vecfld["optimal_t"]
    init_R = vecfld["init_R"]
    init_t = vecfld["init_t"]
    XA = cal_dot(XA, init_R.T, use_chunk=True) + init_t

    beta = vecfld["beta"]
    quary_kernel = con_K(XA, ctrl_pts, beta, True)
    quary_velocities = cal_dot(quary_kernel, Coff, use_chunk=True)
    quary_similarity = cal_dot(XA, R.T, use_chunk=True) + t
    quary_optimal_similarity = cal_dot(XA, optimal_R.T, use_chunk=True) + optimal_t
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

    beta2 = vecfld["beta2"]
    outlier_variance = nx.from_numpy(vecfld["outlier_variance"])
    if small_variance:
        sigma2 = _data(nx, 0.01, type_as_cpu)
    else:
        sigma2 = nx.from_numpy(vecfld["sigma2"])
    gamma = nx.from_numpy(vecfld["gamma"])
    alpha = nx.ones((XA.shape[0]), type_as=type_as_cpu)
    SigmaDiag = nx.zeros((XA.shape[0]), type_as=type_as_cpu)
    P = get_P_chunk(
        XnAHat=XAHat,
        XnB=XB,
        X_A=X_A,
        X_B=X_B,
        sigma2=sigma2,
        beta2=beta2,
        alpha=alpha,
        gamma=gamma,
        Sigma=SigmaDiag,
        outlier_variance=outlier_variance,
    )

    if vecfld["normalize_c"]:
        XAHat = XAHat * normalize_scale + normalize_mean_ref
        quary_velocities = quary_velocities * normalize_scale
        quary_optimal_similarity = quary_optimal_similarity * normalize_scale + normalize_mean_ref
    XAHat = nx.to_numpy(XAHat)
    return XAHat, quary_velocities, quary_optimal_similarity, P.T


def get_P_chunk(
    XnAHat: Union[np.ndarray, torch.Tensor],
    XnB: Union[np.ndarray, torch.Tensor],
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    beta2: Union[int, float, np.ndarray, torch.Tensor],
    alpha: Union[np.ndarray, torch.Tensor],
    gamma: Union[float, np.ndarray, torch.Tensor],
    Sigma: Union[np.ndarray, torch.Tensor],
    samples_s: Optional[List[float]] = None,
    outlier_variance: float = None,
    chunk_size: int = 1000,
    dissimilarity: str = "kl",
) -> Union[np.ndarray, torch.Tensor]:
    """Calculating the generating probability matrix P.

    Args:
        XAHat: Current spatial coordinate of sample A. Shape
    """
    # Get the number of cells in each sample
    NA, NB = XnAHat.shape[0], XnB.shape[0]
    # Get the number of genes
    G = X_A.shape[1]
    # Get the number of spatial dimensions
    D = XnAHat.shape[1]
    chunk_num = int(np.ceil(NA / chunk_size))

    assert XnAHat.shape[1] == XnB.shape[1], "XnAHat and XnB do not have the same number of features."
    assert XnAHat.shape[0] == alpha.shape[0], "XnAHat and alpha do not have the same length."
    assert XnAHat.shape[0] == Sigma.shape[0], "XnAHat and Sigma do not have the same length."

    nx = ot.backend.get_backend(XnAHat, XnB)
    if samples_s is None:
        samples_s = nx.maximum(
            _prod(nx)(nx.max(XnAHat, axis=0) - nx.min(XnAHat, axis=0)),
            _prod(nx)(nx.max(XnB, axis=0) - nx.min(XnB, axis=0)),
        )
    outlier_s = samples_s * NA
    # chunk
    X_Bs = _chunk(nx, X_B, chunk_num, dim=0)
    XnBs = _chunk(nx, XnB, chunk_num, dim=0)

    Ps = []
    for x_Bs, xnBs in zip(X_Bs, XnBs):
        SpatialDistMat = cal_dist(XnAHat, xnBs)
        GeneDistMat = calc_exp_dissimilarity(X_A=X_A, X_B=x_Bs, dissimilarity=dissimilarity)
        if outlier_variance is None:
            exp_SpatialMat = nx.exp(-SpatialDistMat / (2 * sigma2))
        else:
            exp_SpatialMat = nx.exp(-SpatialDistMat / (2 * sigma2 / outlier_variance))
        spatial_term1 = nx.einsum(
            "ij,i->ij",
            exp_SpatialMat,
            (_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
        )
        spatial_outlier = (
            _power(nx)((2 * _pi(nx) * sigma2), _data(nx, D / 2, XnAHat)) * (1 - gamma) / (gamma * outlier_s)
        )
        spatial_inlier = 1 - spatial_outlier / (spatial_outlier + nx.einsum("ij->j", exp_SpatialMat))
        term1 = nx.einsum(
            "ij,i->ij",
            _mul(nx)(nx.exp(-SpatialDistMat / (2 * sigma2)), nx.exp(-GeneDistMat / (2 * beta2))),
            (_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
        )
        P = term1 / (_unsqueeze(nx)(nx.einsum("ij->j", term1), 0) + 1e-8)
        P = nx.einsum("j,ij->ij", spatial_inlier, P)
        Ps.append(nx.to_numpy(P))
    P = np.concatenate(Ps, axis=1)
    return P
