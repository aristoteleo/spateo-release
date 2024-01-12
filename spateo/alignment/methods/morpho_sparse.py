import random

import numpy as np
import ot
import torch
from anndata import AnnData

try:
    from typing import Any, Dict, List, Literal, Optional, Tuple, Union
except ImportError:
    from typing_extensions import Literal

from typing import List, Optional, Tuple, Union

import pandas as pd

from spateo.logging import logger_manager as lm

from .morpho_sparse_utils import (
    _init_guess_beta2,
    _init_guess_sigma2,
    calc_distance,
    calc_P_related,
    get_optimal_R_sparse,
)
from .utils import (
    _data,
    _dot,
    _identity,
    _linalg,
    _mul,
    _pi,
    _pinv,
    _power,
    _prod,
    _psi,
    _randperm,
    _roll,
    _unique,
    align_preprocess,
    cal_dist,
    coarse_rigid_alignment,
    empty_cache,
)


# construct kernel
def con_K(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    beta: Union[int, float] = 0.01,
    use_chunk: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """con_K constructs the Squared Exponential (SE) kernel, where K(i,j)=k(X_i,Y_j)=exp(-beta*||X_i-Y_j||^2).

    Args:
        X: The first vector X\in\mathbb{R}^{N\times d}
        Y: The second vector X\in\mathbb{R}^{M\times d}
        beta: The length-scale of the SE kernel.
        use_chunk (bool, optional): Whether to use chunk to reduce the GPU memory usage. Note that if set to ``True'' it will slow down the calculation. Defaults to False.

    Returns:
        K: The kernel K\in\mathbb{R}^{N\times M}
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)

    K = cal_dist(X, Y)
    K = nx.exp(-beta * K)
    return K


# get the assignment matrix P
def get_P_sparse(
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
    label_mask: Optional[np.ndarray] = None,
    batch_capacity: int = 1,
    labelA: Optional[pd.Series] = None,
    labelB: Optional[pd.Series] = None,
    label_transfer_prior: Optional[dict] = None,
    top_k: int = 1024,
):
    assert XnAHat.shape[1] == XnB.shape[1], "XnAHat and XnB do not have the same number of features."
    assert XnAHat.shape[0] == alpha.shape[0], "XnAHat and alpha do not have the same length."
    assert XnAHat.shape[0] == Sigma.shape[0], "XnAHat and Sigma do not have the same length."
    nx = ot.backend.get_backend(XnAHat, XnB)
    NA, NB, D = XnAHat.shape[0], XnB.shape[0], XnAHat.shape[1]
    if samples_s is None:
        samples_s = nx.maximum(
            _prod(nx)(nx.max(XnAHat, axis=0) - nx.min(XnAHat, axis=0)),
            _prod(nx)(nx.max(XnB, axis=0) - nx.min(XnB, axis=0)),
        )
    outlier_s = samples_s * nx.sum(label_mask, axis=0) if label_mask is not None else samples_s * NA
    spatial_outlier = _power(nx)((2 * _pi(nx) * sigma2), _data(nx, D / 2, XnAHat)) * (1 - gamma) / (gamma * outlier_s)
    K_NA_spatial, K_NA_sigma2, P, sigma2_temp = calc_P_related(
        XnAHat=XnAHat,
        XnB=XnB,
        X_A=X_A,
        X_B=X_B,
        labelA=labelA,
        labelB=labelB,
        label_transfer_prior=label_transfer_prior,
        sigma2=sigma2,
        sigma2_robust=sigma2 / outlier_variance,
        beta2=beta2,
        spatial_outlier=spatial_outlier,
        col_mul=(_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
        batch_capacity=batch_capacity,
        top_k=top_k,
    )

    K_NA = P.sum(1).to_dense()
    K_NB = P.sum(0).to_dense()
    Sp = P.sum()
    Sp_spatial = K_NA_spatial.sum()
    Sp_sigma2 = K_NA_sigma2.sum()

    assignment_results = {
        "K_NA": K_NA,
        "K_NB": K_NB,
        "K_NA_spatial": K_NA_spatial,
        "K_NA_sigma2": K_NA_sigma2,
        "Sp": Sp,
        "Sp_spatial": Sp_spatial,
        "Sp_sigma2": Sp_sigma2,
        "sigma2_temp": sigma2_temp,
    }

    return P, assignment_results


# morpho pairwise alignment
# TO-DO: Calculate the gene dist mat and save it in the cpu. When using the mat, we can use cuda to convert it to gpu
def BA_align_sparse(
    sampleA: AnnData,
    sampleB: AnnData,
    genes: Optional[Union[List, torch.Tensor]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    vecfld_key_added: Optional[str] = "VecFld_morpho",
    layer: str = "X",
    dissimilarity: str = "kl",
    max_iter: int = 200,
    lambdaVF: Union[int, float] = 1e2,
    beta: Union[int, float] = 0.01,
    K: Union[int, float] = 15,
    beta2: Optional[Union[int, float]] = None,
    beta2_end: Optional[Union[int, float]] = None,
    normalize_c: bool = True,
    normalize_g: bool = True,
    dtype: str = "float32",
    device: str = "cpu",
    inplace: bool = True,
    verbose: bool = True,
    nn_init: bool = True,
    partial_robust_level: float = 25,
    use_label_prior: bool = False,
    label_key: Optional[str] = "cluster",
    label_transfer_prior: Optional[dict] = None,
    SVI_mode: bool = True,
    batch_size: int = 1024,
    use_sparse: bool = True,
    pre_compute_dist: bool = False,
) -> Tuple[Optional[Tuple[AnnData, AnnData]], np.ndarray, np.ndarray]:
    empty_cache(device=device)
    # Preprocessing and extract the spatial and expression information
    normalize_g = False if dissimilarity == "kl" else normalize_g
    sampleA, sampleB = (sampleA, sampleB) if inplace else (sampleA.copy(), sampleB.copy())
    (nx, type_as, new_samples, exp_matrices, spatial_coords, normalize_scale, normalize_mean_list,) = align_preprocess(
        samples=[sampleA, sampleB],
        layer=layer,
        genes=genes,
        spatial_key=spatial_key,
        normalize_c=normalize_c,
        normalize_g=normalize_g,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )
    coordsA, coordsB = spatial_coords[1], spatial_coords[0]
    X_A, X_B = exp_matrices[1], exp_matrices[0]
    del spatial_coords, exp_matrices
    NA, NB, D, G = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1], X_A.shape[1]
    # generate label mask by label consistency prior
    if use_label_prior:
        # check the label key
        if label_key not in sampleA.obs.keys():
            raise ValueError(f"adataA does not have label key {label_key}.")
        if label_key not in sampleB.obs.keys():
            raise ValueError(f"adataB does not have label key {label_key}.")
        labelA = pd.Series(sampleB.obs[label_key].values)
        labelB = pd.Series(sampleA.obs[label_key].values)
    else:
        labelA, labelB = None, None
    # construct kernel for inducing variables
    Unique_coordsA = _unique(nx, coordsA, 0)
    idx = random.sample(range(Unique_coordsA.shape[0]), min(K, Unique_coordsA.shape[0]))
    ctrl_pts = Unique_coordsA[idx, :]
    K = ctrl_pts.shape[0]
    GammaSparse = con_K(ctrl_pts, ctrl_pts, beta)
    U = con_K(coordsA, ctrl_pts, beta)
    kernel_dict = {
        "dist": "cdist",
        "X": nx.to_numpy(coordsA),
        "idx": idx,
        "U": nx.to_numpy(U),
        "GammaSparse": nx.to_numpy(GammaSparse),
        "ctrl_pts": nx.to_numpy(ctrl_pts),
    }

    # perform coarse rigid alignment
    if nn_init:
        _cra_kwargs = dict(
            coordsA=coordsA,
            coordsB=coordsB,
            X_A=X_A,
            X_B=X_B,
            transformed_points=None,
        )
        coordsA, inlier_A, inlier_B, inlier_P, init_R, init_t = coarse_rigid_alignment(
            dissimilarity=dissimilarity, top_K=10, verbose=verbose, **_cra_kwargs
        )
        empty_cache(device=device)
        init_R = _data(nx, init_R, type_as)
        init_t = _data(nx, init_t, type_as)
        coordsA = _data(nx, coordsA, type_as)
        inlier_A = _data(nx, inlier_A, type_as)
        inlier_B = _data(nx, inlier_B, type_as)
        inlier_P = _data(nx, inlier_P, type_as)
        # TO-DO: integrate into one function
    else:
        init_R = np.eye(D)
        init_t = np.zeros((D,))
        inlier_A = np.zeros((4, D))
        inlier_B = np.zeros((4, D))
        inlier_P = np.ones((4, 1))
        init_R = _data(nx, init_R, type_as)
        init_t = _data(nx, init_t, type_as)
        inlier_A = _data(nx, inlier_A, type_as)
        inlier_B = _data(nx, inlier_B, type_as)
        inlier_P = _data(nx, inlier_P, type_as)
    coarse_alignment = coordsA
    # Initialize optimization parameters
    kappa = nx.ones((NA), type_as=type_as)
    alpha = nx.ones((NA), type_as=type_as)
    VnA = nx.zeros(coordsA.shape, type_as=type_as)
    Coff = nx.zeros(ctrl_pts.shape, type_as=type_as)
    gamma, gamma_a, gamma_b = (
        _data(nx, 0.5, type_as),
        _data(nx, 1.0, type_as),
        _data(nx, 1.0, type_as),
    )
    minP, sigma2_terc, erc = (
        _data(nx, 1e-5, type_as),
        _data(nx, 1, type_as),
        _data(nx, 1e-4, type_as),
    )
    SigmaDiag = nx.zeros((NA), type_as=type_as)
    XAHat, RnA = coordsA, coordsA
    s = _data(nx, 1, type_as)
    R = _identity(nx, D, type_as)
    # calculate the initial values of sigma2 and beta2
    sigma2 = _init_guess_sigma2(XAHat, coordsB)
    beta2, beta2_end = _init_guess_beta2(nx, X_A, X_B, dissimilarity, partial_robust_level, beta2, beta2_end)
    empty_cache(device=device)
    # initial the sigma2 and beta2 temperature for better performance
    outlier_variance = 1
    max_outlier_variance = partial_robust_level  # 20
    outlier_variance_decrease = _power(nx)(_data(nx, max_outlier_variance, type_as), 1 / (max_iter / 2))
    beta2_decrease = _power(nx)(beta2_end / beta2, 1 / (50))
    # Initial calculation of the gene and spatial similarity (distance) matrix
    spatial_threshold = 6 * sigma2
    # if pre_compute_dist, we compute the full similarity of the expression (NA x NB) and store it, else we will compute this in each iteration.
    if pre_compute_dist:
        GeneDistMat = calc_distance(
            X_A=X_A,
            X_B=X_B,
            metric=dissimilarity,
            use_sparse=use_sparse,
            sparse_method="topk",
            threshold=1000,
        )
    if SVI_mode:
        SVI_deacy = _data(nx, 10.0, type_as)
        # Select a random subset of data
        batch_size = min(max(int(NB / 10), batch_size), NB)
        randomidx = _randperm(nx)(NB)
        randIdx = randomidx[:batch_size]
        randomIdx = _roll(nx)(randomidx, batch_size)
        randcoordsB = coordsB[randIdx, :]  # batch_size x D
        randX_B = X_B[randIdx, :]  # batch_size x G
        randlabelB = labelB.iloc[np.array(randIdx)] if labelB is not None else None

        Sp, Sp_spatial, Sp_sigma2 = 0, 0, 0
        SigmaInv = nx.zeros((K, K), type_as=type_as)  # K x K
        PXB_term = nx.zeros((NA, D), type_as=type_as)  # NA x D

    iteration = (
        lm.progress_logger(range(max_iter), progress_name="Start morpho alignment") if verbose else range(max_iter)
    )
    # intermediate results
    if iter_key_added is not None:
        sampleB.uns[iter_key_added] = dict()
        sampleB.uns[iter_key_added][key_added] = {}
        sampleB.uns[iter_key_added]["sigma2"] = {}
        sampleB.uns[iter_key_added]["beta2"] = {}
        sampleB.uns[iter_key_added]["scale"] = {}
    # main iteration begin
    for iter in iteration:
        # save intermediate results
        if iter_key_added is not None:
            iter_XAHat = XAHat * normalize_scale + normalize_mean_list[0] if normalize_c else XAHat
            sampleB.uns[iter_key_added][key_added][iter] = nx.to_numpy(iter_XAHat)
            sampleB.uns[iter_key_added]["sigma2"][iter] = nx.to_numpy(sigma2)
            sampleB.uns[iter_key_added]["beta2"][iter] = nx.to_numpy(beta2)
            sampleB.uns[iter_key_added]["scale"][iter] = nx.to_numpy(s)
        # update the assignment matrix
        if SVI_mode:
            step_size = nx.minimum(_data(nx, 1.0, type_as), SVI_deacy / (iter + 1.0))
            P, assignment_results = get_P_sparse(
                XnAHat=XAHat,
                XnB=randcoordsB,
                X_A=X_A,
                X_B=randX_B,
                labelA=labelA,
                labelB=randlabelB,
                sigma2=sigma2,
                beta2=beta2,
                alpha=alpha,
                gamma=gamma,
                Sigma=SigmaDiag,
                outlier_variance=outlier_variance,
                label_transfer_prior=label_transfer_prior,
            )
        else:
            P, assignment_results = get_P_sparse(
                XnAHat=XAHat,
                XnB=coordsB,
                X_A=X_A,
                X_B=randX_B,
                labelA=labelA,
                labelB=labelB,
                sigma2=sigma2,
                beta2=beta2,
                alpha=alpha,
                gamma=gamma,
                Sigma=SigmaDiag,
                outlier_variance=outlier_variance,
                label_transfer_prior=label_transfer_prior,
            )

        # update temperature
        if iter > 5:
            beta2 = (
                nx.maximum(beta2 * beta2_decrease, beta2_end)
                if beta2_decrease < 1
                else nx.minimum(beta2 * beta2_decrease, beta2_end)
            )
            outlier_variance = nx.minimum(outlier_variance * outlier_variance_decrease, max_outlier_variance)

        K_NA, K_NB = assignment_results["K_NA"], assignment_results["K_NB"]
        K_NA_spatial = assignment_results["K_NA_spatial"]
        K_NA_sigma2 = assignment_results["K_NA_sigma2"]

        # Update gamma
        if SVI_mode:
            Sp = step_size * assignment_results["Sp"] + (1 - step_size) * Sp
            Sp_spatial = step_size * assignment_results["Sp_spatial"] + (1 - step_size) * Sp_spatial
            Sp_sigma2 = step_size * assignment_results["Sp_sigma2"] + (1 - step_size) * Sp_sigma2
            gamma = nx.exp(_psi(nx)(gamma_a + Sp_spatial) - _psi(nx)(gamma_a + gamma_b + batch_size))
        else:
            Sp = assignment_results["Sp"]
            Sp_spatial = assignment_results["Sp_spatial"]
            Sp_sigma2 = assignment_results["Sp_sigma2"]
            gamma = nx.exp(_psi(nx)(gamma_a + Sp_spatial) - _psi(nx)(gamma_a + gamma_b + NB))
        gamma = _data(nx, 0.99, type_as) if gamma > 0.99 else gamma
        gamma = _data(nx, 0.01, type_as) if gamma < 0.01 else gamma

        # Update alpha
        alpha = nx.exp(_psi(nx)(kappa + K_NA_spatial) - _psi(nx)(kappa * NA + Sp_spatial))

        # Update VnA
        if (sigma2 < 0.015) or (iter > 80):
            if SVI_mode:
                SigmaInv = (
                    step_size * (sigma2 * lambdaVF * GammaSparse + _dot(nx)(U.T, nx.einsum("ij,i->ij", U, K_NA)))
                    + (1 - step_size) * SigmaInv
                )
                term1 = _dot(nx)(_pinv(nx)(SigmaInv), U.T)
                PXB_term = (
                    step_size * (_dot(nx)(P, randcoordsB) - nx.einsum("ij,i->ij", RnA, K_NA))
                    + (1 - step_size) * PXB_term
                )
                Coff = _dot(nx)(term1, PXB_term)
                VnA = _dot(nx)(
                    U,
                    Coff,
                )
                SigmaDiag = sigma2 * nx.einsum("ij->i", nx.einsum("ij,ji->ij", U, term1))
            else:
                term1 = _dot(nx)(
                    _pinv(nx)(sigma2 * lambdaVF * GammaSparse + _dot(nx)(U.T, nx.einsum("ij,i->ij", U, K_NA))),
                    U.T,
                )
                SigmaDiag = sigma2 * nx.einsum("ij->i", nx.einsum("ij,ji->ij", U, term1))
                Coff = _dot(nx)(
                    term1,
                    (_dot(nx)(P, coordsB) - nx.einsum("ij,i->ij", RnA, K_NA)),
                )
                VnA = _dot(nx)(
                    U,
                    Coff,
                )
        # Update R()
        if nn_init:
            lambdaReg = partial_robust_level * 1e0 * Sp / nx.sum(inlier_P)
        else:
            lambdaReg = 0
        if SVI_mode:
            PXA, PVA, PXB = (
                _dot(nx)(K_NA, coordsA)[None, :],
                _dot(nx)(K_NA, VnA)[None, :],
                _dot(nx)(K_NB, randcoordsB)[None, :],
            )
        else:
            PXA, PVA, PXB = (
                _dot(nx)(K_NA, coordsA)[None, :],
                _dot(nx)(K_NA, VnA)[None, :],
                _dot(nx)(K_NB, coordsB)[None, :],
            )
        # if nn_init:
        PCYC, PCXC = _dot(nx)(inlier_P.T, inlier_B), _dot(nx)(inlier_P.T, inlier_A)
        if SVI_mode and iter > 1:
            t = (
                step_size
                * (
                    ((PXB - PVA - _dot(nx)(PXA, R.T)) + 2 * lambdaReg * sigma2 * (PCYC - _dot(nx)(PCXC, R.T)))
                    / (Sp + 2 * lambdaReg * sigma2 * nx.sum(inlier_P))
                )
                + (1 - step_size) * t
            )
        else:
            t = ((PXB - PVA - _dot(nx)(PXA, R.T)) + 2 * lambdaReg * sigma2 * (PCYC - _dot(nx)(PCXC, R.T))) / (
                Sp + 2 * lambdaReg * sigma2 * nx.sum(inlier_P)
            )
        if SVI_mode:
            A = -(
                _dot(nx)(PXA.T, t)
                + _dot(nx)(
                    coordsA.T,
                    nx.einsum("ij,i->ij", VnA, K_NA) - _dot(nx)(P, randcoordsB),
                )
                + 2
                * lambdaReg
                * sigma2
                * (_dot(nx)(PCXC.T, t) - _dot(nx)(nx.einsum("ij,i->ij", inlier_A, inlier_P[:, 0]).T, inlier_B))
            ).T
        else:
            A = -(
                _dot(nx)(PXA.T, t)
                + _dot(nx)(
                    coordsA.T,
                    nx.einsum("ij,i->ij", VnA, K_NA) - _dot(nx)(P, coordsB),
                )
                + 2
                * lambdaReg
                * sigma2
                * (_dot(nx)(PCXC.T, t) - _dot(nx)(nx.einsum("ij,i->ij", inlier_A, inlier_P[:, 0]).T, inlier_B))
            ).T

        svdU, svdS, svdV = _linalg(nx).svd(A)
        C = _identity(nx, D, type_as)
        C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
        if SVI_mode and iter > 1:
            R = step_size * (_dot(nx)(_dot(nx)(svdU, C), svdV)) + (1 - step_size) * R
        else:
            R = _dot(nx)(_dot(nx)(svdU, C), svdV)
        RnA = s * _dot(nx)(coordsA, R.T) + t
        XAHat = RnA + VnA

        # Update sigma2 and beta2 (optional)
        sigma2_old = sigma2
        sigma2 = nx.maximum(
            (assignment_results["sigma2_temp"] + nx.einsum("i,i", K_NA_sigma2, SigmaDiag) / Sp_sigma2),
            _data(nx, 1e-3, type_as),
        )
        if iter < 100:
            sigma2 = nx.maximum(sigma2, _data(nx, 1e-2, type_as))
        sigma2_terc = nx.abs((sigma2 - sigma2_old) / sigma2)

        # SVI next batch
        spatial_threshold = 6 * sigma2
        # if SVI_mode and iter < max_iter - 1:
        if SVI_mode:
            randIdx = randomidx[:batch_size]
            randomidx = _roll(nx)(randomidx, batch_size)
            randcoordsB = coordsB[randIdx, :]
            randX_B = X_B[randIdx, :]  # batch_size x G
            randlabelB = labelB.iloc[np.array(randIdx)] if labelB is not None else None

        empty_cache(device=device)
    # end of the iteration

    # get the full data assignment
    if SVI_mode:
        if not pre_compute_dist:
            P, assignment_results = get_P_sparse(
                XnAHat=XAHat,
                XnB=coordsB,
                X_A=X_A,
                X_B=X_B,
                labelA=labelA,
                labelB=labelB,
                sigma2=sigma2,
                beta2=beta2,
                alpha=alpha,
                gamma=gamma,
                Sigma=SigmaDiag,
                outlier_variance=outlier_variance,
                label_transfer_prior=label_transfer_prior,
                top_k=32,
            )
    # Get optimal Rigid transformation
    optimal_RnA, optimal_R, optimal_t = get_optimal_R_sparse(
        coordsA=coordsA,
        coordsB=coordsB,
        P=P,
        R_init=R,
    )
    # combine the initial rigid transformation and final rigid transformation
    t = _dot(nx)(init_t, R.T) + t
    R = _dot(nx)(R, init_R)
    optimal_t = _dot(nx)(init_t, optimal_R.T) + optimal_t
    optimal_R = _dot(nx)(optimal_R, init_R)

    # output optimization parameters
    if verbose:
        lm.main_info(f"Key Parameters: gamma: {gamma}; beta2: {beta2}; sigma2: {sigma2}")

    # de-normalize
    if normalize_c:
        XAHat = XAHat * normalize_scale + normalize_mean_list[0]
        RnA = RnA * normalize_scale + normalize_mean_list[0]
        optimal_RnA = optimal_RnA * normalize_scale + normalize_mean_list[0]
        coarse_alignment = coarse_alignment * normalize_scale + normalize_mean_list[0]
        output_R = optimal_R
        output_t = optimal_t * normalize_scale + normalize_mean_list[0] - _dot(nx)(normalize_mean_list[1], optimal_R.T)

    # Save aligned coordinates to adata
    sampleB.obsm[key_added + "_nonrigid"] = nx.to_numpy(XAHat).copy()
    sampleB.obsm[key_added + "_rigid"] = nx.to_numpy(optimal_RnA).copy()

    # save vector field and other parameters
    if not (vecfld_key_added is None):
        norm_dict = {
            "mean_transformed": nx.to_numpy(normalize_mean_list[1]),
            "mean_fixed": nx.to_numpy(normalize_mean_list[0]),
            "scale": nx.to_numpy(normalize_scale),
        }
        sampleB.uns[vecfld_key_added] = {
            "R": nx.to_numpy(R),
            "t": nx.to_numpy(t),
            "optimal_R": nx.to_numpy(optimal_R),
            "optimal_t": nx.to_numpy(optimal_t),
            "output_R": nx.to_numpy(output_R),
            "output_t": nx.to_numpy(output_t),
            "beta": beta,
            "C": nx.to_numpy(Coff),
            "X_ctrl": nx.to_numpy(ctrl_pts),
            "norm_dict": norm_dict,
            "kernel_dict": kernel_dict,
            "dissimilarity": dissimilarity,
            "beta2": nx.to_numpy(sigma2),
            "sigma2": nx.to_numpy(sigma2),
            "gamma": nx.to_numpy(gamma),
            "NA": NA,
            "outlier_variance": nx.to_numpy(outlier_variance),
            "method": "morpho",
            "pre_norm_scale": 1,
        }
    empty_cache(device=device)
    return (
        None if inplace else (sampleA, sampleB),
        nx.to_numpy(P.to_dense()),
        nx.to_numpy(sigma2),
    )