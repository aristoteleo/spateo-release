from typing import List, Optional, Tuple, Union

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from sklearn.decomposition import NMF

from spateo.logging import logger_manager as lm

from .utils import (
    align_preprocess,
    calc_exp_dissimilarity,
    check_exp,
    check_spatial_coords,
    filter_common_genes,
    intersect_lsts,
    to_dense_matrix,
)

######################################
# Align spots across pairwise slices #
######################################


def paste_pairwise_align(
    sampleA: AnnData,
    sampleB: AnnData,
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    alpha: float = 0.1,
    dissimilarity: str = "kl",
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm: bool = False,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Calculates and returns optimal alignment of two slices.

    Args:
        sampleA: Sample A to align.
        sampleB: Sample B to align.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
                When α = 0 only the gene expression data is taken into account,
                while when α =1 only the spatial coordinates are taken into account.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        G_init: Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution: Distribution of sampleA spots, otherwise default is uniform.
        b_distribution: Distribution of sampleB spots, otherwise default is uniform.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        verbose: If ``True``, print progress updates.

    Returns:
        pi: Alignment of spots.
        obj: Objective function output of FGW-OT.
    """

    # Preprocessing
    (nx, type_as, new_samples, exp_matrices, spatial_coords, normalize_scale, normalize_mean_list,) = align_preprocess(
        samples=[sampleA, sampleB],
        genes=genes,
        spatial_key=spatial_key,
        layer=layer,
        normalize_c=False,
        normalize_g=False,
        select_high_exp_genes=False,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )

    # Calculate spatial distances
    coordsA, coordsB = spatial_coords[0], spatial_coords[1]
    D_A = ot.dist(coordsA, coordsA, metric="euclidean")
    D_B = ot.dist(coordsB, coordsB, metric="euclidean")

    # Calculate expression dissimilarity
    X_A, X_B = exp_matrices[0], exp_matrices[1]
    M = calc_exp_dissimilarity(X_A=X_A, X_B=X_B, dissimilarity=dissimilarity)

    # init distributions
    a = np.ones((sampleA.shape[0],)) / sampleA.shape[0] if a_distribution is None else np.asarray(a_distribution)
    b = np.ones((sampleB.shape[0],)) / sampleB.shape[0] if b_distribution is None else np.asarray(b_distribution)
    a = nx.from_numpy(a, type_as=type_as)
    b = nx.from_numpy(b, type_as=type_as)

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    constC, hC1, hC2 = ot.gromov.init_matrix(D_A, D_B, a, b, "square_loss")

    if G_init is None:
        G0 = a[:, None] * b[None, :]
    else:
        G_init = nx.from_numpy(G_init, type_as=type_as)
        G0 = (1 / nx.sum(G_init)) * G_init

    pi, log = ot.optim.cg(
        a,
        b,
        (1 - alpha) * M,
        alpha,
        lambda G: ot.gromov.gwloss(constC, hC1, hC2, G),
        lambda G: ot.gromov.gwggrad(constC, hC1, hC2, G),
        G0,
        armijo=False,
        C1=D_A,
        C2=D_B,
        constC=constC,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        log=True,
    )

    pi = nx.to_numpy(pi)
    obj = nx.to_numpy(log["loss"][-1])
    if device != "cpu":
        torch.cuda.empty_cache()

    return pi, obj


###################################################
# Integrate multiple slices into one center slice #
###################################################


def center_NMF(n_components, random_seed, dissimilarity="kl"):
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        model = NMF(n_components=n_components, init="random", random_state=random_seed)
    else:
        model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init="random",
            random_state=random_seed,
        )

    return model


def paste_center_align(
    init_center_sample: AnnData,
    samples: List[AnnData],
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
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
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[AnnData, List[np.ndarray]]:
    """
    Computes center alignment of slices.

    Args:
        init_center_sample: Sample to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        samples: List of samples to use in the center alignment.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        lmbda: List of probability weights assigned to each slice; If ``None``, use uniform weights.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
                When α = 0 only the gene expression data is taken into account,
                while when α =1 only the spatial coordinates are taken into account.
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
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        verbose: If ``True``, print progress updates.

    Returns:
        - Inferred center sample with full and low dimensional representations (W, H) of the gene expression matrix.
        - List of pairwise alignment mappings of the center sample (rows) to each input sample (columns).
    """

    def _generate_center_sample(W, H, genes, coords, layer):
        center_sample = AnnData(np.dot(W, H))
        center_sample.var.index = genes
        center_sample.obsm[spatial_key] = coords
        if layer != "X":
            center_sample.layers[layer] = center_sample.X
        return center_sample

    if lmbda is None:
        lmbda = len(samples) * [1 / len(samples)]

    if distributions is None:
        distributions = len(samples) * [None]

    # get common genes
    all_samples_genes = [s[0].var.index for s in samples]
    all_samples_genes.append(init_center_sample.var.index)
    common_genes = filter_common_genes(*all_samples_genes)
    common_genes = common_genes if genes is None else intersect_lsts(common_genes, genes)

    # subset common genes
    init_center_sample = init_center_sample[:, common_genes]
    samples = [sample[:, common_genes] for sample in samples]

    # Run initial NMF
    if pis_init is None:
        pis = [None for i in range(len(samples))]
        B = check_exp(sample=init_center_sample, layer=layer)
    else:
        pis = pis_init
        B = init_center_sample.shape[0] * sum(
            [
                lmbda[i] * np.dot(pis[i], to_dense_matrix(check_exp(samples[i], layer=layer)))
                for i in range(len(samples))
            ]
        )
    init_NMF_model = center_NMF(n_components=n_components, random_seed=random_seed, dissimilarity=dissimilarity)
    W = init_NMF_model.fit_transform(B)
    H = init_NMF_model.components_
    center_coords = check_spatial_coords(sample=init_center_sample, spatial_key=spatial_key)

    # Minimize R
    iteration_count = 0
    R = 0
    R_diff = 100
    while R_diff > threshold and iteration_count < max_iter:
        lm.main_info(message=f"{iteration_count} iteration of center alignment.", indent_level=1)

        new_pis = []
        r = []
        for i in range(len(samples)):
            p, r_q = paste_pairwise_align(
                sampleA=_generate_center_sample(W=W, H=H, genes=common_genes, coords=center_coords, layer=layer),
                sampleB=samples[i],
                layer=layer,
                spatial_key=spatial_key,
                alpha=alpha,
                dissimilarity=dissimilarity,
                norm=norm,
                G_init=pis[i],
                b_distribution=distributions[i],
                numItermax=numItermax,
                numItermaxEmd=numItermaxEmd,
                dtype=dtype,
                device=device,
                verbose=verbose,
            )
            new_pis.append(p)
            r.append(r_q)

        pis = new_pis.copy()
        NMF_model = center_NMF(n_components, random_seed, dissimilarity=dissimilarity)
        B = W.shape[0] * sum(
            [
                lmbda[i] * np.dot(pis[i], to_dense_matrix(check_exp(samples[i], layer=layer)))
                for i in range(len(samples))
            ]
        )
        W = NMF_model.fit_transform(B)
        H = NMF_model.components_

        R_new = np.dot(r, lmbda)
        iteration_count += 1
        R_diff = abs(R - R_new)
        R = R_new

        lm.main_info(message=f"Objective: {R_new}", indent_level=2)
        lm.main_info(message=f"Difference: {R_diff}", indent_level=2)

    center_sample = init_center_sample.copy()
    center_sample.X = np.dot(W, H)
    center_sample.uns["paste_W"] = W
    center_sample.uns["paste_H"] = H
    center_sample.uns["full_rank"] = center_sample.shape[0] * sum(
        [lmbda[i] * np.dot(pis[i], to_dense_matrix(samples[i].X)) for i in range(len(samples))]
    )
    center_sample.uns["obj"] = R
    return center_sample, pis


########################################
# Generate aligned spatial coordinates #
########################################


def generalized_procrustes_analysis(X, Y, pi):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).

    Args:
        X: np array of spatial coordinates.
        Y: np array of spatial coordinates.
        pi: mapping between the two layers output by PASTE.

    Returns:
        Aligned spatial coordinates of X, Y and the mapping relations.
    """
    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    mapping_dict = {"tX": tX, "tY": tY, "R": R}

    return X, Y, mapping_dict
