from typing import Optional, Tuple, Union

import numpy as np
import ot
from anndata import AnnData

from .utils import align_preprocess, calc_exp_dissimilarity, empty_cache

######################################
# Align spots across pairwise slices #
######################################


def pairwise_align(
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

    pi, log = ot.gromov.cg(
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
    empty_cache(device=device)

    return pi, obj


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
