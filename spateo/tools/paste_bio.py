import os
from typing import List, Optional, Tuple

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import NMF

from ..logging import logger_manager as lm

# Get the intersection of lists
intersect_lsts = lambda *lsts: list(set(lsts[0]).intersection(*lsts[1:]))

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if issparse(X) else np.asarray(X)


def filter_common_genes(*genes) -> list:
    """
    Filters for the intersection of genes between all samples.

    Args:
        genes: List of genes.
    """

    common_genes = intersect_lsts(*genes)
    if len(common_genes) == 0:
        raise ValueError("The number of common gene between all samples is 0.")
    else:
        lm.main_info(
            message=f"Filtered all samples for common genes. There are {(len(common_genes))} common genes.",
            indent_level=2,
        )
        return common_genes


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


def check_backend(device: str = "cpu"):
    """
    Check the proper backend for the device.

    Args:
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        device: The device used to run the program.
        backend: The proper backend.
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
            device = "cpu"
            lm.main_info(message="GPU is not available, resorting to torch cpu.", indent_level=2)
    return device, backend


def check_spatial_coords(sample: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Check spatial coordinate information.

    Args:
        sample: An anndata object.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.

    Returns:
        The spatial coordinates.
    """
    coordinates = sample.obsm[spatial_key]
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.values

    coordinates = coordinates.astype(np.float64)
    return coordinates


def check_exp(sample: AnnData, layer: str = "X") -> np.ndarray:
    """
    Check expression matrix.

    Args:
        sample: An anndata object.
        layer: The key in `.layers` that corresponds to the expression matrix.

    Returns:
        The expression matrix.
    """

    exp_martix = sample.X if layer == "X" else sample.layers[layer]
    exp_martix = to_dense_matrix(exp_martix)
    exp_martix = exp_martix.astype(np.float64)
    return exp_martix


######################################
# Align spots across pairwise slices #
######################################


def pairwise_align(
    sampleA: AnnData,
    sampleB: AnnData,
    layer: str = "X",
    spatial_key: str = "spatial",
    alpha: float = 0.1,
    dissimilarity: str = "kl",
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm: bool = False,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: str = "cpu",
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Calculates and returns optimal alignment of two slices.

    Args:
        sampleA: Sample A to align.
        sampleB: Sample B to align.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sampleA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sampleB spots, otherwise default is uniform.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        pi: Alignment of spots.
        obj: Objective function output of FGW-OT.
    """

    # Determine if gpu or cpu is being used
    device, nx = check_backend(device=device)

    # subset for common genes
    common_genes = filter_common_genes(sampleA.var.index, sampleB.var.index)
    sampleA, sampleB = sampleA[:, common_genes], sampleB[:, common_genes]

    # Calculate spatial distances
    coordinatesA = nx.from_numpy(check_spatial_coords(sample=sampleA, spatial_key=spatial_key))
    coordinatesB = nx.from_numpy(check_spatial_coords(sample=sampleB, spatial_key=spatial_key))

    D_A = ot.dist(coordinatesA, coordinatesA, metric="euclidean")
    D_B = ot.dist(coordinatesB, coordinatesB, metric="euclidean")

    # Calculate expression dissimilarity
    A_X = nx.from_numpy(check_exp(sample=sampleA, layer=layer))
    B_X = nx.from_numpy(check_exp(sample=sampleB, layer=layer))

    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        M = ot.dist(A_X, B_X)
    else:
        M = kl_divergence_backend(A_X + 0.01, B_X + 0.01)

    # init distributions
    a = np.ones((sampleA.shape[0],)) / sampleA.shape[0] if a_distribution is None else a_distribution
    b = np.ones((sampleB.shape[0],)) / sampleB.shape[0] if b_distribution is None else b_distribution
    a = nx.from_numpy(a.astype(np.float64))
    b = nx.from_numpy(b.astype(np.float64))

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    constC, hC1, hC2 = ot.gromov.init_matrix(D_A, D_B, a, b, "square_loss")

    if G_init is None:
        G0 = a[:, None] * b[None, :]
    else:
        G_init = nx.from_numpy(G_init.astype(np.float64))
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


def center_align(
    init_center_sample: AnnData,
    samples: List[AnnData],
    layer: str = "X",
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
    device: str = "cpu",
) -> Tuple[AnnData, List[np.ndarray]]:
    """
    Computes center alignment of slices.

    Args:
        init_center_sample: Sample to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        samples: List of samples to use in the center alignment.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        lmbda: List of probability weights assigned to each slice; If ``None``, use uniform weights.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
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
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        - Inferred center sample with full and low dimensional representations (W, H) of the gene expression matrix.
        - List of pairwise alignment mappings of the center sample (rows) to each input sample (columns).
    """

    def _generate_center_sample(W, H, genes, coords):
        center_sample = AnnData(np.dot(W, H))
        center_sample.var.index = genes
        center_sample.obsm[spatial_key] = coords
        return center_sample

    if lmbda is None:
        lmbda = len(samples) * [1 / len(samples)]

    if distributions is None:
        distributions = len(samples) * [None]

    # get common genes
    all_samples_genes = [s[0].var.index for s in samples]
    all_samples_genes.append(init_center_sample.var.index)
    common_genes = filter_common_genes(*all_samples_genes)

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
            p, r_q = pairwise_align(
                sampleA=_generate_center_sample(W=W, H=H, genes=common_genes, coords=center_coords),
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
                device=device,
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
        Aligned spatial coordinates of X, Y.
    """
    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T

    return X, Y
