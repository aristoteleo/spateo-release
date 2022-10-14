import os
from typing import List, Optional, Tuple, Union

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import NMF

from ..configuration import SKM
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


def check_backend(device: str = "cpu", dtype: str = "float32"):
    """
    Check the proper backend for the device.

    Args:
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        dtype: The floating-point number type. Only float32 and float64.

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
            lm.main_info(message="GPU is not available, resorting to torch cpu.", indent_level=2)

    type_as = backend.__type_list__[0] if dtype == "float32" else backend.__type_list__[1]
    return backend, type_as


# @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "sample")
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


# @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "sample")
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


######################################
# Align spots across pairwise slices #
######################################


# @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "sampleA")
# @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "sampleB")
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
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sampleA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sampleB spots, otherwise default is uniform.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.

    Returns:
        pi: Alignment of spots.
        obj: Objective function output of FGW-OT.
    """

    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)

    # subset for common genes
    common_genes = filter_common_genes(sampleA.var.index, sampleB.var.index)
    common_genes = common_genes if genes is None else intersect_lsts(common_genes, genes)
    sampleA, sampleB = sampleA[:, common_genes], sampleB[:, common_genes]

    # Calculate spatial distances
    coordinatesA = nx.from_numpy(check_spatial_coords(sample=sampleA, spatial_key=spatial_key), type_as=type_as)
    coordinatesB = nx.from_numpy(check_spatial_coords(sample=sampleB, spatial_key=spatial_key), type_as=type_as)

    D_A = ot.dist(coordinatesA, coordinatesA, metric="euclidean")
    D_B = ot.dist(coordinatesB, coordinatesB, metric="euclidean")

    # Calculate expression dissimilarity
    A_X = nx.from_numpy(check_exp(sample=sampleA, layer=layer), type_as=type_as)
    B_X = nx.from_numpy(check_exp(sample=sampleB, layer=layer), type_as=type_as)

    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        M = ot.dist(A_X, B_X)
    else:
        M = kl_divergence_backend(A_X + 0.01, B_X + 0.01)

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


# @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "init_center_sample")
# @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "samples")
def center_align(
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
            p, r_q = pairwise_align(
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


#######################################
# Mapping aligned spatial coordinates #
#######################################


def _get_optimal_mapping_relationship(
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
    X_max_index, X_pi_value, Y_max_index, Y_pi_value = _get_optimal_mapping_relationship(
        X=X, Y=Y, pi=pi, keep_all=keep_all
    )

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


def mapping_center_coords(modelA: AnnData, modelB: AnnData, center_key: str) -> dict:
    """
    Optimal mapping coordinates between X and Y based on intermediate coordinates.

    Args:
        modelA: modelA aligned with center model.
        modelB: modelB aligned with center model.
        center_key: The key in ``.uns`` that corresponds to the alignment info between modelA/modelB and center model.

    Returns:
        A dict of raw_X, raw_Y, mapping_X, mapping_Y, pi_value.
            raw_X is the raw X coordinates.
            raw_Y is the raw Y coordinates.
            mapping_X is the Y coordinates aligned with X coordinates.
            mapping_Y is the X coordinates aligned with Y coordinates.
            pi_value is the value of optimal mapping points.
    """

    modelA_dict = modelA.uns[center_key].copy()
    modelB_dict = modelB.uns[center_key].copy()

    mapping_X_cols = [f"mapping_X_{i}" for i in range(modelA_dict["mapping_Y"].shape[1])]
    raw_X_cols = [f"raw_X_{i}" for i in range(modelA_dict["raw_Y"].shape[1])]
    mapping_Y_cols = [f"mapping_Y_{i}" for i in range(modelB_dict["mapping_Y"].shape[1])]
    raw_Y_cols = [f"raw_Y_{i}" for i in range(modelB_dict["raw_Y"].shape[1])]

    X_cols = mapping_X_cols.copy() + raw_X_cols.copy() + ["mid"]
    X_data = pd.DataFrame(
        np.concatenate([modelA_dict["raw_Y"], modelA_dict["mapping_Y"], modelA_dict["pi_index"][:, [0]]], axis=1),
        columns=X_cols,
    )
    X_data["pi_value_X"] = modelA_dict["pi_value"].astype(np.float64)

    Y_cols = mapping_Y_cols.copy() + raw_Y_cols.copy() + ["mid"]
    Y_data = pd.DataFrame(
        np.concatenate([modelB_dict["raw_Y"], modelB_dict["mapping_Y"], modelB_dict["pi_index"][:, [0]]], axis=1),
        columns=Y_cols,
    )
    Y_data["pi_value_Y"] = modelB_dict["pi_value"].astype(np.float64)

    mapping_data = pd.merge(Y_data, X_data, on=["mid"], how="inner")
    mapping_data["pi_value"] = mapping_data[["pi_value_X"]].values * mapping_data[["pi_value_Y"]].values

    return {
        "raw_X": mapping_data[raw_X_cols].values,
        "raw_Y": mapping_data[raw_Y_cols].values,
        "mapping_X": mapping_data[mapping_X_cols].values,
        "mapping_Y": mapping_data[mapping_Y_cols].values,
        "pi_value": mapping_data["pi_value"].astype(np.float64).values,
    }
