import math
import random
from typing import Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from spateo.logging import logger_manager as lm

from .utils import align_preprocess, calc_exp_dissimilarity


def establish_putative_matches(
    sampleA: AnnData,
    sampleB: AnnData,
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    dissimilarity: str = "kl",
    nearest_neighbor: Union[int, float] = 1,
    mutual: bool = False,
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        sampleA: Sample A to align.
        sampleB: Sample B to align.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        nearest_neighbor:
        mutual:
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        verbose: If ``True``, print progress updates.
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

    coordsA, coordsB = spatial_coords[0], spatial_coords[1]
    X_A, X_B = exp_matrices[0], exp_matrices[1]

    # Calculate gene expression distance matrix using Euclidean distance
    GeneDistMat = calc_exp_dissimilarity(X_A=X_A, X_B=X_B, dissimilarity=dissimilarity)
    MsortIndex = np.argsort(GeneDistMat, axis=1)  # sort the similarity to obtain the most similarity one

    # Resort spatial distances
    X = coordsA
    Y = coordsB[MsortIndex[:, 0], :]

    # For the case that the required nearest neighbor is bigger than 1
    for i in range(1, nearest_neighbor):
        X = np.concatenate((X, coordsA), axis=0)
        Y = np.concatenate((Y, coordsB[MsortIndex[:, i], :]), axis=0)

    # Mutural nearest neighbor
    if mutual:
        MsortIndex = np.argsort(GeneDistMat, axis=0)
        for i in range(1, nearest_neighbor):
            X = np.concatenate((X, coordsA[MsortIndex[i, :], :]), axis=0)
            Y = np.concatenate((Y, coordsB), axis=0)
    return X, Y


def get_model_from_inliers(
    Xt: np.ndarray,
    Yt: np.ndarray,
    inliers: np.ndarray,
    scale_limit: Union[list, tuple] = (0.8, 1.5),
) -> Tuple[np.ndarray, np.ndarray, Union[float, int]]:
    if inliers.shape[0] == 2:
        y_tilde, x_tilde = np.expand_dims(Yt[:, inliers[0]] - Yt[:, inliers[1]], -1), np.expand_dims(
            Xt[:, inliers[0]] - Xt[:, inliers[1]], -1
        )
        s = np.sqrt((y_tilde.T @ y_tilde) / (x_tilde.T @ x_tilde))
        if not scale_limit[0] <= s <= scale_limit[1]:
            return np.identity(2), np.zeros((2, 1)), s

    X_I = Xt[:, inliers]
    Y_I = Yt[:, inliers]
    muX = np.expand_dims(np.mean(X_I, axis=1), -1)
    muY = np.expand_dims(np.mean(Y_I, axis=1), -1)
    X_N = X_I - muX
    Y_N = Y_I - muY
    S = np.dot(X_N, Y_N.T)
    U, sigma, VT = np.linalg.svd(S)
    R = VT.T @ U.T
    t = muY - R @ muX
    # the rotation matrix may have ambiguity
    if np.linalg.det(R) < 0:
        R = np.diag([-1, 1]) @ R
    return R, t, 1


def get_model_from_sample(
    Xt: np.ndarray,
    Yt: np.ndarray,
    rand_sample: Union[np.ndarray, list],
    scale_limit: Union[list, tuple] = (0.8, 1.5),
) -> Tuple[np.ndarray, np.ndarray, bool]:

    R, t, s = get_model_from_inliers(Xt, Yt, np.asarray(rand_sample), scale_limit=scale_limit)
    if scale_limit[0] <= s <= scale_limit[1]:
        return R, t, True
    else:
        return np.identity(2), np.zeros((2, 1)), False


def evaluate_model(
    Xt: np.ndarray,
    Yt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    inlier_threshold: Union[float, int] = 1,
) -> Tuple[int, np.ndarray]:
    Y_hat = R @ Xt + t
    error = np.sqrt(np.sum(np.power(Y_hat - Yt, 2), axis=0))
    inliers = np.where(error < inlier_threshold)[0]
    support = inliers.shape[0]
    return support, inliers


def RANSAC(
    X: np.ndarray,
    Y: np.ndarray,
    max_iter: int = 5000,
    inlier_threshold: Union[float, int] = 1,
    confidence: Union[float] = 0.99,
    polish_iter: int = 10,
    scale_limit: Union[list, tuple] = (0.8, 1.5),
):

    # initialize parameters
    best_support, current_trail, N = 0, 0, X.shape[0]
    Xt, Yt = X.T, Y.T

    best_inliers, best_R, best_t = None, None, None
    while current_trail < max_iter:
        current_trail = current_trail + 1

        # randomly choose two sample from putative matches
        rand_sample = random.sample(range(N), 2)

        # calculate a rigid model from the two sample
        current_R, current_t, continue_flag = get_model_from_sample(
            Xt=Xt, Yt=Yt, rand_sample=rand_sample, scale_limit=scale_limit
        )

        # fast reject the bad model if the scaling value is deviated far from 1
        if not continue_flag:
            continue

        # evaluate the model
        current_support, current_inliers = evaluate_model(Xt, Yt, current_R, current_t, inlier_threshold)

        # update so-far-the-best model
        if current_support > best_support:
            best_support = current_support
            best_R, best_t, best_inliers = current_R, current_t, current_inliers
            # update max iteration
            max_iter = min(
                math.log(1 - confidence) / math.log(1 - pow((best_inliers.shape[0] / N), 2)),
                max_iter,
            )

    # Perform iterative least squares to polish result
    for iter in range(polish_iter):
        best_R, best_t, _ = get_model_from_inliers(Xt, Yt, best_inliers, scale_limit=scale_limit)
        _, best_inliers = evaluate_model(Xt, Yt, best_R, best_t, inlier_threshold)

    lm.main_info(
        message=f"Iter: {current_trail}, Inliers: {best_inliers.shape[0]}.",
        indent_level=1,
    )

    return best_inliers, best_R, best_t


def transform(Y, R, t, s=1):
    return np.transpose(s * np.dot(R, Y.T) + t)


def RANSAC_align(
    sampleA: AnnData,
    sampleB: AnnData,
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    layer: str = "X",
    dissimilarity: str = "kl",
    nearest_neighbor: Union[int, float] = 1,
    mutual: bool = False,
    max_iter: int = 5000,
    inlier_threshold: Union[float, int] = 1,
    confidence: float = 0.99,
    polish_iter: int = 10,
    scale_limit: Union[list, tuple] = (0.8, 1.5),
    dtype: str = "float32",
    device: str = "cpu",
    inplace: bool = True,
    verbose: bool = True,
) -> Optional[Tuple[AnnData, AnnData]]:

    # Establish putative matches according only on the gene espression similarity
    X, Y = establish_putative_matches(
        sampleA=sampleA.copy(),
        sampleB=sampleB.copy(),
        genes=genes,
        spatial_key=spatial_key,
        layer=layer,
        dissimilarity=dissimilarity,
        nearest_neighbor=nearest_neighbor,
        mutual=mutual,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )

    # Perform RANSAC to robustly estimate the transformation model
    inliers, R, t = RANSAC(
        X=X,
        Y=Y,
        max_iter=max_iter,
        inlier_threshold=inlier_threshold,
        confidence=confidence,
        polish_iter=polish_iter,
        scale_limit=scale_limit,
    )

    if inplace:
        sampleA.obsm[key_added] = sampleA.obsm[spatial_key]
        sampleB.obsm[key_added] = transform(sampleB.obsm[spatial_key], R.T, -R.T @ t, 1)
    else:
        sampleA, sampleB = sampleA.copy(), sampleB.copy()

        sampleA.obsm[key_added] = sampleA.obsm[spatial_key]
        sampleB.obsm[key_added] = transform(sampleB.obsm[spatial_key], R.T, -R.T @ t, 1)
        return sampleA, sampleB
