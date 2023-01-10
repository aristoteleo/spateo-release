"""
Todo:
    * @Xiaojieqiu: update with Google style documentation, function typings, tests
"""
import math
from typing import List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import shapely
import shapely.geometry as geometry
from sklearn.decomposition import PCA

from ..configuration import SKM
from ..io.bbs import alpha_shape
from ..logging import logger_manager as lm


def procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    scaling: bool = True,
    reflection: str = "best",
) -> Tuple[float, np.ndarray, dict]:
    """A port of MATLAB's `procrustes` function to Numpy.

    This function will need to be rewritten just with scipy.spatial.procrustes and
    scipy.linalg.orthogonal_procrustes later.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Args:
        X, Y: matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions (columns) than X.
            scaling: if False, the scaling component of the transformation is forced
            to 1
        reflection: if 'best' (default), the transformation solution may or may not include
            a reflection component, depending on which fits the data best. setting
            reflection to True or False forces a solution with reflection or no reflection
            respectively.

    Returns:
        d: the residual sum of squared errors, normalized according to a measure of the scale of X,
            ((X - X.mean(0))**2).sum()
        Z: the matrix of transformed Y-values
        tform: a dict specifying the rotation, translation and scaling that maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = np.linalg.norm(X0, "fro") ** 2  # (X0**2.).sum()
    ssY = np.linalg.norm(Y0, "fro") ** 2  # (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform


def AffineTrans(
    x: np.ndarray,
    y: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    theta: Tuple[None, float],
    R: Tuple[None, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate the x/y coordinates of data points by the translating the centroid to the origin. Then data will be
    rotated with angle theta.

    Args:
        x: x coordinates for the data points (bins). 1D np.array.
        y: y coordinates for the data points (bins). 1D np.array.
        centroid_x: x coordinates for the centroid of data points (bins).
        centroid_y: y coordinates for the centroid of data points (bins).
        theta: the angle of rotation. Unit is is in `np.pi` (so 90 degree is `np.pi / 2` and value is defined in the
            clockwise direction.
        R: the rotation matrix. If `R` is provided, `theta` will be ignored.

    Returns:
        T_t: The translation matrix used in affine transformation.
        T_r: The rotation matrix used in affine transformation.
        trans_xy_coord: The matrix that stores the translated and rotated coordinates.
    """

    if theta is None and R is None:
        lm.EXCEPTION(f"`theta` and `R` cannot be both None!")

    trans_xy_coord = np.zeros((len(x), 2))

    T_t, T_r = np.zeros((3, 3)), np.zeros((3, 3))
    np.fill_diagonal(T_t, 1)
    np.fill_diagonal(T_r, 1)

    T_t[0, 2], T_t[1, 2] = -centroid_x, -centroid_y

    if R is None:
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        T_r[0, 0], T_r[0, 1] = cos_theta, -sin_theta
        T_r[1, 0], T_r[1, 1] = sin_theta, cos_theta
    else:
        T_r[:2, :2] = R

    for cur_x, cur_y, cur_ind in zip(x, y, np.arange(len(x))):
        data = np.array([cur_x, cur_y, 1])
        res = T_t @ data
        res = T_r @ res
        trans_xy_coord[cur_ind, :] = res[:2]

    return T_t, T_r, trans_xy_coord


def pca_align(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use pca to rotate a coordinate matrix to reveal the largest variance on each dimension.

    This can be used to `correct`, for example, embryo slices to the right orientation.

    Args:
        X: The input coordinate matrix.

    Returns:
        Y: The rotated coordinate matrix that has the major variances on each dimension.
        R: The rotation matrix that was used to convert the input X matrix to output Y matrix.

    """
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    R = pca.components_
    Y = (R @ X.T).T

    return Y, R


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def align_slices_pca(
    adata: anndata.AnnData,
    spatial_key: str = "spatial",
    inplace: bool = False,
    result_key: Tuple[None, str] = None,
) -> None:
    """Coarsely align the slices based on the major axis, identified via PCA

    Args:
        adata: the input adata object that contains the spatial key in .obsm.
        spatial_key: the key in .obsm that points to the spatial information.
        inplace: whether the spatial coordinates will be inplace updated or a new key `spatial_.
        result_key: when inplace is False, this points to the key in .obsm that stores the corrected spatial
            coordinates.

    Returns:
        Nothing but updates the spatial coordinates either inplace or with the `result_key` key based on the major axis
        identified via PCA.
    """

    coords = adata.obsm[spatial_key].copy()
    x, y = coords[:, 0], coords[:, 1]

    try:
        adata_concave_hull, _ = alpha_shape(x, y, alpha=1)

        if type(adata_concave_hull) == shapely.geometry.multipolygon.MultiPolygon:
            alpha_shape_x, alpha_shape_y = adata_concave_hull[0].exterior.xy
        else:
            alpha_shape_x, alpha_shape_y = adata_concave_hull.exterior.xy

        centroid_x, centroid_y = adata_concave_hull.centroid.coords.xy
        centroid_x, centroid_y = centroid_x[0], centroid_y[0]

        adata.uns["bbs"] = {"x": alpha_shape_x, "y": alpha_shape_y, "centroid_x": centroid_x, "centroid_y": centroid_y}
    except:
        centroid_x, centroid_y = np.nanmedian(coords, 0)
        adata.uns["bbs"] = {"x": None, "y": None, "centroid_x": centroid_x, "centroid_y": centroid_y}

    coords_correct, R = pca_align(coords)
    _, _, spatial_corrected = AffineTrans(
        coords[:, 0],
        coords[:, 1],
        centroid_x,
        centroid_y,
        None,
        R,
    )

    # rotate 90 degree
    _, _, coords_correct_processed = AffineTrans(
        spatial_corrected[:, 0],
        spatial_corrected[:, 1],
        0,
        0,
        np.pi / 2,
        None,
    )
    # reflect vertically
    coords_correct_processed[:, 1] = -coords_correct_processed[:, 1]

    # reflect vertically again:
    coords_correct_processed[:, 1] = -coords_correct_processed[:, 1]

    # account for the mirror effect when plotting an image
    if inplace:
        adata.obsm["spatial"] = coords_correct_processed
    else:
        key = "spatial_corrected" if result_key is None else result_key
        adata.obsm[key] = coords_correct_processed
