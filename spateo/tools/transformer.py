"""
Todo:
    * @Xiaojieqiu: update with Google style documentation, function typings, tests
"""
import math

import numpy as np
import pandas as pd
import shapely
import shapely.geometry as geometry
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPoint
from shapely.ops import cascaded_union, polygonize
from sklearn.decomposition import PCA


def procrustes(X, Y, scaling=True, reflection="best"):
    """This function will need to be rewritten just with scipy.spatial.procrustes and
    scipy.linalg.orthogonal_procrustes later.

    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Parameters
    ----------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Returns
    -------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

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


def AffineTrans(x, y, centroid_x, centroid_y, theta):
    """Translate the x/y coordinates of data points by the translating the centroid to the origin. Then data will be
    rotated with angle theta.

    Parameters
    ----------
        x: `np.array`
            x coordinates for the data points (bins). 1D np.array.
        y: `np.array`
            y coordinates for the data points (bins). 1D np.array.
        centroid_x: `float`
            x coordinates for the centroid of data points (bins).
        centroid_y: `np.array`
            y coordinates for the centroid of data points (bins).
        theta: `float`
            the angle of rotation. Unit is is in `np.pi` (so 90 degree is `np.pi / 2` and value is defined in the
            clockwise direction.

    Returns
    -------
        T_t: `np.array`
            The translation matrix used in affine transformation.
        T_r: `np.array`
            The rotation matrix used in affine transformation.
        trans_xy_coord: `np.array`
            The matrix that stores the translated and rotated coordinates.
    """

    trans_xy_coord = np.zeros((len(x), 2))

    T_t, T_r = np.zeros((3, 3)), np.zeros((3, 3))
    np.fill_diagonal(T_t, 1)
    np.fill_diagonal(T_r, 1)

    T_t[0, 2], T_t[1, 2] = -centroid_x[0], -centroid_y[0]

    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    T_r[0, 0], T_r[0, 1] = cos_theta, sin_theta
    T_r[1, 0], T_r[1, 1] = -sin_theta, cos_theta

    for cur_x, cur_y, cur_ind in zip(x, y, np.arange(len(x))):
        data = np.array([cur_x, cur_y, 1])
        res = T_t @ data
        res = T_r @ res
        trans_xy_coord[cur_ind, :] = res[:2]

    return T_t, T_r, trans_xy_coord


def add_spatial(adata):
    adata.obsm["spatial"] = adata.obs.loc[:, ["coor_x", "coor_y"]].values
    adata.obsm["X_spatial"] = adata.obsm["spatial"].astype(float).copy()


def add_spatial_intron(adata):
    df = pd.Series(adata.obs.index).str.split(":", expand=True).iloc[:, 1].str.split("_", expand=True)
    adata.obsm["spatial"] = df.values.astype(float)
    adata.obsm["X_spatial"] = df.values.astype(float)


def alpha_shape(x, y, alpha):
    crds = np.array([x.flatten(), y.flatten()]).transpose()
    points = MultiPoint(crds)

    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points


def pca_align(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    R = pca.components_
    Y = (R @ X.T).T

    return Y, R


def AffineTrans(x, y, centroid_x, centroid_y, R, theta=None):
    trans_xy_coord = np.zeros((len(x), 2))

    T_t, T_r = np.zeros((3, 3)), np.zeros((3, 3))
    np.fill_diagonal(T_t, 1)
    np.fill_diagonal(T_r, 1)

    T_t[0, 2], T_t[1, 2] = -centroid_x[0], -centroid_y[0]

    if theta is not None:
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        T_r[0, 0], T_r[0, 1] = cos_theta, sin_theta
        T_r[1, 0], T_r[1, 1] = -sin_theta, cos_theta
    else:
        T_r[:2, :2] = R

    for cur_x, cur_y, cur_ind in zip(x, y, np.arange(len(x))):
        data = np.array([cur_x, cur_y, 1])
        res = T_t @ data
        res = T_r @ res
        trans_xy_coord[cur_ind, :] = res[:2]

    return T_t, T_r, trans_xy_coord


def correct_embryo_coord(adata):
    """

    Args:
        adata:

    Returns:

    """

    if "unspliced" in adata.layers:
        add_spatial_intron(adata)
    else:
        add_spatial(adata)

    adata_coords = adata.obsm["X_spatial"]
    x, y = adata_coords[:, 0], adata_coords[:, 1]

    adata_concave_hull, _ = alpha_shape(x, y, alpha=1)

    if type(adata_concave_hull) == shapely.geometry.multipolygon.MultiPolygon:
        adata_x, adata_y = adata_concave_hull[0].exterior.xy
    else:
        adata_x, adata_y = adata_concave_hull.exterior.xy

    adata_centroid_x, adata_centroid_y = adata_concave_hull.centroid.coords.xy

    adata_coords_correct, adata_R = pca_align(adata_coords)
    adata_spatial_corrected = AffineTrans(
        adata_coords[:, 0],
        adata_coords[:, 1],
        adata_centroid_x,
        adata_centroid_y,
        adata_R,
    )

    # rotate 90 degree
    _, _, adata_coords_correct_2 = AffineTrans(
        adata_spatial_corrected[:, 0],
        adata_spatial_corrected[:, 1],
        [0, 0],
        [0, 0],
        None,
        np.pi / 2,
    )
    # reflect vertically
    adata_coords_correct_2[:, 1] = -adata_coords_correct_2[:, 1]
    adata.obsm["X_spatial"] = adata_coords_correct_2

    # reflect vertically again:
    adata_coords_correct_2[:, 1] = -adata_coords_correct_2[:, 1]

    # account for the mirror effect when plotting an image
    adata.obsm["spatial"] = adata_coords_correct_2
