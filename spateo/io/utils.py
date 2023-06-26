"""IO utility functions.
"""
import math
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.wkb import dumps
from skimage import measure


def bin_indices(coords: np.ndarray, coord_min: float, binsize: int = 50) -> int:
    """Take a DNB coordinate, the mimimum coordinate and the binsize, calculate the index of bins for the current
    coordinate.

    Parameters
    ----------
        coord: `float`
            Current x or y coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        binsize: `float`
            Size of the bins to aggregate data.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """
    num = np.floor((coords - coord_min) / binsize)
    return num.astype(np.uint32)


def centroids(bin_indices: np.ndarray, coord_min: float = 0, binsize: int = 50) -> float:
    """Take a bin index, the mimimum coordinate and the binsize, calculate the centroid of the current bin.

    Parameters
    ----------
        bin_ind: `float`
            The bin index for the current coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        binsize: `int`
            Size of the bins to aggregate data.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """
    coord_centroids = coord_min + bin_indices * binsize + binsize / 2
    return coord_centroids


def contour_to_geo(contour):
    """Transfer contours to `shapely.geometry`"""
    n = contour.shape[0]
    if n >= 3:
        geo = Polygon(contour)
    elif n == 2:
        geo = LineString(contour)
    else:
        geo = Point(contour[0])
    geo = dumps(geo, hex=True)  # geometry object to hex
    return geo


def get_points_props(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate properties of labeled coordinates.

    Args:
        data: Pandas Dataframe containing `x`, `y`, `label` columns.

    Returns:
        A dataframe with properties and contours indexed by label
    """
    rows = []
    for label, _df in data.drop_duplicates(subset=["label", "x", "y"]).groupby("label"):
        points = _df[["x", "y"]].values.astype(int)
        min_offset = points.min(axis=0)
        max_offset = points.max(axis=0)
        min0, min1 = min_offset
        max0, max1 = max_offset
        hull = cv2.convexHull(points, returnPoints=True).squeeze(1)
        contour = contour_to_geo(hull)

        moments = cv2.moments(hull)
        area = moments["m00"]
        if area > 0:
            centroid0 = moments["m10"] / area
            centroid1 = moments["m01"] / area
        elif hull.shape[0] == 2:
            line = hull - min_offset
            mask = cv2.line(np.zeros((max_offset - min_offset + 1)[::-1], dtype=np.uint8), line[0], line[1], color=1).T
            area = mask.sum()
            centroid0, centroid1 = hull.mean(axis=0)
        elif hull.shape[0] == 1:
            area = 1
            centroid0, centroid1 = hull[0] + 0.5
        else:
            raise IOError(f"Convex hull contains {hull.shape[0]} points.")
        rows.append([str(label), area, min0, min1, max0 + 1, max1 + 1, centroid0, centroid1, contour])
    return pd.DataFrame(
        rows, columns=["label", "area", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1", "contour"]
    ).set_index("label")


def get_label_props(labels: np.ndarray) -> pd.DataFrame:
    """Measure properties of labeled cell regions.

    Args:
        labels: cell segmentation label matrix

    Returns:
        A dataframe with properties and contours indexed by label
    """

    def contour(mtx):
        """Get contours of a cell using `cv2.findContours`."""
        mtx = mtx.astype(np.uint8)
        contours = cv2.findContours(mtx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        assert len(contours) == 1
        return contours[0].squeeze(1)

    props = measure.regionprops_table(
        labels, properties=("label", "area", "bbox", "centroid"), extra_properties=[contour]
    )
    props = pd.DataFrame(props)
    props["contour"] = props.apply(lambda x: x["contour"] + x[["bbox-0", "bbox-1"]].to_numpy(), axis=1)
    props["contour"] = props["contour"].apply(contour_to_geo)
    return props.set_index(props["label"].astype(str)).drop(columns="label")


def get_bin_props(data: pd.DataFrame, binsize: int) -> pd.DataFrame:
    """Simulate properties of bin regions.

    Args:
        data: Pandas dataframe containing binned x, y, and cell labels.
            There should not be any duplicate cell labels.
        binsize: Bin size used

    Returns:
        A dataframe with properties and contours indexed by cell label
    """

    def create_geo(row):
        x, y = row["x"] * binsize, row["y"] * binsize
        if binsize > 1:
            geo = Polygon(
                [
                    (x, y),
                    (x + binsize, y),
                    (x + binsize, y + binsize),
                    (x, y + binsize),
                    (x, y),
                ]
            )
        else:
            geo = Point((x, y))
        geo = dumps(geo, hex=True)  # geometry object to hex
        return geo

    props = pd.DataFrame(
        {
            "label": data["label"].copy(),
            "contour": data.apply(create_geo, axis=1),
            "centroid-0": centroids(data["x"], 0, binsize),
            "centroid-1": centroids(data["y"], 0, binsize),
        }
    )
    props["area"] = binsize**2
    props["bbox-0"] = data["x"] * binsize
    props["bbox-1"] = data["y"] * binsize
    props["bbox-2"] = (data["x"] + 1) * binsize + 1
    props["bbox-3"] = (data["y"] + 1) * binsize + 1
    return props.set_index("label")


def in_concave_hull(p: np.ndarray, concave_hull: Union[Polygon, MultiPolygon]) -> np.ndarray:
    """Test if points in `p` are in `concave_hull` using scipy.spatial Delaunay's find_simplex.

    Args:
        p: a `Nx2` coordinates of `N` points in `K` dimensions
        concave_hull: A polygon returned from the concave_hull function (the first value).

    Returns:

    """
    assert p.shape[1] == 2, "this function only works for two dimensional data points."

    res = [concave_hull.intersects(Point(i)) for i in p]

    return np.array(res)


def in_convex_hull(p: np.ndarray, convex_hull: Union[Delaunay, np.ndarray]) -> np.ndarray:
    """Test if points in `p` are in `convex_hull` using scipy.spatial Delaunay's find_simplex.

    Args:
        p: a `NxK` coordinates of `N` points in `K` dimensions
        convex_hull: either a scipy.spatial.Delaunay object or the `MxK` array of the coordinates of `M` points in `K`
              dimensions for which Delaunay triangulation will be computed.

    Returns:

    """
    assert p.shape[1] == convex_hull.shape[1], "the second dimension of p and hull must be the same."

    if not isinstance(convex_hull, Delaunay):
        hull = Delaunay(convex_hull)

    return hull.find_simplex(p) >= 0


def bin_matrix(X: Union[np.ndarray, spmatrix], binsize: int) -> Union[np.ndarray, csr_matrix]:
    """Bin a matrix.

    Args:
        X: Dense or sparse matrix.
        binsize: Bin size

    Returns:
        Dense or spares matrix, depending on what the input was.
    """
    shape = (math.ceil(X.shape[0] / binsize), math.ceil(X.shape[1] / binsize))

    def _bin_sparse(X):
        nz = X.nonzero()
        x, y = nz
        data = X[nz].A.flatten()
        x_bin = bin_indices(x, 0, binsize)
        y_bin = bin_indices(y, 0, binsize)
        return csr_matrix((data, (x_bin, y_bin)), shape=shape, dtype=X.dtype)

    def _bin_dense(X):
        binned = np.zeros(shape, dtype=X.dtype)
        for x in range(X.shape[0]):
            x_bin = bin_indices(x, 0, binsize)
            for y in range(X.shape[1]):
                y_bin = bin_indices(y, 0, binsize)
                binned[x_bin, y_bin] += X[x, y]
        return binned

    if issparse(X):
        return _bin_sparse(X)
    return _bin_dense(X)


def get_coords_labels(labels: np.ndarray) -> pd.DataFrame:
    """Convert labels into sparse-format dataframe.

    Args:
        labels: cell segmentation labels matrix.

    Returns:
        A DataFrame of columns "x", "y", and "label". The coordinates are
        relative to the labels matrix.
    """
    nz = labels.nonzero()
    x, y = nz
    data = labels[nz]
    values = np.vstack((x, y, data)).T
    return pd.DataFrame(values, columns=["x", "y", "label"])
