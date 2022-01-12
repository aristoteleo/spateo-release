"""IO utility functions.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Point, LineString, Polygon
from skimage import measure


def bin_indices(
    coords: np.ndarray, coord_min: float, binsize: Optional[int] = 50
) -> int:
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


def centroids(
    bin_indices: np.ndarray, coord_min: Optional[float] = 0, binsize: Optional[int] = 50
) -> float:
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


def get_label_props(
    label_mtx: np.ndarray,
    properties: Tuple[str, ...] = ("label", "area", "bbox", "centroid"),
) -> pd.DataFrame:
    """Measure properties of labeled cell regions.

    Parameters
    ----------
        label_mtx: `numpy.ndarray`
            cell segmentation label matrix
        properties: `tuple`
            used properties

    Returns
    -------
        props: `pandas.DataFrame`
            A dataframe with properties and contours

    """

    def contours(mtx):
        """Get contours of a cell using `cv2.findContours`."""
        # padding and transfer label mtx to binary mtx
        mtx = np.pad(mtx, 1)
        mtx[mtx > 0] = 255
        mtx = mtx.astype(np.uint8)
        # get contours
        contour = cv2.findContours(mtx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][
            0
        ]
        # shift back coordinates
        contour = contour - np.array([1, 1])
        return contour

    def contour_to_geo(contour):
        """Transfer contours to `shapely.geometry`"""
        n = contour.shape[0]
        contour = np.squeeze(contour)
        if n >= 3:
            geo = Polygon(contour)
        elif n == 2:
            geo = LineString(contour)
        else:
            geo = Point(contour)
        return geo

    props = measure.regionprops_table(
        label_mtx, properties=properties, extra_properties=[contours]
    )
    props = pd.DataFrame(props)
    props["contours"] = props.apply(
        lambda x: x["contours"] + x[["bbox-0", "bbox-1"]].to_numpy(), axis=1
    )
    props["contours"] = props["contours"].apply(contour_to_geo)
    return props


def get_bin_props(data: pd.DataFrame, binsize: int) -> pd.DataFrame:
    """Simulate properties of bin regions.

    Parameters
    ----------
        data :
            The index of coordinates.
        binsize :
            The number of spatial bins to aggregate RNAs captured by DNBs in those bins.

    Returns
    -------
        props: `pandas.DataFrame`
            A dataframe with properties and contours

    """

    def create_geo(row):
        x, y = row["x_ind"], row["y_ind"]
        x *= binsize
        y *= binsize
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
        return geo

    contours = data.apply(create_geo, axis=1)
    props = pd.DataFrame({"contours": contours})
    props["area"] = binsize ** 2
    return props
