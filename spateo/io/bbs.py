"""IO functions for calculating the bounding box.
"""

from typing import Optional, Tuple, Union, List

import math
import numpy as np
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
    multipolygon,
)
from scipy.spatial import Delaunay
from shapely.ops import unary_union, polygonize

from .bgi import read_bgi_agg
from .utils import centroids


def alpha_shape(
    x: np.ndarray,
    y: np.ndarray,
    alpha: Optional[float] = 1,
    buffer: Optional[float] = 1,
) -> Tuple[Polygon, list]:
    """Compute the alpha shape (concave hull) of a set of points.
    Code adapted from: https://gist.github.com/dwyerk/10561690

    Args:
        x: x-coordinates of the DNA nanoballs or buckets, etc.
        y: y-coordinates of the DNA nanoballs or buckets, etc.
        alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
        buffer: the buffer used to smooth and clean up the shapley identified concave hull polygon.

    Returns:
        alpha_hull: The computed concave hull.
        edge_points: The coordinates of the edge of the resultant concave hull.
    """

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

    triangles = list(polygonize(edge_points))
    alpha_hull = unary_union(triangles)

    if buffer != 0:
        alpha_hull.buffer(buffer)

    return alpha_hull, edge_points


def get_concave_hull(
    path: str,
    binsize: Optional[int] = 1,
    min_agg_umi: Optional[int] = 0,
    alpha: Optional[float] = 1.0,
    buffer: Optional[float] = 1.0,
) -> Tuple[Polygon, list]:
    """Return the convex hull of all nanoballs that have non-zero UMI (or at least > min_agg_umi UMI).

    Args:
        path: Path to read file.
        binsize: The number of spatial bins to aggregate RNAs captured by DNBs in those bins. By default it is 1. If
                    stereo-seq chip used is bigger than 1 x 1 mm, you may need to increase the binsize.
        min_agg_umi: the minimal aggregated UMI number for the bucket.
        alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
        buffer: the buffer used to smooth and clean up the shapley identified concave hull polygon.

    Returns:
        alpha_hull: The computed concave hull.
        edge_points: The coordinates of the edge of the resultant concave hull.
    """
    total_agg = read_bgi_agg(path, binsize=binsize)[0]

    i, j = (total_agg > min_agg_umi).nonzero()

    # We use centroids function to get the true stereo-seq chip coordinates.
    if binsize != 1:
        i, j = centroids(i, binsize=binsize), centroids(j, binsize=binsize)

    return alpha_shape(i, j, alpha, buffer)
