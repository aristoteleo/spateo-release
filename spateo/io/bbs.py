"""IO functions for calculating the bounding box.
"""

import math

from typing import Optional, Tuple, Union, List

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
    multipolygon,
)
from shapely.ops import unary_union, polygonize

from .bgi import read_bgi_agg
from .utils import centroids
from ..configuration import SKM


def alpha_shape(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1,
    buffer: float = 1,
    vectorize: bool = True,
) -> Tuple[Union[MultiPolygon, Polygon], List]:
    """Compute the alpha shape (concave hull) of a set of points.
    Code adapted from: https://gist.github.com/dwyerk/10561690

    Args:
        x: x-coordinates of the DNA nanoballs or buckets, etc.
        y: y-coordinates of the DNA nanoballs or buckets, etc.
        alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
        buffer: the buffer used to smooth and clean up the shapley identified concave hull polygon.
        vectorize: Whether to vectorize the alpha-shape calculation instead of looping through.

    Returns:
        alpha_hull: The computed concave hull.
        edge_points: The coordinates of the edge of the resultant concave hull.
    """

    coords = np.array([x.flatten(), y.flatten()]).transpose()
    points = MultiPoint(coords)

    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    tri = Delaunay(coords)
    if vectorize:
        # ia, ib, ic = indices of corner points of the triangle
        triangles = coords[tri.vertices]

        # Lengths of sides of triangle
        a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
        b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
        c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        circums = a * b * c / (4.0 * areas)

        # Here's the radius filter.
        filtered = triangles[circums < (1.0 / alpha)]
        edge1 = filtered[:, (0, 1)]
        edge2 = filtered[:, (1, 2)]
        edge3 = filtered[:, (2, 0)]
        edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    else:

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
    binsize: int = 20,
    min_agg_umi: int = 0,
    alpha: float = 1.0,
    buffer: Optional[float] = None,
) -> Tuple[Polygon, List]:
    """Return the convex hull of all nanoballs that have non-zero UMI (or at least > min_agg_umi UMI).

    Args:
        path: Path to read file.
        binsize: The number of spatial bins to aggregate RNAs captured by DNBs in those bins. By default it is 20, which
                    is close to the size of a single cell. If stereo-seq chip used is bigger than 1 x 1 mm, you may need
                    to increase the binsize.
        min_agg_umi: the minimal aggregated UMI number for the bucket.
        alpha: alpha value to influence the gooeyness of the border. Smaller numbers don't fall inward as much as
                larger numbers. Too large, and you lose everything!
        buffer: the buffer used to smooth and clean up the shapley identified concave hull polygon.

    Returns:
        alpha_hull: The computed concave hull.
        edge_points: The coordinates of the edge of the resultant concave hull.
    """
    adata = read_bgi_agg(path, binsize=binsize)

    i, j = (adata.X > min_agg_umi).nonzero()

    x_min = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_XMIN_KEY)
    y_min = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_YMIN_KEY)

    # We use centroids function to get the true stereo-seq chip coordinates.
    if binsize != 1:
        i = centroids(i, coord_min=x_min, binsize=binsize)
        j = centroids(j, coord_min=y_min, binsize=binsize)
    else:
        i, j = i + x_min, j + y_min

    # TODO: give warnings when alpha_shape returns nothing and ask the users to set alpha to be smaller.
    if buffer is None:
        buffer = binsize

    return alpha_shape(i, j, alpha, buffer, vectorize=True)
