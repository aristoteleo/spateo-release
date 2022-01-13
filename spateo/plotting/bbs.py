"""Plotting functions for creating the bounding box.
"""

from typing import Union, Tuple, List, Optional

import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from shapely.geometry import MultiPolygon, Polygon


def polygon(
    concave_hull: Union[MultiPolygon, Polygon],
    figsize: Union[Tuple, List] = (10, 10),
    margin: float = 0.3,
    fc: str = "#999999",
    ec: str = "#000000",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """Plot the polygon identified by the alpha hull method.

    Args:
        concave_hull: The identified polygon (or multi-polygon) returned from the alpha_shape method or other methods.
        figsize: The size of the figure
        margin: The margin of the figure Axes.
        fc: The facecolor of the PolygonPatch.
        ec: The edgecolor of the PolygonPatch.

    Returns:
        fig: The matplotlib.figure figure object of the figure.
        ax: The matplotlib.axes._subplots AxesSubplot object of the figure.
    """

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    x_min, y_min, x_max, y_max = concave_hull.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(concave_hull, fill=True, zorder=-1, fc=fc, ec=ec)
    ax.add_patch(patch)
    return fig, ax


def delaunay(
    edge_points,
    figsize: Union[Tuple, List] = (10, 10),
    pc: str = "#f16824",
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    """Plot the Delaunay triangulation result.

    Args:
        edge_points:
        figsize: The size of the figure
        fc: The color of the scatter points.
        title: The title of the figure.
        fig: The matplotlib.figure figure object of the figure.
        ax: The matplotlib.axes._subplots AxesSubplot object of the figure.

    Returns:
        fig: The matplotlib.figure figure object of the figure.
        ax: The matplotlib.axes._subplots AxesSubplot object of the figure.
    """

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    lines = LineCollection(edge_points)
    plt.gca().add_collection(lines)
    delaunay_points = np.vstack([point for point in edge_points])
    plt.plot(delaunay_points[:, 0], delaunay_points[:, 1], "o", hold=1, color=pc)
    plt.title(title)

    return fig, ax
