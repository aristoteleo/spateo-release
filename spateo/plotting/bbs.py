"""Plotting functions for creating the bounding box.
"""

from typing import Union, Tuple, List

from descartes import PolygonPatch
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon


def polygon(
    polygon: Union[MultiPolygon, Polygon],
    figsize: Union[Tuple, List] = (10, 10),
    margin: float = 0.3,
    fc: str = "#999999",
    ec: str = "#000000",
):
    """Plot the polygon identified by the alpha hull method.

    Args:
        polygon: The identified polygon (or multi-polygon) returned from the alpha_shape method or other methods.
        figsize: The size of the figure
        margin: The margin of the figure Axes.
        fc: The facecolor of the PolygonPatch.
        ec: The edgecolor of the PolygonPatch.

    Returns:
        The figure of the polygon.

    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fill=True, zorder=-1, fc=fc, ec=ec)
    ax.add_patch(patch)
    return fig
