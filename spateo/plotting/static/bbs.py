"""Plotting functions for creating the bounding box.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from shapely.geometry import MultiPolygon, Polygon
from typing_extensions import Literal

from .utils import save_return_show_fig_utils


def polygon(
    concave_hull: Union[MultiPolygon, Polygon],
    figsize: Union[Tuple, List] = (10, 10),
    margin: float = 0.3,
    fc: str = "#999999",
    ec: str = "#000000",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[Dict] = None,
):
    """Plot the polygon identified by the alpha hull method.

    Args:
        concave_hull: The identified polygon (or multi-polygon) returned from the alpha_shape method or other methods.
        figsize: The size of the figure
        margin: The margin of the figure Axes.
        fc: The facecolor of the PolygonPatch.
        ec: The edgecolor of the PolygonPatch.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        save_show_or_return: Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
            as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs.

    Returns:
        fig: The matplotlib.figure figure object of the figure.
        ax: The matplotlib.axes._subplots AxesSubplot object of the figure.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    x_min, y_min, x_max, y_max = concave_hull.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(concave_hull, fill=True, zorder=-1, fc=fc, ec=ec)
    ax.add_patch(patch)

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)
    else:
        _background = background

    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=False,
        background=_background,
        prefix="scatters",
        save_kwargs=save_kwargs or {},
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )


def delaunay(
    edge_points,
    figsize: Union[Tuple, List] = (10, 10),
    pc: str = "#f16824",
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[Dict] = None,
):
    """Plot the Delaunay triangulation result.

    Args:
        edge_points:
        figsize: The size of the figure
        fc: The color of the scatter points.
        title: The title of the figure.
        fig: The matplotlib.figure figure object of the figure.
        ax: The matplotlib.axes._subplots AxesSubplot object of the figure.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        save_show_or_return: Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
            as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs.

    Returns:
        fig: The matplotlib.figure figure object of the figure.
        ax: The matplotlib.axes._subplots AxesSubplot object of the figure.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    lines = LineCollection([[tuple(i[0]), tuple(i[1])] for i in edge_points], color="blue")
    ax.add_collection(lines)
    delaunay_points = np.vstack([point for point in edge_points])
    ax.plot(delaunay_points[:, 0], delaunay_points[:, 1], "o", color=pc)
    ax.set_title(title)

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)
    else:
        _background = background

    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=False,
        background=_background,
        prefix="scatters",
        save_kwargs=save_kwargs or {},
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )
