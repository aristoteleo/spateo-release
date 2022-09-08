"""
Plots to visualize results from cell-cell colocalization based analyses, as well as cell-cell communication
inference-based analyses. Makes use of dotplot-generating functions
"""
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...configuration import reset_rcParams
from ...plotting.static.dotplot import Dotplot
from ...tools.labels import Label, interlabel_connections


def plot_connections(
    label: Label,
    weights_matrix: Union[scipy.sparse.csr_matrix, np.ndarray],
    ax: Union[None, mpl.axes.Axes] = None,
    figsize: tuple = (8, 8),
    zero_self_connections: bool = True,
    normalize_by_self_connections: bool = False,
    shapes_style: bool = True,
    label_outline: bool = False,
    max_scale: float = 0.46,
    colormap: Union[str, dict, "mpl.colormap"] = "Spectral",
    title_str="connection strengths between types",
    title_fontsize: float = 12,
    label_fontsize: float = 12,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
):
    """
    Plot connections between labels- visualization of how closely labels are colocalized

    Args:
        label : class `Label`
            This class contains attributes related to the labeling of each sample
        weights_matrix : sparse matrix or numpy array
            Spatial distance matrix, weighted by distance between spots
        ax : optional `matplotlib.Axes`
            Existing axes object, if applicable
        figsize : (int, int) tuple
            Width x height of desired figure window in inches
        zero_self_connections : bool, default True
            Ignores intra-label interactions
        normalize_by_self_connections : bool, default False
            Only used if 'zero_self_connections' is False. If True, normalize intra-label connections by the number of
            spots of that label
        shapes_style : bool, default True
            If True plots squares, if False plots heatmap
        label_outline : bool, default False
            If True, gives dark outline to axis tick label text
        max_scale : float, default 0.46
            Only used for the case that 'shape_style' is True, gives maximum size of square
        colormap : str, dict, or matplotlib.colormap
            Specifies colors to use for plotting. If dictionary, keys should be numerical labels corresponding to those
            of the Label object.
        title_str : str
            Give plot a title
        title_fontsize : float
            Size of plot title
        label_fontsize : float
            Size of labels along the axes of the graph
        save_show_or_return : str, default "show"
            Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If "all", the figure will be saved, displayed
            and the associated axis and other object will be return.
        save_kwargs : optional dict
            A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.

    Returns:
        (fig, ax) :
            Returns plot and axis object if 'save_show_or_return' is "all"
    """
    from ...plotting.static.utils import save_fig
    from ...tools.utils import update_dict

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax
        fig = ax.get_figure()
    connections = interlabel_connections(label, weights_matrix)

    if zero_self_connections:
        np.fill_diagonal(connections, 0)
    elif normalize_by_self_connections:
        connections /= connections.diagonal()[:, np.newaxis]

    connections_max = np.amax(connections)

    # Set label colors:
    if isinstance(colormap, str):
        cmap = mpl.cm.get_cmap(colormap)
    else:
        cmap = colormap

    # If colormap is given, map label ID to points along the colormap. If dictionary is given, instead map each label
    # to a color using the dictionary keys as guides.
    if isinstance(cmap, dict):
        if type(list(cmap.keys())[0]) == str:
            id_colors = {n_id: cmap[id] for n_id, id in zip(label.ids, label.str_ids)}
        else:
            id_colors = {id: cmap[id] for id in label.ids}
    else:
        id_colors = {id: cmap(id / label.max_id) for id in label.ids}

    if shapes_style:
        # Cell types/labels will be represented using triangles:
        left_triangle = np.array(
            (
                (-1.0, 1.0),
                # (1., 1.),
                (1.0, -1.0),
                (-1.0, -1.0),
            )
        )

        right_triangle = np.array(
            (
                (-1.0, 1.0),
                (1.0, 1.0),
                (1.0, -1.0),
                # (-1., -1.)
            )
        )

        polygon_list = []
        color_list = []

        ax.set_ylim(-0.55, label.num_labels - 0.45)
        ax.set_xlim(-0.55, label.num_labels - 0.45)

        for label_1 in range(connections.shape[0]):
            for label_2 in range(connections.shape[1]):

                if label_1 <= label_2:

                    for triangle in [left_triangle, right_triangle]:
                        center = np.array((label_1, label_2))[np.newaxis, :]
                        scale_factor = connections[label_1, label_2] / connections_max
                        offsets = triangle * max_scale * scale_factor
                        polygon_list.append(center + offsets)

                    color_list += (id_colors[label.ids[label_2]], id_colors[label.ids[label_1]])

        collection = PolyCollection(polygon_list, facecolors=color_list, edgecolors="face", linewidths=0)

        ax.add_collection(collection)

        # Remove ticks
        ax.tick_params(labelbottom=False, labeltop=True, top=False, bottom=False, left=False)
        ax.xaxis.set_tick_params(pad=-2)
    else:
        # Heatmap of connection strengths
        heatmap = ax.imshow(connections, cmap=colormap, interpolation="nearest")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        fig.colorbar(heatmap, cax=cax)
        cax.tick_params(axis="both", which="major", labelsize=6, rotation=-45)

        # Change formatting if values too small
        if connections_max < 0.001:
            cax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1e}"))

    # Formatting adjustments
    ax.set_aspect("equal")

    ax.set_xticks(
        np.arange(label.num_labels),
    )
    text_outline = [PathEffects.Stroke(linewidth=0.5, foreground="black", alpha=0.8)] if label_outline else None

    # If label has categorical labels associated, use those to label the axes instead:
    if label.str_map is not None:
        ax.set_xticklabels(
            label.str_ids,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=90,
            path_effects=text_outline,
        )
    else:
        ax.set_xticklabels(
            label.ids,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=0,
            path_effects=text_outline,
        )

    ax.set_yticks(np.arange(label.num_labels))
    if label.str_map is not None:
        ax.set_yticklabels(
            label.str_ids,
            fontsize=label_fontsize,
            fontweight="bold",
            path_effects=text_outline,
        )
    else:
        ax.set_yticklabels(
            label.ids,
            fontsize=label_fontsize,
            fontweight="bold",
            path_effects=text_outline,
        )

    for ticklabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for n, id in enumerate(label.ids):
            ticklabels[n].set_color(id_colors[id])

    ax.set_title(title_str, fontsize=title_fontsize, fontweight="bold")
    if save_show_or_return in ["save", "both", "all"]:
        s_kwargs = {
            "path": None,
            "prefix": "connections",
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)

    elif save_show_or_return in ["show", "both", "all"]:
        plt.show()
    elif save_show_or_return in ["return", "all"]:
        return (fig, ax)


def heatmap(data: Union[np.ndarray, pd.DataFrame]):
    """
    Generates and plots heatmap from array or dataframe, where x- and y-axes are two variable categories (e.g. can be
    cell type-gene or cell types on both axes), and the element magnitude is the relation between them.

    Args:
        data : np.ndarray or pd.DataFrame
    """
    reset_rcParams()

    # def

    # def volcano_plot():
    """
    Generates and plots volcano plot for visualization of the relative log-fold change and FDR-corrected p-values for
    features between two groups
    """
