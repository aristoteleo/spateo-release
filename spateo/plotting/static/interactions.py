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
from matplotlib import rcParams
from matplotlib.collections import PolyCollection
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...configuration import config_spateo_rcParams
from ...plotting.static.dotplot import CCDotplot
from ...tools.labels import Label, interlabel_connections


def plot_connections(
    label: Label,
    spatial_weights_matrix: Union[scipy.sparse.csr_matrix, np.ndarray],
    expr_weights_matrix: Union[None, scipy.sparse.csr_matrix, np.ndarray] = None,
    reverse_expr_plot_orientation: bool = False,
    ax: Union[None, mpl.axes.Axes] = None,
    figsize: tuple = (8, 8),
    zero_self_connections: bool = True,
    normalize_by_self_connections: bool = False,
    shapes_style: bool = True,
    label_outline: bool = False,
    max_scale: float = 0.46,
    colormap: Union[str, dict, "mpl.colormap"] = "Spectral",
    title_str: Union[None, str] = None,
    title_fontsize: Union[None, float] = None,
    label_fontsize: Union[None, float] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = {},
):
    """
    Plot spatial_connections between labels- visualization of how closely labels are colocalized

    Args:
        label : class `Label`
            This class contains attributes related to the labeling of each sample
        weights_matrix : sparse matrix or numpy array
            Spatial distance matrix, weighted by distance between spots
        expr_weights_matrix : optional sparse matrix or numpy array
            Gene expression distance matrix, weighted by distance in transcriptomic or PCA space. If not given,
            only the spatial distance matrix will be plotted. If given, will plot the spatial distance matrix in the
            left plot and the gene expression distance matrix in the right plot.
        reverse_expr_plot_orientation : bool, default False
            If True, plot the gene expression connections in the form of a lower right triangle. If False,
            gene expression connections will be an upper left triangle just like the spatial connections.
        ax : optional `matplotlib.Axes`
            Existing axes object, if applicable
        figsize : (int, int) tuple
            Width x height of desired figure window in inches
        zero_self_connections : bool, default True
            Ignores intra-label interactions
        normalize_by_self_connections : bool, default False
            Only used if 'zero_self_connections' is False. If True, normalize intra-label connections by
            the number of spots of that label
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

    config_spateo_rcParams()
    title_fontsize = rcParams.get("axes.titlesize") if title_fontsize is None else title_fontsize
    label_fontsize = rcParams.get("axes.labelsize") if label_fontsize is None else label_fontsize

    if ax is None:
        if expr_weights_matrix is not None:
            figsize = (figsize[0] * 2.25, figsize[1])
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax_sp, ax_expr = axes[0], axes[1]

            if reverse_expr_plot_orientation:
                # Allow subplot boundaries to technically be partially overlapping (for better visual)
                box = ax_expr.get_position()
                box.x0 = box.x0 - 0.3
                box.x1 = box.x1 - 0.3
                ax_expr.set_position(box)
        else:
            fig, ax_sp = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = ax
        if len(ax) > 1:
            ax_sp, ax_expr = ax[0], ax[1]
        else:
            ax_sp = ax
        fig = ax.get_figure()

    # Compute spatial connections array:
    spatial_connections = interlabel_connections(label, spatial_weights_matrix)

    if zero_self_connections:
        np.fill_diagonal(spatial_connections, 0)
    elif normalize_by_self_connections:
        spatial_connections /= spatial_connections.diagonal()[:, np.newaxis]

    spatial_connections_max = np.amax(spatial_connections)

    # Optionally, compute gene expression connections array:
    if expr_weights_matrix is not None:
        expr_connections = interlabel_connections(label, expr_weights_matrix)

        if zero_self_connections:
            np.fill_diagonal(expr_connections, 0)
        elif normalize_by_self_connections:
            expr_connections /= expr_connections.diagonal()[:, np.newaxis]

        expr_connections_max = np.amax(expr_connections)

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


    # -------------------------------- Spatial Connections Plot- Setup -------------------------------- #
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

        ax_sp.set_ylim(-0.55, label.num_labels - 0.45)
        ax_sp.set_xlim(-0.55, label.num_labels - 0.45)

        for label_1 in range(spatial_connections.shape[0]):
            for label_2 in range(spatial_connections.shape[1]):

                if label_1 <= label_2:

                    for triangle in [left_triangle, right_triangle]:
                        center = np.array((label_1, label_2))[np.newaxis, :]
                        scale_factor = spatial_connections[label_1, label_2] / spatial_connections_max
                        offsets = triangle * max_scale * scale_factor
                        polygon_list.append(center + offsets)

                    color_list += (id_colors[label.ids[label_2]], id_colors[label.ids[label_1]])

        collection = PolyCollection(polygon_list, facecolors=color_list, edgecolors="face", linewidths=0)

        ax_sp.add_collection(collection)

        # Remove ticks
        ax_sp.tick_params(labelbottom=False, labeltop=True, top=False, bottom=False, left=False)
        ax_sp.xaxis.set_tick_params(pad=-2)
    else:
        # Heatmap of connection strengths
        heatmap = ax_sp.imshow(spatial_connections, cmap=colormap, interpolation="nearest")

        divider = make_axes_locatable(ax_sp)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        fig.colorbar(heatmap, cax=cax)
        cax.tick_params(axis="both", which="major", labelsize=6, rotation=-45)

        # Change formatting if values too small
        if spatial_connections_max < 0.001:
            cax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1e}"))

    # Formatting adjustments
    ax_sp.set_aspect("equal")

    ax_sp.set_xticks(
        np.arange(label.num_labels),
    )
    text_outline = [PathEffects.Stroke(linewidth=0.5, foreground="black", alpha=0.8)] if label_outline else None

    # If label has categorical labels associated, use those to label the axes instead:
    if label.str_map is not None:
        ax_sp.set_xticklabels(
            label.str_ids,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=90,
            path_effects=text_outline,
        )
    else:
        ax_sp.set_xticklabels(
            label.ids,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=0,
            path_effects=text_outline,
        )

    ax_sp.set_yticks(np.arange(label.num_labels))
    if label.str_map is not None:
        ax_sp.set_yticklabels(
            label.str_ids,
            fontsize=label_fontsize,
            fontweight="bold",
            path_effects=text_outline,
        )
    else:
        ax_sp.set_yticklabels(
            label.ids,
            fontsize=label_fontsize,
            fontweight="bold",
            path_effects=text_outline,
        )

    for ticklabels in [ax_sp.get_xticklabels(), ax_sp.get_yticklabels()]:
        for n, id in enumerate(label.ids):
            ticklabels[n].set_color(id_colors[id])

    title_str_sp = "Spatial Connections" if title_str is None else title_str
    ax_sp.set_title(title_str_sp, fontsize=title_fontsize, fontweight="bold")

    # ------------------------------ Optional Gene Expression Connections Plot- Setup ------------------------------ #
    if expr_weights_matrix is not None:
        if shapes_style:
            polygon_list = []
            color_list = []

            ax_expr.set_ylim(-0.55, label.num_labels - 0.45)
            ax_expr.set_xlim(-0.55, label.num_labels - 0.45)

            for label_1 in range(expr_connections.shape[0]):
                for label_2 in range(expr_connections.shape[1]):

                    if label_1 <= label_2:
                        for triangle in [left_triangle, right_triangle]:
                            center = np.array((label_1, label_2))[np.newaxis, :]
                            scale_factor = expr_connections[label_1, label_2] / expr_connections_max
                            offsets = triangle * max_scale * scale_factor
                            polygon_list.append(center + offsets)

                        color_list += (id_colors[label.ids[label_2]], id_colors[label.ids[label_1]])

                # Remove ticks
                if reverse_expr_plot_orientation:
                    ax_expr.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True,
                                        top=False, bottom=False, left=False)
                    # Flip x- and y-axes of the expression plot:
                    ax_expr.invert_xaxis()
                    ax_expr.invert_yaxis()
                else:
                    ax_expr.tick_params(labelbottom=False, labeltop=True, top=False, bottom=False, left=False)
                ax_expr.xaxis.set_tick_params(pad=-2)

            collection = PolyCollection(polygon_list, facecolors=color_list, edgecolors="face", linewidths=0)

            ax_expr.add_collection(collection)
        else:
            # Heatmap of connection strengths
            heatmap = ax_expr.imshow(expr_connections, cmap=colormap, interpolation="nearest")

            divider = make_axes_locatable(ax_expr)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            fig.colorbar(heatmap, cax=cax)
            cax.tick_params(axis="both", which="major", labelsize=6, rotation=-45)

            # Change formatting if values too small
            if spatial_connections_max < 0.001:
                cax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1e}"))

        # Formatting adjustments
        ax_expr.set_aspect("equal")

        ax_expr.set_xticks(
            np.arange(label.num_labels),
        )
        if reverse_expr_plot_orientation:
            # Despine both spatial connections & gene expression connections plots:
            ax_sp.spines['right'].set_visible(False)
            ax_sp.spines['top'].set_visible(False)
            ax_sp.spines['left'].set_visible(False)
            ax_sp.spines['bottom'].set_visible(False)

            ax_expr.spines['right'].set_visible(False)
            ax_expr.spines['top'].set_visible(False)
            ax_expr.spines['left'].set_visible(False)
            ax_expr.spines['bottom'].set_visible(False)

        text_outline = [PathEffects.Stroke(linewidth=0.5, foreground="black", alpha=0.8)] if label_outline else None

        # If label has categorical labels associated, use those to label the axes instead:
        if label.str_map is not None:
            ax_expr.set_xticklabels(
                label.str_ids,
                fontsize=label_fontsize,
                fontweight="bold",
                rotation=90,
                path_effects=text_outline,
            )
        else:
            ax_expr.set_xticklabels(
                label.ids,
                fontsize=label_fontsize,
                fontweight="bold",
                rotation=0,
                path_effects=text_outline,
            )

        ax_expr.set_yticks(np.arange(label.num_labels))
        if label.str_map is not None:
            ax_expr.set_yticklabels(
                label.str_ids,
                fontsize=label_fontsize,
                fontweight="bold",
                path_effects=text_outline,
            )
        else:
            ax_expr.set_yticklabels(
                label.ids,
                fontsize=label_fontsize,
                fontweight="bold",
                path_effects=text_outline,
            )

        for ticklabels in [ax_expr.get_xticklabels(), ax_expr.get_yticklabels()]:
            for n, id in enumerate(label.ids):
                ticklabels[n].set_color(id_colors[id])

        title_str_expr = "Gene Expression Similarity" if title_str is None else title_str
        if reverse_expr_plot_orientation:
            if label_fontsize <= 8:
                y = -0.35
            elif label_fontsize > 8:
                y = -0.25
        else:
            y = None
        ax_expr.set_title(title_str_expr, fontsize=title_fontsize, fontweight="bold", y=y)

    prefix = "spatial_connections" if expr_weights_matrix is None else "spatial_and_expr_connections"
    if save_show_or_return in ["save", "both", "all"]:
        s_kwargs = {
            "path": None,
            "prefix": prefix,
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
        if expr_weights_matrix is not None:
            ax = axes
        else:
            ax = ax_sp
        return (fig, ax)


def ligrec():
    """
    Dotplot for visualizing results of ligand-receptor interaction analysis

    """


def heatmap(data: Union[np.ndarray, pd.DataFrame]):
    """
    Generates and plots heatmap from array or dataframe, where x- and y-axes are two variable categories (e.g. can be
    cell type-gene or cell types on both axes), and the element magnitude is the relation between them.

    Args:
        data : np.ndarray or pd.DataFrame
    """
    config_spateo_rcParams()

    # def

    # def volcano_plot():
    """
    Generates and plots volcano plot for visualization of the relative log-fold change and FDR-corrected p-values for
    features between two groups
    """
