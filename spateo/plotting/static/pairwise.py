"""
Plots for pairwise comparison analyses between groups of samples and across different conditions (e.g. biological
conditions or using different clustering parameters).

These functions build on the Label class- see spateo.utils.Label for documentation
"""
from ...utils import Label, interlabel_connections

from typing import Union, List, Tuple
import numpy as np
import scipy

from anndata import AnnData

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patheffects as PathEffects
plt.rcParams["figure.dpi"] = 300


def plot_connections(label: Label,
                     weights_matrix: Union[scipy.sparse.csr_matrix, np.ndarray],
                     ax: Union[None, mpl.axes.Axes] = None,
                     figsize: tuple = (8, 8),
                     zero_self_connections: bool = True,
                     normalize_by_self_connections: bool = False,
                     shapes_style: bool = True,
                     max_scale: float = 0.46,
                     colormap: Union[str, dict, 'mpl.colormap'] = "Spectral",
                     title_str="connection strengths between types",
                     title_fontsize: float = 12,
                     label_fontsize: float = 12,
                     verbose: bool = True,
                     ) -> None:
    """
    Plot connections between labels- visualization of how closely labels are colocalized

    Parameters
    ----------
    shapes_style : bool, default True
        If True plots squares, if False plots heatmap
    max_scale : float, default 0.46
        Only used for the case that 'shape_style' is True, gives maximum size of square
    colormap : str, dict, or matplotlib.colormap
        Specifies colors to use for plotting. If dictionary, keys should be numerical labels corresponding to those
        of the Label object.
    """
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

    # If colormap is given, map label ID to points along the colormap. If dictionary is given, instead map
    if isinstance(cmap, dict):
        if type(list(cmap.keys())[0]) == str:
            id_colors = {n_id: cmap[id] for n_id, id in zip(label.ids, label.str_ids)}
        else:
            id_colors = {id: cmap[id] for id in label.ids}
    else:
        id_colors = {id: cmap(id / label.max_id) for id in label.ids}

    if shapes_style:
        # Cell types/labels will be represented using triangles:
        left_triangle = np.array((
            (-1., 1.),
            # (1., 1.),
            (1., -1.),
            (-1., -1.)
        ))

        right_triangle = np.array((
            (-1., 1.),
            (1., 1.),
            (1., -1.),
            # (-1., -1.)
        ))

        polygon_list = []
        color_list = []

        ax.set_ylim(- 0.55, label.num_labels - 0.45)
        ax.set_xlim(- 0.55, label.num_labels - 0.45)

        for label_1 in range(connections.shape[0]):
            for label_2 in range(connections.shape[1]):

                if label_1 <= label_2:

                    for triangle in [left_triangle, right_triangle]:
                        center = np.array((label_1, label_2))[np.newaxis, :]
                        scale_factor = connections[label_1, label_2] / connections_max
                        offsets = triangle * max_scale * scale_factor
                        polygon_list.append(center + offsets)

                    color_list += (id_colors[label.ids[label_2]],
                                    id_colors[label.ids[label_1]])

        collection = PolyCollection(polygon_list, facecolors=color_list, edgecolors="face", linewidths=0)

        ax.add_collection(collection)

        ax.tick_params(labelbottom=False, labeltop=True)
        ax.xaxis.set_tick_params(pad=-2)
    else:
        # Heatmap of connection strengths
        heatmap = ax.imshow(connections, cmap=colormap, interpolation="nearest")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        fig.colorbar(heatmap, cax=cax)
        cax.tick_params(axis='both', which='major', labelsize=6, rotation=-45)

        # Change formatting if values too small
        if connections_max < 0.001:
            cax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))

    # Formatting adjustments
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(label.num_labels), )
    # If label has categorical labels associated, use those to label the axes instead:
    if label.str_map is not None:
        ax.set_xticklabels(label.str_ids, fontsize=label_fontsize, fontweight="bold", rotation=90,
                           path_effects=[PathEffects.Stroke(linewidth=0.5, foreground='black', alpha=0.8)])
    else:
        ax.set_xticklabels(label.ids, fontsize=label_fontsize, fontweight="bold", rotation=0,
                           path_effects=[PathEffects.Stroke(linewidth=0.5, foreground='black', alpha=0.8)])

    ax.set_yticks(np.arange(label.num_labels))
    if label.str_map is not None:
        ax.set_yticklabels(label.str_ids, fontsize=label_fontsize, fontweight="bold",
                           path_effects=[PathEffects.Stroke(linewidth=0.5, foreground='black', alpha=0.8)])
    else:
        ax.set_yticklabels(label.ids, fontsize=label_fontsize, fontweight="bold",
                           path_effects=[PathEffects.Stroke(linewidth=0.5, foreground='black', alpha=0.8)])

    for ticklabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for n, id in enumerate(label.ids):
            ticklabels[n].set_color(id_colors[id])

    ax.set_title(title_str, fontsize=title_fontsize, fontweight="bold")
    plt.show()


def plot_specific_labels(label: Label,
                         specific_labels: Union[str, List[str]],
                         locations: np.ndarray,
                         plots_per_row: int = 3,
                         cmap_name: str = "Spectral",
                         default_color: str = "tab:red",
                         background_color: str = "whitesmoke",
                         subplot_size: Tuple[float, float] = (5, 5),
                         spot_size: float = 0.2,
                         alpha: float = 1,
                         flip_axes: bool = False,
                         show_axes: bool = False,
                         verbose: bool = False,
                         ) -> None:
    """
    For the chosen labels, plot their spatial locations independently of all other labels

    Args:
        label : Label object
            Categorical labels for each spot
        specific_labels : str or list of str
            Labels to plot in isolation from all other labels
        locations : np.ndarray
            x/y coordinates
        cmap_name : str, default "Spectral"
            Name of standard colormap or reference to custom colormap
        default_color : str or list of str, default "tab:red"
            Color for clusters if no colormap given
        background_color: str, default "whitesmoke"
            Color for spots not in cluster
        subplot_size : float tuple, default (5, 5)
            Size of individual subplot
        flip_axes : bool, default False
            If True, flip y axis
        show_axes : bool, default False
            If True, show x and y axis borders
    """
    assert label.str_map is not None, "Objects of class Label are encoded with numerical labels and need to be " \
                                      "initialized with a dictionary mapping to be able to connect back to string " \
                                      "labels- see :class `Label`."

    max_id = label.max_id
    if len(specific_labels) % plots_per_row == 0:
        num_rows = len(specific_labels) // plots_per_row
    else:
        num_rows = len(specific_labels) // plots_per_row + 1
    figsize_x = (subplot_size[0] * plots_per_row) * 1.1
    figsize_y = subplot_size[1] * num_rows

    fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=True)
    grid = fig.add_gridspec(ncols=plots_per_row, nrows=num_rows)

    if cmap_name is not None:
        cmap = mpl.cm.get_cmap(cmap_name)

    # Get numerical labels corresponding to provided string labels:
    for i, clust in enumerate(specific_labels):
        grid_y = i // plots_per_row
        grid_x = i % plots_per_row

        label_id = list(label.str_map.values()).index(clust)
        print(f"Numerical label corresponding to {clust}: {label_id}")

        # Find location of 'label_id' in the ID list:
        for n, id in enumerate(label.ids):
            if id == label_id:
                label_idx = n

        if verbose:
            print(f"Plotting grid position: {grid_y}, {grid_x}")

        ax = fig.add_subplot(grid[grid_y, grid_x])

        onehot = label.get_onehot()
        label_mask = np.squeeze(onehot[label_idx, :].toarray().astype(bool))

        if cmap_name is None:
            c = default_color
        else:
            c = np.expand_dims(cmap((i + 1)/ (len(specific_labels) + 0.5)), axis=0)

        other_spots = ax.scatter(
            locations[~label_mask, 0], locations[~label_mask, 1],
            c=background_color, s=spot_size, alpha=alpha,
        )

        cluster_spots = ax.scatter(
            locations[label_mask, 0], locations[label_mask, 1],
            c=c, s=spot_size, edgecolors='black', linewidths=0.25, alpha=alpha,
        )

        if flip_axes:
            ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_aspect('equal', 'datalim')
        ax.set_title(f'{clust}', fontsize=4 * subplot_size[0], fontweight="bold")

        if not show_axes:
            ax.axis("off")