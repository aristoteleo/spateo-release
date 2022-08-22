"""NOTES: code still needs touched up- but putting a pull request through to start putting code into the main repo.
- DZ"""
"""Plots for pairwise comparison analyses between groups of samples"""
from spateo.utils import Label, interlabel_connections

from typing import Union, Tuple
import numpy as np
import scipy.sparse as sp

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
plt.rcParams["figure.dpi"] = 300


def plot_connections(label: Label,
                     weights_matrix: Union[sp.csr_matrix, np.ndarray],
                     ax: Union[None, mpl.axes.Axes] = None,
                     figsize: tuple = (8, 8),
                     zero_self_connections: bool = True,
                     normalize_by_self_connections: bool = False,
                     shapes_style: bool = True,
                     max_scale: float = 0.46,
                     colormap_name: str = "Spectral",
                     title_str="connection strengths between types",
                     title_fontsize: float = 12,
                     label_fontsize: float = 12,
                     verbose: bool = True,
                     ) -> None:
    """
    Plot connections between labels- visualization of how closely labels are colocalized. Function integrates with
    the `Label` class

    Args:
        label :
        weights_matrix : scipy sparse matrix or np.ndarray
            Pairwise adjacency matrix, weighted by e.g. spatial distance between points
        ax : optional class: `~mpl.axes.Axes`
            Can provide predefined ax to plot onto. If None, will create ax at runtime
        figsize : tuple[int, int]
            Width and height of the plotting window
        zero_self_connections : bool, default True
            Set True to ignore the distance between a sample and itself
        normalize_by_self_connections : bool, default False
            If values along the diagonal are nonzero/not one, normalize by the value along the diagonal
        shapes_style : bool, default True
            If True plots squares, if False plots heatmap
        max_scale : float, default 0.46
            Only used for the case that 'shape_style' is True, gives maximum size of square
        colormap_name : str, default "Spectral"
            Name of the Matplotlib colormap/palette to use for coloring
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
    cmap = mpl.cm.get_cmap(colormap_name)
    id_colours = {id: cmap(id / label.max_id) for id in label.ids}

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
        colour_list = []

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

                    colour_list += (id_colours[label.ids[label_2]],
                                    id_colours[label.ids[label_1]])

        collection = PolyCollection(polygon_list, facecolors=colour_list, edgecolors="face", linewidths=0)

        ax.add_collection(collection)

        ax.tick_params(labelbottom=False, labeltop=True)
        ax.xaxis.set_tick_params(pad=-2)
    else:
        # Heatmap of connection strengths
        heatmap = ax.imshow(connections, cmap=colormap_name, interpolation="nearest")

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
    ax.set_xticklabels(label.ids, fontsize=label_fontsize, fontweight="bold", rotation=0)

    ax.set_yticks(np.arange(label.num_labels))
    ax.set_yticklabels(label.ids, fontsize=label_fontsize, fontweight="bold")

    for ticklabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for n, id in enumerate(label.ids):
            ticklabels[n].set_color(id_colours[id])

    ax.set_title(title_str, fontsize=title_fontsize, fontweight="bold")
    plt.show()