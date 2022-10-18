from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from ...configuration import SKM


# Heatmap class, compatible with dataframes
def df_heatmap(
    data: pd.DataFrame,
    col_cluster: bool = False,
    row_cluster: bool = False,
    figsize: tuple = (5, 5),
    save_show_or_return: str = "show",
    save_kwargs: Optional[dict] = None,
    swap_axis: bool = False,
    cbar_pos: Optional[tuple] = None,
    theme: Optional[str] = None,
    cmap: str = "viridis",
    **kwargs
):
    """

    Args:
        data:
        col_cluster:
        row_cluster:
        figsize:
        save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.
        swap_axis: Set True to switch the orientation of the x- and y-axes (such that x-labels are along the vertical
            axis and y-labels are along the horizontal).
        cbar_pos:
        theme:
        cmap: Name of the colormap to use
        kwargs: A dictionary that will be passed to :func `sns.clustermap`. Valid arguments are any arguments that
            can be found in the signature for :func `sns.heatmap`.
    """

    if swap_axis:
        data = data.T

    heatmap_kwargs = dict()

    # If 'col_cluster' or 'row_cluster', plot a clustermap. Otherwise, plot a heatmap:
