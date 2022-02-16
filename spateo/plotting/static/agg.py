"""Plotting functions for aggregated UMI counts.
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.sparse import issparse
from skimage.color import label2rgb

from ...configuration import SKM
from ...errors import PlottingError


def imshow(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    ax: Optional[Axes] = None,
    show_cbar: bool = False,
    use_scale: bool = True,
    labels: bool = False,
    **kwargs
) -> Tuple[Figure, Axes]:
    """Display raw data within an AnnData.

    Args:
        layer: Layer to display. Defaults to X.
        ax: Axes to plot.
        show_cbar: Whether or not to show a colorbar next to the plot.
        use_scale: Whether or not to plot in physical units. Only valid when
            appropriate scale keys are present in .uns
        labels: Whether the input data contains labels, encoded as positive
            integers.

    Returns:
        The figure and axis
    """
    if SKM.get_adata_type(adata) != SKM.ADATA_AGG_TYPE:
        raise PlottingError("Only `AGG` type AnnDatas are supported.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
    else:
        fig = ax.get_figure()

    mtx = SKM.select_layer_data(adata, layer, make_dense=True)
    if labels:
        mtx = label2rgb(mtx)

    im = ax.imshow(mtx, interpolation="none", **kwargs)
    if show_cbar:
        fig.colorbar(im)
    if use_scale and SKM.has_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY):
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        scale = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY)
        unit = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY)
        ax.set_xticklabels(xticks * scale)
        ax.set_yticklabels(yticks * scale)
        ax.set_xlabel(unit)
        ax.set_ylabel(unit)

    return fig, ax
