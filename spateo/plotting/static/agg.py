"""Plotting functions for aggregated UMI counts.
"""
import math
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import issparse
from skimage.color import label2rgb

from ...configuration import SKM
from ...errors import PlottingError
from ...warnings import PlottingWarning


def imshow(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    ax: Optional[Axes] = None,
    show_cbar: bool = False,
    use_scale: bool = True,
    labels: bool = False,
    **kwargs,
) -> Optional[Tuple[Figure, Axes]]:
    """Display raw data within an AnnData.

    Args:
        adata: Anndata containing aggregated UMI counts.
        layer: Layer to display. Defaults to X.
        ax: Axes to plot.
        show_cbar: Whether or not to show a colorbar next to the plot.
        use_scale: Whether or not to plot in physical units. Only valid when
            appropriate scale keys are present in .uns
        labels: Whether the input data contains labels, encoded as positive
            integers.

    Returns:
        The figure and axis if `ax` is not provided.
    """
    if SKM.get_adata_type(adata) != SKM.ADATA_AGG_TYPE:
        raise PlottingError("Only `AGG` type AnnDatas are supported.")

    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
    else:
        fig = ax.get_figure()

    mtx = SKM.select_layer_data(adata, layer, make_dense=True)
    if labels:
        mtx = label2rgb(mtx)

    kwargs.update({"interpolation": "none"})
    im = ax.imshow(mtx, **kwargs)
    ax.set_title(layer)
    if show_cbar:
        fig.colorbar(im)
    unit = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY)
    if use_scale and unit is not None:
        binsize = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
        scale = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY) * binsize
        im.set_extent((0, (mtx.shape[1] - 1) * scale, (mtx.shape[0] - 1) * scale, 0))
        ax.set_xlabel(unit)
        ax.set_ylabel(unit)

    if return_fig_ax:
        return fig, ax


def qc_regions(
    adata: AnnData, layer: str = SKM.X_LAYER, axes: Optional[np.ndarray] = None, ncols: int = 1, **kwargs
) -> Optional[Tuple[Figure, np.ndarray]]:
    """Display QC regions.

    Args:
        adata: Input Anndata
        layer: Layer to display
        axes: Numpy array (possibly 2D) of Matplotlib axes to plot each region.
            This option is useful when trying to overlay multiple layers together.
        ncols: Number of columns when displaying multiple panels.
        **kwargs: Additional keyword arguments are all passed to :func:`imshow`.

    Returns:
        The figure and axes if `axes` is not provided.
    """
    if SKM.get_adata_type(adata) != SKM.ADATA_AGG_TYPE:
        raise PlottingError("Only `AGG` type AnnDatas are supported.")

    regions = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_QC_KEY)
    n_regions = regions.shape[0]
    return_fig_axes = False
    if axes is None:
        return_fig_axes = True
        nrows = math.ceil(n_regions / ncols)
        fig, axes = plt.subplots(figsize=(4 * ncols, 4 * nrows), ncols=ncols, nrows=nrows, tight_layout=True)
    elif axes.size < n_regions:
        raise PlottingError(f"`fig` must have at least {n_regions} axes.")

    for ax, region in zip(axes.flatten(), regions):
        xmin, xmax, ymin, ymax = region
        if (
            str(xmin) not in adata.obs_names
            or str(xmax - 1) not in adata.obs_names
            or str(ymin) not in adata.var_names
            or str(ymax - 1) not in adata.var_names
        ):
            warnings.warn(f"Region {region} not in AnnData bounds.", PlottingWarning)
            continue
        imshow(
            adata[
                adata.obs_names.get_loc(str(xmin)) : adata.obs_names.get_loc(str(xmax - 1)) + 1,
                adata.var_names.get_loc(str(ymin)) : adata.var_names.get_loc(str(ymax - 1)) + 1,
            ],
            layer,
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"{layer} [{xmin}:{xmax},{ymin}:{ymax}]")

    if return_fig_axes:
        return fig, axes
