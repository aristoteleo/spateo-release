"""Plotting functions for aggregated UMI counts.
"""
import math
import warnings
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import issparse
from skimage.color import label2rgb

from ...configuration import SKM
from ...errors import PlottingError
from ...logging import logger_manager as lm
from .utils import save_return_show_fig_utils


def imshow(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    ax: Optional[Axes] = None,
    show_cbar: bool = False,
    use_scale: bool = True,
    labels: bool = False,
    background: Union[None, str] = None,
    save_show_or_return: str = "show",
    save_kwargs: Dict = {},
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
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        save_show_or_return: `str` {'save', 'show', 'return', 'both', 'all'} (default: `show`)
            Whether to save, show or return the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent":
            True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs.
        **kwargs: Additional keyword arguments are all passed to :func:`imshow`.

    Returns:
        The figure and axis if `ax` is not provided.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if SKM.get_adata_type(adata) != SKM.ADATA_AGG_TYPE:
        raise PlottingError("Only `AGG` type AnnDatas are supported.")

    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    else:
        fig = ax.get_figure()

    mtx = SKM.select_layer_data(adata, layer, make_dense=True)
    if labels:
        mtx = label2rgb(mtx, bg_label=0)

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
        save_kwargs=save_kwargs,
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )


def qc_regions(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    axes: Optional[np.ndarray] = None,
    ncols: int = 1,
    background: Union[None, str] = None,
    save_show_or_return: str = "show",
    save_kwargs: Dict = {},
    **kwargs,
) -> Optional[Tuple[Figure, np.ndarray]]:
    """Display QC regions.

    Args:
        adata: Input Anndata
        layer: Layer to display
        axes: Numpy array (possibly 2D) of Matplotlib axes to plot each region.
            This option is useful when trying to overlay multiple layers together.
        ncols: Number of columns when displaying multiple panels.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        save_show_or_return: `str` {'save', 'show', 'return', 'both', 'all'} (default: `show`)
            Whether to save, show or return the figure. If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: `dict` (default: `{}`)
            A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
            save_fig function will use the {"path": None, "prefix": 'qc_regions', "dpi": None, "ext": 'pdf', "transparent":
            True, "close": True, "verbose": True} as its parameters. Otherwise you can provide a dictionary that
            properly modify those keys according to your needs.
        **kwargs: Additional keyword arguments are all passed to :func:`imshow`.

    Returns:
        The figure and axes if `axes` is not provided.
    """
    if SKM.get_adata_type(adata) != SKM.ADATA_AGG_TYPE:
        raise PlottingError("Only `AGG` type AnnDatas are supported.")

    regions = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_QC_KEY)
    n_regions = regions.shape[0]

    if axes is None:
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
            lm.main_warning(f"Region {region} not in AnnData bounds.")
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

    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=False,
        background=background,
        prefix="scatters",
        save_kwargs=save_kwargs,
        total_panels=1,
        fig=fig,
        axes=axes,
        return_all=False,
        return_all_list=None,
    )
