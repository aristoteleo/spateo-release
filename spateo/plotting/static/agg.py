"""Plotting functions for aggregated UMI counts.
"""
import math
import warnings
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import issparse
from skimage.color import label2rgb
from typing_extensions import Literal

from ...configuration import SKM
from ...errors import PlottingError
from ...logging import logger_manager as lm
from .utils import save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def imshow(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    ax: Optional[Axes] = None,
    show_cbar: bool = False,
    use_scale: bool = True,
    absolute: bool = False,
    labels: bool = False,
    downscale: float = 1.0,
    downscale_interpolation: Optional[int] = None,
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[Dict] = None,
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
        absolute: Whether to set the axes to be in absolute coordinates. By
            default, relative coordinates are used (i.e. the axes start at
            zero).
        labels: Whether the input data contains labels, encoded as positive
            integers.
        downscale: Downscale image by this amount for faster plotting.
        downscale_interpolation: Use this CV2 interpolation method when downscaling.
            By default, bilinear interpolation is used when `labels=True` and nearest
            neighbor interpolation when `labels=False`.
            Available options are located here:
            https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
            Only has an effect when `downscale` < 1.
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
        **kwargs: Additional keyword arguments are all passed to :func:`imshow`.

    Returns:
        The figure and axis if `ax` is not provided.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    else:
        fig = ax.get_figure()

    mtx = SKM.select_layer_data(adata, layer, make_dense=True)
    if downscale < 1:
        default = cv2.INTER_NEAREST if labels else cv2.INTER_LINEAR
        if mtx.dtype == np.dtype(bool):
            mtx = mtx.astype(np.uint8)
            default = cv2.INTER_NEAREST
        mtx = cv2.resize(
            mtx,
            dsize=None,
            fx=downscale,
            fy=downscale,
            interpolation=default if downscale_interpolation is None else downscale_interpolation,
        )

    if labels:
        mtx = label2rgb(mtx, bg_label=0)

    unit = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY)
    adata_bounds = SKM.get_agg_bounds(adata)
    # Note that we +1 to the xmax and ymax values because the first and last
    # ticks are at exactly these locations.
    extent = (
        [adata_bounds[0], adata_bounds[1] + 1, adata_bounds[3] + 1, adata_bounds[2]]
        if absolute
        else [0, mtx.shape[1] / downscale, mtx.shape[0] / downscale, 0]
    )
    xlabel = "Y"
    ylabel = "X"
    if use_scale and unit is not None:
        binsize = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
        scale = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY) * binsize
        extent = [val * scale for val in extent]
        xlabel += f" ({unit})"
        ylabel += f" ({unit})"

    # Make sure any existing images match the size and extent of the one we are about to plot.
    if any(mtx.shape[:2] != im.get_array().shape[:2] for im in ax.get_images()):
        raise PlottingError(
            f"The dimensions of the matrix, {mtx.shape[:2]} must be equal to the dimensions of "
            "the images present in the axis. Make sure you are using the same AnnData and the `downscale` argument "
            "as you used to show the previous image(s)."
        )
    if any(not np.allclose(extent, im.get_extent(), atol=0.5) for im in ax.get_images()):
        raise PlottingError(
            f"The extent of the matrix, {extent} must be equal to the extent of "
            "the images present in the axis. Make sure you are using the same AnnData and the "
            "`use_scale` and `absolute` arguments as you used to show the previous image(s)."
        )

    kwargs.update({"interpolation": "none"})
    im = ax.imshow(mtx, **kwargs)
    ax.set_title(layer)
    if show_cbar:
        fig.colorbar(im)
    im.set_extent(tuple(extent))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
        prefix="imshow",
        save_kwargs=save_kwargs or {},
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def box_qc_regions(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    use_scale: bool = True,
    box_kwargs: Optional[Dict] = None,
    ax: Optional[Axes] = None,
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[Dict] = None,
    **kwargs,
):
    """Indicate selected QC regions with boxes on the full tissue.

    Args:
        adata: Input Anndata
        layer: Layer to display
        use_scale: Whether or not to plot in physical units. Only valid when
            appropriate scale keys are present in .uns
        box_kwargs: Keyword arguments to pass to :func:`patches.Rectangle`. By default,
            the boxes will be transparent with red outlines of 1 point thickness.
        ax: Axes to plot.
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
        **kwargs: Additional keyword arguments are all passed to :func:`imshow`.
    """
    regions = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_QC_KEY)
    unit = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY)

    kwargs.update(dict(ax=ax, use_scale=use_scale, save_show_or_return="return"))
    fig, ax = imshow(adata, layer, **kwargs)

    scale = 1
    if use_scale and unit is not None:
        binsize = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
        scale = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY) * binsize

    _box_kwargs = dict(edgecolor="red", linewidth=1.0, fill=False)
    _box_kwargs.update(box_kwargs or {})
    for region in regions:
        xmin, xmax, ymin, ymax = region
        if (
            str(xmin) not in adata.obs_names
            or str(xmax) not in adata.obs_names
            or str(ymin) not in adata.var_names
            or str(ymax) not in adata.var_names
        ):
            lm.main_warning(f"Region {region} not in AnnData bounds.")
            continue

        # Modify bounds to match anndata bounds
        xmin = adata.obs_names.get_loc(str(xmin))
        xmax = adata.obs_names.get_loc(str(xmax))
        ymin = adata.var_names.get_loc(str(ymin))
        ymax = adata.var_names.get_loc(str(ymax))
        box = patches.Rectangle(
            (ymin * scale, xmin * scale), (ymax - ymin + 1) * scale, (xmax - xmin + 1) * scale, **_box_kwargs
        )
        ax.add_patch(box)

    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=False,
        background=background,
        prefix="box_qc_regions",
        save_kwargs=save_kwargs or {},
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def qc_regions(
    adata: AnnData,
    layer: str = SKM.X_LAYER,
    axes: Optional[np.ndarray] = None,
    ncols: int = 1,
    background: Optional[str] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[Dict] = None,
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
        save_show_or_return: Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If
            "all", the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True, "verbose": True}
            as its parameters. Otherwise you can provide a dictionary that
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
    else:
        fig = axes.flatten()[0].get_figure()

    for ax, region in zip(axes.flatten(), regions):
        xmin, xmax, ymin, ymax = region
        if (
            str(xmin) not in adata.obs_names
            or str(xmax) not in adata.obs_names
            or str(ymin) not in adata.var_names
            or str(ymax) not in adata.var_names
        ):
            lm.main_warning(f"Region {region} not in AnnData bounds.")
            continue
        imshow(
            adata[
                adata.obs_names.get_loc(str(xmin)) : adata.obs_names.get_loc(str(xmax)) + 1,
                adata.var_names.get_loc(str(ymin)) : adata.var_names.get_loc(str(ymax)) + 1,
            ],
            layer,
            ax=ax,
            save_show_or_return="return",
            **kwargs,
        )
        ax.set_title(f"{layer} [{xmin}:{xmax},{ymin}:{ymax}]")

    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=False,
        background=background,
        prefix="qc_regions",
        save_kwargs=save_kwargs or {},
        total_panels=1,
        fig=fig,
        axes=axes,
        return_all=False,
        return_all_list=None,
    )
