"""Interactive plotting functions for aggregated UMI counts.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from skimage.color.colorlabel import DEFAULT_COLORS, color_dict
from typing_extensions import Literal

from ...configuration import SKM
from ...errors import PlottingError
from ..static import imshow
from ..static.utils import save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def contours(adata: AnnData, layer: str, colors: Optional[List] = None, scale: float = 0.05) -> go.Figure:
    """Interactively display UMI density bins.

    Args:
        adata: Anndata containing aggregated UMI counts.
        layer: Layer to display
        colors: List of colors.
        scale: Scale width and height by this amount.

    Returns:
        A Plotly figure
    """
    if SKM.get_adata_type(adata) != SKM.ADATA_AGG_TYPE:
        raise PlottingError("Only `AGG` type AnnDatas are supported.")
    bins = SKM.select_layer_data(adata, layer)
    if colors is None:
        colors = DEFAULT_COLORS

    figure = go.Figure()
    color_i = 0
    for bin in np.unique(bins):
        if bin > 0:
            mask = bins == bin
            mtx = mask.astype(np.uint8)
            mtx[mtx > 0] = 255
            contours = cv2.findContours(mtx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                contour = contour.squeeze(1)
                figure.add_trace(
                    go.Scatter(
                        x=contour[:, 0],
                        y=-contour[:, 1],
                        text=str(bin),
                        line_width=0,
                        fill="toself",
                        mode="lines",
                        showlegend=False,
                        hoverinfo="text",
                        hoveron="fills",
                        fillcolor=mpl.colors.to_hex(colors[color_i % len(colors)]),
                    )
                )
            color_i += 1

    figure.update_layout(
        width=bins.shape[1] * scale,
        height=bins.shape[0] * scale,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return figure


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def select_polygon(
    adata: AnnData,
    layer: str,
    out_layer: Optional[str] = None,
    ax: Optional[Axes] = None,
    background: Optional[str] = None,
    **kwargs,
) -> PolygonSelector:
    """Display raw data within an AnnData with interactive polygon selection.

    Args:
        adata: Anndata containing aggregated UMI counts.
        layer: Layer to display. Defaults to X.
        out_layer: Layer to output selection result as a boolean mask. Defaults to
            `{layer}_selection`.
        ax: Axes to plot.
        background: string or None (optional, default 'None`)
            The color of the background. Usually this will be either
            'white' or 'black', but any color name will work. Ideally
            one wants to match this appropriately to the colors being
            used for points etc. This is one of the things that themes
            handle for you. Note that if theme
            is passed then this value will be overridden by the
            corresponding option of the theme.
        **kwargs: Additional keyword arguments are all passed to :func:`spateo.pl.imshow`.
    """
    from matplotlib import rcParams
    from matplotlib.colors import to_hex

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    else:
        fig = ax.get_figure()

    # Don't show figure immediately because we need to do some bookkeeping.
    kwargs["save_show_or_return"] = "return"
    kwargs["interpolation"] = "none"
    imshow(adata, layer, ax=ax, **kwargs)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Overlay a completely transparent image on top. This image will be modified
    # in-place to highlight selected regions.
    mask_shape = ax.get_images()[-1].get_array().shape[:2] + (4,)
    mask_placeholder = np.zeros(mask_shape, dtype=np.uint8)
    mask_im = ax.imshow(mask_placeholder, extent=ax.get_images()[-1].get_extent())

    factor = mask_shape[0] / abs(ylim[0] - ylim[1])
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.SELECTION_SUFFIX)

    def onselect(data):
        points = np.array(data)
        points[:, 0] -= min(xlim)
        points[:, 1] -= min(ylim)
        points *= factor

        alpha = np.full(mask_shape[:2], 126, dtype=np.uint8)
        cv2.fillPoly(alpha, [points.astype(int)], 0)
        SKM.set_layer_data(
            adata,
            out_layer,
            cv2.resize((alpha == 0).astype(np.uint8), dsize=adata.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(
                bool
            ),
        )

        mask = np.zeros_like(mask_placeholder)
        mask[:, :, 3] = alpha
        mask_im.set_data(mask)
        mask_im.set_extent(ax.get_images()[-1].get_extent())
        fig.canvas.draw()

    def key_press_event(event):
        if event.key == "escape":
            mask_im.set_data(np.zeros_like(mask_placeholder))
            del adata.layers[out_layer]
            fig.canvas.draw()

    lasso = PolygonSelector(ax=ax, onselect=onselect)
    fig.canvas.mpl_connect("key_press_event", key_press_event)
    ax.set_title("Draw polygon with mouse.\nHold Ctrl to click and drag vertices.\nPress Esc to reset selection.")

    if background is None:
        _background = rcParams.get("figure.facecolor")
        _background = to_hex(_background) if type(_background) is tuple else _background
        # if save_show_or_return != 'save': set_figure_params('dynamo', background=_background)
    else:
        _background = background
    save_return_show_fig_utils(
        save_show_or_return="show",
        show_legend=False,
        background=_background,
        prefix="select_polygon",
        save_kwargs={},
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )
    return lasso
