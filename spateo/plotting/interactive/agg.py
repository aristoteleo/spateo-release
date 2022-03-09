"""Interactive plotting functions for aggregated UMI counts.
"""
from typing import List, Optional

import cv2
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
from anndata import AnnData
from skimage.color.colorlabel import color_dict, DEFAULT_COLORS

from ...configuration import SKM
from ...errors import PlottingError


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
                contour = contour.squeeze()
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
