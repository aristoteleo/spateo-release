# -*- coding: utf-8 -*-
"""
@File    :   cluster_lasso.py
@Time    :   2022/06/29 10:43:55
@Author  :   LuluZuo XueWang
@Version :   1.0
@Desc    :   spatial cluster lasso
"""

from typing import Optional

import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from ipywidgets import VBox


class Lasso:
    """Lasso an region of interest (ROI) based on spatial cluster.

    Examples:
        L = st.tl.Lasso(adata)
        L.vi_plot(group='group', group_color='group_color')
    """

    __sub_inde = []
    sub_adata = None

    def __init__(self, adata):
        """
        Args:
            adata: An Anndata object.
        """
        self.adata = adata

    def vi_plot(
        self,
        key="spatial",
        group: Optional[str] = None,
        group_color: Optional[str] = None,
    ):
        """Plot spatial cluster result and lasso ROI.

        Args:
            key: The column key in .obsm, default to be 'spatial'.
            group: The column key/name that identifies the grouping information
                (for example, clusters that correspond to different cell types)
                of buckets.
            group_color: The key in .uns, corresponds to a dictionary that map group names to group colors.

        Returns:
            sub_adata: subset of adata.
        """
        if group and group_color:

            df = pd.DataFrame()
            df["group_ID"] = self.adata.obs_names
            df["labels"] = self.adata.obs[group].values
            df["spatial_0"] = self.adata.obsm[key][:, 0]
            df["spatial_1"] = self.adata.obsm[key][:, 1]
            df["color"] = df.labels.map(self.adata.uns[group_color])

            py.init_notebook_mode()

            f = go.FigureWidget(
                [go.Scatter(x=df["spatial_0"], y=df["spatial_1"], mode="markers", marker_color=df["color"])]
            )
            scatter = f.data[0]
            scatter.marker.opacity = 0.5
            f.layout.plot_bgcolor = "rgb(255,255,255)"
            f.layout.autosize = True

            axis_dict = dict(
                showticklabels=False,
                autorange=True,
            )
            f.layout.yaxis = axis_dict
            f.layout.xaxis = axis_dict

            # Create a table FigureWidget that updates on selection from points in the scatter plot of f
            t = go.FigureWidget(
                [
                    go.Table(
                        header=dict(
                            values=["group_ID", "labels", "spatial_0", "spatial_1"],
                            fill=dict(color="#C2D4FF"),
                            align=["left"] * 5,
                        ),
                        cells=dict(
                            values=[df[col] for col in ["group_ID", "labels", "spatial_0", "spatial_1"]],
                            fill=dict(color="#F5F8FF"),
                            align=["left"] * 5,
                        ),
                    )
                ]
            )

            def selection_fn(trace, points, selector):

                t.data[0].cells.values = [
                    df.loc[points.point_inds][col] for col in ["group_ID", "labels", "spatial_0", "spatial_1"]
                ]

                Lasso.__sub_index = t.data[0].cells.values[0]
                Lasso.sub_adata = self.adata[
                    Lasso.__sub_index,
                ]

            scatter.on_selection(selection_fn)

            # Put everything together
            return VBox((f, t))

        else:
            raise ValueError(f"adata.obsm doesn't have {group} or {group_color} is not in adata.uns")
