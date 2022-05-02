from typing import Union, Optional

from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .baseplot import BasePlot
from .utils import _to_hex


class HeatMap(BasePlot):

    def __init__(self,
                 adata: AnnData,
                 markers: list,
                 group: Optional[str] = None,
                 group_mean: bool = False,
                 group_cmap: str = "tab20",
                 col_cluster: bool = False,
                 row_cluster: bool = False,
                 layer: Union[str, list] = "X",
                 slices: Union[str, list] = None,
                 slices_split: bool = False,
                 slices_key: str = 'slices',
                 background: Optional[str] = None,
                 dpi: int = 100,
                 figsize: tuple = (11, 5),
                 save_show_or_return: str = "show",
                 save_kwargs: Optional[dict] = None,
                 swap_axis: bool = False,
                 cbar_pos: Optional[tuple] = None,
                 theme: Optional[str] = None,
                 cmap: str = "viridis",
                 **kwargs
                 ):
        super().__init__(
            adata=adata,
            color=[markers],
            basis=group,
            layer=layer,
            slices=slices,
            slices_split=slices_split,
            slices_key=slices_key,
            background=background,
            dpi=dpi,
            figsize=figsize,
            save_show_or_return=save_show_or_return,
            save_kwargs=save_kwargs,
            theme=theme,
            gridspec=False
        )
        self.group_mean = group_mean
        self.group_cmap = group_cmap
        self.col_cluster = col_cluster
        self.row_cluster = row_cluster
        self.cmap = cmap
        self.cbar_pos = cbar_pos
        self.swap_axis = swap_axis
        self.kwargs = kwargs

    def _plot_basis_layer(self, adata: AnnData, markers, cells_group, cur_l):
        value_df, colors = self._fetch_data(adata, markers, cells_group, cur_l)

        if self.swap_axis:
            value_df = value_df.T

        heatmap_kwargs = dict(
            xticklabels=1,
            yticklabels=False,
            col_colors=colors if self.swap_axis else None,
            row_colors=None if self.swap_axis else colors,
            row_linkage=None,
            col_linkage=None,
            method="average",
            metric="euclidean",
            z_score=None,
            standard_scale=None,
            cbar_pos=self.cbar_pos,
        )
        if self.kwargs is not None:
            heatmap_kwargs.update(self.kwargs)

        sns_heatmap = sns.clustermap(
            value_df,
            col_cluster=self.col_cluster,
            row_cluster=self.row_cluster,
            cmap=self.cmap,
            figsize=self.figsize,
            **heatmap_kwargs,
        )

        # if not self.show_legend:
        #     sns_heatmap.cax.set_visible(False)

    def _fetch_data(self, adata: AnnData, markers, cells_group, cur_l):
        layer = None if cur_l == 'X' else cur_l
        value_df = pd.DataFrame()
        for i, marker in enumerate(markers):
            v = adata.obs_vector(marker, layer=layer)
            value_df[marker] = v
        value_df.index = adata.obs.index
        colors = None
        if cells_group is not None:
            value_df[cells_group] = adata.obs_vector(cells_group, layer=layer)
            value_df = value_df.sort_values(cells_group)
            if self.group_mean:
                value_df = value_df.groupby(cells_group, as_index=False).mean()
            num_labels = len(value_df[cells_group].unique())

            color_key = _to_hex(plt.get_cmap(self.group_cmap)(np.linspace(0, 1, num_labels)))
            cell_lut = dict(zip(value_df[cells_group].unique().tolist(), color_key))
            colors = value_df[cells_group].map(cell_lut)
            value_df = value_df.drop(cells_group, axis=1)

        return value_df, colors


def heatmap(adata: AnnData,
            markers: list,
            group: Optional[str] = None,
            group_mean: bool = False,
            group_cmap: str = "tab20",
            col_cluster: bool = False,
            row_cluster: bool = False,
            layer: Union[str, list] = "X",
            slices: Union[str, list] = None,
            slices_split: bool = False,
            slices_key: str = 'slices',
            background: Optional[str] = None,
            dpi: int = 100,
            figsize: tuple = (11, 5),
            save_show_or_return: str = "show",
            save_kwargs: Optional[dict] = None,
            swap_axis: bool = False,
            cbar_pos: Optional[tuple] = None,
            theme: Optional[str] = None,
            cmap: str = "viridis",
            **kwargs):
    hm = HeatMap(adata=adata,
                 markers=markers,
                 group=group,
                 group_mean=group_mean,
                 group_cmap=group_cmap,
                 col_cluster=col_cluster,
                 row_cluster=row_cluster,
                 layer=layer,
                 slices=slices,
                 slices_split=slices_split,
                 slices_key=slices_key,
                 background=background,
                 dpi=dpi,
                 figsize=figsize,
                 save_show_or_return=save_show_or_return,
                 save_kwargs=save_kwargs,
                 swap_axis=swap_axis,
                 cbar_pos=cbar_pos,
                 theme=theme,
                 cmap=cmap,
                 **kwargs)
    return hm.plot()
