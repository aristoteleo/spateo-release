from typing import Union, Optional

from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import to_hex
import numpy as np


from .utils import (
    _select_font_color,
    save_fig
)
from spateo.tools.utils import (
    update_dict
)


class BasePlot:
    def __init__(self,
                 adata: AnnData,
                 color: Union[str, list] = "ntr",
                 layer: Union[str, list] = "X",
                 basis: Union[str, list] = "umap",
                 slices: Union[str, list] = None,
                 slices_split: bool = False,
                 slices_key: str = "slices",
                 stack_colors=False,
                 stack_colors_threshold=0.001,
                 stack_colors_title="stacked colors",
                 stack_colors_legend_size=2,
                 stack_colors_cmaps=None,
                 ncols: int = 4,
                 aspect: str = "auto",
                 axis_on: bool = False,
                 background: Optional[str] = None,
                 dpi: int = 100,
                 figsize: tuple = (6, 4),
                 gridspec: bool = True,
                 pointsize: Optional[int] = None,
                 save_show_or_return: str = "show",
                 save_kwargs: Optional[dict] = None,
                 show_legend="on data",
                 theme: Optional[str] = None,
                 ):
        self.adata = adata.copy()
        self.stack_colors = stack_colors
        self.stack_colors_threshold = stack_colors_threshold
        self.stack_colors_title = stack_colors_title
        self.stack_colors_legend_size = stack_colors_legend_size
        self.show_legend = show_legend
        self.aspect = aspect
        self.axis_on = axis_on
        self.dpi = dpi
        self.figsize =figsize
        self.save_show_or_return = save_show_or_return
        self.save_kwargs = save_kwargs
        self.slices_split = slices_split
        self.slices_key = slices_key
        self.theme = theme
        self.basis = self._check_iterable(basis)
        self.color = self._check_iterable(color)
        self.layer = self._check_iterable(layer)
        if slices is None and slices_split:
            self.slices = self.adata.obs[self.slices_key].unique().tolist()
        self.slices = self._check_iterable(slices)
        self.prefix = "baseplot"

        if background is None:
            _background = rcParams.get("figure.facecolor")
            self._background = to_hex(_background) if type(_background) is tuple else _background
        else:
            self._background = background
        self.font_color = _select_font_color(self._background)

        if stack_colors and stack_colors_cmaps is None:
            self.stack_colors_cmaps = [
                "Greys",
                "Purples",
                "Blues",
                "Greens",
                "Oranges",
                "Reds",
                "YlOrBr",
                "YlOrRd",
                "OrRd",
                "PuRd",
                "RdPu",
                "BuPu",
                "GnBu",
                "PuBu",
                "YlGnBu",
                "PuBuGn",
                "BuGn",
                "YlGn",
            ]
        self.stack_legend_handles = []
        if stack_colors:
            self.color_key = None

        n_s = len(self.slices) if slices_split else 1
        n_c = len(self.color) if not stack_colors else 1
        n_l = len(self.layer)
        n_b = len(self.basis)
        total_panels, ncols = (
            n_s * n_c * n_l * n_b,
            min(max([n_s, n_c, n_l, n_b]), ncols),
        )
        nrow, ncol = int(np.ceil(total_panels / ncols)), ncols

        if pointsize is None:
            self.pointsize = 16000.0 / np.sqrt(adata.shape[0])
        else:
            self.pointsize = 16000.0 / np.sqrt(adata.shape[0]) * pointsize

        if gridspec:
            if total_panels > 1:
                self.fig = plt.figure(
                    None,
                    (figsize[0] * ncol, figsize[1] * nrow),
                    facecolor=self._background,
                    dpi=self.dpi,
                )
                self.gs = plt.GridSpec(nrow, ncol, wspace=0.12)
            else:
                self.fig, ax = plt.subplots(figsize=figsize)
                self.gs = [ax]
        self.ax_index = 0

    def plot(self):
        if self.slices_split:
            for cur_s in self.slices:
                adata = self.adata[self.adata.obs[self.slices_key] == cur_s, :]
                for cur_b in self.basis:
                    for cur_l in self.layer:
                        for cur_c in self.color:
                            self._plot_basis_layer(
                                adata, cur_c, cur_b, cur_l
                            )
                            if not self.stack_colors:
                                self.ax_index += 1
                        if self.stack_colors:
                            self.ax_index += 1

        else:
            for cur_b in self.basis:
                for cur_l in self.layer:
                    for cur_c in self.color:
                        self._plot_basis_layer(
                            self.adata, cur_c, cur_b, cur_l
                        )
                        if not self.stack_colors:
                            self.ax_index += 1
                    if self.stack_colors:
                        self.ax_index += 1

        clf = self._save_show_or_return()
        return clf

    def _plot_basis_layer(self, *args, **kwargs):
        raise NotImplementedError

    def _save_show_or_return(self):
        if self.save_show_or_return in ["save", "both", "all"]:
            s_kwargs = {
                "path": None,
                "prefix": self.prefix,
                "dpi": self.dpi,
                "ext": "pdf",
                "transparent": True,
                "close": True,
                "verbose": True,
            }
            s_kwargs = update_dict(s_kwargs, self.save_kwargs)

            save_fig(**s_kwargs)
        elif self.save_show_or_return in ["show", "both", "all"]:
            if self.show_legend:
                plt.subplots_adjust(right=0.85)
            plt.show()
        elif self.save_show_or_return in ["return", "all"]:
            return plt.clf()

    def _check_iterable(self, arg):
        if arg is None or isinstance(arg, str):
            return [arg]
        else:
            return list(arg)
