import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from anndata import AnnData
from scipy.interpolate import interp1d
from statsmodels.graphics.api import abline_plot

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import save_return_show_fig_utils


def glm_fit(
    adata: AnnData,
    gene: Union[str, list] = None,
    feature_x: str = None,
    feature_y: Optional[str] = None,
    feature_fit: Optional[str] = "mu",
    glm_key: str = "glm_degs",
    lowess_smooth: bool = True,
    frac: float = 0.1,
    point_size: float = 1,
    point_color: str = "skyblue",
    line_size: float = 1,
    line_color: str = "black",
    ax_size: Union[tuple, list] = (3, 3),
    background_color: str = "white",
    ncol: int = 4,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):

    assert not (gene is None), "``gene`` cannot be None."
    assert not (feature_x is None), "``feature_x`` cannot be None."
    assert (
        glm_key in adata.uns
    ), f"``glm_key`` does not exist in adata.uns, please replace ``glm_key`` or run st.tl.glm_degs(key_added={glm_key})."

    genes = list(gene) if isinstance(gene, list) else [gene]
    genes_data = [adata.uns[glm_key]["correlation"][g].copy() for g in genes]

    ncol = len(genes) if len(genes) < ncol else ncol
    nrow = math.ceil(len(genes) / ncol)
    fig = plt.figure(figsize=(ax_size[0] * ncol, ax_size[1] * nrow))

    axes_list = []
    for i, data in enumerate(genes_data):
        data.sort_values(by=feature_x, ascending=True, axis=0, inplace=True)
        feature_x_values = np.asarray(data[feature_x]).flatten()

        ax = plt.subplot(nrow, ncol, i + 1)
        ax.set_title(f"Gene: {genes[i]}")
        if not (feature_y is None):
            feature_y_values = np.asarray(data[feature_y]).flatten()
            if lowess_smooth:
                tmp = sm.nonparametric.lowess(
                    feature_y_values, feature_x_values, frac=frac, is_sorted=True, return_sorted=True
                )
                f = interp1d(tmp[:, 0], tmp[:, 1], bounds_error=False)
                feature_y_values = f(feature_x_values)
            ax.scatter(feature_x_values, feature_y_values, c=point_color, s=point_size, **kwargs)

        if not (feature_fit is None):
            feature_fit_values = np.asarray(data[feature_fit]).flatten()
            ax.plot(feature_x_values, feature_fit_values, color=line_color, linewidth=line_size)
            # fit_line = sm.OLS(data[feature_y], sm.add_constant(data[feature_x], prepend=True)).fit()
            # abline_plot(model_results=fit_line, ax=ax, color=line_color, linewidth=line_size)
        axes_list.append(ax)

    fig.supxlabel(feature_x)
    fig.supylabel(feature_y)
    added_pad = nrow * 0.1 if ncol * 2 < nrow else ncol * 0.2
    plt.tight_layout(pad=1 + added_pad)
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=False,
        background=background_color,
        prefix="glm_degs",
        save_kwargs=save_kwargs,
        total_panels=len(genes),
        fig=fig,
        axes=axes_list,
        return_all=False,
        return_all_list=None,
    )
