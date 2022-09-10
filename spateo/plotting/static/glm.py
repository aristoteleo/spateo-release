import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import statsmodels.api as sm
from anndata import AnnData
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

        ax = plt.subplot(nrow, ncol, i + 1)
        ax.set_title(f"Gene: {genes[i]}")
        if not (feature_y is None):
            ax.scatter(data[feature_x], data[feature_y], c=point_color, s=point_size, **kwargs)
        if not (feature_fit is None):
            ax.plot(data[feature_x], data[feature_fit], color=line_color, linewidth=line_size)
        # fit_line = sm.OLS(data[feature_y], sm.add_constant(data[feature_x], prepend=True)).fit()
        # abline_plot(model_results=fit_line, ax=ax, color=line_color, linewidth=line_size)
        axes_list.append(ax)

    fig.supxlabel(feature_x)
    fig.supylabel(feature_y)
    plt.tight_layout()
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
