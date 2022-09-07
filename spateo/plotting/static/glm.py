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

import seaborn as sns

from .utils import save_return_show_fig_utils


def glm_abline(
    adata: AnnData,
    gene: Union[str, list] = None,
    feature_x: str = None,
    feature_y: str = None,
    glm_key: str = "glm_degs",
    point_size: Optional[float] = 1,
    point_color: Optional[str] = "skyblue",
    line_size: Optional[float] = 1,
    line_color: Optional[str] = "black",
    ncol: int = 4,
    ax_height: Union[float, int] = 2,
    ax_width: Union[float, int] = 3,
    dpi: int = 100,
    show_legend: bool = True,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):
    assert not (gene is None), "``gene`` cannot be None."
    assert not (feature_x is None), "``feature_x`` cannot be None."
    assert not (feature_y is None), "``feature_y`` cannot be None."
    assert (
        glm_key in adata.uns
    ), f"``glm_key`` does not exist in adata.uns, please replace ``glm_key`` or run st.tl.glm_degs(key_added={glm_key})."

    genes = list(gene) if isinstance(gene, list) else [gene]
    genes_data = [adata.uns[glm_key]["correlation"][g].copy() for g in genes]

    nrow = math.ceil(len(genes) / ncol)
    fig = plt.figure(figsize=(ax_width * ncol, ax_height * nrow))
    sns.set_theme(
        context="paper",
        style="white",
        font="Arial",
        font_scale=1,
        rc={
            "font.size": 10.0,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["Arial", "sans-serif", "Helvetica", "DejaVu Sans", "Bitstream Vera Sans"],
        },
    )
    for i, data in enumerate(genes_data):
        ax = plt.subplot(nrow, ncol, i + 1)
        ax.set_title(f"Model Fit Plot of  {genes[i]}")
        ax.scatter(data[feature_x], data[feature_y], c=point_color, s=point_size, **kwargs)

        fit_line = sm.OLS(data[feature_y], sm.add_constant(data[feature_x], prepend=True)).fit()
        abline_plot(model_results=fit_line, ax=ax, color=line_color, linewidth=line_size)

    plt.ylabel(feature_y)
    plt.xlabel(feature_x)

    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=show_legend,
        background="white",
        prefix="multi_slices",
        save_kwargs=save_kwargs,
        total_panels=len(genes),
        fig=fig,
        return_all=False,
        return_all_list=None,
    )
