import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from anndata import AnnData
from scipy.interpolate import interp1d

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import save_return_show_fig_utils


def glm_fit(
    adata: AnnData,
    gene: Union[str, list] = None,
    feature_x: str = None,
    feature_y: str = "expression",
    glm_key: str = "glm_degs",
    frac: float = 0.6,
    point_size: float = 1,
    point_color: Union[str, np.ndarray, list] = "skyblue",
    line_size: float = 1,
    line_color: str = "black",
    ax_size: Union[tuple, list] = (3, 3),
    background_color: str = "white",
    ncols: int = 4,
    show_point: bool = True,
    show_line: bool = True,
    show_legend: bool = True,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Visualize the glm_degs result.

    Args:
        adata: An Anndata object contain glm_degs result in ``.uns[glm_key]``.
        gene: A gene name or a list of genes that will be used to plot.
        feature_x: The key in ``.uns[glm_key]['correlation'][gene]`` that corresponds to the independent variables, such as ``'torsion'``, etc.
        feature_y: The key in ``.uns[glm_key]['correlation'][gene]`` that corresponds to the dependent variables, such as ``'expression'``, etc.
        glm_key: The key in ``.uns`` that corresponds to the glm_degs result.
        frac: Between 0 and 1. The fraction of the data used when estimating each feature_y-value.
        show_legend: Whether to show the legend.
        point_size: The scale of the feature_y point size.
        point_color: The color of the feature_y point.
        line_size: The scale of the fitted line width.
        line_color: The color of the fitted line.
        ax_size: The width and height of each ax.
        background_color: The background color of the figure.
        ncols: Number of columns for the figure.
        show_point: Whether to show the scatter plot.
        show_line: Whether to show the line plot.
        show_legend: Whether to show the legend.
        save_show_or_return: If ``'both'``, it will save and plot the figure at the same time.

                             If ``'all'``, the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will be passed to the save_fig function.

                     By default, it is an empty dictionary and the save_fig function will use the ``{"path": None, "prefix": 'scatter',
                     "dpi": None, "ext": 'pdf', "transparent": True, "close": True, "verbose": True}`` as its parameters.

                     Otherwise, you can provide a dictionary that properly modify those keys according to your needs.
        **kwargs: Additional parameters that will be passed into the ``statsmodels.nonparametric.smoothers_lowess.lowess`` function.
    """

    assert not (gene is None), "``gene`` cannot be None."
    assert not (feature_x is None), "``feature_x`` cannot be None."
    assert not (feature_y is None), "``feature_y`` cannot be None."
    assert (
        glm_key in adata.uns
    ), f"``glm_key`` does not exist in adata.uns, please replace ``glm_key`` or run st.tl.glm_degs(key_added={glm_key})."

    genes = list(gene) if isinstance(gene, list) else [gene]
    genes_data = [adata.uns[glm_key]["correlation"][g].copy() for g in genes]

    ncols = len(genes) if len(genes) < ncols else ncols
    nrows = math.ceil(len(genes) / ncols)
    fig = plt.figure(figsize=(ax_size[0] * ncols, ax_size[1] * nrows))

    axes_list = []
    for i, data in enumerate(genes_data):
        data.sort_values(by=feature_x, ascending=True, axis=0, inplace=True)
        feature_x_values = np.asarray(data[feature_x]).flatten()

        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_title(f"Gene: {genes[i]}")

        feature_y_values = np.asarray(data[feature_y]).flatten()

        plot_feature_x_values = feature_x_values[feature_y_values != 0]
        plot_feature_y_values = feature_y_values[feature_y_values != 0]
        if show_point:
            ax.scatter(
                plot_feature_x_values,
                plot_feature_y_values,
                c=point_color,
                s=point_size,
            )

        if show_line:
            # lowess: Locally Weighted Scatter-plot Smoothing. Whether to use the lowess function on the feature_y value.
            tmp = sm.nonparametric.lowess(
                plot_feature_y_values, plot_feature_x_values, frac=frac, is_sorted=True, return_sorted=True, **kwargs
            )
            f = interp1d(tmp[:, 0], tmp[:, 1], bounds_error=False)
            lowess_feature_y_values = f(plot_feature_x_values)
            ax.plot(
                plot_feature_x_values,
                lowess_feature_y_values,
                color=line_color,
                linewidth=line_size,
            )
        axes_list.append(ax)

    fig.supxlabel(feature_x)
    fig.supylabel(feature_y)
    if show_legend:
        fig.legend(loc="center right")

    added_pad = nrows * 0.1 if ncols * 2 < nrows else ncols * 0.2
    plt.tight_layout(pad=1 + added_pad)
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=show_legend,
        background=background_color,
        prefix="glm_degs",
        save_kwargs=save_kwargs,
        total_panels=len(genes),
        fig=fig,
        axes=axes_list,
        return_all=False,
        return_all_list=None,
    )
