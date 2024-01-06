import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import save_return_show_fig_utils


def glm_fit(
    adata: AnnData,
    genes: Optional[Union[str, list]] = None,
    feature_x: str = None,
    feature_y: str = "expression",
    glm_key: str = "glm_degs",
    remove_zero: bool = False,
    color_key: Optional[str] = None,
    color_key_cmap: Optional[str] = "vlag",
    point_size: float = 14,
    point_color: Union[str, np.ndarray, list] = "skyblue",
    line_size: float = 2,
    line_color: str = "black",
    ax_size: Union[tuple, list] = (6, 4),
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
    Plot the glm_degs result in a scatterplot.

    Args:
        adata: An Anndata object contain glm_degs result in ``.uns[glm_key]``.
        genes: A gene name or a list of genes that will be used to plot.
        feature_x: The key in ``.uns[glm_key]['correlation'][gene]`` that corresponds to the independent variables, such as ``'torsion'``, etc.
        feature_y: The key in ``.uns[glm_key]['correlation'][gene]`` that corresponds to the dependent variables, such as ``'expression'``, etc.
        glm_key: The key in ``.uns`` that corresponds to the glm_degs result.
        remove_zero: Whether to remove the data equal to 0 saved in ``.uns[glm_key]['correlation'][gene][feature_y]``.
        color_key: This can either be an explicit dict mapping labels to colors (as strings of form ‘#RRGGBB’), or an array like object providing one color for each distinct category being provided in labels.
        color_key_cmap: The name of a matplotlib colormap to use for categorical coloring.
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
        **kwargs: Additional parameters that will be passed into the ``seaborn.scatterplot`` function.
    """

    assert not (feature_x is None), "``feature_x`` cannot be None."
    assert not (feature_y is None), "``feature_y`` cannot be None."
    assert (
        glm_key in adata.uns
    ), f"``glm_key`` does not exist in adata.uns, please replace ``glm_key`` or run st.tl.glm_degs(key_added={glm_key})."

    genes = list(adata.uns[glm_key]["glm_result"].index) if genes is None else genes
    genes = list(genes) if isinstance(genes, list) else [genes]
    genes_data = [adata.uns[glm_key]["correlation"][g].copy() for g in genes]

    ncols = len(genes) if len(genes) < ncols else ncols
    nrows = math.ceil(len(genes) / ncols)
    fig = plt.figure(figsize=(ax_size[0] * ncols, ax_size[1] * nrows))

    axes_list = []
    for i, data in enumerate(genes_data):
        data.sort_values(by=feature_x, ascending=True, axis=0, inplace=True)
        if remove_zero:
            data = data[np.asarray(data[feature_y]).flatten() != 0]

        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_title(f"Gene: {genes[i]}")

        if show_point:
            sns.scatterplot(
                data=data,
                x=feature_x,
                y=feature_y,
                hue=color_key,
                palette=color_key_cmap,
                color=point_color,
                s=point_size,
                legend=show_legend,
                ax=ax,
                **kwargs,
            )
            ax.set_ylabel(feature_y)

        if show_line:
            ax = ax.twinx() if show_point is True else ax
            sns.lineplot(
                data=data,
                x=feature_x,
                y="mu",
                color=line_color,
                lw=line_size,
                legend=False,
                ax=ax,
            )
            ax.set_ylabel("mu")
        axes_list.append(ax)

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


def glm_heatmap(
    adata: AnnData,
    genes: Optional[Union[str, list]] = None,
    feature_x: str = None,
    feature_y: str = "expression",
    glm_key: str = "glm_degs",
    lowess_smooth: bool = True,
    frac: float = 0.2,
    robust: bool = True,
    colormap: str = "vlag",
    figsize: tuple = (6, 6),
    background_color: str = "white",
    show_legend: bool = True,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Plot the glm_degs result in a heatmap.

    Args:
        adata: An Anndata object contain glm_degs result in ``.uns[glm_key]``.
        genes: A gene name or a list of genes that will be used to plot.
        feature_x: The key in ``.uns[glm_key]['correlation'][gene]`` that corresponds to the independent variables, such as ``'torsion'``, etc.
        feature_y: The key in ``.uns[glm_key]['correlation'][gene]`` that corresponds to the dependent variables, such as ``'expression'``, etc.
        glm_key: The key in ``.uns`` that corresponds to the glm_degs result.
        lowess_smooth: If True, use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression).
        frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
        robust: If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.
        colormap: The name of a matplotlib colormap.
        figsize: The width and height of figure.
        background_color: The background color of the figure.
        show_legend: Whether to show the legend.
        save_show_or_return: If ``'both'``, it will save and plot the figure at the same time.

                             If ``'all'``, the figure will be saved, displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will be passed to the save_fig function.

                     By default, it is an empty dictionary and the save_fig function will use the ``{"path": None, "prefix": 'scatter',
                     "dpi": None, "ext": 'pdf', "transparent": True, "close": True, "verbose": True}`` as its parameters.

                     Otherwise, you can provide a dictionary that properly modify those keys according to your needs.
        **kwargs: Additional parameters that will be passed into the ``seaborn.heatmap`` function.
    """
    assert not (feature_x is None), "``feature_x`` cannot be None."
    assert not (feature_y is None), "``feature_y`` cannot be None."
    assert (
        glm_key in adata.uns
    ), f"``glm_key`` does not exist in adata.uns, please replace ``glm_key`` or run st.tl.glm_degs(key_added={glm_key})."

    genes = list(adata.uns[glm_key]["glm_result"].index) if genes is None else genes
    genes = list(genes) if isinstance(genes, list) else [genes]

    genes_data = []
    for g in genes:
        gene_data = adata.uns[glm_key]["correlation"][g].copy()
        gene_data.sort_values(by=feature_x, ascending=True, axis=0, inplace=True)
        gene_data = gene_data.loc[:, [feature_x, feature_y]]
        data = pd.DataFrame(gene_data.groupby(by=feature_x)[feature_y].mean())
        if lowess_smooth:
            import statsmodels.api as sm

            data = pd.DataFrame(sm.nonparametric.lowess(exog=data.index, endog=data[feature_y], frac=frac))[1]
        genes_data.append(data)
    genes_data = pd.concat(genes_data, axis=1)
    genes_data.fillna(value=0, inplace=True)
    genes_data.columns = genes
    genes_data = genes_data.T

    max_sort = np.argsort(np.argmax(genes_data.values, axis=1))
    genes_data = genes_data.iloc[max_sort]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(genes_data, cmap=colormap, robust=robust, ax=ax, **kwargs)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)

    plt.tight_layout(pad=1)
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=show_legend,
        background=background_color,
        prefix="glm_degs",
        save_kwargs=save_kwargs,
        total_panels=len(genes),
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )
