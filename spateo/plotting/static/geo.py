"""Plotting functions for spatial geometry plots.
"""

import anndata
import geopandas as gpd
import lack
from typing import Union
from .scatters import scatters

slog = lack.LoggerManager(namespace="spateo")

# from .scatters import (
#     scatters,
#     docstrings,
# )

# from ..dynamo_logger import main_critical, main_info, main_finish_progress, main_log_time, main_warning

# docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


# @docstrings.with_indent(4)
def geo(
    adata: anndata.AnnData,
    color: Union[list, str, None] = None,
    genes: Union[list, None] = [],
    gene_cmaps=None,
    dpi: int = 100,
    alpha: float = 0.8,
    boundary_width: float = 0.2,
    boundary_color="black",
    stack_genes: bool = False,
    stack_genes_threshold: float = 0.01,
    stack_colors_legend_size: int = 10,
    figsize=(6, 6),
    aspect: str = "equal",
    *args,
    **kwargs
):
    """\
    Geometry plot for cell segmentation on physical space.

    Args:
        adata:
            an Annodata object that contain the physical coordinates for each bin/cell, etc.
        genes:
            The gene list that will be used to plot the gene expression on the same geometry plot. Each gene will have a
            different color. Can be a single gene name string and we will convert it to a list.
        color: `string` (default: `ntr`)
            Any or any list of column names or gene name, etc. that will be used for coloring cells.
            If `color` is not None, stack_genes will be disabled automatically because `color` can contain non numerical
            values.
        stack_genes:
            whether to show all gene plots on the same plot.
        stack_genes_threshold:
            lower bound of gene values that will be drawn on the plot.
        stack_colors_legend_size:
            control the size of legend when stacking genes.
        alpha: `float`
            The alpha value of the cells.
        boundary_width: `float`, (default: 0.2)
            The line width of boundary.
        boundary_color: (default: "black")
            The color value of boundary.
        dpi: `float`, (default: 100.0)
            The resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
            comprises. dpi is different from ppi or points per inches. Note that most elements like lines, markers,
            texts have a size given in points so you can convert the points to inches. Matplotlib figures use Points per
            inch (ppi) of 72. A line with thickness 1 point will be 1./72. inch wide. A text with fontsize 12 points
            will be 12./72. inch height. Of course if you change the figure size in inches, points will not change, so a
            larger figure in inches still has the same size of the elements.Changing the figure size is thus like taking
            a piece of paper of a different size. Doing so, would of course not change the width of the line drawn with
            the same pen. On the other hand, changing the dpi scales those elements. At 72 dpi, a line of 1 point size
            is one pixel strong. At 144 dpi, this line is 2 pixels strong. A larger dpi will therefore act like a
            magnifying glass. All elements are scaled by the magnifying power of the lens. see more details at answer 2
            by @ImportanceOfBeingErnest:
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        aspect: `str`
            Set the aspect of the axis scaling, i.e. the ratio of y-unit to x-unit. In physical spatial plots, the
            default is 'equal'. See more details at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_aspect.html
        %(scatters.parameters.no_adata|basis|figsize)s

        Args:
            plots gene or cell feature of the adata object on the physical spatial coordinates with cell segmentations.
    """
    slog.main_info("Plotting geometry info on adata")
    slog.main_log_time()
    if not isinstance(adata.obs, gpd.GeoDataFrame):
        slog.main_critical(
            "The obs of your adata is not a `geopandas.GeoDataFrame`. Please check your argument passed " "in."
        )
        return

    if color is not None and stack_genes:
        slog.main_warning(
            "Set `stack_genes` to False because `color` argument cannot be used with stack_genes. If you would like to"
            "stack genes (or other numerical values), please pass gene expression like column names into `gene` "
            "argument."
        )
        stack_genes = False

    genes = [genes] if type(genes) is str else list(genes)
    # concatenate genes and colors for scatters plot
    if color is not None and genes is not None:
        color = [color] if type(color) is str else list(color)
        genes.extend(color)

    show_colorbar = True
    if stack_genes:
        slog.main_warning("disable side colorbar due to colorbar scale (numeric tick) related issue.")
        show_colorbar = False

    if genes is None or (len(genes) == 0):
        if color is not None:
            genes = color
        else:
            slog.main_critical("No genes provided. Please check your argument passed in.")
            return

    res = scatters(
        adata,
        color=genes,
        figsize=figsize,
        dpi=dpi,
        alpha=alpha,
        stack_colors=stack_genes,
        stack_colors_threshold=stack_genes_threshold,
        stack_colors_title="stacked spatial genes",
        show_colorbar=show_colorbar,
        stack_colors_legend_size=stack_colors_legend_size,
        stack_colors_cmaps=gene_cmaps,
        geo=True,
        boundary_width=boundary_width,
        boundary_color=boundary_color,
        aspect=aspect,
        *args,
        **kwargs,
    )

    slog.main_finish_progress("geometry plot")
    return res
