import itertools
from typing import Optional

import anndata

from .scatters import scatters
from .utils import _convert_to_geo_dataframe

# DEFAULT COLORS from
# https://github.com/scikit-image/scikit-image/blob/8faf62d677e73f5911cfc5232d5de02f1da7486b/skimage/color/colorlabel.py
DEFAULT_COLORS = ("red", "blue", "yellow", "magenta", "green", "indigo", "darkorange", "cyan", "pink", "yellowgreen")


def color_label(
    adata: anndata.AnnData,
    basis="contour",
    color_key: Optional[list] = None,
    dpi: int = 100,
    boundary_width: float = 0.2,
    boundary_color="black",
    figsize=(6, 6),
    aspect: str = "equal",
    *args,
    **kwargs
):
    """Color the segmented cells with different colors.

    Args:
        adata: ~anndata.AnnData
            An Annodata object.
        basis: str
            The key to the column in the adata.obs, from which the contour of the cell segmentation will be generated.
        color_key: list
            List of colors. If the number of labels exceeds the number of colors, then the colors are cycled.
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
        boundary_width: `float`, (default: 0.2)
            The line width of boundary.
        boundary_color: (default: "black")
            The color value of boundary.
        figsize: `tuple`
            The size of the figure in inches.
        aspect: `str`
            Set the aspect of the axis scaling, i.e. the ratio of y-unit to x-unit. In physical spatial plot, the
            default is 'equal'. See more details at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_aspect.html
    """

    # main_info("Plotting geometry info on adata")
    # main_log_time()

    adata = adata.copy()  # make sure don't modify the original data.
    adata = _convert_to_geo_dataframe(adata, basis)

    if color_key is None:
        color_key = DEFAULT_COLORS

    color_cycle = itertools.cycle(color_key)
    adata.obs["color_label"] = list(itertools.islice(color_cycle, adata.n_obs))

    res = scatters(
        adata,
        color="color_label",
        figsize=figsize,
        dpi=dpi,
        show_colorbar=False,
        show_legend=False,
        geo=True,
        boundary_width=boundary_width,
        boundary_color=boundary_color,
        aspect=aspect,
        color_key=color_key,
        *args,
        **kwargs,
    )

    # main_finish_progress("color label plot")
    return res
