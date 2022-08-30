import itertools
from typing import Optional

import anndata
from matplotlib import cm, colors

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


# Custom bright colors palette:
bright_10 = [
    "#9d00fe",
    "#0000ff",
    "#ff0000",
    "#21b20c",
    "#f2e50b",
    "#6e260e",
    "#cd7f32",
    "#ff7518",
    "#ff0000",
    "#feb3c6",
]

# Colormaps used in Scanpy:
# Vega10:
vega_10 = list(map(colors.to_hex, cm.tab10.colors))
vega_10_scanpy = vega_10.copy()
vega_10_scanpy[2] = "#279e68"  # green
vega_10_scanpy[4] = "#aa40fc"  # purple
vega_10_scanpy[8] = "#b5bd61"  # kakhi


# Vega20:
vega_20 = list(map(colors.to_hex, cm.tab20.colors))

# reorderd, some removed, some added
vega_20_scanpy = [
    # dark without grey:
    *vega_20[0:14:2],
    *vega_20[16::2],
    # light without grey:
    *vega_20[1:15:2],
    *vega_20[17::2],
    # manual additions:
    "#ad494a",
    "#8c6d31",
]
vega_20_scanpy[2] = vega_10_scanpy[2]
vega_20_scanpy[4] = vega_10_scanpy[4]
vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey

default_20 = vega_20_scanpy


# Additional colormaps:
# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
# update 1
# orig reference http://epub.wu.ac.at/1692/1/document.pdf
zeileis_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    # these last ones were added:
    "#7f7f7f",
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]

default_28 = zeileis_28


# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
godsnot_102 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]
