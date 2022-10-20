# code adapted from https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/configuration.py
import inspect
import logging
import os
import warnings
from functools import wraps
from typing import Optional, Tuple, Union

import colorcet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import cycler
from matplotlib import cm, colors, rcParams
from scipy import sparse

from .errors import ConfigurationError
from .logging import logger_manager as lm


class SpateoConfig:
    def __init__(
        self,
        logging_level: int = logging.INFO,
        n_threads: int = os.cpu_count(),
    ):
        self.logging_level = logging_level
        self.n_threads = n_threads

    @property
    def logging_level(self):
        return self.__logging_level

    @property
    def n_threads(self):
        return self.__n_threads

    @logging_level.setter
    def logging_level(self, level):
        lm.main_debug(f"Setting logging level to {level}.")
        if isinstance(level, str):
            level = level.lower()
            if level == "debug":
                level = logging.DEBUG
            elif level == "info":
                level = logging.INFO
            elif level == "warning":
                level = logging.WARNING
            elif level == "error":
                level = logging.ERROR
            elif level == "critical":
                level = logging.CRITICAL
        lm.main_set_level(level)
        self.__logging_level = level

    @n_threads.setter
    def n_threads(self, n):
        lm.main_debug(f"Setting n_threads to {n}.")
        try:
            import torch

            torch.set_num_threads(n)
        except:
            pass
        try:
            import cv2

            cv2.setNumThreads(n)
        except:
            pass
        try:
            import tensorflow as tf

            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(n)
        except:
            pass
        self.__n_threads = n


config = SpateoConfig()


class SpateoAdataKeyManager:
    # This key will be present in the .uns of the AnnData to indicate the type of
    # information the AnnData holds..
    ADATA_TYPE_KEY = "__type"
    ADATA_DEFAULT_TYPE = None
    ADATA_AGG_TYPE = "AGG"  # This is an AnnData containing aggregated UMI counts
    ADATA_UMI_TYPE = "UMI"  # This is an obs x genes AnnData (canonical)

    UNS_PP_KEY = "pp"
    UNS_SPATIAL_KEY = "spatial"
    UNS_SPATIAL_BINSIZE_KEY = "binsize"
    UNS_SPATIAL_SCALE_KEY = "scale"
    UNS_SPATIAL_SCALE_UNIT_KEY = "scale_unit"
    UNS_SPATIAL_SEGMENTATION_KEY = "segmentation"
    UNS_SPATIAL_ALIGNMENT_KEY = "alignment"
    UNS_SPATIAL_QC_KEY = "qc"

    SPLICED_LAYER_KEY = "spliced"
    UNSPLICED_LAYER_KEY = "unspliced"
    STAIN_LAYER_KEY = "stain"
    LABELS_LAYER_KEY = "labels"
    MASK_SUFFIX = "mask"
    MARKERS_SUFFIX = "markers"
    DISTANCES_SUFFIX = "distances"
    BINS_SUFFIX = "bins"
    LABELS_SUFFIX = "labels"
    SCORES_SUFFIX = "scores"
    EXPANDED_SUFFIX = "expanded"
    AUGMENTED_SUFFIX = "augmented"
    SELECTION_SUFFIX = "selection"

    X_LAYER = "X"

    def gen_new_layer_key(layer_name, key, sep="_") -> str:
        if layer_name == "":
            return key
        if layer_name[-1] == sep:
            return layer_name + key
        return sep.join([layer_name, key])

    def select_layer_data(
        adata: AnnData, layer: str, copy: bool = False, make_dense: bool = False
    ) -> Union[np.ndarray, sparse.spmatrix]:
        lm.main_info(f"<select> {layer} layer in AnnData Object")
        if layer is None:
            layer = SpateoAdataKeyManager.X_LAYER
        res_data = None
        if layer == SpateoAdataKeyManager.X_LAYER:
            res_data = adata.X
        else:
            res_data = adata.layers[layer]
        if make_dense and sparse.issparse(res_data):
            return res_data.A
        if copy:
            return res_data.copy()
        return res_data

    def set_layer_data(
        adata: AnnData, layer: str, vals: np.ndarray, var_indices: Optional[np.ndarray] = None, replace: bool = False
    ):
        lm.main_info_insert_adata_layer(layer)

        # Mostly for testing
        if replace:
            adata.layers[layer] = vals
            return

        if var_indices is None:
            var_indices = slice(None)
        if layer == SpateoAdataKeyManager.X_LAYER:
            adata.X[:, var_indices] = vals
        elif layer in adata.layers:
            adata.layers[layer][:, var_indices] = vals
        else:
            # layer does not exist in adata
            # ignore var_indices and set values as a new layer
            adata.layers[layer] = vals

    def get_adata_type(adata: AnnData) -> str:
        return adata.uns[SpateoAdataKeyManager.ADATA_TYPE_KEY]

    def adata_is_type(adata: AnnData, t: str) -> bool:
        return SpateoAdataKeyManager.get_adata_type(adata) == t

    def check_adata_is_type(t: str, argname: str = "adata", optional: bool = False):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get original, unwrapped function in case multiple decorators
                # are applied.
                unwrapped = inspect.unwrap(func)
                # Obtain arguments by name.
                call_args = inspect.getcallargs(unwrapped, *args, **kwargs)
                adata = call_args[argname]
                passing = (
                    all(SpateoAdataKeyManager.adata_is_type(_adata, t) for _adata in adata)
                    if isinstance(adata, (list, tuple))
                    else SpateoAdataKeyManager.adata_is_type(adata, t)
                    if type(adata) == AnnData
                    else False
                )
                if (not optional or adata is not None) and not passing:
                    if isinstance(adata, (list, tuple)):
                        raise ConfigurationError(
                            f"AnnDatas provided to `{argname}` argument must be of `{t}` type, but some are not."
                        )
                    elif type(adata) == AnnData:
                        raise ConfigurationError(
                            f"AnnData provided to `{argname}` argument must be of `{t}` type, but received "
                            f"`{SpateoAdataKeyManager.get_adata_type(adata)}` type."
                        )
                    else:
                        raise ConfigurationError(f"AnnData is not AnnData object, but {type(adata)}.")
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def init_adata_type(adata: AnnData, t: Optional[str] = None):
        lm.main_info_insert_adata_uns(SpateoAdataKeyManager.ADATA_TYPE_KEY)
        if t is None:
            t = SpateoAdataKeyManager.ADATA_DEFAULT_TYPE
        adata.uns[SpateoAdataKeyManager.ADATA_TYPE_KEY] = t

    def init_uns_pp_namespace(adata: AnnData):
        lm.main_info_insert_adata_uns(SpateoAdataKeyManager.UNS_PP_KEY)
        if SpateoAdataKeyManager.UNS_PP_KEY not in adata.uns:
            adata.uns[SpateoAdataKeyManager.UNS_PP_KEY] = {}

    def init_uns_spatial_namespace(adata: AnnData):
        lm.main_info_insert_adata_uns(SpateoAdataKeyManager.UNS_SPATIAL_KEY)
        if SpateoAdataKeyManager.UNS_SPATIAL_KEY not in adata.uns:
            adata.uns[SpateoAdataKeyManager.UNS_SPATIAL_KEY] = {}

    def set_uns_spatial_attribute(adata: AnnData, key: str, value: object):
        if SpateoAdataKeyManager.UNS_SPATIAL_KEY not in adata.uns:
            SpateoAdataKeyManager.init_uns_spatial_namespace(adata)
        adata.uns[SpateoAdataKeyManager.UNS_SPATIAL_KEY][key] = value

    def get_uns_spatial_attribute(adata: AnnData, key: str) -> object:
        return adata.uns[SpateoAdataKeyManager.UNS_SPATIAL_KEY][key]

    def has_uns_spatial_attribute(adata: AnnData, key: str) -> bool:
        return key in adata.uns[SpateoAdataKeyManager.UNS_SPATIAL_KEY]

    def get_agg_bounds(adata: AnnData) -> Tuple[int, int, int, int]:
        """Get (xmin, xmax, ymin, ymax) for AGG type anndatas."""
        atype = SpateoAdataKeyManager.get_adata_type(adata)
        if atype != SpateoAdataKeyManager.ADATA_AGG_TYPE:
            raise ConfigurationError(f"AnnData has incorrect type: {atype}")
        return int(adata.obs_names[0]), int(adata.obs_names[-1]), int(adata.var_names[0]), int(adata.var_names[-1])


SKM = SpateoAdataKeyManager


# Means to shift the scale of colormaps:
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fire", colorcet.fire)
darkblue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("darkblue", colorcet.kbc)
darkgreen_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("darkgreen", colorcet.kgy)
darkred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "darkred", colors=colorcet.linear_kry_5_95_c72[:192], N=256
)
darkpurple_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("darkpurple", colorcet.linear_bmw_5_95_c89)
# add gkr theme
div_blue_black_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_black_red", colorcet.diverging_gkr_60_10_c40
)
# add RdBu_r theme
div_blue_red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div_blue_red", colorcet.diverging_bwr_55_98_c37
)
# add glasbey_bw for cell annotation in white background
glasbey_white_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("glasbey_white", colorcet.glasbey_bw_minc_20)
# add glasbey_bw_minc_20_maxl_70 theme for cell annotation in dark background
glasbey_dark_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "glasbey_dark", colorcet.glasbey_bw_minc_20_maxl_70
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if "fire" not in matplotlib.colormaps():
        plt.register_cmap("fire", fire_cmap)
    if "darkblue" not in matplotlib.colormaps():
        plt.register_cmap("darkblue", darkblue_cmap)
    if "darkgreen" not in matplotlib.colormaps():
        plt.register_cmap("darkgreen", darkgreen_cmap)
    if "darkred" not in matplotlib.colormaps():
        plt.register_cmap("darkred", darkred_cmap)
    if "darkpurple" not in matplotlib.colormaps():
        plt.register_cmap("darkpurple", darkpurple_cmap)
    if "div_blue_black_red" not in matplotlib.colormaps():
        plt.register_cmap("div_blue_black_red", div_blue_black_red_cmap)
    if "div_blue_red" not in matplotlib.colormaps():
        plt.register_cmap("div_blue_red", div_blue_red_cmap)
    if "glasbey_white" not in matplotlib.colormaps():
        plt.register_cmap("glasbey_white", glasbey_white_cmap)
    if "glasbey_dark" not in matplotlib.colormaps():
        plt.register_cmap("glasbey_dark", glasbey_dark_cmap)

_themes = {
    "fire": {
        "cmap": "fire",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "fire",
    },
    "viridis": {
        "cmap": "viridis",
        "color_key_cmap": "Spectral",
        "background": "white",
        "edge_cmap": "gray",
    },
    "inferno": {
        "cmap": "inferno",
        "color_key_cmap": "Spectral",
        "background": "black",
        "edge_cmap": "gray",
    },
    "blue": {
        "cmap": "Blues",
        "color_key_cmap": "tab20",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "red": {
        "cmap": "Reds",
        "color_key_cmap": "tab20b",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "green": {
        "cmap": "Greens",
        "color_key_cmap": "tab20c",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "darkblue": {
        "cmap": "darkblue",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkred",
    },
    "darkred": {
        "cmap": "darkred",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkblue",
    },
    "darkgreen": {
        "cmap": "darkgreen",
        "color_key_cmap": "rainbow",
        "background": "black",
        "edge_cmap": "darkpurple",
    },
    "div_blue_black_red": {
        "cmap": "div_blue_black_red",
        "color_key_cmap": "div_blue_black_red",
        "background": "black",
        "edge_cmap": "gray_r",
    },
    "div_blue_red": {
        "cmap": "div_blue_red",
        "color_key_cmap": "div_blue_red",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "glasbey_dark": {
        "cmap": "glasbey_dark",
        "color_key_cmap": "glasbey_dark",
        "background": "black",
        "edge_cmap": "gray",
    },
    "glasbey_white_zebrafish": {
        "cmap": "zebrafish",
        "color_key_cmap": "zebrafish",
        "background": "white",
        "edge_cmap": "gray_r",
    },
    "glasbey_white": {
        "cmap": "glasbey_white",
        "color_key_cmap": "glasbey_white",
        "background": "white",
        "edge_cmap": "gray_r",
    },
}


def reset_rcParams():
    """Reset `matplotlib.rcParams` to defaults."""
    from matplotlib import rcParamsDefault

    rcParams.update(rcParamsDefault)


# create cmap
zebrafish_colors = [
    "#4876ff",
    "#85C7F2",
    "#cd00cd",
    "#911eb4",
    "#000080",
    "#808080",
    "#008080",
    "#ffc125",
    "#262626",
    "#3cb44b",
    "#ff4241",
    "#b77df9",
]

# https://github.com/vega/vega/wiki/Scales#scale-range-literals
cyc_10 = list(map(colors.to_hex, cm.tab10.colors))
cyc_20 = list(map(colors.to_hex, cm.tab20c.colors))
zebrafish_256 = list(map(colors.to_hex, zebrafish_colors))


def spateo_theme(background="white"):
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/dark_background.mplstyle

    if background == "black":
        rcParams.update(
            {
                "lines.color": "w",
                "patch.edgecolor": "w",
                "text.color": "w",
                "axes.facecolor": background,
                "axes.edgecolor": "white",
                "axes.labelcolor": "w",
                "xtick.color": "w",
                "ytick.color": "w",
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "savefig.facecolor": background,
                "savefig.edgecolor": background,
                "grid.color": "w",
                "axes.grid": False,
            }
        )
    else:
        rcParams.update(
            {
                "lines.color": "k",
                "patch.edgecolor": "k",
                "text.color": "k",
                "axes.facecolor": background,
                "axes.edgecolor": "black",
                "axes.labelcolor": "k",
                "xtick.color": "k",
                "ytick.color": "k",
                "figure.facecolor": background,
                "figure.edgecolor": background,
                "savefig.facecolor": background,
                "savefig.edgecolor": background,
                "grid.color": "k",
                "axes.grid": False,
            }
        )


def config_spateo_rcParams(
    background="white",
    prop_cycle=zebrafish_256,
    fontsize=8,
    color_map=None,
    frameon=None,
):
    """Configure matplotlib.rcParams to spateo defaults (based on ggplot style and scanpy).
    Parameters
    ----------
        background: `str` (default: `white`)
            The background color of the plot. By default we use the white ground
            which is suitable for producing figures for publication. Setting it to `black` background will
            be great for presentation.
        prop_cycle: `list` (default: zebrafish_256)
            A list with hex color codes
        fontsize: float (default: 6)
            Size of font
        color_map: `plt.cm` or None (default: None)
            Color map
        frameon: `bool` or None (default: None)
            Whether to have frame for the figure.
    Returns
    -------
        Nothing but configure the rcParams globally.
    """

    # from http://www.huyng.com/posts/sane-color-scheme-for-matplotlib/

    rcParams["patch.linewidth"] = 0.5
    rcParams["patch.facecolor"] = "348ABD"  # blue
    rcParams["patch.edgecolor"] = "EEEEEE"
    rcParams["patch.antialiased"] = True

    rcParams["font.size"] = 10.0

    rcParams["axes.facecolor"] = "E5E5E5"
    rcParams["axes.edgecolor"] = "white"
    rcParams["axes.linewidth"] = 1
    rcParams["axes.grid"] = True
    # rcParams['axes.titlesize'] =  "x-large"
    # rcParams['axes.labelsize'] = "large"
    rcParams["axes.labelcolor"] = "555555"
    rcParams["axes.axisbelow"] = True  # grid/ticks are below elements (e.g., lines, text)

    # rcParams['axes.prop_cycle'] = cycler('color', ['E24A33', '348ABD', '988ED5', '777777', 'FBC15E', '8EBA42', 'FFB5B8'])
    # # E24A33 : red
    # # 348ABD : blue
    # # 988ED5 : purple
    # # 777777 : gray
    # # FBC15E : yellow
    # # 8EBA42 : green
    # # FFB5B8 : pink

    # rcParams['xtick.color'] = "555555"
    rcParams["xtick.direction"] = "out"

    # rcParams['ytick.color'] = "555555"
    rcParams["ytick.direction"] = "out"

    rcParams["grid.color"] = "white"
    rcParams["grid.linestyle"] = "-"  # solid line

    rcParams["figure.facecolor"] = "white"
    rcParams["figure.edgecolor"] = "white"  # 0.5

    # the following code is modified from scanpy
    # https://github.com/theislab/scanpy/blob/178a0981405ba8ccfd5031eb15bc07b3a45d2730/scanpy/plotting/_rcmod.py

    # dpi options (mpl default: 100, 100)
    rcParams["figure.dpi"] = 100
    rcParams["savefig.dpi"] = 300

    # figure (default: 0.125, 0.96, 0.15, 0.91)
    rcParams["figure.figsize"] = (6, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    # lines (defaults:  1.5, 6, 1)
    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "sans-serif",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
    ]
    fontsize = fontsize
    labelsize = 0.90 * fontsize

    # fonsizes (default: 10, medium, large, medium)
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = labelsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = labelsize

    # legend (default: 1, 1, 2, 0.8)
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycle
    rcParams["axes.prop_cycle"] = cycler(color=prop_cycle)  # use tab20c by default

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks (default: k, k, medium, medium)
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = labelsize
    rcParams["ytick.labelsize"] = labelsize

    # axes grid (default: False, #b0b0b0)
    rcParams["axes.grid"] = False
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = "RdBu_r" if color_map is None else color_map

    spateo_theme(background)

    # frame (default: True)
    frameon = False if frameon is None else frameon
    global _frameon
    _frameon = frameon


def set_figure_params(
    spateo=True,
    background="white",
    fontsize=8,
    figsize=(6, 4),
    dpi=None,
    dpi_save=None,
    frameon=None,
    vector_friendly=True,
    color_map=None,
    format="pdf",
    transparent=False,
    ipython_format="png2x",
):
    """Set resolution/size, styling and format of figures.
       This function is adapted from: https://github.com/theislab/scanpy/blob/f539870d7484675876281eb1c475595bf4a69bdb/scanpy/_settings.py
    Arguments
    ---------
        spateo: `bool` (default: `True`)
            Init default values for :obj:`matplotlib.rcParams` suited for spateo.
        background: `str` (default: `white`)
            The background color of the plot. By default we use the white ground
            which is suitable for producing figures for publication. Setting it to `black` background will
            be great for presentation.
        fontsize: `[float, float]` or None (default: `6`)
        figsize: `(float, float)` (default: `(6.5, 5)`)
            Width and height for default figure size.
        dpi: `int` or None (default: `None`)
            Resolution of rendered figures - this influences the size of figures in notebooks.
        dpi_save: `int` or None (default: `None`)
            Resolution of saved figures. This should typically be higher to achieve
            publication quality.
        frameon: `bool` or None (default: `None`)
            Add frames and axes labels to scatter plots.
        vector_friendly: `bool` (default: `True`)
            Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
        color_map: `str` (default: `None`)
            Convenience method for setting the default color map.
        format: {'png', 'pdf', 'svg', etc.} (default: 'pdf')
            This sets the default format for saving figures: `file_format_figs`.
        transparent: `bool` (default: `False`)
            Save figures with transparent back ground. Sets `rcParams['savefig.transparent']`.
        ipython_format : list of `str` (default: 'png2x')
            Only concerns the notebook/IPython environment; see
            `IPython.core.display.set_matplotlib_formats` for more details.
    """

    try:
        import IPython

        if isinstance(ipython_format, str):
            ipython_format = [ipython_format]
        IPython.display.set_matplotlib_formats(*ipython_format)
    except Exception:
        pass

    from matplotlib import rcParams

    global _vector_friendly, file_format_figs

    _vector_friendly = vector_friendly
    file_format_figs = format

    if spateo:
        config_spateo_rcParams(background=background, fontsize=fontsize, color_map=color_map)
    if figsize is not None:
        rcParams["figure.figsize"] = figsize

    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if frameon is not None:
        global _frameon
        _frameon = frameon


def set_pub_style(scaler=1):
    """formatting helper function that can be used to save publishable figures"""
    set_figure_params("spateo", background="white")
    matplotlib.use("cairo")
    matplotlib.rcParams.update({"font.size": 6 * scaler})
    params = {
        "font.size": 6 * scaler,
        "legend.fontsize": 6 * scaler,
        "legend.handlelength": 0.5 * scaler,
        "axes.labelsize": 8 * scaler,
        "axes.titlesize": 8 * scaler,
        "xtick.labelsize": 8 * scaler,
        "ytick.labelsize": 8 * scaler,
        "axes.titlepad": 1 * scaler,
        "axes.labelpad": 1 * scaler,
    }
    matplotlib.rcParams.update(params)


def set_pub_style_mpltex():
    """formatting helper function based on mpltex package that can be used to save publishable figures"""
    set_figure_params("spateo", background="white")
    matplotlib.use("cairo")
    # the following code is adapted from https://github.com/liuyxpp/mpltex
    # latex_preamble = r"\usepackage{siunitx}\sisetup{detect-all}\usepackage{helvet}\usepackage[eulergreek,EULERGREEK]{sansmath}\sansmath"
    params = {
        "font.family": "sans-serif",
        "font.serif": ["Times", "Computer Modern Roman"],
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "sans-serif",
            "Computer Modern Sans serif",
        ],
        "font.size": 6,
        # "text.usetex": True,
        # "text.latex.preamble": latex_preamble,  # To force LaTeX use Helvetica
        # "axes.prop_cycle": default_color_cycler,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 1,
        "figure.subplot.left": 0.125,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.95,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        # "savefig.bbox": "tight",
        # this will crop white spaces around images that will make
        # width/height no longer the same as the specified one.
        "legend.fontsize": 6,
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.handlelength": 0.5,
        "legend.scatterpoints": 1,
        "legend.labelspacing": 0.5,
        "legend.markerscale": 0.9,
        "legend.handletextpad": 0.5,  # pad between handle and text
        "legend.borderaxespad": 0.5,  # pad between legend and axes
        "legend.borderpad": 0.5,  # pad between legend and legend content
        "legend.columnspacing": 1,  # pad between each legend column
        # "text.fontsize" : 4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1,
        "lines.markersize": 6,
        # "lines.markeredgewidth": 0,
        # 0 will make line-type markers, such as "+", "x", invisible
        # Revert some properties to mpl v1 which is more suitable for publishing
        "axes.autolimit_mode": "round_numbers",
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.right": True,
        "axes.titlepad": 1,
        "axes.labelpad": 1,
    }
    matplotlib.rcParams.update(params)
