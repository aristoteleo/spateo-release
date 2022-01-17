# code adapted from https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/configuration.py

import colorcet
import matplotlib
from matplotlib import rcParams, colors

# from cycler import cycler
import matplotlib.pyplot as plt

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

# register cmap
plt.register_cmap("fire", fire_cmap)
plt.register_cmap("darkblue", darkblue_cmap)
plt.register_cmap("darkgreen", darkgreen_cmap)
plt.register_cmap("darkred", darkred_cmap)
plt.register_cmap("darkpurple", darkpurple_cmap)
plt.register_cmap("div_blue_black_red", div_blue_black_red_cmap)
plt.register_cmap("div_blue_red", div_blue_red_cmap)
plt.register_cmap("glasbey_white", glasbey_white_cmap)
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
