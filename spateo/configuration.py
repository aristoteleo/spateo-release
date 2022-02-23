# code adapted from https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/configuration.py

from typing import Optional, Union

import colorcet
import matplotlib
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import rcParams, colors
from scipy import sparse

# from cycler import cycler
import matplotlib.pyplot as plt


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

    SPLICED_LAYER_KEY = "spliced"
    UNSPLICED_LAYER_KEY = "unspliced"
    STAIN_LAYER_KEY = "stain"
    MASK_SUFFIX = "mask"
    MARKERS_SUFFIX = "markers"
    BINS_SUFFIX = "bins"
    LABELS_SUFFIX = "labels"
    SCORES_SUFFIX = "scores"
    EXPANDED_SUFFIX = "expanded"

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

    def init_adata_type(adata: AnnData, t: Optional[str] = None):
        if t is None:
            t = SpateoAdataKeyManager.ADATA_DEFAULT_TYPE
        adata.uns[SpateoAdataKeyManager.ADATA_TYPE_KEY] = t

    def init_uns_pp_namespace(adata: AnnData):
        if SpateoAdataKeyManager.UNS_PP_KEY not in adata.uns:
            adata.uns[SpateoAdataKeyManager.UNS_PP_KEY] = {}

    def init_uns_spatial_namespace(adata: AnnData):
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


SKM = SpateoAdataKeyManager

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
