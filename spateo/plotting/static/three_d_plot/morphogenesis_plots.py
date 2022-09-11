from typing import Optional, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from ....tdr import add_model_labels
from .three_dims_plots import three_d_multi_plot

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _get_default_cmap():
    if "default_cmap" not in mpl.colormaps():
        colors = ["#4B0082", "#800080", "#F97306", "#FFA500", "#FFD700", "#FFFFCB"]
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        mpl.colormaps.register(LinearSegmentedColormap.from_list("default_cmap", list(zip(nodes, colors))))
    return "default_cmap"


def _check_index_in_adata(adata, model):
    adata_obs_index = pd.DataFrame(range(len(adata.obs.index)), index=adata.obs.index, columns=["ind"])
    obs_index = (
        np.asarray(model.point_data["obs_index"]) if "obs_index" in model.point_data else np.asarray(adata.obs.index)
    )
    obs_index_ind = adata_obs_index.loc[obs_index, "ind"].values
    return obs_index_ind


def _check_key_in_adata(adata: AnnData, key: str, where: str):
    if where == "obs":
        assert key in adata.obs_keys(), f"``{key}`` does not exist in adata.obs."
        return adata.obs[key]
    elif where == "obsm":
        assert key in adata.obsm_keys(), f"``{key}`` does not exist in adata.obsm."
        return adata.obsm[key]
    elif where == "obsp":
        assert key in adata.obsp, f"``{key}`` does not exist in adata.layers."
        return adata.obsp[key]
    elif where == "var":
        assert key in adata.var_keys(), f"``{key}`` does not exist in adata.var."
        return adata.var[key]
    elif where == "varm":
        assert key in adata.varm_keys(), f"``{key}`` does not exist in adata.varm."
        return adata.varm[key]
    elif where == "varp":
        assert key in adata.varp, f"``{key}`` does not exist in adata.varp."
        return adata.varp[key]
    elif where == "uns":
        assert key in adata.uns_keys(), f"``{key}`` does not exist in adata.uns."
        return adata.uns[key]
    elif where == "layers":
        assert key in adata.layers, f"``{key}`` does not exist in adata.layers."
        return adata.layers[key]
    else:
        raise ValueError("``where`` value is error.")


def jacobian(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    jacobian_key: str = "jacobian",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    shape: Union[str, list, tuple] = (3, 3),
    window_size: Optional[tuple] = (512 * 3, 512 * 3),
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    adata, model = adata.copy(), model.copy()

    obs_index_ind = _check_index_in_adata(adata=adata, model=model)
    jacobian_martix = _check_key_in_adata(adata=adata, key=jacobian_key, where="uns")
    jacobian_martix = jacobian_martix[:, :, obs_index_ind]

    # Add values in the jacobian matrix to the model separately.
    j_keys = []
    for f_i, f in enumerate(["fx", "fy", "fz"]):
        for i_i, i in enumerate(["x", "y", "z"]):
            sub_key = f"∂{f}/∂{i}"
            sub_matrix = jacobian_martix[f_i, i_i, :]
            add_model_labels(model=model, labels=sub_matrix, key_added=sub_key, where="point_data", inplace=True)
            j_keys.append(sub_key)

    # Visualization.
    colormap = _get_default_cmap() if colormap is None or colormap == "default_cmap" else colormap
    three_d_multi_plot(
        model=model,
        key=j_keys,
        filename=filename,
        jupyter=jupyter,
        off_screen=off_screen,
        shape=shape,
        window_size=window_size,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=j_keys if text is True else text,
        text_kwargs=text_kwargs,
        **kwargs,
    )
