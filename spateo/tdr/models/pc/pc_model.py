from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import add_model_labels

###############################
# Construct point cloud model #
###############################


def construct_pc(
    adata: AnnData,
    layer: str = "X",
    spatial_key: str = "spatial",
    groupby: Union[str, tuple] = None,
    key_added: str = "groups",
    mask: Union[str, int, float, list] = None,
    colormap: Union[str, list, dict] = "rainbow",
    alphamap: Union[float, list, dict] = 1.0,
) -> Tuple[PolyData, Optional[str]]:
    """
    Construct a point cloud model based on 3D coordinate information.

    Args:
        adata: AnnData object.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        groupby: The key that stores clustering or annotation information in ``.obs``,
                 a gene name or a list of gene names in ``.var``.
        key_added: The key under which to add the labels.
        mask: The part that you don't want to be displayed.
        colormap: Colors to use for plotting pc. The default colormap is ``'rainbow'``.
        alphamap: The opacity of the colors to use for plotting pc. The default alphamap is ``1.0``.

    Returns:
        pc: A point cloud, which contains the following properties:
            ``pc.point_data[key_added]``, the ``groupby`` information.
            ``pc.point_data[f'{key_added}_rgba']``, the rgba colors of the ``groupby`` information.
            ``pc.point_data['obs_index']``, the obs_index of each coordinate in the original adata.
        plot_cmap: Recommended colormap parameter values for plotting.
    """

    # create an initial pc.
    adata = adata.copy()
    bucket_xyz = adata.obsm[spatial_key].astype(np.float64)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pc = pv.PolyData(bucket_xyz)

    # The`groupby` array in original adata.obs or adata.X.
    mask_list = mask if isinstance(mask, list) else [mask]

    obs_names = set(adata.obs_keys())
    gene_names = set(adata.var_names.tolist())
    if groupby is None:
        groups = np.asarray(["same"] * adata.obs.shape[0], dtype=str)
    elif groupby in obs_names:
        groups = np.asarray(adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values)
    elif groupby in gene_names or set(groupby) <= gene_names:
        adata.X = adata.X if layer == "X" else adata.layers[layer]
        groups = np.asarray(adata[:, groupby].X.sum(axis=1).flatten())
    else:
        raise ValueError(
            "`groupby` value is wrong."
            "\n`groupby` can be a string and one of adata.obs_names or adata.var_names."
            "\n`groupby` can also be a list and is a subset of adata.var_names."
        )

    _, plot_cmap = add_model_labels(
        model=pc,
        labels=groups,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
        inplace=True,
    )

    # The obs_index of each coordinate in the original adata.
    pc.point_data["obs_index"] = np.array(adata.obs_names.tolist())

    return pc, plot_cmap
