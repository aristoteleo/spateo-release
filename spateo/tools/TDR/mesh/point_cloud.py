from typing import Union

import numpy as np
import PVGeo
import pyvista as pv
from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista import PolyData, UnstructuredGrid

from .utils import add_mesh_labels


def construct_pc(
    adata: AnnData,
    spatial_key: str = "spatial",
    groupby: Union[str, list] = None,
    key_added: str = "groups",
    mask: Union[str, int, float, list] = None,
    colormap: Union[str, list, dict] = "rainbow",
    alphamap: Union[float, list, dict] = 1.0,
) -> PolyData:
    """
    Construct a point cloud model based on 3D coordinate information.

    Args:
        adata: AnnData object.
        spatial_key: The key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        groupby: The key that stores clustering or annotation information in adata.obs,
                 a gene's name or a list of genes' name in adata.var.
        key_added: The key under which to add the labels.
        mask: The part that you don't want to be displayed.
        colormap: Colors to use for plotting pcd. The default colormap is `'rainbow'`.
        alphamap: The opacity of the colors to use for plotting pcd. The default alphamap is `1.0`.

    Returns:
        pc: A point cloud, which contains the following properties:
            `pc.point_data[key_added]`, the `groupby` information.
            `pc.point_data[f'{key_added}_rgba']`, the rgba colors of the `groupby` information.
            `pc.point_data["obs_index"]`, the obs_index of each coordinate in the original adata.
    """

    # create an initial pc.
    bucket_xyz = adata.obsm[spatial_key].astype(np.float64)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pc = pv.PolyData(bucket_xyz)

    # The`groupby` array in original adata.obs or adata.X
    mask_list = mask if isinstance(mask, list) else [mask]

    obs_names = set(adata.obs_keys())
    gene_names = set(adata.var_names.tolist())
    if groupby in obs_names:
        groups = adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values
    elif groupby in gene_names or set(groupby) <= gene_names:
        groups = adata[:, groupby].X.sum(axis=1).flatten().round(2)
    elif groupby is None:
        groups = np.array(["same"] * adata.obs.shape[0])
    else:
        raise ValueError(
            "\n`groupby` value is wrong."
            "\n`groupby` can be a string and one of adata.obs_names or adata.var_names. "
            "\n`groupby` can also be a list and is a subset of adata.var_names"
        )

    add_mesh_labels(
        mesh=pc,
        labels=groups,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
    )

    # The obs_index of each coordinate in the original adata.
    pc.point_data["obs_index"] = np.array(adata.obs_names.tolist())

    return pc


def voxelize_pc(
    pc: PolyData,
    voxel_size: Union[list, tuple] = None,
) -> UnstructuredGrid:
    """
    Voxelize the point cloud.

    Args:
        pc: A point cloud.
        voxel_size: The size of the voxelized points. A list of three elements.
    Returns:
        voxel: A voxelized point cloud.
    """
    voxelizer = PVGeo.filters.VoxelizePoints()

    if not (voxel_size is None):
        voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
        voxelizer.set_estimate_grid(False)

    voxel = voxelizer.apply(pc)

    # add labels
    for key in pc.point_data.keys():
        voxel.cell_data[key] = pc.point_data[key]

    return voxel
