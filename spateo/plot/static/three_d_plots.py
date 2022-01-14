import re
import math
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import pyvista as pv
import PVGeo
import matplotlib as mpl
import seaborn as sns
from anndata import AnnData
from typing import Union, Optional, Sequence, List


def clip_3d_coords(adata: AnnData,
                   coordsby: Optional[List] = None
                   ):

    if coordsby is None:
        coordsby = ["x", "y", "z"]
    points_data = adata.obs[coordsby]
    points_arr = points_data.values.astype(float)
    grid = pv.PolyData(points_arr)
    grid["index"] = adata.obs_names.tolist()
    # Clip mesh using a pyvista.PolyData surface mesh.
    surf = grid.delaunay_3d().extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clipped_grid = grid.clip_surface(surf)
    clipped_adata = adata[adata.obs.index.isin(clipped_grid["index"].tolist()), :]

    clipped_points = pd.DataFrame()
    clipped_points[0] = list(map(tuple, clipped_grid.points))
    surf_points = list(map(tuple, surf.points.round(1)))
    clipped_points[1] = clipped_points[0].isin(surf_points)
    clipped_adata = clipped_adata[~clipped_points[1].values, :]

    return clipped_adata, surf


def groups_color(groups,
                 colormap: Union[str, list, dict] = "viridis",
                 alphamap: Union[float, list, dict] = 1.0,
                 mask_color: Optional[str] = "whitesmoke",
                 mask_alpha: Optional[float] = 0.5
                 ):

    color_groups = groups.unique().tolist()
    color_groups.sort()
    colordict = {}
    if "mask" in color_groups:
        color_groups.remove("mask")
        rgb_color = mpl.colors.to_rgb(mask_color)
        colordict["mask"] = [rgb_color[0], rgb_color[1], rgb_color[2], mask_alpha]

    # Set group color
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {group: alphamap for group in color_groups}
    elif isinstance(alphamap, list):
        alphamap = {group: alpha for group, alpha in zip(color_groups, alphamap)}
    if isinstance(colormap, str):
        colormap = [mpl.colors.to_hex(i, keep_alpha=False) for i in
                    sns.color_palette(palette=colormap, n_colors=len(color_groups), as_cmap=False)]
    if isinstance(colormap, list):
        colormap = {group: color for group, color in zip(color_groups, colormap)}
    for group in color_groups:
        rgb_color = mpl.colors.to_rgb(colormap[group])
        colordict[group] = [rgb_color[0], rgb_color[1], rgb_color[2], alphamap[group]]

    return colordict


def genes_color(genes_exp,
                colormap: Union[str, list, dict] = "hot_r",
                alphamap: Union[float, list, dict] = 1.0,
                mask_color: Optional[str] = "whitesmoke",
                mask_alpha: Optional[float] = 0.5):


    color_genes = genes_exp.unique().tolist()
    color_genes.sort()
    colordict = {}

    # Set gene color
    if isinstance(colormap, str):
        colormap = [mpl.colors.to_hex(i, keep_alpha=False) for i in
                    sns.color_palette(palette=colormap, n_colors=len(color_genes), as_cmap=False)]
    if isinstance(colormap, list):
        colormap = {group: color for group, color in zip(color_genes, colormap)}
    for gene in color_genes:
        rgb_color = mpl.colors.to_rgb(colormap[gene])
        colordict[gene] = [rgb_color[0], rgb_color[1], rgb_color[2], alphamap]
    mask_rgb_color = mpl.colors.to_rgb(mask_color)
    colordict[float(0)] = [mask_rgb_color[0], mask_rgb_color[1], mask_rgb_color[2], mask_alpha]
    return colordict


def build_3Dmodel(adata: AnnData,
                  coordsby: Optional[list] = None,
                  groupby: Optional[str] = "cluster",
                  group_show: Union[str, list] = "all",
                  group_cmap: Union[str, list, dict] = "viridis",
                  group_amap: Union[float, list, dict] = 1.0,
                  gene_show: Union[str, list] = "all",
                  gene_cmap: Union[str, list, dict] = "hot_r",
                  gene_amap: Union[float, list, dict] = 1.0,
                  mask_color: Optional[str] = "gainsboro",
                  mask_alpha: Optional[float] = 0.5,
                  smoothing: Optional[bool] = True,
                  voxelize: Optional[bool] = False,
                  voxel_size: Optional[list] = None,
                  unstructure: Optional[bool] = False
                  ):

    # Clip mesh using a pyvista.PolyData surface mesh.
    if smoothing:
        _adata, _ = clip_3d_coords(adata=adata, coordsby=coordsby)
    else:
        _adata = adata

    # filter group info
    if isinstance(group_show, str) and group_show is "all":
        groups = _adata.obs[groupby]
    elif isinstance(group_show, str) and group_show is not "all":
        groups = _adata.obs[groupby].map(lambda x: str(x) if x == group_show else "mask")
    elif isinstance(group_show, list) or isinstance(group_show, tuple):
        groups = _adata.obs[groupby].map(lambda x: str(x) if x in group_show else "mask")
    else:
        raise ValueError("`group_show` value is wrong.")
    # Set group color(rgba)
    groups_rgba = groups_color(groups=groups, colormap=group_cmap, alphamap=group_amap,
                               mask_color=mask_color, mask_alpha=mask_alpha)

    # filter gene info
    genes_exp = _adata.X.sum(axis=1) if gene_show == "all" else _adata[:, gene_show].X.sum(axis=1)
    genes_data = pd.DataFrame([groups.values.tolist(), genes_exp.tolist()]).stack().unstack(0)
    genes_data.columns = ["group", "genes_exp"]
    genes_data["filter"] = genes_data[["group", "genes_exp"]].apply(
        lambda x: 0 if x["group"] is "mask" else x["genes_exp"], axis=1).astype(float)
    new_genes_exp = genes_data["filter"].round(5)
    # Set gene color(rgba)
    genes_rgba = genes_color(genes_exp=new_genes_exp, colormap=gene_cmap, alphamap=gene_amap,
                             mask_color=mask_color, mask_alpha=mask_alpha)

    # Create a point cloud(pyvista.PolyData) or a voxelized volume(pyvista.UnstructuredGrid).
    if coordsby is None:
        coordsby = ["x", "y", "z"]
    points_data = _adata.obs[coordsby]
    points_data = points_data.astype(float)
    points = PVGeo.points_to_poly_data(points_data)
    surface = points.delaunay_3d().extract_geometry()

    if voxelize is True:
        voxelizer = PVGeo.filters.VoxelizePoints()
        voxel_size = [1, 1, 1] if voxel_size is None else voxel_size
        voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
        voxelizer.set_estimate_grid(False)
        mesh = voxelizer.apply(points)
    else:
        mesh = points.cast_to_unstructured_grid() if unstructure else points

    mesh["points_coords"] = points_data
    mesh["genes_exp"] = new_genes_exp.values
    mesh["genes_rgba"] = np.array([genes_rgba[g] for g in new_genes_exp.tolist()])
    mesh["groups"] = groups.values
    mesh["groups_rgba"] = np.array([groups_rgba[g] for g in groups.tolist()])

    return mesh, surface


def threeDslicing(mesh,
                  axis: Union[str, int] = "x",
                  n_slices: Union[str, int] = 10,
                  center: Optional[Sequence[float]] = None):

    if isinstance(mesh, pv.core.pointset.UnstructuredGrid) is False:
        warnings.warn("The model should be a pyvista.UnstructuredGrid (voxelized) object.")
        mesh = mesh.cast_to_unstructured_grid()

    if n_slices is "orthogonal":
        # Create three orthogonal slices through the dataset on the three cartesian planes.
        if center is None:
            return mesh.slice_orthogonal(x=None, y=None, z=None)
        else:
            return mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])
    elif n_slices == 1:
        # Slice a dataset by a plane at the specified origin and normal vector orientation.
        return mesh.slice(normal=axis, origin=center, contour=True)
    else:
        # Create many slices of the input dataset along a specified axis.
        return mesh.slice_along_axis(n=n_slices, axis=axis, center=center)