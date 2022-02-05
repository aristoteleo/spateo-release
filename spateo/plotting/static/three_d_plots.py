import math
import re
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
import PVGeo
import pyacvd
import pyvista as pv
import seaborn as sns

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista.core.pointset import PolyData, UnstructuredGrid
from typing import Optional, Sequence, Tuple, Union


def smoothing_mesh(
    adata: AnnData,
    coordsby: str = "spatial",
    n_surf: int = 10000,
) -> Tuple[AnnData, PolyData]:
    """
    Takes a uniformly meshed surface using voronoi clustering and clip the original mesh using the reconstructed surface.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        n_surf: The number of faces obtained using voronoi clustering. The larger the number, the smoother the surface.

    Returns:
        clipped_adata: AnnData object that is clipped.
        uniform_surf: A uniformly meshed surface.
    """

    float_type = np.float64

    bucket_xyz = adata.obsm[coordsby].astype(float_type)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    grid = pv.PolyData(bucket_xyz)
    grid["index"] = adata.obs_names.to_numpy()

    # takes a surface mesh and returns a uniformly meshed surface using voronoi clustering.
    surf = grid.delaunay_3d().extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clustered = pyacvd.Clustering(surf)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh()

    # Clip the original mesh using the reconstructed surface.
    clipped_grid = grid.clip_surface(uniform_surf)
    clipped_adata = adata[clipped_grid["index"], :]

    clipped_points = pd.DataFrame()
    clipped_points[0] = list(map(tuple, clipped_grid.points.round(5)))
    surf_points = list(map(tuple, uniform_surf.points.round(5)))
    clipped_points[1] = clipped_points[0].isin(surf_points)
    clipped_adata = clipped_adata[~clipped_points[1].values, :]

    return clipped_adata, uniform_surf


def three_d_color(
    series,
    colormap: Union[str, list, dict] = None,
    alphamap: Union[float, list, dict] = None,
    mask_color: Optional[str] = None,
    mask_alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Set the color of groups or gene expression.
    Args:
        series: Pandas sereis (e.g. cell groups or gene names).
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Colors to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
    Returns:
        rgba: The rgba values mapped to groups or gene expression.
    """

    color_types = series.unique().tolist()
    colordict = {}

    # set mask rgba
    if "mask" in color_types:
        color_types.remove("mask")
        colordict["mask"] = mpl.colors.to_rgba(mask_color, alpha=mask_alpha)
    color_types.sort()

    # set alpha
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {t: alphamap for t in color_types}
    elif isinstance(alphamap, list):
        alphamap = {t: alpha for t, alpha in zip(color_types, alphamap)}

    # set rgb
    if isinstance(colormap, str):
        colormap = [
            mpl.colors.to_hex(i, keep_alpha=False)
            for i in sns.color_palette(palette=colormap, n_colors=len(color_types), as_cmap=False)
        ]
    if isinstance(colormap, list):
        colormap = {t: color for t, color in zip(color_types, colormap)}

    # set rgba
    for t in color_types:
        colordict[t] = mpl.colors.to_rgba(colormap[t], alpha=alphamap[t])
    rgba = np.array([colordict[g] for g in series.tolist()])

    return rgba


def build_three_d_model(
    adata: AnnData,
    coordsby: str = "spatial",
    groupby: Optional[str] = None,
    group_show: Union[str, list] = "all",
    group_cmap: Union[str, list, dict] = "rainbow",
    group_amap: Union[float, list, dict] = 1.0,
    gene_show: Union[str, list] = "all",
    gene_cmap: str = "hot_r",
    gene_amap: float = 1.0,
    mask_color: str = "gainsboro",
    mask_alpha: float = 0,
    surf_alpha: float = 0.5,
    smoothing: bool = True,
    n_surf: int = 10000,
    voxelize: bool = True,
    voxel_size: Optional[list] = None,
    voxel_smooth: Optional[int] = 200,
) -> UnstructuredGrid:
    """
    Reconstruct a voxelized 3D model.
    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        groupby: The key of the observations grouping to consider.
        group_show: Subset of groups used for display, e.g. [`'g1'`, `'g2'`, `'g3'`]. The default group_show is `'all'`, for all groups.
        group_cmap: Colors to use for plotting groups. The default group_cmap is `'rainbow'`.
        group_amap: The opacity of the colors to use for plotting groups. The default group_amap is `1.0`.
        gene_show: Subset of genes used for display, e.g. [`'g1'`, `'g2'`, `'g3'`]. The default gene_show is `'all'`, for all groups.
        gene_cmap: Colors to use for plotting genes. The default gene_cmap is `'hot_r'`.
        gene_amap: The opacity of the colors to use for plotting genes. The default gene_amap is `1.0`.
        mask_color: Color to use for plotting mask. The default mask_color is `'gainsboro'`.
        mask_alpha: The opacity of the color to use for plotting mask. The default mask_alpha is `0.0`.
        surf_alpha: The opacity of the color to use for surface. The default mask_alpha is `0.5`.
        smoothing: Smoothing the surface of the reconstructed 3D structure.
        n_surf: The number of faces obtained using voronoi clustering. The larger the n_surf, the smoother the surface. Only valid when smoothing is True.
        voxelize: Voxelize the reconstructed 3D structure.
        voxel_size: The size of the voxelized points. A list of three elements.
        voxel_smooth: The smoothness of the voxelized surface. Only valid when voxelize is True.
    Returns:
        mesh: Reconstructed 3D structure, which contains the following properties:
            groups: `mesh['groups']`, the mask and the groups used for display.
            genes_exp: `mesh['genes']`, the gene expression.
            groups_rgba: `mesh['groups_rgba']`, the rgba colors for plotting groups and mask.
            genes_rgba: `mesh['genes_rgba']`, the rgba colors for plotting genes and mask.
    """

    float_type = np.float64

    # takes a uniformly meshed surface and clip the original mesh using the reconstructed surface if smoothing is True.
    _adata, uniform_surf = smoothing_mesh(adata=adata, coordsby=coordsby, n_surf=n_surf) if smoothing else (adata, None)

    # filter group info
    if groupby is None:
        n_points = _adata.obs.shape[0]
        groups = pd.Series(["same"] * n_points, index=_adata.obs.index, dtype=str)
    else:
        if isinstance(group_show, str) and group_show is "all":
            groups = _adata.obs[groupby]
        elif isinstance(group_show, str) and group_show is not "all":
            groups = _adata.obs[groupby].map(lambda x: str(x) if x == group_show else "mask")
        elif isinstance(group_show, list) or isinstance(group_show, tuple):
            groups = _adata.obs[groupby].map(lambda x: str(x) if x in group_show else "mask")
        else:
            raise ValueError("`group_show` value is wrong.")

    # filter gene expression info
    genes_exp = _adata.X.sum(axis=1) if gene_show == "all" else _adata[:, gene_show].X.sum(axis=1)
    genes_exp = pd.DataFrame(genes_exp, index=groups.index, dtype=float_type)
    genes_data = pd.concat([groups, genes_exp], axis=1)
    genes_data.columns = ["groups", "genes_exp"]
    new_genes_exp = (
        genes_data[["groups", "genes_exp"]]
        .apply(lambda x: 0 if x["groups"] is "mask" else round(x["genes_exp"], 2), axis=1)
        .astype(float_type)
    )

    # Create a point cloud(Unstructured) and its surface.
    bucket_xyz = _adata.obsm[coordsby].astype(float_type)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    points = pv.PolyData(bucket_xyz).cast_to_unstructured_grid()
    surface = points.delaunay_3d().extract_geometry() if uniform_surf is None else uniform_surf

    # Voxelize the cloud and the surface
    if voxelize:
        voxelizer = PVGeo.filters.VoxelizePoints()
        voxel_size = [1, 1, 1] if voxel_size is None else voxel_size
        voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
        voxelizer.set_estimate_grid(False)
        points = voxelizer.apply(points)

        density = surface.length / voxel_smooth
        surface = pv.voxelize(surface, density=density, check_surface=False)

    # Add some properties of the 3D model
    points.cell_data["groups"] = groups.astype(str).values
    points.cell_data["groups_rgba"] = three_d_color(
        series=groups,
        colormap=group_cmap,
        alphamap=group_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(float_type)

    points.cell_data["genes"] = new_genes_exp.values
    points.cell_data["genes_rgba"] = three_d_color(
        series=new_genes_exp,
        colormap=gene_cmap,
        alphamap=gene_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(float_type)

    surface.cell_data["groups"] = np.array(["mask"] * surface.n_cells).astype(str)
    surface.cell_data["groups_rgba"] = np.array(
        [mpl.colors.to_rgba(mask_color, alpha=surf_alpha)] * surface.n_cells
    ).astype(float_type)

    surface.cell_data["genes"] = np.array([0] * surface.n_cells).astype(float_type)
    surface.cell_data["genes_rgba"] = np.array(
        [mpl.colors.to_rgba(mask_color, alpha=surf_alpha)] * surface.n_cells
    ).astype(float_type)

    # Merge points and surface into a single mesh.
    mesh = surface.merge(points)

    return mesh


def three_d_slicing(
    mesh: UnstructuredGrid,
    scalar: str = "groups",
    axis: Union[str, int] = "x",
    n_slices: Union[str, int] = 10,
    center: Optional[Sequence[float]] = None,
) -> PolyData:
    """
    Create many slices of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.
    Args:
        mesh: Reconstructed 3D structure (voxelized object).
        scalar: Types used to “color” the mesh. Available scalars are:
                * `'groups'`
                * `'genes'`
        axis: The axis to generate the slices along. Available axes are:
                * `'x'` or `0`
                * `'y'` or `1`
                * `'z'` or `2`
        n_slices: The number of slices to create along a specified axis.
                  If n_slices is `"orthogonal"`, create three orthogonal slices.
        center: A 3-length sequence specifying the position which slices are taken. Defaults to the center of the mesh.
    Returns:
        Sliced dataset.
    """

    if isinstance(mesh, pv.core.pointset.UnstructuredGrid) is False:
        warnings.warn("The model should be a pyvista.UnstructuredGrid (voxelized) object.")
        mesh = mesh.cast_to_unstructured_grid()

    mesh.set_active_scalars(f"{scalar}_rgba")

    if n_slices is "orthogonal":
        # Create three orthogonal slices through the dataset on the three cartesian planes.
        if center is None:
            return mesh.slice_orthogonal(x=None, y=None, z=None)
        else:
            return mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])
    elif n_slices == 1:
        # Slice a dataset by a plane at the specified origin and normal vector orientation.
        return mesh.slice(normal=axis, origin=center)
    else:
        # Create many slices of the input dataset along a specified axis.
        return mesh.slice_along_axis(n=n_slices, axis=axis, center=center)


def easy_three_d_plot(
    mesh: Optional[pv.DataSet] = None,
    scalar: str = "groups",
    outline: bool = False,
    ambient: float = 0.3,
    opacity: float = 0.5,
    background: str = "black",
    background_r: str = "white",
    save: Optional[str] = None,
    notebook: bool = False,
    shape: Optional[list] = None,
    off_screen: bool = False,
    window_size: Optional[list] = None,
    cpos: Union[str, tuple, list] = "iso",
    legend_loc: str = "lower right",
    legend_size: Optional[Sequence] = None,
    view_up: Optional[list] = None,
    framerate: int = 15,
):
    """
    Create a plotting object to display pyvista/vtk mesh.
    Args:
        mesh: Reconstructed 3D structure.
        scalar: Types used to “color” the mesh. Available scalars are:
                * `'groups'`
                * `'genes'`
        outline: Produce an outline of the full extent for the input dataset.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        background: The background color of the window.
        background_r: A color that is clearly different from the background color.
        save: If a str, save the figure. Infer the file type if ending on
             {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".gif", ".mp4"}.
        notebook: When True, the resulting plot is placed inline a jupyter notebook.
        shape: Number of sub-render windows inside of the main window. Available shape formats are:
                * `shape=(2, 1)`: 2 plots in two rows and one column
                * `shape='3|1'`: 3 plots on the left and 1 on the right,
                * `shape='4/2'`: 4 plots on top and 2 at the bottom.
        off_screen: Renders off screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        cpos: Camera position of the window. Available cpos are:
                * `'xy'`, `'xz'`, `'yz'`, `'yx'`, `'zx'`, `'zy'`, `'iso'`
                * Customize a tuple. E.g.: (7, 0, 20.).
        legend_loc: The location of the legend in the window. Available legend_loc are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
        legend_size: The size of the legend in the window. Two float sequence, each float between 0 and 1.
                     E.g.: (0.1, 0.1) would make the legend 10% the size of the entire figure window.
        view_up: The normal to the orbital plane. The default view_up is `[0.5, 0.5, 1]`.
        framerate: Frames per second.
    """

    if shape is None:
        shape = (1, 1)

    if isinstance(shape, str):
        n = re.split("[|/]", shape)
        subplot_indices = [i for i in range(int(n[0]) + int(n[1]))]
    else:
        subplot_indices = [[i, j] for i in range(shape[0]) for j in range(shape[1])]

    if window_size is None:
        window_size = (1024, 768)

    if type(cpos) in [str, tuple]:
        cpos = [cpos]

    if len(cpos) != len(subplot_indices):
        raise ValueError("The number of cpos does not match the number of subplots drawn.")

    # Create a plotting object to display pyvista/vtk mesh.
    p = pv.Plotter(
        shape=shape,
        off_screen=off_screen,
        lighting="light_kit",
        window_size=window_size,
        notebook=notebook,
        border=True,
        border_color=background_r,
    )
    for subplot_index, cpo in zip(subplot_indices, cpos):

        # Add a reconstructed 3D structure.
        p.add_mesh(
            mesh,
            scalars=f"{scalar}_rgba",
            rgba=True,
            render_points_as_spheres=True,
            ambient=ambient,
            opacity=opacity,
        )

        # Add a legend to render window.
        mesh[f"{scalar}_hex"] = np.array([mpl.colors.to_hex(i) for i in mesh[f"{scalar}_rgba"]])
        _data = pd.concat([pd.Series(mesh[scalar]), pd.Series(mesh[f"{scalar}_hex"])], axis=1)
        _data.columns = ["label", "hex"]
        _data = _data[_data["label"] != "mask"]
        _data.drop_duplicates(inplace=True)
        _data.sort_values(by=["label", "hex"], inplace=True)
        _data = _data.astype(str)

        gap = math.ceil(len(_data.index) / 5) if scalar is "genes" else 1
        legend_entries = [[_data["label"].iloc[i], _data["hex"].iloc[i]] for i in range(0, len(_data.index), gap)]
        if scalar is "genes":
            legend_entries.append([_data["label"].iloc[-1], _data["hex"].iloc[-1]])

        legend_size = (0.1, 0.1) if legend_size is None else legend_size

        p.add_legend(
            legend_entries,
            face="circle",
            bcolor=None,
            loc=legend_loc,
            size=legend_size,
        )

        if outline:
            p.add_mesh(mesh.outline(), color=background_r, line_width=3)

        p.camera_position = cpo
        p.background_color = background
        p.show_axes()

    # Save as image or gif or mp4
    save = "three_d_structure.jpg" if save is None else save
    save_format = save.split(".")[-1]
    if save_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
        p.show(screenshot=save)
    else:
        view_up = [0.5, 0.5, 1] if view_up is None else view_up
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
        if save.endswith(".gif"):
            p.open_gif(save)
        elif save.endswith(".mp4"):
            p.open_movie(save, framerate=framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()
