import matplotlib as mpl
import numpy as np
import open3d as o3d
import pandas as pd
import pyacvd
import pymeshfix as mf
import pyvista as pv
import PVGeo
import seaborn as sns

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista import PolyData, UnstructuredGrid, MultiBlock
from typing import Optional, Tuple, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def mesh_type(
    mesh: Union[PolyData, UnstructuredGrid],
    mtype: Literal["polydata", "unstructured"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """Get a new representation of this mesh as a new type."""
    if mtype == "polydata":
        return mesh if isinstance(mesh, PolyData) else pv.PolyData(mesh.points, mesh.cells)
    elif mtype == "unstructured":
        return mesh.cast_to_unstructured_grid() if isinstance(mesh, PolyData) else mesh
    else:
        raise ValueError("\n`mtype` value is wrong." "\nAvailable `mtype` are: `'polydata'` and `'unstructured'`.")


def construct_pcd(
    adata: AnnData,
    coordsby: str = "spatial",
    mtype: Literal["polydata", "unstructured"] = "polydata",
    coodtype: type = np.float64,
) -> PolyData or UnstructuredGrid:
    """
    Construct a point cloud model based on 3D coordinate information.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        mtype: The type of the reconstructed surface. Available `mtype` are:
                * `'polydata'`
                * `'unstructured'`
        coodtype: Data type of 3D coordinate information.

    Returns:
        A point cloud.
    """

    bucket_xyz = adata.obsm[coordsby].astype(coodtype)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pcd = pv.PolyData(bucket_xyz)

    return mesh_type(mesh=pcd, mtype=mtype)


def voxelize_pcd(
    pcd: PolyData,
    voxel_size: Optional[list] = None,
) -> UnstructuredGrid:
    """
    Voxelize the point cloud.

    Args:
        pcd: A point cloud.
        voxel_size: The size of the voxelized points. A list of three elements.
    Returns:
        A voxelized point cloud.
    """

    voxel_size = [1, 1, 1] if voxel_size is None else voxel_size

    voxelizer = PVGeo.filters.VoxelizePoints()
    voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
    voxelizer.set_estimate_grid(False)

    return voxelizer.apply(pcd)


def construct_surface(
    pcd: PolyData,
    cs_method: Literal["basic", "slide", "alpha_shape", "ball_pivoting", "poisson"] = "basic",
    cs_method_args: dict = None,
    surface_smoothness: int = 100,
    n_surf: int = 10000,
    mtype: Literal["polydata", "unstructured"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pcd: A point cloud.
        cs_method: The methods of creating a surface mesh. Available `cs_method` are:
                * `'basic'`
                * `'slide'`
                * `'alpha_shape'`
                * `'ball_pivoting'`
                * `'poisson'`
        cs_method_args: Parameters for various surface reconstruction methods. Available `cs_method_args` are:
                * `'slide'` method: {"n_slide": 3}
                * `'alpha_shape'` method: {"al_alpha": 10}
                * `'ball_pivoting'` method: {"ba_radii": [1, 1, 1, 1]}
                * `'poisson'` method: {"po_depth": 5, "po_threshold": 0.1}
        surface_smoothness: Adjust surface point coordinates using Laplacian smoothing.
                            If smoothness==0, do not smooth the reconstructed surface.
        n_surf: The number of faces obtained using voronoi clustering. The larger the number, the smoother the surface.
        mtype: The type of the reconstructed surface. Available `mtype` are:
                * `'polydata'`
                * `'unstructured'`

    Returns:
        A surface mesh.
    """

    _cs_method_args = {
        "n_slide": 3,
        "al_alpha": 10,
        "ba_radii": [1, 1, 1, 1],
        "po_depth": 5,
        "po_threshold": 0.1,
    }
    if cs_method_args is not None:
        _cs_method_args.update(cs_method_args)

    # Reconstruct surface mesh.
    if cs_method == "basic":
        surf = pcd.delaunay_3d().extract_surface()

    elif cs_method == "slide":
        n_slide = _cs_method_args["n_slide"]

        z_data = pd.Series(pcd.points[:, 2])
        layers = np.unique(z_data.tolist())
        n_layer_groups = len(layers) - n_slide + 1
        layer_groups = [layers[i : i + n_slide] for i in range(n_layer_groups)]

        points = np.empty(shape=[0, 3])
        for layer_group in layer_groups:
            lg_points = pcd.extract_points(z_data.isin(layer_group))

            lg_grid = lg_points.delaunay_3d().extract_surface()
            lg_grid.subdivide(nsub=2, subfilter="loop", inplace=True)

            points = np.concatenate((points, lg_grid.points), axis=0)

        surf = pv.PolyData(points).delaunay_3d().extract_surface()

    elif cs_method in ["alpha_shape", "ball_pivoting", "poisson"]:
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(pcd.points)

        if cs_method == "alpha_shape":
            alpha = _cs_method_args["al_alpha"]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(_pcd, alpha)

        elif cs_method == "ball_pivoting":
            radii = _cs_method_args["ba_radii"]

            _pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            _pcd.estimate_normals()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                _pcd, o3d.utility.DoubleVector(radii)
            )

        else:
            depth, density_threshold = (
                _cs_method_args["po_depth"],
                _cs_method_args["po_threshold"],
            )

            _pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            _pcd.estimate_normals()

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                (
                    mesh,
                    densities,
                ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(_pcd, depth=depth)
            mesh.remove_vertices_by_mask(np.asarray(densities) < np.quantile(densities, density_threshold))

        _vertices = np.asarray(mesh.vertices)
        _faces = np.asarray(mesh.triangles)
        _faces = np.concatenate((np.ones((_faces.shape[0], 1), dtype=np.int64) * 3, _faces), axis=1)
        surf = pv.PolyData(_vertices, _faces.ravel()).extract_surface()

    else:
        raise ValueError(
            "\n`cs_method` value is wrong."
            "\nAvailable `cs_method` are: `'basic'` , `'slide'` ,`'alpha_shape'`, `'ball_pivoting'`, `'poisson'`."
        )

    # Get an all triangle mesh.
    surf.triangulate(inplace=True)

    # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
    meshfix = mf.MeshFix(surf)
    meshfix.repair(verbose=False)
    surf = meshfix.mesh

    # Smooth the reconstructed surface.
    if surface_smoothness != 0:
        surf.smooth(n_iter=surface_smoothness, inplace=True)
        surf.subdivide_adaptive(max_n_passes=3, inplace=True)

    # Get a uniformly meshed surface using voronoi clustering.
    clustered = pyacvd.Clustering(surf)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh()

    return mesh_type(mesh=uniform_surf, mtype=mtype)


def clip_pcd(adata: AnnData, pcd: PolyData, surface: PolyData, invert: bool = True) -> Tuple[PolyData, AnnData]:
    """Clip the original pcd using the reconstructed surface and reconstruct new point cloud."""
    pcd.point_data["index"] = adata.obs_names.to_numpy()
    clipped_pcd = pcd.clip_surface(surface, invert=invert)
    clipped_adata = adata[clipped_pcd.point_data["index"], :]

    return clipped_pcd, clipped_adata


def construct_volume(
    mesh: Union[PolyData, UnstructuredGrid],
    volume_smoothness: Optional[int] = 200,
) -> UnstructuredGrid:
    """Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh.
        volume_smoothness: The smoothness of the volumetric mesh.

    Returns:
        A volumetric mesh.
    """

    density = mesh.length / volume_smoothness
    volume = pv.voxelize(mesh, density=density, check_surface=False)

    return volume


def three_d_color(
    arr: np.ndarray,
    colormap: Union[str, list, dict] = None,
    alphamap: Union[float, list, dict] = None,
    mask_color: Optional[str] = None,
    mask_alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Set the color of groups or gene expression.
    Args:
        arr: NumPy ndarray.
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Color to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
    Returns:
        The rgba values mapped to groups or gene expression.
    """

    cu_arr = np.unique(arr)
    cu_arr = np.sort(cu_arr, axis=0)
    cu_dict = {}

    # Set mask rgba.
    mask_ind = np.argwhere(cu_arr == "mask")
    if len(mask_ind) != 0:
        cu_arr = np.delete(cu_arr, mask_ind[0])
        cu_dict["mask"] = mpl.colors.to_rgba(mask_color, alpha=mask_alpha)

    # Set alpha.
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {t: alphamap for t in cu_arr}
    elif isinstance(alphamap, list):
        alphamap = {t: alpha for t, alpha in zip(cu_arr, alphamap)}

    # Set rgb.
    if isinstance(colormap, str):
        colormap = [
            mpl.colors.to_hex(i, keep_alpha=False)
            for i in sns.color_palette(palette=colormap, n_colors=len(cu_arr), as_cmap=False)
        ]
    if isinstance(colormap, list):
        colormap = {t: color for t, color in zip(cu_arr, colormap)}

    # Set rgba.
    for t in cu_arr:
        cu_dict[t] = mpl.colors.to_rgba(colormap[t], alpha=alphamap[t])

    return np.array([cu_dict[g] for g in arr])


def construct_three_d_mesh(
    adata: AnnData,
    coordsby: str = "spatial",
    groupby: Optional[str] = None,
    key_added: str = "groups",
    mask: Union[str, int, float, list] = None,
    mesh_style: Literal["pcd", "surf", "volume"] = "volume",
    mesh_color: Optional[str] = "gainsboro",
    mesh_alpha: Optional[float] = 1.0,
    pcd_cmap: Union[str, list, dict] = "rainbow",
    pcd_amap: Union[float, list, dict] = 1.0,
    pcd_voxelize: bool = False,
    pcd_voxel_size: Optional[list] = None,
    cs_method: Literal["basic", "slide", "alpha_shape", "ball_pivoting", "poisson"] = "basic",
    cs_method_args: dict = None,
    surf_smoothness: int = 100,
    n_surf: int = 10000,
    vol_smoothness: Optional[int] = 200,
) -> Tuple[UnstructuredGrid, UnstructuredGrid or None]:
    """
    Reconstruct a voxelized 3D model.
    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        groupby: The key of the observations grouping to consider.
        key_added: The key under which to add the labels.
        mask: The part that you don't want to be displayed.
        mesh_style: The style of the reconstructed mesh. Available `mesh_style` are:
                * `'pcd'`
                * `'surface'`
                * `'volume'`
        mesh_color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        mesh_alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.5`.
        pcd_cmap: Colors to use for plotting pcd. The default pcd_cmap is `'rainbow'`.
        pcd_amap: The opacity of the colors to use for plotting pcd. The default pcd_amap is `1.0`.
        pcd_voxelize: Voxelize the point cloud.
        pcd_voxel_size: The size of the voxelized points. A list of three elements.
        cs_method: The methods of creating a surface mesh. Available `cs_method` are:
                * `'basic'`
                * `'slide'`
                * `'alpha_shape'`
                * `'ball_pivoting'`
                * `'poisson'`
        cs_method_args: Parameters for various surface reconstruction methods. Available `cs_method_args` are:
                * `'slide'` method: {"n_slide": 3}
                * `'alpha_shape'` method: {"al_alpha": 10}
                * `'ball_pivoting'` method: {"ba_radii": [1, 1, 1, 1]}
                * `'poisson'` method: {"po_depth": 5, "po_threshold": 0.1}
        surf_smoothness: Adjust surface point coordinates using Laplacian smoothing.
                         If surf_smoothness==0, do not smooth the reconstructed surface.
        n_surf: The number of faces obtained using voronoi clustering. The larger the n_surf, the smoother the surface.
                Only valid when smoothing is True.
        vol_smoothness: The smoothness of the volumetric mesh.
    Returns:
        pcd: Reconstructed 3D point cloud, which contains the following properties:
            `pcd[key_added]`, the data which under the groupby;
            `pcd[f'{key_added}_rgba']`, the rgba colors of pcd.
        mesh: Reconstructed surface mesh or volumetric mesh, which contains the following properties:
            `mesh[key_added]`, the "mask" array;
            `mesh[f'{key_added}_rgba']`, the rgba colors of mesh.
    """

    # Reconstruct a point cloud, surface or volumetric mesh.
    pcd = construct_pcd(adata=adata, coordsby=coordsby, mtype="polydata", coodtype=np.float64)

    if mesh_style == "pcd":
        surface = None
    else:
        surface = construct_surface(
            pcd=pcd,
            cs_method=cs_method,
            cs_method_args=cs_method_args,
            surface_smoothness=surf_smoothness,
            n_surf=n_surf,
            mtype="polydata",
        )
        clip_invert = True if cs_method in ["basic", "slide"] else False
        pcd, adata = clip_pcd(adata=adata, pcd=pcd, surface=surface, invert=clip_invert)

    mesh = construct_volume(mesh=surface, volume_smoothness=vol_smoothness) if mesh_style == "volume" else surface

    # add `groupby` data
    mask_list = mask if isinstance(mask, list) else [mask]
    if groupby in adata.obs.columns:
        groups = adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values
    elif groupby in adata.var.index:
        groups = adata[:, groupby].X.flatten()
    elif groupby is None:
        groups = np.array(["same"] * adata.obs.shape[0])
    else:
        raise ValueError(
            "\n`groupby` value is wrong." "\n`groupby` should be one of adata.obs.columns, or one of adata.var.index\n"
        )

    # pcd
    pcd = voxelize_pcd(pcd=pcd, voxel_size=pcd_voxel_size) if pcd_voxelize else mesh_type(pcd, mtype="unstructured")
    pcd.cell_data[key_added] = groups
    pcd.cell_data[f"{key_added}_rgba"] = three_d_color(
        arr=groups,
        colormap=pcd_cmap,
        alphamap=pcd_amap,
        mask_color="gainsboro",
        mask_alpha=0,
    ).astype(np.float64)

    # surface mesh or volumetric mesh
    if mesh is not None:
        mesh = mesh_type(mesh, mtype="unstructured")
        mesh.cell_data[key_added] = np.array(["mask"] * mesh.n_cells).astype(str)
        mesh.cell_data[f"{key_added}_rgba"] = np.array(
            [mpl.colors.to_rgba(mesh_color, alpha=mesh_alpha)] * mesh.n_cells
        ).astype(np.float64)

    return pcd, mesh


def merge_mesh(
    meshes: List[PolyData or UnstructuredGrid],
) -> PolyData or UnstructuredGrid:
    """Merge all meshes in the `meshes` list. The format of all meshes must be the same."""

    merged_mesh = meshes[0]
    for mesh in meshes[1:]:
        merged_mesh.merge(mesh, inplace=True)

    return merged_mesh


def collect_mesh(
    meshes: List[PolyData or UnstructuredGrid],
    meshes_name: Optional[List[str]] = None,
) -> MultiBlock:
    """
    A composite class to hold many data sets which can be iterated over.
    You can think of MultiBlock like lists or dictionaries as we can iterate over this data structure by index and we can also access blocks by their string name.

    If the input is a dictionary, it can be iterated in the following ways:
        >>> blocks = collect_mesh(meshes, meshes_name)
        >>> for name in blocks.keys():
        ...     print(blocks[name])

    If the input is a list, it can be iterated in the following ways:
        >>> blocks = collect_mesh(meshes)
        >>> for block in blocks:
        ...    print(block)
    """

    if meshes_name is not None:
        meshes = {name: mesh for mesh, name in zip(meshes, meshes_name)}

    return pv.MultiBlock(meshes)


def mesh_to_ply(
    mesh: Union[PolyData, UnstructuredGrid],
    filename: str,
    binary: bool = True,
    texture: Union[str, np.ndarray] = None,
):
    """
    Save the vtk object to PLY files.
    Args:
        mesh: A reconstructed mesh.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
        binary: If True, write as binary. Otherwise, write as ASCII. Binary files write much faster than ASCII and have a smaller file size.
        texture: Write a single texture array to file when using a PLY file.
                 Texture array must be a 3 or 4 component array with the datatype np.uint8.
                 Array may be a cell array or a point array, and may also be a string if the array already exists in the PolyData.
                 If a string is provided, the texture array will be saved to disk as that name.
                 If an array is provided, the texture array will be saved as 'RGBA'
    """

    if filename.endswith(".ply"):
        mesh.save(filename=filename, binary=binary, texture=texture)
    else:
        raise ValueError(
            "\nFilename is wrong. This function is only available when saving PLY files, "
            "\nplease enter a filename ending with `.ply`."
        )
