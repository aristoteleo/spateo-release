import matplotlib as mpl
import numpy as np
import open3d as o3d
import pandas as pd
import pyacvd
import pymeshfix as mf
import pyvista as pv
import PVGeo

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista import PolyData, UnstructuredGrid, MultiBlock, DataSet
from typing import Optional, Tuple, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def mesh_type(
    mesh: Union[PolyData, UnstructuredGrid],
    mtype: Literal["polydata", "unstructuredgrid"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """Get a new representation of this mesh as a new type."""
    if mtype == "polydata":
        return mesh if isinstance(mesh, PolyData) else pv.PolyData(mesh.points, mesh.cells)
    elif mtype == "unstructured":
        return mesh.cast_to_unstructured_grid() if isinstance(mesh, PolyData) else mesh
    else:
        raise ValueError("\n`mtype` value is wrong." "\nAvailable `mtype` are: `'polydata'` and `'unstructuredgrid'`.")


def construct_pcd(
    adata: AnnData,
    coordsby: str = "spatial",
    groupby: Union[str, list] = None,
    key_added: str = "groups",
    mask: Union[str, int, float, list] = None,
    colormap: Union[str, list, dict] = "rainbow",
    alphamap: Union[float, list, dict] = 1.0,
    coodtype: type = np.float64,
) -> PolyData:
    """
    Construct a point cloud model based on 3D coordinate information.

    Args:
        adata: AnnData object.
        coordsby: The key that stores 3D coordinate information in adata.obsm.
        groupby: The key that stores clustering or annotation information in adata.obs, a gene's name or a list of genes' name in adata.var.
        key_added: The key under which to add the labels.
        mask: The part that you don't want to be displayed.
        colormap: Colors to use for plotting pcd. The default pcd_cmap is `'rainbow'`.
        alphamap: The opacity of the colors to use for plotting pcd. The default pcd_amap is `1.0`.
        coodtype: Data type of 3D coordinate information.

    Returns:
        pcd: A point cloud, which contains the following properties:
            `pcd.point_data[key_added]`, the `groupby` information.
            `pcd.point_data[f'{key_added}_rgba']`, the rgba colors of the `groupby` information.
            `pcd.point_data["obs_index"]`, the obs_index of each coordinate in the original adata.
    """

    # create an initial pcd.
    bucket_xyz = adata.obsm[coordsby].astype(coodtype)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pcd = pv.PolyData(bucket_xyz)

    # The`groupby` array in original adata.obs or adata.X
    mask_list = mask if isinstance(mask, list) else [mask]

    obs_names = set(adata.obs_keys())
    gene_names = set(adata.var_names.tolist())
    if groupby in obs_names:
        groups = adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values
    elif groupby in gene_names or set(groupby) <= gene_names:
        groups = adata[:, groupby].X.sum(axis=1).flatten()
    elif groupby is None:
        groups = np.array(["same"] * adata.obs.shape[0])
    else:
        raise ValueError(
            "\n`groupby` value is wrong." 
            "\n`groupby` can be a string and one of adata.obs_names or adata.var_names. "
            "\n`groupby` can also be a list and is a subset of adata.var_names"
        )

    pcd = add_mesh_labels(
        mesh=pcd,
        labels=groups,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
    )

    # The obs_index of each coordinate in the original adata.
    pcd.point_data["obs_index"] = adata.obs_names.to_numpy()

    return pcd


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
        v_pcd: A voxelized point cloud.
    """

    voxel_size = [1, 1, 1] if voxel_size is None else voxel_size

    voxelizer = PVGeo.filters.VoxelizePoints()
    voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
    voxelizer.set_estimate_grid(False)
    v_pcd = voxelizer.apply(pcd)

    # add labels
    for key in pcd.array_names:
        v_pcd.cell_data[key] = pcd.point_data[key]

    return v_pcd


def construct_surface(
    pcd: PolyData,
    key_added: str = "groups",
    color: Optional[str] = "gainsboro",
    alpha: Optional[float] = 0.8,
    cs_method: Literal["basic", "slide", "alpha_shape", "ball_pivoting", "poisson"] = "basic",
    cs_method_args: dict = None,
    surface_smoothness: int = 100,
    n_surf: int = 10000,
) -> Tuple[PolyData, PolyData]:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pcd: A point cloud.
        key_added: The key under which to add the labels.
        color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.8`.
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
    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            `surf.cell_data[key_added]`, the "surface" array;
            `surf.cell_data[f'{key_added}_rgba']`, the rgba colors of the "surface" array.
        clipped_pcd: A point cloud, which contains the following properties:
            `clipped_pcd.point_data["obs_index"]`, the obs_index of each coordinate in the original adata.
            `clipped_pcd.point_data[key_added]`, the `groupby` information.
            `clipped_pcd.point_data[f'{key_added}_rgba']`, the rgba colors of the `groupby` information.
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
            al_alpha = _cs_method_args["al_alpha"]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(_pcd, al_alpha)

        elif cs_method == "ball_pivoting":
            ba_radii = _cs_method_args["ba_radii"]

            _pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            _pcd.estimate_normals()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                _pcd, o3d.utility.DoubleVector(ba_radii)
            )

        else:
            po_depth, po_density_threshold = (
                _cs_method_args["po_depth"],
                _cs_method_args["po_threshold"],
            )

            _pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            _pcd.estimate_normals()

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                (
                    mesh,
                    densities,
                ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(_pcd, depth=po_depth)
            mesh.remove_vertices_by_mask(np.asarray(densities) < np.quantile(densities, po_density_threshold))

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

    # Add labels and the colormap of the surface mesh
    labels = np.array(["surface"] * uniform_surf.n_cells).astype(str)
    uniform_surf = add_mesh_labels(
        mesh=uniform_surf,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
    )

    # Clip the original pcd using the reconstructed surface and reconstruct new point cloud.
    clip_invert = True if cs_method in ["basic", "slide"] else False
    clipped_pcd = pcd.clip_surface(uniform_surf, invert=clip_invert)

    return uniform_surf, clipped_pcd


def construct_volume(
    mesh: Union[PolyData, UnstructuredGrid],
    key_added: str = "groups",
    color: Optional[str] = "gainsboro",
    alpha: Optional[float] = 0.8,
    volume_smoothness: Optional[int] = 200,
) -> UnstructuredGrid:
    """
    Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh.
        key_added: The key under which to add the labels.
        color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.8`.
        volume_smoothness: The smoothness of the volumetric mesh.

    Returns:
        volume: A reconstructed volumetric mesh, which contains the following properties:
            `volume.cell_data[key_added]`, the "volume" array;
            `volume.cell_data[f'{key_added}_rgba']`,  the rgba colors of the "volume" array.

    """

    density = mesh.length / volume_smoothness
    volume = pv.voxelize(mesh, density=density, check_surface=False)

    # Add labels and the colormap of the volumetric mesh
    labels = np.array(["volume"] * volume.n_cells).astype(str)
    volume = add_mesh_labels(
        mesh=volume,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
    )

    return volume


def add_mesh_labels(
    mesh: Union[PolyData, UnstructuredGrid],
    labels: np.ndarray,
    key_added: str = "groups",
    where: Literal["point_data", "cell_data"] = "cell_data",
    colormap: Union[str, list] = None,
    alphamap: Union[float, list] = None,
    mask_color: Optional[str] = "gainsboro",
    mask_alpha: Optional[float] = 0,
) -> PolyData or UnstructuredGrid:
    """
    Add rgba color to each point of mesh based on labels.

    Args:
        mesh: A reconstructed mesh.
        labels: An array of labels of interest.
        key_added: The key under which to add the labels.
        where: The location where the label information is recorded in the mesh.
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Color to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
    Returns:
         A mesh, which contains the following properties:
            `mesh.cell_data[key_added]` or `mesh.point_data[key_added]`, the labels array;
            `mesh.cell_data[f'{key_added}_rgba']` or `mesh.point_data[f'{key_added}_rgba']`, the rgba colors of the labels.
    """

    cu_arr = np.unique(labels)
    cu_arr = np.sort(cu_arr, axis=0)
    cu_dict = {}

    # Set mask rgba.
    mask_ind = np.argwhere(cu_arr == "mask")
    if len(mask_ind) != 0:
        cu_arr = np.delete(cu_arr, mask_ind[0])
        cu_dict["mask"] = mpl.colors.to_rgba(mask_color, alpha=mask_alpha)

    cu_arr_num = cu_arr.shape[0]
    if cu_arr_num != (0,):
        # Set alpha.
        alpha_list = alphamap if isinstance(alphamap, list) else [alphamap] * cu_arr_num

        # Set raw rgba.
        if isinstance(colormap, list):
            raw_rgba_list = [mpl.colors.to_rgba(color) for color in colormap]
        elif colormap in list(mpl.colormaps):
            lscmap = mpl.cm.get_cmap(colormap)
            raw_rgba_list = [lscmap(i) for i in np.linspace(0, 1, cu_arr_num)]
        else:
            raw_rgba_list = [mpl.colors.to_rgba(colormap)] * cu_arr_num

        # Set new rgba.
        for t, c, a in zip(cu_arr, raw_rgba_list, alpha_list):
            cu_dict[t] = mpl.colors.to_rgba(c, alpha=a)

    # Added labels and rgba of the labels
    if where == "point_data":
        mesh.point_data[key_added] = labels
        mesh.point_data[f"{key_added}_rgba"] = np.array([cu_dict[g] for g in labels]).astype(np.float64)
    else:
        mesh.cell_data[key_added] = labels
        mesh.cell_data[f"{key_added}_rgba"] = np.array([cu_dict[g] for g in labels]).astype(np.float64)

    return mesh
