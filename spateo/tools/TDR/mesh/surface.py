from typing import List, Optional, Tuple, Union

import mcubes
import numpy as np
import open3d as o3d
import pyacvd
import pymeshfix as mf
import pyvista as pv
from open3d import geometry
from pyvista import PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import add_mesh_labels, merge_mesh


def _pv2o3d(pc: PolyData):
    """Convert a point cloud in PyVista to Open3D."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc.points)
    if "norms" in pc.point_data.keys():
        cloud.normals = o3d.utility.Vector3dVector(pc["norms"])
    else:
        cloud.estimate_normals()
    return cloud


def _o3d2pv(trimesh: geometry.TriangleMesh) -> PolyData:
    """Convert a triangle mesh in Open3D to PyVista."""
    v = np.asarray(trimesh.vertices)
    f = np.array(trimesh.triangles)
    f = np.c_[np.full(len(f), 3), f]

    surface = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    surface.clean(inplace=True)
    return surface


def uniform_point_cloud(pc: PolyData, alpha: float = 2.0, nsub: Optional[int] = 5, nclus: int = 20000):
    """
    Generates a uniform point cloud with a larger number of points.
    If the number of points in the original point cloud is too small or the distribution of the original point cloud is
    not uniform, making it difficult to construct the surface, this method can be used for preprocessing.

    Args:
        pc: A point cloud.
        alpha: Specify alpha (or distance) value to control output of this filter.
               For a non-zero alpha value, only edges or triangles contained within a sphere centered at mesh vertices
               will be output. Otherwise, only triangles will be output.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        uniform_cloud: A uniform point cloud with a larger number of points.
    """
    coords = np.asarray(pc.points)
    coords_z = np.unique(coords[:, 2])
    slices = []
    for z in coords_z:
        slice_coords = coords[coords[:, 2] == z]
        if len(slice_coords) >= 3:
            slice_cloud = pv.PolyData(slice_coords)
            slice_plane = slice_cloud.delaunay_2d(alpha=alpha).triangulate().clean()
            slices.append(slice_plane)
        else:
            raise ValueError(
                f"\nWhen the z-axis is {z}, the number of coordinates is less than 3 and cannot be uniform."
            )
    slices = merge_mesh(meshes=slices)

    uniform_surf = uniform_surface(surf=slices, nsub=nsub, nclus=nclus)
    uniform_cloud = pv.PolyData(uniform_surf.points)
    uniform_cloud.clean()
    return uniform_cloud


def uniform_surface(surf: PolyData, nsub: Optional[int] = 3, nclus: int = 20000) -> PolyData:
    """
    Generate a uniformly meshed surface using voronoi clustering.

    Args:
        surf: A surface mesh.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        uniform_surf: A uniform surface mesh.
    """
    # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
    if not (nsub is None):
        surf.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

    # Uniformly remeshing.
    clustered = pyacvd.Clustering(surf)
    clustered.cluster(nclus)

    uniform_surf = clustered.create_mesh().triangulate()
    uniform_surf.clean(inplace=True)
    return uniform_surf


def fix_surface(surf: PolyData) -> PolyData:
    """Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh."""
    meshfix = mf.MeshFix(surf)
    meshfix.repair(verbose=False)

    surf = meshfix.mesh.triangulate()

    if surf.n_points == 0:
        raise ValueError(
            f"\nThe surface cannot be Repaired. " f"\nPlease change the method or parameters of surface reconstruction."
        )

    surf.clean(inplace=True)
    return surf


def scale_mesh(
    mesh: Union[PolyData, UnstructuredGrid],
    scale_factor: Union[float, list] = 1,
) -> Union[PolyData, UnstructuredGrid]:
    """
    Scale the mesh around the center of the mesh.

    Args:
        mesh: Reconstructed 3D mesh.
        scale_factor: scale of scaling. If `scale factor` is float, the mesh is scaled along the xyz axis at the same
                      scale; when the `scale factor` is list, the mesh is scaled along the xyz axis at different scales.

    Returns:
        mesh_s: Scaled mesh.
    """
    mesh_s = mesh.copy()

    if isinstance(scale_factor, float):
        factor_x = factor_y = factor_z = scale_factor
    else:
        factor_x = scale_factor[0]
        factor_y = scale_factor[1]
        factor_z = scale_factor[2]

    mesh_s.points[:, 0] = (mesh_s.points[:, 0] - mesh_s.center[0]) * factor_x + mesh_s.center[0]
    mesh_s.points[:, 1] = (mesh_s.points[:, 1] - mesh_s.center[1]) * factor_y + mesh_s.center[1]
    mesh_s.points[:, 2] = (mesh_s.points[:, 2] - mesh_s.center[2]) * factor_z + mesh_s.center[2]

    return mesh_s


def pv_surface(pc: PolyData, alpha: float = 2.0) -> PolyData:
    """
    Generate a 3D tetrahedral mesh from a scattered points and extract surface mesh of the 3D tetrahedral mesh.
    Args:
        pc: A point cloud.
        alpha: Distance value to control output of this filter.
               For a non-zero alpha value, only vertices, edges, faces,
               or tetrahedra contained within the circumsphere (of radius alpha) will be output.
               Otherwise, only tetrahedra will be output.
    Returns:
        surface: Surface mesh.
    """
    surface = pc.delaunay_3d(alpha=alpha).extract_surface().triangulate()

    if surface.n_points == 0:
        raise ValueError(
            f"\nThe point cloud cannot generate a surface mesh with `pyvista` method and alpha == {alpha}."
        )

    surface.clean(tolerance=1.5, inplace=True)
    return surface


def marching_cube_surface(pc: PolyData):
    """
    Computes a triangle mesh from a point cloud based on the marching cube algorithm.
    Algorithm Overview:
        The algorithm proceeds through the scalar field, taking eight neighbor locations at a time (thus forming an
        imaginary cube), then determining the polygon(s) needed to represent the part of the isosurface that passes
        through this cube. The individual polygons are then fused into the desired surface.

    Args:
        pc: A point cloud.

    Returns:
        surface: Surface mesh.
    """

    pc_points = np.asarray(pc.points)
    pc_points_int = np.ceil(pc_points).astype(np.int64)

    rmd = pc_points_int - pc_points
    rmd_mean = np.mean(rmd, axis=0)

    volume_array = np.zeros(
        shape=[
            pc_points_int[:, 0].max() + 3,
            pc_points_int[:, 1].max() + 3,
            pc_points_int[:, 2].max() + 3,
        ]
    )
    volume_array[pc_points_int[:, 0], pc_points_int[:, 1], pc_points_int[:, 2]] = 1

    # Extract the iso-surface based on marching cubes algorithm.
    vertices, triangles = mcubes.marching_cubes(volume_array, 0)

    if len(vertices) == 0:
        raise ValueError(f"\nThe point cloud cannot generate a surface mesh with `marching_cube` method.")

    vertices = vertices - rmd_mean

    v = np.asarray(vertices).astype(np.float64)
    f = np.asarray(triangles).astype(np.int64)
    f = np.c_[np.full(len(f), 3), f]

    surface = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    surface.clean(inplace=True)
    return surface


def alpha_shape_surface(pc: PolyData, alpha: float = 2.0) -> PolyData:
    """
    Computes a triangle mesh from a point cloud based on the alpha shape algorithm.
    Algorithm Overview:
        For each real number α, define the concept of a generalized disk of radius 1/α as follows:
            If α = 0, it is a closed half-plane;
            If α > 0, it is a closed disk of radius 1/α;
            If α < 0, it is the closure of the complement of a disk of radius −1/α.
        Then an edge of the alpha-shape is drawn between two members of the finite point set whenever there exists a
        generalized disk of radius 1/α containing none of the point set and which has the property that the two points
        lie on its boundary.
        If α = 0, then the alpha-shape associated with the finite point set is its ordinary convex hull.

    Args:
        pc: A point cloud.
        alpha: Parameter to control the shape.
               With decreasing alpha value the shape shrinks and creates cavities.
               A very big value will give a shape close to the convex hull.

    Returns:
        surface: Surface mesh.
    """
    cloud = _pv2o3d(pc)
    trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)

    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"\nThe point cloud cannot generate a surface mesh with `alpha shape` method and alpha == {alpha}."
        )

    surface = _o3d2pv(trimesh=trimesh)
    return surface


def ball_pivoting_surface(pc: PolyData, radii: List[float] = None):
    """
    Computes a triangle mesh from an oriented point cloud based on the Ball Pivoting algorithm.
    Algorithm Overview:
        The main assumption this algorithm is based on is the following: Given three vertices, and a ball of radius r,
        the three vertices form a triangle if the ball is getting "caught" and settle between the points, without
        containing any other point.
        The algorithm stimulates a virtual ball of radius r. Each iteration consists of two steps:
            * Seed triangle - The ball rolls over the point cloud until it gets "caught" between three vertices and
                              settles between in them. Choosing the right r promises no other point is contained in the
                              formed triangle. This triangle is called "Seed triangle".
            * Expanding triangle - The ball pivots from each edge in the seed triangle, looking for a third point. It
                                   pivots until it gets "caught" in the triangle formed by the edge and the third point.
                                   A new triangle is formed, and the algorithm tries to expand from it. This process
                                   continues until the ball can't find any point to expand to.
        At this point, the algorithm looks for a new seed triangle, and the process described above starts all over.
    Useful Notes:
        1. The point cloud is "dense enough";
        2. The chosen r size should be "slightly" larger than the average space between points.

    Args:
        pc: A point cloud.
        radii: The radii of the ball that are used for the surface reconstruction.
               This is a list of multiple radii that will create multiple balls of different radii at the same time.

    Returns:
        surface: Surface mesh.
    """

    cloud = _pv2o3d(pc)
    radii = [1] if radii is None else radii
    radii = o3d.utility.DoubleVector(radii)
    trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)

    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"\nThe point cloud cannot generate a surface mesh with `ball pivoting` method and radii == {radii}."
        )

    surface = _o3d2pv(trimesh=trimesh)
    return surface


def poisson_surface(
    pc: PolyData,
    depth: int = 8,
    width: float = 0,
    scale: float = 1.1,
    linear_fit: bool = False,
    density_threshold: Optional[float] = None,
) -> PolyData:
    """
    Computes a triangle mesh from an oriented point cloud based on thee Screened Poisson Reconstruction.

    Args:
        pc: A point cloud.
        depth: Maximum depth of the tree that will be used for surface reconstruction.
               Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d.
               Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction
               depth is only an upper bound.
               The depth that defines the depth of the octree used for the surface reconstruction and hence implies the
               resolution of the resulting triangle mesh. A higher depth value means a mesh with more details.
        width: Specifies the target width of the finest level octree cells.
               This parameter is ignored if depth is specified.
        scale: Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the
               samples’ bounding cube.
        linear_fit: If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.
        density_threshold: The threshold of the low density.

    Returns:
        surface: Surface mesh.
    """
    cloud = _pv2o3d(pc)
    trimesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"\nThe point cloud cannot generate a surface mesh with `poisson` method and depth == {depth}."
        )

    # A low density value means that the vertex is only supported by a low number of points from the input point cloud.
    # Remove all vertices (and connected triangles) that have a lower density value than the density_threshold quantile
    # of all density values.
    if not (density_threshold is None):
        trimesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density, density_threshold))

    surface = _o3d2pv(trimesh=trimesh)
    return surface


def construct_surface(
    pc: PolyData,
    key_added: str = "groups",
    color: Optional[str] = "gainsboro",
    alpha: Optional[float] = 0.8,
    uniform_pc: bool = True,
    uniform_pc_alpha: float = 3.0,
    cs_method: Literal["pyvista", "alpha_shape", "ball_pivoting", "poisson", "marching_cube"] = "marching_cube",
    cs_args: Optional[dict] = None,
    nsub: Optional[int] = 3,
    nclus: int = 20000,
    smooth: Optional[int] = 500,
    scale_factor: Union[float, list] = 1.1,
) -> Tuple[PolyData, PolyData]:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pc: A point cloud.
        key_added: The key under which to add the labels.
        color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.8`.
        uniform_pc: Generates a uniform point cloud with a larger number of points.
        uniform_pc_alpha: Specify alpha (or distance) value to control output of this filter.
        cs_method: The methods of generating a surface mesh. Available `cs_method` are:
                * `'pyvista'`: Generate a 3D tetrahedral mesh based on pyvista.
                * `'alpha_shape'`: Computes a triangle mesh on the alpha shape algorithm.
                * `'ball_pivoting'`: Computes a triangle mesh based on the Ball Pivoting algorithm.
                * `'poisson'`: Computes a triangle mesh based on thee Screened Poisson Reconstruction.
                * `'marching_cube'`: Computes a triangle mesh based on the marching cube algorithm.
        cs_args: Parameters for various surface reconstruction methods. Available `cs_args` are:
                * `'pyvista'`: {"alpha": 0}
                * `'alpha_shape'`: {"alpha": 2.0}
                * `'ball_pivoting'`: {"radii": [1]}
                * `'poisson'`: {'depth': 8, 'width'=0, 'scale'=1.1, 'linear_fit': False, 'density_threshold': 0.01}
                * `'marching_cube'`: None
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.
        smooth: Number of iterations for Laplacian smoothing.
        scale_factor: scale of scaling. If `scale factor` is float, the mesh is scaled along the xyz axis at the same
                      scale; when the `scale factor` is list, the mesh is scaled along the xyz axis at different scales.


    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            `uniform_surf.cell_data[key_added]`, the "surface" array;
            `uniform_surf.cell_data[f'{key_added}_rgba']`, the rgba colors of the "surface" array.
        clipped_pcd: A point cloud, which contains the following properties:
            `clipped_pcd.point_data["obs_index"]`, the obs_index of each coordinate in the original adata.
            `clipped_pcd.point_data[key_added]`, the `groupby` information.
            `clipped_pcd.point_data[f'{key_added}_rgba']`, the rgba colors of the `groupby` information.
    """

    # Generates a uniform point cloud with a larger number of points or not.
    cloud = uniform_point_cloud(pc=pc, alpha=uniform_pc_alpha, nsub=3, nclus=20000) if uniform_pc else pc.copy()

    # Reconstruct surface mesh.
    if cs_method == "pyvista":
        _cs_args = {"alpha": 0}
        if not (cs_args is None):
            _cs_args.update(cs_args)
        surf = pv_surface(pc=cloud, alpha=_cs_args["alpha"])
    elif cs_method == "alpha_shape":
        _cs_args = {"alpha": 2.0}
        if not (cs_args is None):
            _cs_args.update(cs_args)
        surf = alpha_shape_surface(pc=cloud, alpha=_cs_args["alpha"])
    elif cs_method == "ball_pivoting":
        _cs_args = {"radii": [1]}
        if not (cs_args is None):
            _cs_args.update(cs_args)
        surf = ball_pivoting_surface(pc=cloud, radii=_cs_args["radii"])
    elif cs_method == "poisson":
        _cs_args = {
            "depth": 8,
            "width": 0,
            "scale": 1.1,
            "linear_fit": False,
            "density_threshold": None,
        }
        if not (cs_args is None):
            _cs_args.update(cs_args)
        surf = poisson_surface(
            pc=cloud,
            depth=_cs_args["depth"],
            width=_cs_args["width"],
            scale=_cs_args["scale"],
            linear_fit=_cs_args["linear_fit"],
            density_threshold=_cs_args["density_threshold"],
        )
    elif cs_method == "marching_cube":
        surf = marching_cube_surface(pc=cloud)
    else:
        raise ValueError(
            "\n`cs_method` value is wrong."
            "\nAvailable `cs_method` are: `'pyvista'`, `'alpha_shape'`, `'ball_pivoting'`, `'poisson'`, `'marching_cube'`."
        )

    # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
    fix_surf = fix_surface(surf=surf)

    # Get a uniformly meshed surface using voronoi clustering.
    uniform_surf = uniform_surface(surf=fix_surf, nsub=nsub, nclus=nclus)

    # Adjust point coordinates using Laplacian smoothing.
    if not (smooth is None):
        uniform_surf.smooth(n_iter=smooth, inplace=True)

    # Add labels and the colormap of the surface mesh.
    labels = np.array(["surface"] * uniform_surf.n_cells).astype(str)
    add_mesh_labels(
        mesh=uniform_surf,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
    )

    # Scale the surface mesh.
    uniform_surf = scale_mesh(mesh=uniform_surf, scale_factor=scale_factor)

    # Clip the original pcd using the reconstructed surface and reconstruct new point cloud.
    clipped_invert = True if cs_method in ["pyvista", "marching_cube"] else False
    clipped_pcd = pc.clip_surface(uniform_surf, invert=clipped_invert)

    return uniform_surf, clipped_pcd
