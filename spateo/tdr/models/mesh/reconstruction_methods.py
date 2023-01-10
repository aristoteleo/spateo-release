"""
This code contains methods to reconstruct the mesh model based on the 3D point cloud.
    1. 3D Delaunay
    2. marching cube algorithm
    3. alpha shape algorithm
    4. ball pivoting algorithm
    5. poisson reconstruction
"""

from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData
from scipy.spatial.distance import cdist

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ....tools.three_dims_align import rigid_transform_3D
from ..utilities import scale_model

###############
# 3D Delaunay #
###############


def pv_mesh(pc: PolyData, alpha: float = 2.0) -> PolyData:
    """
    Generate a 3D tetrahedral mesh from a scattered points and extract surface mesh of the 3D tetrahedral mesh.

    Args:
        pc: A point cloud model.
        alpha: Distance value to control output of this filter.
               For a non-zero alpha value, only vertices, edges, faces,
               or tetrahedron contained within the circumspect (of radius alpha) will be output.
               Otherwise, only tetrahedron will be output.

    Returns:
        A mesh model.
    """
    mesh = pc.delaunay_3d(alpha=alpha).extract_surface().triangulate().clean(tolerance=1.5)

    if mesh.n_points == 0:
        raise ValueError(
            f"\nThe point cloud cannot generate a surface mesh with `pyvista` method and alpha == {alpha}."
        )

    return mesh


###########################
# marching cube algorithm #
###########################


def marching_cube_mesh(pc: PolyData, levelset: Union[int, float] = 0, mc_scale_factor: Union[int, float] = 1.0):
    """
    Computes a triangle mesh from a point cloud based on the marching cube algorithm.
    Algorithm Overview:
        The algorithm proceeds through the scalar field, taking eight neighbor locations at a time (thus forming an
        imaginary cube), then determining the polygon(s) needed to represent the part of the iso-surface that passes
        through this cube. The individual polygons are then fused into the desired surface.

    Args:
        pc: A point cloud model.
        levelset: The levelset of iso-surface. It is recommended to set levelset to 0 or 0.5.
        mc_scale_factor: The scale of the model. The scaled model is used to construct the mesh model.

    Returns:
        A mesh model.
    """
    try:
        import mcubes
    except ImportError:
        raise ImportError(
            "You need to install the package `mcubes`." "\nInstall mcubes via `pip install --upgrade PyMCubes`"
        )

    pc = pc.copy()

    # Move the model so that the coordinate minimum is at (0, 0, 0).
    raw_points = np.asarray(pc.points)
    pc.points = new_points = raw_points - np.min(raw_points, axis=0)

    # Generate new models for calculatation.
    dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
    row, col = np.diag_indices_from(dist)
    dist[row, col] = None
    max_dist = np.nanmin(dist, axis=1).max()
    mc_sf = max_dist * mc_scale_factor

    scale_pc = scale_model(model=pc, scale_factor=1 / mc_sf)
    scale_pc_points = scale_pc.points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)

    # Generate grid for calculatation based on new model.
    volume_array = np.zeros(
        shape=[
            scale_pc_points[:, 0].max() + 3,
            scale_pc_points[:, 1].max() + 3,
            scale_pc_points[:, 2].max() + 3,
        ]
    )
    volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1

    # Extract the iso-surface based on marching cubes algorithm.
    # volume_array = mcubes.smooth(volume_array)
    vertices, triangles = mcubes.marching_cubes(volume_array, levelset)

    if len(vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `marching_cube` method.")

    v = np.asarray(vertices).astype(np.float64)
    f = np.asarray(triangles).astype(np.int64)
    f = np.c_[np.full(len(f), 3), f]

    # Generate mesh model.
    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    mesh.clean(inplace=True)
    mesh = scale_model(model=mesh, scale_factor=mc_sf)

    # Transform.
    scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf)
    mesh.points = rigid_transform_3D(
        coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
    )
    return mesh


##############################
# Docking open3d and pyvista #
##############################


def _pv2o3d(pc: PolyData):
    """Convert a point cloud in PyVista to Open3D."""

    # Check open3d package
    try:
        import open3d as o3d
        from open3d import geometry
    except ImportError:
        raise ImportError("You need to install the package `open3d`. \nInstall open3d via `pip install open3d`")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc.points)
    if "norms" in pc.point_data.keys():
        cloud.normals = o3d.utility.Vector3dVector(pc["norms"])
    else:
        cloud.estimate_normals()
    return cloud


def _o3d2pv(trimesh) -> PolyData:
    """Convert a triangle mesh in Open3D to PyVista."""
    v = np.asarray(trimesh.vertices)
    f = np.array(trimesh.triangles)
    f = np.c_[np.full(len(f), 3), f]

    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate().clean()
    return mesh


#########################
# Alpha shape algorithm #
#########################


def alpha_shape_mesh(pc: PolyData, alpha: float = 2.0) -> PolyData:
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
        pc: A point cloud model.
        alpha: Parameter to control the shape.
               With decreasing alpha value the shape shrinks and creates cavities.
               A very big value will give a shape close to the convex hull.

    Returns:
        A mesh model.
    """
    # Check open3d package
    try:
        import open3d as o3d
        from open3d import geometry
    except ImportError:
        raise ImportError("You need to install the package `open3d`. \nInstall open3d via `pip install open3d`")

    cloud = _pv2o3d(pc)
    trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)

    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"The point cloud cannot generate a surface mesh with `alpha shape` method and alpha == {alpha}."
        )

    return _o3d2pv(trimesh=trimesh)


###########################
# ball pivoting algorithm #
###########################


def ball_pivoting_mesh(pc: PolyData, radii: List[float] = None):
    """
    Computes a triangle mesh from an oriented point cloud based on the ball pivoting algorithm.
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
        pc: A point cloud model.
        radii: The radii of the ball that are used for the surface reconstruction.
               This is a list of multiple radii that will create multiple balls of different radii at the same time.

    Returns:
        A mesh model.
    """
    # Check open3d package
    try:
        import open3d as o3d
        from open3d import geometry
    except ImportError:
        raise ImportError("You need to install the package `open3d`. \nInstall open3d via `pip install open3d`")

    cloud = _pv2o3d(pc)
    radii = [1] if radii is None else radii
    radii = o3d.utility.DoubleVector(radii)
    trimesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)

    if len(trimesh.vertices) == 0:
        raise ValueError(
            f"The point cloud cannot generate a surface mesh with `ball pivoting` method and radii == {radii}."
        )

    return _o3d2pv(trimesh=trimesh)


##########################
# poisson reconstruction #
##########################


def poisson_mesh(
    pc: PolyData,
    depth: int = 8,
    width: float = 0,
    scale: float = 1.1,
    linear_fit: bool = False,
    density_threshold: Optional[float] = None,
) -> PolyData:
    """
    Computes a triangle mesh from an oriented point cloud based on the screened poisson reconstruction.

    Args:
        pc: A point cloud model.
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
        A mesh model.
    """
    # Check open3d package
    try:
        import open3d as o3d
        from open3d import geometry
    except ImportError:
        raise ImportError("You need to install the package `open3d`. \nInstall open3d via `pip install open3d`")

    cloud = _pv2o3d(pc)
    trimesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    if len(trimesh.vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `poisson` method and depth == {depth}.")

    # A low density value means that the vertex is only supported by a low number of points from the input point cloud.
    # Remove all vertices (and connected triangles) that have a lower density value than the density_threshold quantile
    # of all density values.
    if not (density_threshold is None):
        trimesh.remove_vertices_by_mask(np.asarray(density) < np.quantile(density, density_threshold))

    return _o3d2pv(trimesh=trimesh)


############################################################
# spherical harmonics: scipy.special.sph_harm
# Reference:
#   https://www.nature.com/articles/s41467-017-00023-7#Abs1
#   https://www.hindawi.com/journals/mpe/2015/582870/  # related-articles
############################################################
