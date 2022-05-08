"""
This code contains methods to optimize the final reconstructed mesh model.
    1. uniform mesh
    2. fix mesh
"""
from typing import Optional, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import merge_models

################
# uniform mesh #
################


def uniform_larger_pc(
    pc: PolyData,
    alpha: Union[float, int] = 0,
    nsub: Optional[int] = 5,
    nclus: int = 20000,
) -> PolyData:
    """
    Generates a uniform point cloud with a larger number of points.
    If the number of points in the original point cloud is too small or the distribution of the original point cloud is
    not uniform, making it difficult to construct the surface, this method can be used for preprocessing.

    Args:
        pc: A point cloud model.
        alpha: Specify alpha (or distance) value to control output of this filter.
               For a non-zero alpha value, only edges or triangles contained within a sphere centered at mesh vertices
               will be output. Otherwise, only triangles will be output.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        new_pc: A uniform point cloud with a larger number of points.
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
            raise ValueError(f"When the z-axis is {z}, the number of coordinates is less than 3 and cannot be uniform.")
    slices_mesh = merge_models(models=slices)
    uniform_slices_mesh = uniform_mesh(mesh=slices_mesh, nsub=nsub, nclus=nclus)

    new_pc = pv.PolyData(uniform_slices_mesh.points).clean()
    return new_pc


def uniform_mesh(mesh: PolyData, nsub: Optional[int] = 3, nclus: int = 20000) -> PolyData:
    """
    Generate a uniformly meshed surface using voronoi clustering.

    Args:
        mesh: A mesh model.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        new_mesh: A uniform mesh model.
    """
    # Check pyacvd package
    try:
        import pyacvd
    except ImportError:
        raise ImportError("You need to install the package `pyacvd`. \nInstall pyacvd via `pip install pyacvd`")

    # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
    if not (nsub is None):
        mesh.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

    # Uniformly remeshing.
    clustered = pyacvd.Clustering(mesh)
    clustered.cluster(nclus)

    new_mesh = clustered.create_mesh().triangulate().clean()
    return new_mesh


###############
# Smooth mesh #
###############


def smooth_mesh(mesh: PolyData, n_iter: int = 100, **kwargs) -> PolyData:
    """
    Adjust point coordinates using Laplacian smoothing.
    https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.smooth.html#pyvista.PolyData.smooth

    Args:
        mesh: A mesh model.
        n_iter: Number of iterations for Laplacian smoothing.
        **kwargs: The rest of the parameters in pyvista.PolyData.smooth.

    Returns:
        smoothed_mesh: A smoothed mesh model.
    """

    smoothed_mesh = mesh.smooth(n_iter=n_iter, **kwargs)

    return smoothed_mesh


############
# fix mesh #
############


def fix_mesh(mesh: PolyData) -> PolyData:
    """Repair the mesh where it was extracted and subtle holes along complex parts of the mesh."""

    # Check pymeshfix package
    try:
        import pymeshfix as mf
    except ImportError:
        raise ImportError(
            "You need to install the package `pymeshfix`. \nInstall pymeshfix via `pip install pymeshfix`"
        )

    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=False)
    fixed_mesh = meshfix.mesh.triangulate().clean()

    if fixed_mesh.n_points == 0:
        raise ValueError(
            f"The surface cannot be Repaired. " f"\nPlease change the method or parameters of surface reconstruction."
        )

    return fixed_mesh
