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
        slice_cloud = pv.PolyData(slice_coords)
        if len(slice_coords) >= 3:
            slice_plane = slice_cloud.delaunay_2d(alpha=alpha).triangulate().clean()
            uniform_plane = uniform_mesh(mesh=slice_plane, nsub=nsub, nclus=nclus)
            slices.append(uniform_plane)
        else:
            slices.append(slice_cloud)

    slices_mesh = merge_models(models=slices)
    new_pc = pv.PolyData(slices_mesh.points).clean()
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


##############
# clean mesh #
##############


def clean_mesh(mesh: PolyData) -> PolyData:
    """Removes unused points and degenerate cells."""

    sub_meshes = mesh.split_bodies()
    n_mesh = len(sub_meshes)

    if n_mesh == 1:
        return mesh
    else:
        inside_number = []
        for i, main_mesh in enumerate(sub_meshes[:-1]):
            main_mesh = pv.PolyData(main_mesh.points, main_mesh.cells)
            for j, check_mesh in enumerate(sub_meshes[i + 1 :]):
                check_mesh = pv.PolyData(check_mesh.points, check_mesh.cells)
                inside = check_mesh.select_enclosed_points(main_mesh, check_surface=False).threshold(0.5)
                inside = pv.PolyData(inside.points, inside.cells)
                if check_mesh == inside:
                    inside_number.append(i + 1 + j)

        cm_number = list(set([i for i in range(n_mesh)]).difference(set(inside_number)))
        if len(cm_number) == 1:
            cmesh = sub_meshes[cm_number[0]]
        else:
            cmesh = merge_models([sub_meshes[i] for i in cm_number])

        return pv.PolyData(cmesh.points, cmesh.cells)
