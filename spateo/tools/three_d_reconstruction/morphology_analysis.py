import numpy as np
import pandas as pd
import pyvista as pv

from pyvista import PolyData, UnstructuredGrid
from sklearn.neighbors import KernelDensity
from typing import Optional, Union, Tuple


def _log(m):
    print(f"\n[-----{m}")


def mesh_morphology(mesh: Union[PolyData, UnstructuredGrid], verbose: bool = True) -> Tuple[float, float, float]:
    """
    Return the basic morphological characteristics of mesh,
    including mesh volume, mesh surface area, volume / surface area ratio.

    Args:
        mesh: A reconstructed mesh.
        verbose: Print information along iterations.
    Returns:
        mesh_v: The volume of the reconstructed 3D mesh.
        mesh_sa: The surface area of the reconstructed 3D mesh.
        mesh_vsa: The volume / surface area ratio of the reconstructed 3D mesh.
    """

    mesh_surf = mesh.extract_surface()

    # mesh volume
    mesh_v = round(mesh_surf.volume, 5)

    # mesh surface area
    mesh_sa = round(mesh_surf.area, 5)

    # volume / surface area ratio
    mesh_vsa = round(mesh_v / mesh_sa, 5)

    if verbose:
        _log(f"mesh volume: {mesh_v};")
        _log(f"mesh surface area: {mesh_sa};")
        _log(f"mesh volume / surface area ratio: {mesh_vsa}.")

    return mesh_v, mesh_sa, mesh_vsa


def mesh_memory_size(mesh: Union[PolyData, UnstructuredGrid], verbose: bool = True):
    """
    Return the actual memory size of the mesh.

    Args:
        mesh: A reconstructed mesh.
        verbose: Print information along iterations.
    Returns:
        m_memory: The actual memory size of the mesh.
    """

    m_memory = mesh.actual_memory_size

    if m_memory < 1024:
        m_memory = f"{m_memory} KiB"
    elif 1024 <= m_memory < 1024 * 1024:
        m_memory = f"{round(m_memory / 1024, 2)} MiB"
    else:
        m_memory = f"{round(m_memory / 1024 * 1024, 2)} GiB"

    if verbose:
        _log(f"The actual memory size of the mesh: {m_memory}.")

    return m_memory


def pcd_KDE(
    pcd: Union[PolyData, UnstructuredGrid],
    key_added: str = "kde",
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
):
    """
    Calculate the kernel density of a 3D point cloud.

    Args:
        pcd: A point cloud.
        key_added: The key under which to add the labels.
        kernel: The kernel to use. Available `kernel` are:
                * `'gaussian'`
                * `'tophat'`
                * `'epanechnikov'`
                * `'exponential'`
                * `'linear'`
                * `'cosine'`
        bandwidth: The bandwidth of the kernel.
    Returns:
        pcd: Reconstructed 3D point cloud, which contains the following properties:
            `pcd[key_added]`, the kernel density;
    """

    coords = pcd.points
    pcd_kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(coords).score_samples(coords)
    pcd.point_data[key_added] = pcd_kde

    return pcd
