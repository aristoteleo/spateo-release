from typing import Any, Dict, Optional, Union

import numpy as np
from pyvista import PolyData, UnstructuredGrid
from sklearn.neighbors import KernelDensity

from ....logging import logger_manager as lm
from ..mesh.utils import add_mesh_labels


def mesh_morphology(
    mesh: Union[PolyData, UnstructuredGrid],
    pcd: Optional[PolyData or UnstructuredGrid] = None,
) -> Dict[str, Union[float, Any]]:
    """
    Return the basic morphological characteristics of mesh,
    including mesh volume, mesh surface area, volume / surface area ratio，etc..

    Args:
        mesh: A reconstructed surface mesh or volume mesh.
        pcd: A point cloud representing the number of cells.
    Returns:
        morphology: A dictionary containing the following mesh morphological features:
            morphology['Length(x)']: Length (x) of mesh.
            morphology['Width(y)']: Width (y) of mesh.
            morphology['Height(z)']: Height (z) of mesh.
            morphology['Surface_area']: Surface area of mesh.
            morphology['Volume']: Volume of mesh.
            morphology['V/SA_ratio']: Volume / surface area ratio of mesh;
            morphology['cell_density']: Cell density of mesh.
    """

    mesh_surf = mesh.extract_surface()
    morphology = {}

    # Length, width and height of mesh
    mesh_outline = mesh.outline()
    mo_points = np.asarray(mesh_outline.points)
    mesh_x = mo_points[:, 0].max() - mo_points[:, 0].min()
    mesh_y = mo_points[:, 1].max() - mo_points[:, 1].min()
    mesh_z = mo_points[:, 2].max() - mo_points[:, 2].min()
    mesh_x, mesh_y, mesh_z = (
        round(mesh_x.astype(float), 5),
        round(mesh_y.astype(float), 5),
        round(mesh_z.astype(float), 5),
    )
    morphology["Length(x)"], morphology["Width(y)"], morphology["Height(z)"] = (
        mesh_x,
        mesh_y,
        mesh_z,
    )
    lm.main_info(f"Length (x) of mesh: {morphology['Length(x)']};", indent_level=1)
    lm.main_info(f"Width (y) of mesh: {morphology['Width(y)']};", indent_level=1)
    lm.main_info(f"Height (z) of mesh: {morphology['Height(z)']};", indent_level=1)

    # Surface area of mesh
    mesh_sa = round(mesh_surf.area, 5)
    morphology["Surface_area"] = mesh_sa
    lm.main_info(f"Surface area of mesh: {morphology['Surface_area']};", indent_level=1)

    # Volume of mesh
    mesh_v = round(mesh_surf.volume, 5)
    morphology["Volume"] = mesh_v
    lm.main_info(f"Volume of mesh: {morphology['Volume']};", indent_level=1)

    # Volume / surface area ratio of mesh
    mesh_vsa = round(mesh_v / mesh_sa, 5)
    morphology["V/SA_ratio"] = mesh_vsa
    lm.main_info(f"Volume / surface area ratio of mesh: {morphology['V/SA_ratio']}.", indent_level=1)

    # cell density
    if not (pcd is None):
        mesh_cd = round(pcd.n_points / mesh_v, 5)
        morphology["cell_density"] = mesh_cd
        lm.main_info(f"Cell density of mesh: {morphology['cell_density']}.", indent_level=1)

    return morphology


def pc_KDE(
    pc: PolyData,
    key_added: str = "kde",
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
    colormap: Union[str, list, dict] = "hot_r",
    alphamap: Union[float, list, dict] = 1.0,
    copy: bool = False,
) -> Union[PolyData, UnstructuredGrid]:
    """
    Calculate the kernel density of a 3D point cloud.

    Args:
        pc: A point cloud.
        key_added: The key under which to add the labels.
        kernel: The kernel to use. Available `kernel` are:
                * `'gaussian'`
                * `'tophat'`
                * `'epanechnikov'`
                * `'exponential'`
                * `'linear'`
                * `'cosine'`
        bandwidth: The bandwidth of the kernel.
        colormap: Colors to use for plotting pcd. The default colormap is `'hot_r'`.
        alphamap: The opacity of the colors to use for plotting pcd. The default alphamap is `1.0`.
        copy: Whether to copy `pcd` or modify it inplace.
    Returns:
        pcd: Reconstructed 3D point cloud, which contains the following properties:
            `pcd[key_added]`, the kernel density.
    """

    pc = pc.copy() if copy else pc
    coords = pc.points
    pc_kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(coords).score_samples(coords)

    add_mesh_labels(
        mesh=pc,
        labels=pc_kde,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
    )

    return pc if copy else None
