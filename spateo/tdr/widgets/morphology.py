from typing import Any, Dict, Optional, Union

import numpy as np
from pyvista import PolyData, UnstructuredGrid
from sklearn.neighbors import KernelDensity

from ...logging import logger_manager as lm
from ..models import add_model_labels


def model_morphology(
    model: Union[PolyData, UnstructuredGrid],
    pc: Optional[PolyData or UnstructuredGrid] = None,
) -> Dict[str, Union[float, Any]]:
    """
    Return the basic morphological characteristics of model,
    including model volume, model surface area, volume / surface area ratio，etc.

    Args:
        model: A reconstructed surface model or volume model.
        pc: A point cloud representing the number of cells.
    Returns:
        morphology: A dictionary containing the following model morphological features:
            morphology['Length(x)']: Length (x) of model.
            morphology['Width(y)']: Width (y) of model.
            morphology['Height(z)']: Height (z) of model.
            morphology['Surface_area']: Surface area of model.
            morphology['Volume']: Volume of model.
            morphology['V/SA_ratio']: Volume / surface area ratio of model;
            morphology['cell_density']: Cell density of model.
    """

    model_surf = model.extract_surface()
    morphology = {}

    # Length, width and height of model
    model_bounds = np.asarray(model.bounds)
    model_x = round(abs(model_bounds[1] - model_bounds[0]), 5)
    model_y = round(abs(model_bounds[3] - model_bounds[2]), 5)
    model_z = round(abs(model_bounds[5] - model_bounds[4]), 5)
    morphology["Length(x)"], morphology["Width(y)"], morphology["Height(z)"] = (
        model_x,
        model_y,
        model_z,
    )
    lm.main_info(f"Length (x) of model: {morphology['Length(x)']};", indent_level=1)
    lm.main_info(f"Width (y) of model: {morphology['Width(y)']};", indent_level=1)
    lm.main_info(f"Height (z) of model: {morphology['Height(z)']};", indent_level=1)

    # Surface area of model
    model_sa = round(model_surf.area, 5)
    morphology["Surface_area"] = model_sa
    lm.main_info(f"Surface area of model: {morphology['Surface_area']};", indent_level=1)

    # Volume of model
    model_v = round(model_surf.volume, 5)
    morphology["Volume"] = model_v
    lm.main_info(f"Volume of model: {morphology['Volume']};", indent_level=1)

    # Volume / surface area ratio of model
    model_vsa = round(model_v / model_sa, 5)
    morphology["V/SA_ratio"] = model_vsa
    lm.main_info(f"Volume / surface area ratio of model: {morphology['V/SA_ratio']}.", indent_level=1)

    # cell density
    if not (pc is None):
        model_cd = round(pc.n_points / model_v, 5)
        morphology["cell_density"] = model_cd
        lm.main_info(f"Cell density of model: {morphology['cell_density']}.", indent_level=1)

    return morphology


def pc_KDE(
    pc: PolyData,
    key_added: str = "kde",
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
    colormap: Union[str, list, dict] = "hot_r",
    alphamap: Union[float, list, dict] = 1.0,
    inplace: bool = False,
) -> Union[PolyData, UnstructuredGrid]:
    """
    Calculate the kernel density of a 3D point cloud model.

    Args:
        pc: A point cloud model.
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
        inplace: Updates model in-place.

    Returns:
        pc: Reconstructed 3D point cloud, which contains the following properties:
            `pc[key_added]`, the kernel density.
    """

    pc = pc.copy() if not inplace else pc
    coords = pc.points
    pc_kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(coords).score_samples(coords)

    add_model_labels(
        model=pc,
        labels=pc_kde,
        key_added=key_added,
        where="point_data",
        colormap=colormap,
        alphamap=alphamap,
        inplace=True,
    )

    return pc if not inplace else None
