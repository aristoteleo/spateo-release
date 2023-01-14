import numpy as np
import pyvista as pv
from pyvista import PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional, Union

from ..utilities import add_model_labels, merge_models

#########################################################
# Construct cell-level voxel model based on point cloud #
#########################################################


def voxelize_pc(
    pc: PolyData,
    voxel_size: Optional[np.ndarray] = None,
) -> UnstructuredGrid:
    """
    Voxelize the point cloud.

    Args:
        pc: A point cloud model.
        voxel_size: The size of the voxelized points. The shape of voxel_size is (pc.n_points, 3).

    Returns:
        voxel: A voxel model.
    """
    # Check open3d package
    try:
        import PVGeo
    except ImportError:
        raise ImportError("You need to install the package `PVGeo`. \nInstall PVGeo via `pip install PVGeo`")

    voxelizer = PVGeo.filters.VoxelizePoints()

    if not (voxel_size is None):
        voxelizer.set_deltas(voxel_size[:, 0], voxel_size[:, 1], voxel_size[:, 2])
        voxelizer.set_estimate_grid(False)

    voxel_pc = voxelizer.apply(pc)

    # add labels
    pc_keys = pc.point_data.keys()
    if not (pc_keys is None):
        for key in pc_keys:
            voxel_pc.cell_data[key] = pc.point_data[key]

    return voxel_pc


##################################################################
# Construct cell-level or tissue-level voxel model based on mesh #
##################################################################


def voxelize_mesh(
    mesh: Union[PolyData, UnstructuredGrid],
    voxel_pc: Union[PolyData, UnstructuredGrid] = None,
    key_added: str = "groups",
    label: str = "voxel",
    color: Optional[str] = "gainsboro",
    alpha: Union[float, int] = 1.0,
    smooth: Optional[int] = 200,
) -> UnstructuredGrid:
    """
    Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh model.
        voxel_pc: A voxel model which contains the ``voxel_pc.cell_data['obs_index']`` and ``voxel_pc.cell_data[key_added]``.
        key_added: The key under which to add the labels.
        label: The label of reconstructed voxel model.
        color: Color to use for plotting mesh. The default color is ``'gainsboro'``.
        alpha: The opacity of the color to use for plotting model. The default alpha is ``0.8``.
        smooth: The smoothness of the voxel model.

    Returns:
        voxel_model: A reconstructed voxel model, which contains the following properties:
            `voxel_model.cell_data[key_added]`, the `label` array;
            `voxel_model.cell_data[f'{key_added}_rgba']`,  the rgba colors of the `label` array.
            `voxel_model.cell_data['obs_index']`, the cell labels if not (voxel_pc is None).
    """

    density = mesh.length / smooth
    voxel_model = pv.voxelize(mesh, density=density, check_surface=False)

    # Add labels and the colormap of the volumetric mesh
    labels = np.array([label] * voxel_model.n_cells).astype(str)
    add_model_labels(
        model=voxel_model,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
        inplace=True,
    )
    if not (voxel_pc is None):
        voxel_model.cell_data["obs_index"] = np.asarray(["no_cell"] * voxel_model.n_cells).astype(str)
        voxel_model = merge_models(models=[voxel_model, voxel_pc])

    return voxel_model
