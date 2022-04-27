from typing import Optional, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData, UnstructuredGrid

from .utils import add_mesh_labels, merge_mesh


def construct_volume(
    surf: Union[PolyData, UnstructuredGrid],
    voxel_pc: Union[PolyData, UnstructuredGrid] = None,
    key_added: str = "groups",
    color: Optional[str] = "gainsboro",
    alpha: Optional[float] = 0.8,
    volume_smoothness: Optional[int] = 200,
) -> UnstructuredGrid:
    """
    Construct a volumetric mesh based on surface mesh.

    Args:
        surf: A surface mesh.
        voxel_pc: A voxelized point cloud which contains the `pc.cell_data['obs_index']` and `pc.cell_data[key_added]`.
        key_added: The key under which to add the labels.
        color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.8`.
        volume_smoothness: The smoothness of the volumetric mesh.

    Returns:
        volume: A reconstructed volumetric mesh, which contains the following properties:
            `volume.cell_data[key_added]`, the "volume" array;
            `volume.cell_data[f'{key_added}_rgba']`,  the rgba colors of the "volume" array.
            `volume.cell_data['obs_index']`, the cell labels if not (voxel_pc is None).
    """

    density = surf.length / volume_smoothness
    volume = pv.voxelize(surf, density=density, check_surface=False)

    # Add labels and the colormap of the volumetric mesh
    labels = np.array(["volume"] * volume.n_cells).astype(str)
    add_mesh_labels(
        mesh=volume,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
    )
    if not (voxel_pc is None):
        volume.cell_data["obs_index"] = np.asarray(["no_cell"] * volume.n_cells).astype(str)
        volume = merge_mesh(meshes=[volume, voxel_pc])

    return volume
