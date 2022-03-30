import numpy as np
import pyvista as pv

from pyvista import PolyData, UnstructuredGrid
from typing import Optional, Union

from .utils import add_mesh_labels


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
    add_mesh_labels(
        mesh=volume,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
    )

    return volume
