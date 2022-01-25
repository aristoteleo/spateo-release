import numpy as np
import pandas as pd
import pyvista as pv

from typing import Optional, Sequence, Tuple, Union


def compute_volume(
    mesh: Optional[pv.DataSet] = None,
    group_show: Union[str, list] = "all",
) -> float:
    """
    Calculate the volume of the reconstructed 3D structure.

    Args:
        mesh: Reconstructed 3D structure (voxelized object).
        group_show: Subset of groups used for calculation, e.g. [`'g1'`, `'g2'`, `'g3'`]. The default group_show is `'all'`, for all groups.

    Returns:
        volume_size: The volume of the reconstructed 3D structure.
    """

    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    volume_data = pd.concat(
        [pd.Series(mesh.cell_data["groups"]), pd.Series(mesh.cell_data["Volume"])],
        axis=1,
    )

    if group_show != "all":
        group_show = [group_show] if isinstance(group_show, str) else group_show
        volume_data = volume_data[volume_data[0].isin(group_show)]

    volume_size = float(np.sum(volume_data[1]))
    print(f"{group_show} volume: {volume_size}")

    return volume_size
