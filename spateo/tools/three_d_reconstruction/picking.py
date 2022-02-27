import warnings

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

from pyvista import PolyData, UnstructuredGrid
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .reconstruct_mesh import merge_mesh


def three_d_pick(
    mesh: Union[PolyData, UnstructuredGrid],
    key: str = "groups",
    pick_method: Literal["rectangle", "box"] = "rectangle",
    invert: bool = False,
    merge: bool = True,
) -> List[PolyData or UnstructuredGrid]:
    """
    Pick the interested part of a reconstructed 3D mesh by interactive approach.
    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
        pick_method: Pick the interested part of a mesh using a 2D rectangle widget or 3D box widget. Available `pick_method` are:
                * `'rectangle'`: Pick the interested part of a mesh using a 2D rectangle widget. Multiple meshes can be generated at the same time.
                * `'box'`: Pick the interested part of a mesh using a 3D box widget. Only one mesh can be generated.
        invert: Flag on whether to flip/invert the pick.
        merge: Flag on  whether to merge all picked meshes.
    Returns:
        A list of meshes or a merged mesh. If merge is True, return a merged mesh; else return a list of meshes.
    """

    if isinstance(mesh, UnstructuredGrid) is False:
        warnings.warn("The mesh should be a pyvista.UnstructuredGrid object.")
        mesh = mesh.cast_to_unstructured_grid()

    p = pv.Plotter()

    if pick_method == "rectangle":
        # Clip a mesh using a 2D rectangle widget.
        p.add_mesh(mesh, scalars=f"{key}_rgba", rgba=True)
        picked_meshes, invert_meshes, legend = [], [], []

        def split_mesh(original_mesh):
            """Adds a new mesh to the plotter each time cells are picked, and
            removes them from the original mesh"""

            # if nothing selected
            if not original_mesh.n_cells:
                return

            # remove the picked cells from main grid
            ghost_cells = np.zeros(mesh.n_cells, np.uint8)
            ghost_cells[original_mesh["orig_extract_id"]] = 1
            mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
            mesh.RemoveGhostCells()

            # add the selected mesh this to the main plotter
            color = np.random.random(3)
            legend.append(["picked mesh %d" % len(picked_meshes), color])
            p.add_mesh(original_mesh, color=color)
            p.add_legend(legend)

            # track the picked meshes and label them
            original_mesh["picked_index"] = np.ones(original_mesh.n_points) * len(picked_meshes)
            picked_meshes.append(original_mesh)

        p.enable_cell_picking(
            mesh=mesh,
            callback=split_mesh,
            show=False,
            font_size=12,
            show_message="Press `r` to enable retangle based selection. Press `r` again to turn it off. ",
        )
        p.show()
        picked_meshes = [invert_meshes[0]] if invert else picked_meshes
    else:
        # Clip a mesh using a 3D box widget.
        p.add_mesh_clip_box(mesh, invert=invert, scalars=f"{key}_rgba", rgba=True)
        p.show()
        picked_meshes = p.box_clipped_meshes

    if merge:
        return merge_mesh(picked_meshes) if len(picked_meshes) > 1 else picked_meshes[0]
    else:
        return picked_meshes
