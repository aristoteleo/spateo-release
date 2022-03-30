from typing import Union

import numpy as np
import vtk
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from ..mesh.utils import collect_mesh, multiblock2mesh
from .pick import _interactive_pick
from .utils import _interactive_plotter


def _interactive_rectangle_clip(
    plotter,
    mesh,
    picking_list,
    picking_r_list,
):
    """Add a 2D rectangle widget to the scene."""

    legend = []

    def _split_mesh(original_mesh):
        """Adds a new mesh to the plotter each time cells are picked, and removes them from the original mesh"""

        # If nothing selected.
        if not original_mesh.n_cells:
            return

        # Remove the picked cells from main grid.
        ghost_cells = np.zeros(mesh.n_cells, np.uint8)
        ghost_cells[original_mesh["orig_extract_id"]] = 1
        mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        mesh.RemoveGhostCells()

        # Add the selected mesh this to the main plotter.
        color = np.random.random(3)
        legend.append(["picked mesh %d" % len(picking_list), color])
        plotter.add_mesh(original_mesh, color=color)
        plotter.add_legend(legend, bcolor=None, face="circle", loc="lower right")

        # Track the picked meshes and label them.
        original_mesh["picked_index"] = np.ones(original_mesh.n_points) * len(picking_list)
        picking_list.append(original_mesh)
        picking_r_list.append(mesh)

    plotter.enable_cell_picking(mesh=mesh, callback=_split_mesh, show=False, color="black")

    plotter.add_text(
        "Please double-click the camera orientation widget in the upper right corner first."
        "\nPress `r` to enable retangle based selection. Press `r` again to turn it off."
        "\nPress `q` to exit the interactive window. ",
        font_size=12,
        color="black",
        font="arial",
    )


def interactive_rectangle_clip(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    invert: bool = False,
    bg_mesh=None,
) -> MultiBlock:
    """
    Pick the interested part of a mesh using a 2D rectangle widget.
    Multiple meshes can be generated at the same time.

    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
        invert: Flag on whether to flip/invert the pick.
        bg_mesh: A visualization-only background mesh to help clip our target mesh.
    Returns:
        picked_mesh: A MultiBlock that contains all meshes you picked.
    """
    # Check input mesh.
    mesh = multiblock2mesh(mesh=mesh, message="rectangle clipping") if isinstance(mesh, MultiBlock) else mesh.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter(message=True)

    # Add a visualization-only background mesh with checkbox button widgets
    label_meshes = []
    if bg_mesh is not None:
        bg_mesh = multiblock2mesh(mesh=bg_mesh, message=None) if isinstance(bg_mesh, MultiBlock) else bg_mesh
        for label in np.unique(bg_mesh[key]):
            if label not in np.unique(mesh[key]):
                label_meshes.append(bg_mesh.remove_cells(bg_mesh[key] != label))

        checkbox_size = 27
        checkbox_start_pos = 5.0
        for label_mesh in label_meshes:
            _interactive_pick(
                plotter=p,
                mesh=label_mesh,
                key=key,
                checkbox_color=label_mesh[f"{key}_rgba"][0][:3],
                checkbox_position=(5.0, checkbox_start_pos),
            )
            checkbox_start_pos = checkbox_start_pos + checkbox_size + (checkbox_size // 10)

    # Clip mesh via a 2D rectangle widget.
    picked_meshes, picking_r_list = [], []
    p.add_mesh(
        mesh,
        scalars=f"{key}_rgba",
        rgba=True,
        render_points_as_spheres=True,
        point_size=10,
    )
    _interactive_rectangle_clip(plotter=p, mesh=mesh, picking_list=picked_meshes, picking_r_list=picking_r_list)
    p.show()

    # Obtain final picked mesh
    picked_mesh = collect_mesh([picking_r_list[0]]) if invert else collect_mesh(picked_meshes)

    return picked_mesh


def interactive_box_clip(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    invert: bool = False,
) -> MultiBlock:
    """
    Pick the interested part of a mesh using a 3D box widget. Only one mesh can be generated.

    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
        invert: Flag on whether to flip/invert the pick.
    Returns:
        picked_mesh: A MultiBlock that contains all meshes you picked.
    """
    # Check input mesh.
    mesh = multiblock2mesh(mesh=mesh, message="box clipping") if isinstance(mesh, MultiBlock) else mesh.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter(message=True)

    # Clip a mesh using a 3D box widget.
    p.add_mesh_clip_box(
        mesh,
        invert=invert,
        scalars=f"{key}_rgba",
        rgba=True,
        render_points_as_spheres=True,
        point_size=10,
        widget_color="black",
    )
    p.show()

    # obtain final picked meshes
    picked_mesh = collect_mesh(p.box_clipped_meshes)

    return picked_mesh
