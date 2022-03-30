from typing import Optional, Union

import numpy as np
import pyvista as pv
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from ..mesh.utils import collect_mesh, multiblock2mesh
from .utils import _interactive_plotter


def three_d_pick(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    picked_groups: Union[str, list] = None,
) -> MultiBlock:
    """Pick the desired groups."""
    # Check input mesh.
    mesh = multiblock2mesh(mesh=mesh, message=None) if isinstance(mesh, MultiBlock) else mesh.copy()

    # Pick groups.
    picked_groups = [picked_groups] if isinstance(picked_groups, str) else picked_groups
    picked_meshes = []
    for group in picked_groups:
        if key in mesh.point_data.keys():
            picked_mesh = mesh.extract_points(mesh.point_data[key] == group)
        elif key in mesh.cell_data.keys():
            picked_mesh = mesh.extract_cells(mesh.cell_data[key] == group)
        else:
            raise ValueError(
                "\n`key` value is wrong." "\nThe `key` value must be contained in mesh.point_data or mesh.cell_data."
            )

        if isinstance(mesh, PolyData):
            picked_mesh_poly = pv.PolyData(picked_mesh.points, picked_mesh.cells)
            for pd_key in picked_mesh.point_data.keys():
                if pd_key != "vtkOriginalPointIds":
                    picked_mesh_poly.point_data[pd_key] = picked_mesh.point_data[pd_key]
            for cd_key in picked_mesh.cell_data.keys():
                if cd_key != "vtkOriginalCellIds":
                    picked_mesh_poly.cell_data[cd_key] = picked_mesh.cell_data[cd_key]
            picked_meshes.append(picked_mesh_poly)
        else:
            picked_meshes.append(picked_mesh)

    return collect_mesh(picked_meshes)


def _interactive_pick(
    plotter,
    mesh,
    picking_list: Optional[list] = None,
    key: str = "groups",
    label_size: int = 12,
    checkbox_size: int = 27,
    checkbox_color: Union[str, tuple] = "blue",
    checkbox_position: tuple = (5.0, 5.0),
):
    """Add a checkbox button widget to the scene."""

    def toggle_vis(flag):
        actor.SetVisibility(flag)
        if picking_list is not None:
            if flag is True:
                picking_list.append(mesh)
            elif flag is False and mesh in picking_list:
                picking_list.remove(mesh)

    # Make a separate callback for each widget
    actor = plotter.add_mesh(
        mesh,
        scalars=f"{key}_rgba",
        rgba=True,
        render_points_as_spheres=True,
        point_size=10,
    )
    plotter.add_checkbox_button_widget(
        toggle_vis,
        value=True,
        position=checkbox_position,
        size=checkbox_size,
        border_size=3,
        color_on=checkbox_color,
        color_off="white",
        background_color=checkbox_color,
    )

    plotter.add_text(
        f"\n     {mesh[key][0]}",
        position=checkbox_position,
        font_size=label_size,
        color="black",
        font="arial",
    )


def interactive_pick(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    checkbox_size: int = 27,
    label_size: int = 12,
) -> MultiBlock:
    """
    Add a checkbox button widget to pick the desired groups through the interactive window and output the picked groups.

    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the groups.
        checkbox_size: The size of the button in number of pixels.
        label_size: The font size of the checkbox labels.
    Returns:
        A MultiBlock that contains all meshes you picked.
    """
    # Check input mesh.
    mesh = multiblock2mesh(mesh=mesh, message=None) if isinstance(mesh, MultiBlock) else mesh.copy()

    label_meshes = []
    for label in np.unique(mesh[key]):
        label_meshes.append(mesh.remove_cells(mesh[key] != label))

    picked_meshes = label_meshes.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter(message=True)

    checkbox_start_pos = 5.0
    for label_mesh in label_meshes:
        _interactive_pick(
            plotter=p,
            mesh=label_mesh,
            picking_list=picked_meshes,
            key=key,
            label_size=label_size,
            checkbox_size=checkbox_size,
            checkbox_color=label_mesh[f"{key}_rgba"][0][:3],
            checkbox_position=(5.0, checkbox_start_pos),
        )
        checkbox_start_pos = checkbox_start_pos + checkbox_size + (checkbox_size // 10)
    p.show()

    return collect_mesh(picked_meshes)
