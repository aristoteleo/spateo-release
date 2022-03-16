import warnings

import numpy as np
import pyvista as pv
import vtk

from pyvista import Plotter, PolyData, UnstructuredGrid, MultiBlock
from typing import Union, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import collect_mesh, _MultiBlock


def _create_plotter(message=True) -> Plotter:
    """Create an interactive window for using widgets."""

    plotter = pv.Plotter(
        off_screen=False,
        window_size=(1024, 768),
        notebook=False,
        lighting="light_kit",
    )

    plotter.camera_position = "iso"
    plotter.background_color = "white"
    plotter.add_camera_orientation_widget()
    if message is True:
        plotter.add_text(
            "Please double-click the camera orientation widget in the upper right corner first.",
            font_size=15,
            color="black",
            font="arial",
        )

    return plotter


def _show_final(mesh, key, legend=None, message=True):
    """Create a presentation window to display the modified model results through the interactive window."""
    pf = _create_plotter(message=message)
    pf.add_mesh(mesh, scalars=f"{key}_rgba", rgba=True, render_points_as_spheres=True, point_size=10)

    if legend is not None:
        pf.add_legend(legend, bcolor=None, face="circle", loc="lower right")

    pf.show()


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
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock], key: str = "groups", invert: bool = False, bg_mesh=None
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
    mesh = _MultiBlock(mesh=mesh, message="rectangle clipping") if isinstance(mesh, MultiBlock) else mesh

    # Create an interactive window for using widgets.
    p = _create_plotter(message=False)

    # Add a A visualization-only background mesh with checkbox button widgets
    label_meshes = []
    if bg_mesh is not None:
        bg_mesh = _MultiBlock(mesh=bg_mesh, message=None) if isinstance(bg_mesh, MultiBlock) else bg_mesh
        for label in np.unique(bg_mesh[key]):
            if label not in np.unique(mesh[key]):
                label_meshes.append(bg_mesh.remove_cells(bg_mesh[key] != label))

        checkbox_size = 27
        checkbox_start_pos = 5.0
        for label_mesh in label_meshes:
            _interactive_checkbox_pick(
                plotter=p,
                mesh=label_mesh,
                key=key,
                checkbox_color=label_mesh[f"{key}_rgba"][0][:3],
                checkbox_position=(5.0, checkbox_start_pos),
            )
            checkbox_start_pos = checkbox_start_pos + checkbox_size + (checkbox_size // 10)

    # Clip mesh via a 2D rectangle widget.
    picked_meshes, picking_r_list = [], []
    p.add_mesh(mesh, scalars=f"{key}_rgba", rgba=True, render_points_as_spheres=True, point_size=10)
    _interactive_rectangle_clip(plotter=p, mesh=mesh, picking_list=picked_meshes, picking_r_list=picking_r_list)

    p.show()

    # Obtain and plot final picked mesh
    picked_mesh = collect_mesh([picking_r_list[0]]) if invert else collect_mesh(picked_meshes)
    _show_final(mesh=picked_mesh, key=key, legend=None, message=True)

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
    mesh = _MultiBlock(mesh=mesh, message="box clipping") if isinstance(mesh, MultiBlock) else mesh

    # Create an interactive window for using widgets.
    p = _create_plotter(message=True)

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

    # obtain amd plot final picked meshes
    picked_mesh = collect_mesh(p.box_clipped_meshes)
    _show_final(mesh=picked_mesh, key=key, legend=None, message=True)

    return picked_mesh


def interactive_slice(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    method: Literal["axis", "orthogonal"] = "axis",
    axis: Literal["x", "y", "z"] = "x",
) -> MultiBlock:
    """
    Create a slice of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.

    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
        method: The methods of slicing a mesh. Available `method` are:
                * `'axis'`: Create a slice of the input dataset along a specified axis.
                * `'orthogonal'`: Create three orthogonal slices through the dataset on the three cartesian planes.
        axis: The axis to generate the slices along. Only works when `method` is 'axis'.
    Returns:
        sliced_mesh: A MultiBlock that contains all meshes you sliced.
    """
    # Check input mesh.
    mesh = _MultiBlock(mesh=mesh, message="slicing") if isinstance(mesh, MultiBlock) else mesh

    # Create an interactive window for using widgets.
    p = _create_plotter(message=True)

    # Slice a mesh using a slicing widget.
    if method == "axis":
        p.add_mesh_slice(mesh, normal=axis, scalars=f"{key}_rgba", rgba=True, tubing=True, widget_color="black")
    else:
        p.add_mesh_slice_orthogonal(mesh, scalars=f"{key}_rgba", rgba=True, tubing=True, widget_color="black")
    p.show()

    # Obtain and plot final picked meshes.
    sliced_mesh = collect_mesh(p.plane_sliced_meshes)
    _show_final(mesh=sliced_mesh, key=key, legend=None, message=True)

    return sliced_mesh


def _interactive_checkbox_pick(
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
    actor = plotter.add_mesh(mesh, scalars=f"{key}_rgba", rgba=True, render_points_as_spheres=True, point_size=10)
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
        f"\n     {mesh[key][0]}", position=checkbox_position, font_size=label_size, color="black", font="arial"
    )


def interactive_checkbox_pick(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    checkbox_size: int = 27,
    label_size: int = 12,
) -> MultiBlock:
    """
    Add a checkbox button widget to pick the desired groups through the interactive window and output the pick groups.

    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the groups.
        checkbox_size: The size of the button in number of pixels.
        label_size: The font size of the checkbox labels.
    Returns:
        A MultiBlock that contains all meshes you picked.
    """
    # Check input mesh.
    mesh = _MultiBlock(mesh=mesh, message=None) if isinstance(mesh, MultiBlock) else mesh

    label_meshes = []
    for label in np.unique(mesh[key]):
        label_meshes.append(mesh.remove_cells(mesh[key] != label))

    picked_meshes = label_meshes.copy()

    # Create an interactive window for using widgets.
    p = _create_plotter(message=False)

    checkbox_start_pos = 5.0
    for label_mesh in label_meshes:
        _interactive_checkbox_pick(
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
