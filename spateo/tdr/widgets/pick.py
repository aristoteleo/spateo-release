from typing import Optional, Union

import numpy as np
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from ..models import collect_models, multiblock2model
from .utils import _interactive_plotter

##################################
# Picking by entering parameters #
##################################


def three_d_pick(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    picked_groups: Union[str, list] = None,
) -> MultiBlock:
    """Pick the desired groups."""
    # Check input model.
    model = multiblock2model(model=model, message=None) if isinstance(model, MultiBlock) else model.copy()

    # Pick groups.
    picked_groups = [picked_groups] if isinstance(picked_groups, str) else picked_groups
    picked_models = []
    for group in picked_groups:
        if key in model.point_data.keys():
            picked_model = model.extract_points(model.point_data[key] == group)
        elif key in model.cell_data.keys():
            picked_model = model.extract_cells(model.cell_data[key] == group)
        else:
            raise ValueError(
                "`key` value is wrong." "\nThe `key` value must be contained in model.point_data or model.cell_data."
            )

        if isinstance(model, PolyData):
            picked_model = picked_model.extract_surface()

        picked_models.append(picked_model)

    return collect_models(picked_models)


#################################
# Picking by interactive window #
#################################


def _interactive_pick(
    plotter,
    model,
    picking_list: Optional[list] = None,
    key: str = "groups",
    label_size: int = 12,
    checkbox_size: int = 27,
    checkbox_color: Union[str, tuple, list] = "blue",
    checkbox_position: tuple = (5.0, 5.0),
):
    """Add a checkbox button widget to the scene."""

    def toggle_vis(flag):
        actor.SetVisibility(flag)
        if picking_list is not None:
            if flag is True:
                picking_list.append(model)
            elif flag is False and model in picking_list:
                picking_list.remove(model)

    # Make a separate callback for each widget
    actor = plotter.add_mesh(
        model,
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
        f"\n     {model[key][0]}",
        position=checkbox_position,
        font_size=label_size,
        color="black",
        font="arial",
    )


def interactive_pick(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    checkbox_size: int = 27,
    label_size: int = 12,
) -> MultiBlock:
    """
    Add a checkbox button widget to pick the desired groups through the interactive window and output the picked groups.

    Args:
        model: Reconstructed 3D model.
        key: The key under which are the groups.
        checkbox_size: The size of the button in number of pixels.
        label_size: The font size of the checkbox labels.

    Returns:
        A MultiBlock that contains all models you picked.
    """
    # Check input model.
    model = multiblock2model(model=model, message=None) if isinstance(model, MultiBlock) else model.copy()

    label_models = []
    if key in model.point_data.keys():
        for label in np.unique(model.point_data[key]):
            label_model = model.remove_points(model.point_data[key] != label)
            print(label_model[0])
            label_models.append(label_model[0])
    elif key in model.cell_data.keys():
        for label in np.unique(model.cell_data[key]):
            label_models.append(model.remove_cells(model.cell_data[key] != label))
    else:
        raise ValueError(
            "`key` value is wrong." "\nThe `key` value must be contained in model.point_data or model.cell_data."
        )

    picked_models = label_models.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter()

    checkbox_start_pos = 5.0
    for label_model in label_models:
        _interactive_pick(
            plotter=p,
            model=label_model,
            picking_list=picked_models,
            key=key,
            label_size=label_size,
            checkbox_size=checkbox_size,
            checkbox_color=tuple(label_model[f"{key}_rgba"][0][:3]),
            checkbox_position=(5.0, checkbox_start_pos),
        )
        checkbox_start_pos = checkbox_start_pos + checkbox_size + (checkbox_size // 10)
    p.show(cpos="iso")

    return collect_models(picked_models)


#############################
# Picking overlapping parts #
#############################


def overlap_pc_pick(
    pc: PolyData,
    mesh: PolyData,
) -> [PolyData, PolyData]:
    """
    Pick the point cloud inside the mesh model and point cloud outside the mesh model.

    Args:
        pc: Reconstructed 3D point cloud model corresponding to mesh.
        mesh: Reconstructed 3D mesh model.

    Returns:
        inside_pc: Point cloud inside the mesh model.
        outside_pc: Point cloud outside the mesh model.
    """

    select_pc = pc.select_enclosed_points(surface=mesh, check_surface=False)
    inside_pc = select_pc.threshold(0.5, scalars="SelectedPoints")
    outside_pc = select_pc.threshold(0.5, invert=True, scalars="SelectedPoints")

    return inside_pc, outside_pc


def overlap_mesh_pick(
    mesh1: PolyData,
    mesh2: PolyData,
) -> PolyData:
    """
    Pick the intersection between two mesh models.

    Args:
        mesh1: Reconstructed 3D mesh model.
        mesh2: Reconstructed 3D mesh model.

    Returns:
        select_mesh: The intersection mesh model.
    """

    select_mesh = mesh1.boolean_intersection(mesh2)

    return select_mesh


def overlap_pick(
    main_mesh: PolyData,
    other_mesh: PolyData,
    main_pc: Optional[PolyData] = None,
    other_pc: Optional[PolyData] = None,
):
    """
    Add a checkbox button widget to pick the desired groups through the interactive window and output the picked groups.

    Args:
        main_mesh: Reconstructed 3D mesh model.
        other_mesh: Reconstructed 3D mesh model.
        main_pc: Reconstructed 3D point cloud model corresponding to main_mesh.
        other_pc: Reconstructed 3D point cloud model corresponding to other_mesh.

    Returns:
        A MultiBlock that contains all models you picked.
    """

    select_mesh = overlap_mesh_pick(mesh1=main_mesh, mesh2=other_mesh)
    if not (main_pc is None and other_pc is None):
        inside_pc1, _ = overlap_pc_pick(pc=main_pc, mesh=select_mesh)
        inside_pc2, _ = overlap_pc_pick(pc=other_pc, mesh=select_mesh)
        select_pc = collect_models(models=[inside_pc1, inside_pc2])
    else:
        select_pc = None
    return select_mesh, select_pc
