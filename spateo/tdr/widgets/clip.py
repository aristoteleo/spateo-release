from typing import Union

import numpy as np
import vtk
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from ..models import collect_models, multiblock2model
from .pick import _interactive_pick
from .utils import _interactive_plotter


def _interactive_rectangle_clip(
    plotter,
    model,
    picking_list,
    picking_r_list,
):
    """Add a 2D rectangle widget to the scene."""

    legend = []

    def _split_model(original_model):
        """Adds a new model to the plotter each time cells are picked, and removes them from the original model"""

        # If nothing selected.
        if not original_model.n_cells:
            return

        # Remove the picked cells from main grid.
        ghost_cells = np.zeros(model.n_cells, np.uint8)
        ghost_cells[original_model["orig_extract_id"]] = 1
        model.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        model.RemoveGhostCells()

        # Add the selected model this to the main plotter.
        color = np.random.random(3)
        legend.append(["picked model %d" % len(picking_list), color])
        plotter.add_mesh(original_model, color=color)
        plotter.add_legend(legend, bcolor=None, face="circle", loc="lower right")

        # Track the picked models and label them.
        original_model["picked_index"] = np.ones(original_model.n_points) * len(picking_list)
        picking_list.append(original_model)
        picking_r_list.append(model)

    plotter.enable_cell_picking(mesh=model, callback=_split_model, show=False, color="black")

    plotter.add_text(
        "Please double-click the camera orientation widget in the upper right corner first."
        "\nPress `r` to enable retangle based selection. Press `r` again to turn it off."
        "\nPress `q` to exit the interactive window. ",
        font_size=12,
        color="black",
        font="arial",
    )


def interactive_rectangle_clip(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    invert: bool = False,
    bg_model=None,
) -> MultiBlock:
    """
    Pick the interested part of a model using a 2D rectangle widget.
    Multiple models can be generated at the same time.

    Args:
        model: Reconstructed 3D model.
        key: The key under which are the labels.
        invert: Flag on whether to flip/invert the pick.
        bg_model: A visualization-only background model to help clip our target model.
    Returns:
        picked_model: A MultiBlock that contains all models you picked.
    """
    # Check input model.
    model = (
        multiblock2model(model=model, message="rectangle clipping") if isinstance(model, MultiBlock) else model.copy()
    )

    # Create an interactive window for using widgets.
    p = _interactive_plotter()

    # Add a visualization-only background model with checkbox button widgets
    label_models = []
    if bg_model is not None:
        bg_model = multiblock2model(model=bg_model, message=None) if isinstance(bg_model, MultiBlock) else bg_model
        for label in np.unique(bg_model[key]):
            if label not in np.unique(model[key]):
                label_models.append(bg_model.remove_cells(bg_model[key] != label))

        checkbox_size = 27
        checkbox_start_pos = 5.0
        for label_model in label_models:
            _interactive_pick(
                plotter=p,
                model=label_model,
                key=key,
                checkbox_color=label_model[f"{key}_rgba"][0][:3],
                checkbox_position=(5.0, checkbox_start_pos),
            )
            checkbox_start_pos = checkbox_start_pos + checkbox_size + (checkbox_size // 10)

    # Clip model via a 2D rectangle widget.
    picked_models, picking_r_list = [], []
    p.add_mesh(
        model,
        scalars=f"{key}_rgba",
        rgba=True,
        render_points_as_spheres=True,
        point_size=10,
    )
    _interactive_rectangle_clip(
        plotter=p,
        model=model,
        picking_list=picked_models,
        picking_r_list=picking_r_list,
    )
    p.show(cpos="iso")

    # Obtain final picked model
    picked_model = collect_models([picking_r_list[0]]) if invert else collect_models(picked_models)

    return picked_model


def interactive_box_clip(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    invert: bool = False,
) -> MultiBlock:
    """
    Pick the interested part of a model using a 3D box widget. Only one model can be generated.

    Args:
        model: Reconstructed 3D model.
        key: The key under which are the labels.
        invert: Flag on whether to flip/invert the pick.
    Returns:
        picked_model: A MultiBlock that contains all models you picked.
    """
    # Check input model.
    model = multiblock2model(model=model, message="box clipping") if isinstance(model, MultiBlock) else model.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter()

    # Clip a model using a 3D box widget.
    p.add_mesh_clip_box(
        model,
        invert=invert,
        scalars=f"{key}_rgba",
        rgba=True,
        render_points_as_spheres=True,
        point_size=10,
        widget_color="black",
    )
    p.show(cpos="iso")

    # obtain final picked models
    picked_model = collect_models(models=p.box_clipped_meshes)

    return picked_model
