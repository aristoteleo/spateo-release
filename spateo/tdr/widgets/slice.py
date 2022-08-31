import math
from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import MultiBlock, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ...logging import logger_manager as lm
from ..models import collect_models, multiblock2model
from .utils import _interactive_plotter

###############
# Create line #
###############


def find_plane_equation(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray):
    xo1, yo1, zo1 = point1
    xo2, yo2, zo2 = point2
    xo3, yo3, zo3 = point3

    a = (yo2 - yo1) * (zo3 - zo1) - (zo2 - zo1) * (yo3 - yo1)
    b = (xo3 - xo1) * (zo2 - zo1) - (xo2 - xo1) * (zo3 - zo1)
    c = (xo2 - xo1) * (yo3 - yo1) - (xo3 - xo1) * (yo2 - yo1)
    d = -(a * xo1 + b * yo1 + c * zo1)

    equation_args = np.array([a, b, c, d])
    return equation_args


def find_model_outline_planes(model) -> dict:
    x1, x2, y1, y2, z1, z2 = model.bounds
    vertices = np.asarray(
        [
            [x1, y1, z1],
            [x1, y1, z2],
            [x1, y2, z1],
            [x1, y2, z2],
            [x2, y1, z1],
            [x2, y1, z2],
            [x2, y2, z1],
            [x2, y2, z2],
        ]
    )

    x_plane = find_plane_equation(point1=vertices[0, :], point2=vertices[1, :], point3=vertices[2, :])
    x_plane_opposite = find_plane_equation(point1=vertices[4, :], point2=vertices[5, :], point3=vertices[6, :])
    y_plane = find_plane_equation(point1=vertices[0, :], point2=vertices[1, :], point3=vertices[4, :])
    y_plane_opposite = find_plane_equation(point1=vertices[2, :], point2=vertices[3, :], point3=vertices[6, :])
    z_plane = find_plane_equation(point1=vertices[0, :], point2=vertices[2, :], point3=vertices[4, :])
    z_plane_opposite = find_plane_equation(point1=vertices[1, :], point2=vertices[3, :], point3=vertices[5, :])

    planes = {
        "x": [x_plane, x_plane_opposite],
        "y": [y_plane, y_plane_opposite],
        "z": [z_plane, z_plane_opposite],
    }
    return planes


def find_intersection(model, vec, center, plane):

    # Normalize the vector
    normal = vec / np.linalg.norm(vec)
    normal_x, normal_y, normal_z = normal

    center = model.center if center is None else center
    center_x, center_y, center_z = center

    a, b, c, d = plane
    t = (-a * center_x - b * center_y - c * center_z - d) / (a * normal_x + b * normal_y + c * normal_z)

    intersection_x = normal_x * t + center_x
    intersection_y = normal_y * t + center_y
    intersection_z = normal_z * t + center_z
    intersection = np.asarray([intersection_x, intersection_y, intersection_z])
    return intersection


def euclidean_distance(instance1, instance2, dimension):
    distance = 0
    for i in range(dimension):
        distance += (instance1[i] - instance2[i]) ** 2

    return math.sqrt(distance)


def create_line(model, vec, center, n_points):
    planes = find_model_outline_planes(model=model)

    line_dict = {}
    for key, value in planes.items():
        center = model.center if center is None else center
        intersection1 = find_intersection(model=model, vec=vec, center=center, plane=value[0])
        intersection2 = find_intersection(model=model, vec=vec, center=center, plane=value[1])

        ed = euclidean_distance(instance1=model.center, instance2=intersection2, dimension=3)
        if not math.isnan(ed):
            line_dict[ed] = [intersection1, intersection2]

    if len(line_dict.keys()) != 0:
        min_ed = np.min([i for i in line_dict.keys()])

        intersection1 = line_dict[min_ed][0]
        intersection2 = line_dict[min_ed][1]
        # line_new = pv.Line(intersection1, intersection2, n_points - 1)
        line = pv.Line(intersection1, intersection2, n_points + 1)
        line_new = pv.Line(line.points[1], line.points[-2], n_points - 1)

        return line_new, intersection1, intersection2
    else:
        raise ValueError("`vec` value is wrong. Please input another `vec` value.")


#########
# Slice #
#########


def three_d_slice(
    model: Union[PolyData, UnstructuredGrid],
    method: Literal["axis", "orthogonal", "line"] = "axis",
    n_slices: int = 10,
    axis: Literal["x", "y", "z"] = "x",
    vec: Union[tuple, list] = (1, 0, 0),
    center: Union[tuple, list] = None,
) -> Union[PolyData, Tuple[MultiBlock, MultiBlock, PolyData]]:
    """
    Create many slices of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes or
    slice a model along a vector direction perpendicularly.

    Args:
        model: Reconstructed 3D model.
        method: The methods of slicing a model. Available `method` are:
                * `'axis'`: Create many slices of the input dataset along a specified axis.
                * `'orthogonal'`: Create three orthogonal slices through the dataset on the three cartesian planes.
                                  This method is usually used interactively without entering a position which slices are taken.
                * `'line'`: Slice a model along a vector direction perpendicularly.
        n_slices: The number of slices to create along a specified axis. Only works when `method` is 'axis' or 'line'.
        axis: The axis to generate the slices along. Only works when `method` is 'axis'.
        vec: The vector direction. Only works when `method` is 'line'.
        center: A 3-length sequence specifying the position which slices are taken. Defaults to the center of the model.

    Returns:
        If method is 'axis' or 'orthogonal', return a MultiBlock that contains all models you sliced; else return a
        tuple that contains line model, all models you sliced and intersections of slices model and line model.
    """
    # Check input model.
    model = multiblock2model(model=model, message="slicing") if isinstance(model, MultiBlock) else model.copy()
    center = model.center if center is None else center

    if method == "axis":
        # Create many slices of the input dataset along a specified axis.
        return model.slice_along_axis(n=n_slices, axis=axis, center=center)
    elif method == "orthogonal":
        # Create three orthogonal slices through the dataset on the three cartesian planes.
        return model.slice_orthogonal(x=center[0], y=center[1], z=center[2])
    elif method == "line":
        # Slice a model along a vector direction perpendicularly.
        line, intersection1, intersection2 = create_line(model=model, vec=vec, center=center, n_points=n_slices)

        slices = pv.MultiBlock()
        line_points = pv.MultiBlock()
        for point in line.points:
            normal = vec / np.linalg.norm(vec)
            point_slice = model.slice(normal=normal, origin=point)

            if point_slice.n_points != 0:
                slices.append(point_slice)
                line_points.append(point)

        lm.main_info(
            f"Slice the model uniformly along the vector `vec` and generate {n_slices} slices. "
            f"There are {n_slices-len(slices)} empty slices, {len(slices)} valid slices in all slices.",
            indent_level=1,
        )

        return slices, line_points, line
    else:
        raise ValueError("`method` value is wrong. " "\nAvailable `method` are: `'axis'`, `'orthogonal'`, `'line'`.")


#####################
# Interactive slice #
#####################


def interactive_slice(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    method: Literal["axis", "orthogonal"] = "axis",
    axis: Literal["x", "y", "z"] = "x",
) -> MultiBlock:
    """
    Create a slice of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.

    Args:
        model: Reconstructed 3D model.
        key: The key under which are the labels.
        method: The methods of slicing a model. Available `method` are:
                * `'axis'`: Create a slice of the input dataset along a specified axis.
                * `'orthogonal'`: Create three orthogonal slices through the dataset on the three cartesian planes.
        axis: The axis to generate the slices along. Only works when `method` is 'axis'.

    Returns:
        sliced_model: A MultiBlock that contains all models you sliced.
    """
    # Check input model.
    model = multiblock2model(model=model, message="slicing") if isinstance(model, MultiBlock) else model.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter()
    p.add_mesh(model, opacity=0.2, scalars=f"{key}_rgba", rgba=True, style="surface")
    # Slice a model using a slicing widget.
    if method == "axis":
        p.add_mesh_slice(
            model,
            normal=axis,
            scalars=f"{key}_rgba",
            rgba=True,
            tubing=True,
            widget_color="black",
            color="black",
            line_width=3.0,
        )
    else:
        p.add_mesh_slice_orthogonal(
            model,
            scalars=f"{key}_rgba",
            rgba=True,
            tubing=True,
            widget_color="black",
            color="black",
            line_width=3.0,
        )
    p.show(cpos="iso")

    return collect_models(p.plane_sliced_meshes)
