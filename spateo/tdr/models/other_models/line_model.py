from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import add_model_labels, collect_model, merge_models


def construct_line(
    start_point: Union[list, tuple, np.ndarray],
    end_point: Union[list, tuple, np.ndarray],
    style: Literal["line", "arrow"] = "line",
) -> PolyData:
    """
    Create a 3D line model.

    Args:
        start_point: Start location in [x, y, z] of the line.
        end_point: End location in [x, y, z] of the line.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.

    Returns:
        Line mesh.
    """

    if style == "line":
        model = pv.Line(pointa=start_point, pointb=end_point, resolution=1)
    elif style == "arrow":
        model = pv.Arrow(
            start=start_point,
            direction=end_point - start_point,
            scale="auto",
            tip_length=0.1,
            tip_radius=0.02,
            shaft_radius=0.01,
        )
    else:
        raise ValueError("`style` value is wrong.")

    return model


def construct_polyline(
    points: Union[list, np.ndarray],
    style: Literal["line", "arrow"] = "line",
) -> PolyData:
    """
    Create a 3D polyline model.

    Args:
        points: List of points defining a broken line.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.

    Returns:
        Line mesh.
    """

    if style == "line":
        model = pv.MultipleLines(points=points)
    elif style == "arrow":
        arrows = [
            construct_line(start_point=start_point, end_point=end_point, style=style)
            for start_point, end_point in zip(points[:-1], points[1:])
        ]
        model = merge_models(models=arrows)
    else:
        raise ValueError("`style` value is wrong.")

    return model


def construct_align_lines(
    model1_points: np.ndarray,
    model2_points: np.ndarray,
    key_added: str = "check_alignment",
    label: Union[str, list] = "align_mapping",
    color: str = "gainsboro",
    style: Literal["line", "arrow"] = "line",
) -> PolyData:
    """
    Construct alignment lines between models after model alignment.

    Args:
        model1_points: Start location in model1 of the line.
        model2_points: End location in model2 of the line.
        key_added: The key under which to add the labels.
        label: The label of alignment lines model.
        color: Color to use for plotting model.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.

    Returns:
        Alignment lines model.
    """
    assert model1_points.shape == model2_points.shape, "model1_points.shape is not equal to model2_points.shape"
    labels = [label] * model1_points.shape[0] if isinstance(label, str) else label

    lines_model = []
    for m1p, m2p, l in zip(model1_points, model2_points, labels):
        line = construct_line(start_point=m1p, end_point=m2p, style=style)
        add_model_labels(
            model=line,
            key_added=key_added,
            labels=np.asarray([l] * line.n_points),
            where="point_data",
            colormap=color,
            inplace=True,
        )
        lines_model.append(line)

    lines_model = merge_models(lines_model)
    return lines_model


def construct_axis_line(
    axis_points: np.ndarray,
    key_added: str = "axis",
    label: str = "axis_line",
    color: str = "gainsboro",
    style: Literal["line", "arrow"] = "line",
) -> PolyData:
    """
    Construct axis line.

    Args:
        axis_points:  List of points defining an axis.
        key_added: The key under which to add the labels.
        label: The label of axis line model.
        color: Color to use for plotting model.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.

    Returns:
        Axis line model.
    """

    axis_line = construct_polyline(points=axis_points, style=style)
    add_model_labels(
        model=axis_line,
        key_added=key_added,
        labels=np.asarray([label] * axis_line.n_points),
        where="point_data",
        colormap=color,
        inplace=True,
    )
    return axis_line
