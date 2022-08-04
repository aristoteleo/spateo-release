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
    key_added: str = "line",
    label: str = "line",
    color: str = "gainsboro",
) -> PolyData:
    """
    Create a 3D line model.

    Args:
        start_point: Start location in [x, y, z] of the line.
        end_point: End location in [x, y, z] of the line.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.
        key_added: The key under which to add the labels.
        label: The label of lines model.
        color: Color to use for plotting model.

    Returns:
        Line model.
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

    add_model_labels(
        model=model,
        key_added=key_added,
        labels=np.asarray([label] * model.n_cells),
        where="cell_data",
        colormap=color,
        inplace=True,
    )

    return model


def construct_multi_lines(
    points: np.ndarray,
    edges: np.ndarray,
    key_added: str = "line",
    label: Union[str, list, np.ndarray] = "multi_lines",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
) -> PolyData:
    """
    Create multiple 3D lines model.

    Args:
        points: List of points.
        edges: The edges between points.
        key_added: The key under which to add the labels.
        label: The label of line models.
        color: Color to use for plotting model.

    Returns:
        Multiple-lines model.
    """

    padding = np.array([2] * edges.shape[0], int)
    edges_w_padding = np.vstack((padding, edges.T)).T
    model = pv.PolyData(points, edges_w_padding)

    labels = np.asarray([label] * edges.shape[0]) if isinstance(label, str) else label
    assert len(labels) == edges.shape[0], "The number of labels is not equal to the number of edges."

    add_model_labels(
        model=model,
        key_added=key_added,
        labels=labels,
        where="cell_data",
        colormap=color,
        inplace=True,
    )

    return model


def construct_multi_arrows(
    points: np.ndarray,
    edges: np.ndarray,
    key_added: str = "arrow",
    label: Union[str, list, np.ndarray] = "multi_arrows",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
) -> PolyData:
    """
    Create multiple 3D arrows model.

    Args:
        points: List of points.
        edges: The edges between points.
        key_added: The key under which to add the labels.
        label: The label of arrow model.
        color: Color to use for plotting model.

    Returns:
        Multiple-arrows model.
    """

    labels = np.asarray([label] * edges.shape[0]) if isinstance(label, str) else label
    assert len(labels) == edges.shape[0], "The number of labels is not equal to the number of edges."

    arrows = [
        construct_line(
            start_point=points[edge[0]],
            end_point=points[edge[1]],
            style="arrow",
            key_added=key_added,
            label=l,
            color=color,
        )
        for edge, l in zip(edges, labels)
    ]

    model = merge_models(models=arrows)
    return model


def construct_tree(
    points: np.ndarray,
    edges: np.ndarray,
    style: Literal["line", "arrow"] = "line",
    key_added: str = "tree",
    label: Union[str, list, np.ndarray] = "tree",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
) -> PolyData:
    """
    Create a 3D tree model of multiple discontinuous line segments.

    Args:
        points: List of points defining a tree.
        edges: The edges between points in the tree.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.
        key_added: The key under which to add the labels.
        label: The label of tree model.
        color: Color to use for plotting model.

    Returns:
        Tree model.
    """

    if style == "line":
        model = construct_multi_lines(points=points, edges=edges, key_added=key_added, label=label, color=color)
    elif style == "arrow":
        model = construct_multi_arrows(points=points, edges=edges, key_added=key_added, label=label, color=color)
    else:
        raise ValueError("`style` value is wrong.")

    return model


def construct_align_lines(
    model1_points: np.ndarray,
    model2_points: np.ndarray,
    style: Literal["line", "arrow"] = "line",
    key_added: str = "check_alignment",
    label: Union[str, list, np.ndarray] = "align_mapping",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
) -> PolyData:
    """
    Construct alignment lines between models after model alignment.

    Args:
        model1_points: Start location in model1 of the line.
        model2_points: End location in model2 of the line.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.
        key_added: The key under which to add the labels.
        label: The label of alignment lines model.
        color: Color to use for plotting model.

    Returns:
        Alignment lines model.
    """

    assert model1_points.shape == model2_points.shape, "model1_points.shape is not equal to model2_points.shape"
    spoints_shape = model1_points.shape[0]
    epoints_shape = model2_points.shape[0]

    points = np.concatenate([model1_points, model2_points], axis=0)
    edges_x = np.arange(spoints_shape).reshape(-1, 1)
    edges_y = np.arange(spoints_shape, spoints_shape + epoints_shape).reshape(-1, 1)
    edges = np.concatenate([edges_x, edges_y], axis=1)

    model = construct_tree(points=points, edges=edges, style=style, key_added=key_added, label=label, color=color)
    return model


def construct_axis_line(
    axis_points: np.ndarray,
    style: Literal["line", "arrow"] = "line",
    key_added: str = "axis",
    label: str = "axis_line",
    color: str = "gainsboro",
) -> PolyData:
    """
    Construct axis line.

    Args:
        axis_points: List of points defining an axis.
        style: Line style. According to whether there is an arrow, it is divided into `'line'` and `'arrow'`.
        key_added: The key under which to add the labels.
        label: The label of axis line model.
        color: Color to use for plotting model.

    Returns:
        Axis line model.
    """

    start_point = axis_points.min(axis=0)
    end_point = axis_points.max(axis=0)
    axis_line = construct_line(
        start_point=start_point, end_point=end_point, style=style, key_added=key_added, label=label, color=color
    )

    return axis_line
