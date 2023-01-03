from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import add_model_labels


def _construct_line(
    start_point: Union[list, tuple, np.ndarray] = (-0.5, 0.0, 0.0),
    end_point: Union[list, tuple, np.ndarray] = (0.5, 0.0, 0.0),
) -> PolyData:
    """
    Create a 3D line model.

    Args:
        start_point: Start location in [x, y, z] of the line.
        end_point: End location in [x, y, z] of the line.

    Returns:
        Line model.
    """

    return pv.Line(pointa=start_point, pointb=end_point, resolution=1)


def construct_line(
    start_point: Union[list, tuple, np.ndarray],
    end_point: Union[list, tuple, np.ndarray],
    key_added: Optional[str] = "line",
    label: str = "line",
    color: str = "gainsboro",
    alpha: float = 1.0,
) -> PolyData:
    """
    Create a 3D line model.

    Args:
        start_point: Start location in [x, y, z] of the line.
        end_point: End location in [x, y, z] of the line.
        key_added: The key under which to add the labels.
        label: The label of line model.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        Line model.
    """

    model = _construct_line(start_point=start_point, end_point=end_point)

    if not (key_added is None):
        add_model_labels(
            model=model,
            key_added=key_added,
            labels=np.asarray([label] * model.n_points),
            where="point_data",
            colormap=color,
            alphamap=alpha,
            inplace=True,
        )

    return model


def construct_lines(
    points: np.ndarray,
    edges: np.ndarray,
    key_added: Optional[str] = "line",
    label: Union[str, list, np.ndarray] = "lines",
    color: Union[str, list, dict] = "gainsboro",
    alpha: Union[float, int, list, dict] = 1.0,
) -> PolyData:
    """
    Create 3D lines model.

    Args:
        points: List of points.
        edges: The edges between points.
        key_added: The key under which to add the labels.
        label: The label of lines model.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        Lines model.
    """

    padding = np.array([2] * edges.shape[0], int)
    edges_w_padding = np.vstack((padding, edges.T)).T
    model = pv.PolyData(points, edges_w_padding)

    labels = np.asarray([label] * points.shape[0]) if isinstance(label, str) else np.asarray(label)
    assert len(labels) == points.shape[0], "The number of labels is not equal to the number of points."
    if not (key_added is None):
        add_model_labels(
            model=model,
            key_added=key_added,
            labels=labels,
            where="point_data",
            colormap=color,
            alphamap=alpha,
            inplace=True,
        )

    return model


def generate_edges(
    points1: np.ndarray,
    points2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    assert points1.shape == points2.shape, "points1.shape is not equal to points2.shape"
    spoints_shape = points1.shape[0]
    epoints_shape = points2.shape[0]

    points = np.concatenate([points1, points2], axis=0)
    edges_x = np.arange(spoints_shape).reshape(-1, 1)
    edges_y = np.arange(spoints_shape, spoints_shape + epoints_shape).reshape(-1, 1)
    edges = np.concatenate([edges_x, edges_y], axis=1)
    return points, edges


def construct_align_lines(
    model1_points: np.ndarray,
    model2_points: np.ndarray,
    key_added: str = "check_alignment",
    label: Union[str, list, np.ndarray] = "align_mapping",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
    alpha: Union[float, int, list, dict, np.ndarray] = 1.0,
) -> PolyData:
    """
    Construct alignment lines between models after model alignment.

    Args:
        model1_points: Start location in model1 of the line.
        model2_points: End location in model2 of the line.
        key_added: The key under which to add the labels.
        label: The label of alignment lines model.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        Alignment lines model.
    """

    points, edges = generate_edges(points1=model1_points, points2=model2_points)
    model = construct_lines(points=points, edges=edges, key_added=key_added, label=label, color=color, alpha=alpha)
    return model


def construct_axis_line(
    axis_points: np.ndarray,
    key_added: str = "axis",
    label: str = "axis_line",
    color: str = "gainsboro",
    alpha: Union[float, int, list, dict, np.ndarray] = 1.0,
) -> PolyData:
    """
    Construct axis line.

    Args:
        axis_points: List of points defining an axis.
        key_added: The key under which to add the labels.
        label: The label of axis line model.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.

    Returns:
        Axis line model.
    """

    start_point = axis_points.min(axis=0)
    end_point = axis_points.max(axis=0)
    axis_line = construct_line(
        start_point=start_point, end_point=end_point, key_added=key_added, label=label, color=color, alpha=alpha
    )

    return axis_line
