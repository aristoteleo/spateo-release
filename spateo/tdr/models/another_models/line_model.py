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
    resolution: int = 1,
) -> PolyData:
    """
    Create a 3D line model.

    Args:
        start_point: Start location in [x, y, z] of the line.
        end_point: End location in [x, y, z] of the line.
        resolution: Number of pieces to divide line into.

    Returns:
        Line mesh.
    """

    line = pv.Line(pointa=start_point, pointb=end_point, resolution=resolution)
    return line


def construct_polyline(
    points: Union[list, np.ndarray],
) -> PolyData:
    """
    Create a 3D polyline model.

    Args:
        points: List of points defining a broken line.

    Returns:
        Line mesh.
    """

    line = pv.MultipleLines(points=points)
    return line


def construct_align_lines(
    model1_points: np.ndarray,
    model2_points: np.ndarray,
    key_added: str = "check_alignment",
    label: Union[str, list] = "align_mapping",
    color: str = "gainsboro",
) -> PolyData:
    """
    Construct alignment lines between models after model alignment.

    Args:
        model1_points: Start location in model1 of the line.
        model2_points: End location in model2 of the line.
        key_added: The key under which to add the labels.
        label: The label of alignment lines model.
        color: Color to use for plotting model.

    Returns:
        Alignment lines model.
    """
    assert model1_points.shape == model2_points.shape, "model1_points.shape is not equal to model2_points.shape"
    labels = [label] * model1_points.shape[0] if isinstance(label, str) else label

    lines_model = []
    for m1p, m2p, l in zip(model1_points, model2_points, labels):
        line = construct_line(start_point=m1p, end_point=m2p, resolution=1)
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


def construct_tree_model(
    nodes: np.ndarray,
    edges: np.ndarray,
    key_added: Optional[str] = "nodes",
) -> PolyData:
    """
    Construct a principal tree model.

    Args:
        nodes: The nodes in the principal tree.
        edges: The edges between nodes in the principal tree.
        key_added: The key under which to add the nodes labels.

    Returns:
         A three-dims principal tree model, which contains the following properties:
            `tree_model.point_data[key_added]`, the nodes labels array.
    """

    padding = np.array([2] * edges.shape[0], int)
    edges_w_padding = np.vstack((padding, edges.T)).T
    tree_model = pv.PolyData(nodes, edges_w_padding)
    tree_model.point_data[key_added] = np.arange(0, len(nodes), 1)

    return tree_model
