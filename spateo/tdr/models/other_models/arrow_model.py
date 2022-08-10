from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import add_model_labels


def _construct_arrow(
    start_point: Union[list, tuple, np.ndarray] = (0.0, 0.0, 0.0),
    direction: Union[list, tuple, np.ndarray] = (1.0, 0.0, 0.0),
    tip_length: float = 0.1,
    tip_radius: float = 0.02,
    tip_resolution: int = 20,
    shaft_radius: float = 0.01,
    shaft_resolution: int = 20,
    scale: Optional[Union[str, float]] = "auto",
) -> PolyData:
    """
    Create a 3D arrow model.

    Args:
        start_point: Start location in [x, y, z] of the arrow.
        direction: Direction the arrow points to in [x, y, z].
        tip_length: Length of the tip.
        tip_radius: Radius of the tip.
        tip_resolution: Number of faces around the tip.
        shaft_radius: Radius of the shaft.
        shaft_resolution: Number of faces around the shaft.
        scale: Scale factor of the entire object. 'auto' scales to length of direction array.

    Returns:
        Arrow model.
    """

    return pv.Arrow(
        start=start_point,
        direction=direction,
        tip_length=tip_length,
        tip_radius=tip_radius,
        tip_resolution=tip_resolution,
        shaft_radius=shaft_radius,
        shaft_resolution=shaft_resolution,
        scale=scale,
    )


def construct_arrow(
    start_point: Union[list, tuple, np.ndarray],
    direction: Union[list, tuple, np.ndarray],
    arrow_scale: Optional[Union[int, float]] = None,
    key_added: str = "arrow",
    label: str = "arrow",
    color: str = "gainsboro",
    **kwargs
) -> PolyData:
    """
    Create a 3D arrow model.

    Args:
        start_point: Start location in [x, y, z] of the arrow.
        direction: Direction the arrow points to in [x, y, z].
        arrow_scale: Scale factor of the entire object. 'auto' scales to length of direction array.
        key_added: The key under which to add the labels.
        label: The label of arrow model.
        color: Color to use for plotting model.

    Returns:
        Arrow model.
    """

    model = _construct_arrow(
        start_point=start_point,
        direction=direction,
        scale="auto" if arrow_scale is None else arrow_scale,
        **kwargs
    )

    add_model_labels(
        model=model,
        key_added=key_added,
        labels=np.asarray([label] * model.n_points),
        where="point_data",
        colormap=color,
        inplace=True,
    )

    return model


def construct_arrows(
    start_points: np.ndarray,
    direction: np.ndarray = None,
    arrows_scale: Optional[np.ndarray] = None,
    factor: float = 1.0,
    key_added: str = "arrow",
    label: Union[str, list, np.ndarray] = "arrows",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
) -> PolyData:
    """
    Create multiple 3D arrows model.

    Args:
        start_points: List of Start location in [x, y, z] of the arrows.
        direction: Direction the arrows points to in [x, y, z].
        arrows_scale: Scale factor of the entire object.
        factor: Scale factor applied to scaling array.
        key_added: The key under which to add the labels.
        label: The label of arrows model.
        color: Color to use for plotting model.

    Returns:
        Arrows model.
    """

    model = pv.PolyData(start_points)
    model.point_data["direction"] = direction
    model.point_data["scale"] = np.linalg.norm(direction.copy(), axis=1) if arrows_scale is None else arrows_scale

    glyph = model.glyph(orient="direction", geom=_construct_arrow(), scale="scale", factor=factor)

    labels = np.asarray([label] * glyph.n_points) if isinstance(label, str) else label
    assert len(labels) == glyph.n_points, "The number of labels is not equal to the number of edges."

    add_model_labels(
        model=glyph,
        key_added=key_added,
        labels=np.asarray([label] * glyph.n_points),
        where="point_data",
        colormap=color,
        inplace=True,
    )

    return glyph


def construct_vectorfield(
    model: PolyData,
    vector_key: str,
    arrows_scale_key: Optional[str] = None,
    factor: float = 1.0,
    key_added: str = "vector",
    label: Union[str, list, np.ndarray] = "vector",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
) -> PolyData:
    """
    Create 3D vector field model.

    Args:
        model: A model that provides coordinate information and vector information for constructing vector field models.
        vector_key: The key under which are the vector information.
        arrows_scale_key: The key under which are scale factor of the entire object.
        factor: Scale factor applied to scaling array.
        key_added: The key under which to add the labels.
        label: The label of arrows model.
        color: Color to use for plotting model.

    Returns:
        3D vector field model.
    """

    return construct_arrows(
        start_points=np.asarray(model.points),
        direction=np.asarray(model[vector_key]),
        arrows_scale=None if arrows_scale_key is None else np.asarray(model[arrows_scale_key]),
        factor=factor,
        key_added=key_added,
        label=label,
        color=color,
    )
