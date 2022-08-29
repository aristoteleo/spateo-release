from typing import Optional, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ....logging import logger_manager as lm
from ..utilities import add_model_labels


def _construct_arrow(
    start_point: Union[list, tuple, np.ndarray] = (0.0, 0.0, 0.0),
    direction: Union[list, tuple, np.ndarray] = (1.0, 0.0, 0.0),
    tip_length: float = 0.25,
    tip_radius: float = 0.1,
    tip_resolution: int = 20,
    shaft_radius: float = 0.05,
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
        scale: Scale factor of the entire object. ``'auto'`` scales to length of direction array.

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
    key_added: Optional[str] = "arrow",
    label: str = "arrow",
    color: str = "gainsboro",
    alpha: float = 1.0,
    **kwargs,
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
        alpha: The opacity of the color to use for plotting model.
        **kwargs: Additional parameters that will be passed to ``_construct_arrow`` function.

    Returns:
        Arrow model.
    """

    model = _construct_arrow(
        start_point=start_point, direction=direction, scale="auto" if arrow_scale is None else arrow_scale, **kwargs
    )

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


def construct_arrows(
    start_points: np.ndarray,
    direction: np.ndarray = None,
    arrows_scale: Optional[np.ndarray] = None,
    n_sampling: Optional[int] = None,
    sampling_method: str = "trn",
    factor: float = 1.0,
    key_added: Optional[str] = "arrow",
    label: Union[str, list, np.ndarray] = "arrows",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
    alpha: Union[float, int, list, dict, np.ndarray] = 1.0,
    **kwargs,
) -> PolyData:
    """
    Create multiple 3D arrows model.

    Args:
        start_points: List of Start location in [x, y, z] of the arrows.
        direction: Direction the arrows points to in [x, y, z].
        arrows_scale: Scale factor of the entire object.
        n_sampling: n_sampling is the number of coordinates to keep after sampling. If there are too many coordinates
                    in start_points, the generated arrows model will be too complex and unsightly, so sampling is
                    used to reduce the number of coordinates.
        sampling_method: The method to sample data points, can be one of ``['trn', 'kmeans', 'random']``.
        factor: Scale factor applied to scaling array.
        key_added: The key under which to add the labels.
        label: The label of arrows models.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.
        **kwargs: Additional parameters that will be passed to ``_construct_arrow`` function.

    Returns:
        Arrows model.
    """

    from dynamo.tools.sampling import sample

    index_arr = np.arange(0, start_points.shape[0])
    if not (n_sampling is None):
        index_arr = sample(
            arr=index_arr,
            n=n_sampling,
            method=sampling_method,
            X=start_points,
        )
    else:
        if len(start_points) > 500:
            lm.main_warning(
                f"The number of start_points is more than 500. You may want to "
                f"lower the max number of arrows to draw."
            )

    start_points = start_points[index_arr, :].copy()
    direction = direction[index_arr, :].copy()
    model = pv.PolyData(start_points)
    model.point_data["direction"] = direction
    model.point_data["scale"] = np.linalg.norm(direction, axis=1) if arrows_scale is None else arrows_scale[index_arr]

    labels = np.asarray([label] * len(start_points)) if isinstance(label, str) else np.asarray(label)[index_arr]
    assert len(labels) == len(start_points), "The number of labels is not equal to the number of start points."
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

    glyph = model.glyph(orient="direction", geom=_construct_arrow(**kwargs), scale="scale", factor=factor)
    return glyph
