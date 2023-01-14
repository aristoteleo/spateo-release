from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import add_model_labels, merge_models
from .arrow_model import construct_arrows


def construct_field(
    model: PolyData,
    vf_key: str = "VecFld_morpho",
    arrows_scale_key: Optional[str] = None,
    n_sampling: Optional[int] = None,
    sampling_method: str = "trn",
    factor: float = 1.0,
    key_added: str = "v_arrows",
    label: Union[str, list, np.ndarray] = "vector field",
    color: Union[str, list, dict, np.ndarray] = "gainsboro",
    alpha: float = 1.0,
    **kwargs,
) -> PolyData:
    """
    Create a 3D vector field arrows model.

    Args:
        model: A model that provides coordinate information and vector information for constructing vector field models.
        vf_key: The key under which are the vector information.
        arrows_scale_key: The key under which are scale factor of the entire object.
        n_sampling: n_sampling is the number of coordinates to keep after sampling. If there are too many coordinates
                    in start_points, the generated arrows model will be too complex and unsightly, so sampling is
                    used to reduce the number of coordinates.
        sampling_method: The method to sample data points, can be one of ``['trn', 'kmeans', 'random']``.
        factor: Scale factor applied to scaling array.
        key_added: The key under which to add the labels.
        label: The label of arrows models.
        color: Color to use for plotting model.
        alpha: The opacity of the color to use for plotting model.
        **kwargs: Additional parameters that will be passed to ``construct_arrows`` function.

    Returns:
        A 3D vector field arrows model.
    """

    return construct_arrows(
        start_points=np.asarray(model.points),
        direction=np.asarray(model[vf_key]),
        arrows_scale=np.ones(shape=(model.n_points,))
        if arrows_scale_key is None
        else np.asarray(model[arrows_scale_key]),
        n_sampling=n_sampling,
        sampling_method=sampling_method,
        factor=factor,
        key_added=key_added,
        label=label,
        color=color,
        alpha=alpha,
        **kwargs,
    )


def construct_field_streams(
    model: PolyData,
    vf_key: str = "VecFld_morpho",
    source_center: Optional[Tuple[float]] = None,
    source_radius: Optional[float] = None,
    tip_factor: Union[int, float] = 10,
    tip_radius: float = 0.2,
    key_added: str = "v_streams",
    label: Union[str, list, np.ndarray] = "vector field",
    stream_color: str = "gainsboro",
    tip_color: str = "orangered",
    alpha: float = 1.0,
    **kwargs,
):
    """
    Integrate a vector field to generate streamlines.

    Args:
        model: A model that provides coordinate information and vector information for constructing vector field models.
        vf_key: The key under which are the active vector field information.
        source_center: Length 3 tuple of floats defining the center of the source particles. Defaults to the center of the dataset.
        source_radius: Float radius of the source particle cloud. Defaults to one-tenth of the diagonal of the dataset’s spatial extent.
        tip_factor: Scale factor applied to scaling the tips.
        tip_radius: Radius of the tips.
        key_added: The key under which to add the labels.
        label: The label of arrows models.
        stream_color: Color to use for plotting streamlines.
        tip_color: Color to use for plotting tips.
        alpha: The opacity of the color to use for plotting model.
        **kwargs: Additional parameters that will be passed to ``streamlines`` function.

    Returns:
        streams_model: 3D vector field streamlines model.
        src: The source particles as pyvista.PolyData as well as the streamlines.
    """

    # generate the streamlines based on the vector field.
    streamlines, src = model.streamlines(
        vf_key, return_source=True, source_center=source_center, source_radius=source_radius, **kwargs
    )
    add_model_labels(
        model=streamlines,
        key_added=key_added,
        labels=np.asarray([label] * streamlines.n_points),
        where="point_data",
        colormap=stream_color,
        alphamap=alpha,
        inplace=True,
    )

    # generate the tips of the streamlines.
    tips_points, tips_vectors = [], []
    for streamline in streamlines.split_bodies():
        tips_points.append(streamline.points[-1])
        tips_vectors.append(streamline.point_data[vf_key][-1])

    arrows = construct_arrows(
        start_points=np.asarray(tips_points),
        direction=np.asarray(tips_vectors),
        arrows_scale=np.ones(shape=(len(tips_points), 1)),
        factor=tip_factor,
        tip_length=1,
        tip_radius=tip_radius,
        key_added=key_added,
        label=label,
        color=tip_color,
        alpha=alpha,
    )

    streams_model = merge_models([streamlines, arrows])
    return streams_model, src
