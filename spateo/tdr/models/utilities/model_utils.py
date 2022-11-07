from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from pyvista import DataSet, MultiBlock, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

####################
# Integrate models #
####################


def merge_models(
    models: List[PolyData or UnstructuredGrid or DataSet],
) -> PolyData or UnstructuredGrid:
    """Merge all models in the `models` list. The format of all models must be the same."""

    merged_model = models[0]
    for model in models[1:]:
        merged_model = merged_model.merge(model)

    return merged_model


def collect_models(
    models: List[PolyData or UnstructuredGrid or DataSet],
    models_name: Optional[List[str]] = None,
) -> MultiBlock:
    """
    A composite class to hold many data sets which can be iterated over.
    You can think of MultiBlock like lists or dictionaries as we can iterate over this data structure by index
    and we can also access blocks by their string name.
    If the input is a dictionary, it can be iterated in the following ways:
        >>> blocks = collect_models(models, models_name)
        >>> for name in blocks.keys():
        ...     print(blocks[name])
    If the input is a list, it can be iterated in the following ways:
        >>> blocks = collect_models(models)
        >>> for block in blocks:
        ...    print(block)
    """

    if models_name is not None:
        models = {name: model for name, model in zip(models_name, models)}

    return pv.MultiBlock(models)


def multiblock2model(model, message=None):
    """Merge all models in MultiBlock into one model"""
    if message is not None:
        import warnings

        warnings.warn(
            f"MultiBlock does not support {message}. "
            f"\nHere, all models contained in MultiBlock will be automatically merged into one model before {message}."
        )
    models = [model[name] for name in model.keys()]
    return merge_models(models=models)


###############
# Split model #
###############


def split_model(
    model: Union[PolyData, UnstructuredGrid, DataSet],
    label: Optional[bool] = False,
) -> MultiBlock:
    """
    Find, label, and split connected bodies/volumes.
    This splits different connected bodies into blocks in a pyvista.MultiBlock dataset.
    """

    return model.split_bodies(label=label)


###############
# Scale model #
###############


def _scale_model_by_distance(
    model: DataSet,
    distance: Union[int, float, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
) -> DataSet:

    # Check the distance.
    distance = distance if isinstance(distance, (tuple, list)) else [distance] * 3
    if len(distance) != 3:
        raise ValueError(
            "`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the distance.
    for i, (d, c) in enumerate(zip(distance, scale_center)):
        p2c_bool = np.asarray(model.points[:, i] - c) > 0
        model.points[:, i][p2c_bool] += d
        model.points[:, i][~p2c_bool] -= d

    return model


def _scale_model_by_scale_factor(
    model: DataSet,
    scale_factor: Union[int, float, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
) -> DataSet:

    # Check the scaling factor.
    scale_factor = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 3
    if len(scale_factor) != 3:
        raise ValueError(
            "`scale_factor` value is wrong."
            "\nWhen `scale_factor` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the scale center.
    for i, (f, c) in enumerate(zip(scale_factor, scale_center)):
        model.points[:, i] = (model.points[:, i] - c) * f + c

    return model


def scale_model(
    model: Union[PolyData, UnstructuredGrid],
    distance: Union[float, int, list, tuple] = None,
    scale_factor: Union[float, int, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
    inplace: bool = False,
) -> Union[PolyData, UnstructuredGrid, None]:
    """
    Scale the model around the center of the model.

    Args:
        model: A 3D reconstructed model.
        distance: The distance by which the model is scaled. If `distance` is float, the model is scaled same distance
                  along the xyz axis; when the `scale factor` is list, the model is scaled along the xyz axis at
                  different distance. If `distance` is None, there will be no scaling based on distance.
        scale_factor: The scale by which the model is scaled. If `scale factor` is float, the model is scaled along the
                      xyz axis at the same scale; when the `scale factor` is list, the model is scaled along the xyz
                      axis at different scales. If `scale_factor` is None, there will be no scaling based on scale factor.
        scale_center: Scaling center. If `scale factor` is None, the `scale_center` will default to the center of the model.
        inplace: Updates model in-place.

    Returns:
        model_s: The scaled model.
    """

    model_s = model.copy() if not inplace else model

    if not (distance is None):
        model_s = _scale_model_by_distance(model=model_s, distance=distance, scale_center=scale_center)

    if not (scale_factor is None):
        model_s = _scale_model_by_scale_factor(model=model_s, scale_factor=scale_factor, scale_center=scale_center)

    model_s = model_s.triangulate()

    return model_s if not inplace else None


###################
# Translate model #
###################


def translate_model(
    model: Union[PolyData, UnstructuredGrid],
    distance: Union[list, tuple] = (0, 0, 0),
    inplace: bool = False,
) -> Union[PolyData, UnstructuredGrid, None]:
    """
    Translate the mesh.

    Args:
        model: A 3D reconstructed model.
        distance: Distance to translate about the x-axis, y-axis, z-axis. Length 3 list or tuple.
        inplace: Updates model in-place.

    Returns:
        model_t: The translated model.
    """
    if len(distance) != 3:
        raise ValueError(
            "`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only contain three elements."
        )

    model_t = model.copy() if not inplace else model

    model_t.points[:, 0] = model_t.points[:, 0] + distance[0]
    model_t.points[:, 1] = model_t.points[:, 1] + distance[1]
    model_t.points[:, 2] = model_t.points[:, 2] + distance[2]

    return model_t if not inplace else None


def center_to_zero(
    model: Union[PolyData, UnstructuredGrid],
    inplace: bool = False,
):
    """
    Translate the center point of the model to the (0, 0, 0).

    Args:
        model: A 3D reconstructed model.
        inplace: Updates model in-place.

    Returns:
        model_z: Model with center point at (0, 0, 0).
    """
    model_z = model.copy() if not inplace else model

    translate_distance = (-model.center[0], -model.center[1], -model.center[2])
    translate_model(model=model_z, distance=translate_distance, inplace=True)

    return model_z if not inplace else None


################
# Rotate model #
################


def rotate_model(
    model: Union[PolyData, UnstructuredGrid],
    angle: Union[list, tuple] = (0, 0, 0),
    rotate_center: Union[list, tuple] = None,
    inplace: bool = False,
) -> Union[PolyData, UnstructuredGrid, None]:
    """
    Rotate the model around the rotate_center.

    Args:
        model: A 3D reconstructed model.
        angle: Angles in degrees to rotate about the x-axis, y-axis, z-axis. Length 3 list or tuple.
        rotate_center: Rotation center point. The default is the center of the model. Length 3 list or tuple.
        inplace: Updates model in-place.

    Returns:
        model_r: The rotated model.
    """
    model_r = model.copy() if not inplace else model

    rotate_center = model_r.center if rotate_center is None else rotate_center

    model_r.rotate_x(angle=angle[0], point=rotate_center, inplace=True)
    model_r.rotate_y(angle=angle[1], point=rotate_center, inplace=True)
    model_r.rotate_z(angle=angle[2], point=rotate_center, inplace=True)

    return model_r if not inplace else None
