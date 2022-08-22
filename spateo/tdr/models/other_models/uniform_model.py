from typing import List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import DataSet, MultiBlock, PolyData, UniformGrid

from ..utilities import add_model_labels


def construct_space(
    model: Union[DataSet, MultiBlock],
    expand_dist: Union[int, float, list, tuple] = (0, 0, 0),
    grid_num: Optional[Union[List[int], Tuple[int]]] = None,
    key_added: Optional[str] = "space",
    label: str = "space",
    color: str = "gainsboro",
    alpha: float = 0.5,
) -> UniformGrid:
    """
    Construct a model(space-model) with uniform spacing in the three coordinate directions.
    The six surfaces of the commonly generated space-model are exactly the boundaries of the model, but the space-model
    can also be expanded by `expand_dist`.

    Args:
        model: A three dims model.
        expand_dist: The length of space-model to be extended in all directions.
        grid_num: Number of grid to generate.
        key_added: The key under which to add the labels.
        label: The label of space-model.
        color: Color to use for plotting space-model.
        alpha: The opacity of the color to use for plotting space-model.

    Returns:
        A space-model with uniform spacing in the three coordinate directions.
    """

    # return the bounds of the model.
    model_bounds = np.asarray(model.bounds)
    min_bounds = np.floor(model_bounds[[0, 2, 4]])
    max_bounds = np.ceil(model_bounds[[1, 3, 5]])

    # expanding bounds.
    if isinstance(expand_dist, (int, float)):
        expand_dist = [expand_dist] * 3
    expand_dist = np.asarray(expand_dist)
    assert len(expand_dist) == 3, "The number of `expand_dist` list is not equal to 3."
    min_bounds = min_bounds - expand_dist
    max_bounds = max_bounds + expand_dist

    # Set the grid dimensions and the cell sizes along each axis.
    axis_length = max_bounds - min_bounds
    if grid_num is None:
        dims = axis_length.astype(int)
        spacing = np.ones(shape=(3,), dtype=int)
    else:
        assert len(grid_num) == 3, "The number of `grid_num` list is not equal to 3."
        dims = np.asarray(grid_num, dtype=int)
        spacing = axis_length / grid_num

    # Create the 3D grid model.
    grid = pv.UniformGrid(
        dims=dims, origin=min_bounds.tolist(), spacing=spacing  # The bottom left corner of the grid model.
    )

    if not (key_added is None):
        add_model_labels(
            model=grid,
            key_added=key_added,
            labels=np.asarray([label] * grid.n_points),
            where="point_data",
            colormap=color,
            alphamap=alpha,
            inplace=True,
        )

    return grid


def construct_bounding_box(
    model: Union[DataSet, MultiBlock],
    expand_dist: Union[int, float, list, tuple] = (0, 0, 0),
    grid_num: Optional[Union[List[int], Tuple[int]]] = None,
    key_added: str = "bounding_box",
    label: str = "bounding_box",
    color: str = "gainsboro",
    alpha: float = 0.5,
) -> PolyData:
    """
    Construct a bounding box model of the model.

    Args:
        model: A three dims model.
        expand_dist: The length of space-model to be extended in all directions.
        grid_num: Number of grid to generate.
        key_added: The key under which to add the labels.
        label: The label of space-model.
        color: Color to use for plotting space-model.
        alpha: The opacity of the color to use for plotting space-model.

    Returns:
        A bounding box model.
    """

    grid_model = construct_space(
        model=model,
        expand_dist=expand_dist,
        grid_num=grid_num,
        key_added=key_added,
        label=label,
        color=color,
        alpha=alpha,
    )
    return grid_model.triangulate().extract_surface().clean()
