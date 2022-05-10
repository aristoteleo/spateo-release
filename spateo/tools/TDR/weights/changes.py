import math
from typing import Optional, Tuple, Union

import numpy as np
from pyvista import MultiBlock, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .ddrtree import DDRTree, cal_ncenter
from .slice import euclidean_distance, three_d_slice

####################################
# Changes along a vector direction #
####################################


def changes_along_line(
    model: Union[PolyData, UnstructuredGrid],
    key: Union[str, list] = None,
    n_points: int = 100,
    vec: Union[tuple, list] = (1, 0, 0),
    center: Union[tuple, list] = None,
) -> Tuple[np.ndarray, np.ndarray, MultiBlock, MultiBlock]:
    slices, line_points, line = three_d_slice(model=model, method="line", n_slices=n_points, vec=vec, center=center)

    x, y = [], []
    x_length = 0
    for slice, (point_i, point) in zip(slices, enumerate(line_points)):
        change_value = np.asarray(slice[key]).sum()
        y.append(change_value)

        if point_i == 0:
            x.append(0)
        else:
            point1 = line_points[point_i - 1].points.flatten()
            point2 = line_points[point_i].points.flatten()

            ed = euclidean_distance(instance1=point1, instance2=point2, dimension=3)

            x_length += ed
            x.append(x_length)

    return np.asarray(x), np.asarray(y), slices, line


#################################
# Changes along the model shape #
#################################


def changes_along_shape(
    model: Union[PolyData, UnstructuredGrid],
    spatial_key: Optional[str] = None,
    key_added: Optional[str] = "rd_spatial",
    dim: int = 3,
    inplace: bool = False,
    **kwargs,
):
    model = model.copy() if not inplace else model
    X = model.points if spatial_key is None else model[spatial_key]

    DDRTree_kwargs = {
        "maxIter": 10,
        "sigma": 0.001,
        "gamma": 10,
        "eps": 0,
        "dim": dim,
        "Lambda": 5 * X.shape[1],
        "ncenter": cal_ncenter(X.shape[1]),
    }
    DDRTree_kwargs.update(kwargs)
    Z, Y, stree, R, W, Q, C, objs = DDRTree(X, **DDRTree_kwargs)

    model[key_added] = W

    return model if not inplace else None
