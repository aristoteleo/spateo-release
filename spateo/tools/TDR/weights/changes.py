import math
from typing import Optional, Tuple, Union

import numpy as np
from pyvista import PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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
):
    slices, line_points, line = three_d_slice(model=model, method="line", n_slices=n_points, vec=vec, center=center)

    x, y = [], []
    x_length = 0
    for slice, (point_i, point) in zip(slices, enumerate(line_points)):
        gene_exp = np.asarray(slice[key]).sum()
        y.append(gene_exp)

        if point_i == 0:
            x.append(0)
        else:
            point1 = line_points[point_i - 1].points.flatten()
            point2 = line_points[point_i].points.flatten()

            ed = euclidean_distance(instance1=point1, instance2=point2, dimension=3)

            x_length += ed
            x.append(x_length)

    return np.asarray(x), np.asarray(y)
