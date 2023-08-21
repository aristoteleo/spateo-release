try:
    from typing import Any, List, Literal, Tuple, Union
except ImportError:
    from typing_extensions import Literal

from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
from anndata import AnnData
from scipy.spatial import ConvexHull, Delaunay

from .transform import BA_transform


def _merge_models(models):
    merged_model = models[0]
    for model in models[1:]:
        merged_model = merged_model.merge(model)
    return merged_model


def grid_deformation(
    model: AnnData,
    spatial_key: str = "spatial",
    vecfld_key: str = "VecFld_morpho",
    key_added: str = "deformation",
    deformation_scale: int = 3,
    grid_num: Optional[np.asarray] = None,
    dtype: str = "float64",
    device: str = "cpu",
):
    # Check the number of lines
    grid_num = np.asarray([20, 20]) if grid_num is None else grid_num

    # Generate grid
    grid, deformed_grid = [], []
    x_min, y_min = np.min(model.obsm[spatial_key], axis=0)
    x_max, y_max = np.max(model.obsm[spatial_key], axis=0)

    x_level_list = np.linspace(x_min, x_max, grid_num[0], endpoint=True)  # np.arange(x_min, x_max, grid_num[0])
    for x_level in x_level_list:
        liney = np.linspace(y_min, y_max, 1000)[:, np.newaxis]
        liney = np.concatenate((x_level * np.ones_like(liney), liney), axis=1)
        deform_liney, quary_velocities, _ = BA_transform(
            vecfld=model.uns[vecfld_key],
            quary_points=liney,
            deformation_scale=deformation_scale,
            device=device,
            dtype=dtype,
        )

        liney = np.c_[liney, np.zeros(shape=(liney.shape[0], 1))]
        liney = np.asarray(list(chain.from_iterable(zip(liney[:-1, :], liney[1:, :]))))
        pv_liney = pv.line_segments_from_points(liney)
        pv_liney.point_data[key_added] = np.zeros(shape=(liney.shape[0],))
        grid.append(pv_liney)

        deform_liney = np.c_[deform_liney, np.zeros(shape=(deform_liney.shape[0], 1))]
        deform_liney = np.asarray(list(chain.from_iterable(zip(deform_liney[:-1, :], deform_liney[1:, :]))))
        pv_deform_liney = pv.line_segments_from_points(deform_liney)

        velocities = np.mean(np.abs(quary_velocities), axis=1).flatten()
        velocities = np.asarray(list(chain.from_iterable(zip(velocities[:-1], velocities[1:]))))
        pv_deform_liney.point_data[key_added] = velocities
        deformed_grid.append(pv_deform_liney)

    y_level_list = np.linspace(y_min, y_max, grid_num[1], endpoint=True)  # np.arange(y_min, y_max, grid_num[1])
    for y_level in y_level_list:
        linex = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
        linex = np.concatenate((linex, y_level * np.ones_like(linex)), axis=1)
        deform_linex, quary_velocities, _ = BA_transform(
            vecfld=model.uns[vecfld_key],
            quary_points=linex,
            deformation_scale=deformation_scale,
            device=device,
            dtype=dtype,
        )

        linex = np.c_[linex, np.zeros(shape=(linex.shape[0], 1))]
        linex = np.asarray(list(chain.from_iterable(zip(linex[:-1, :], linex[1:, :]))))
        pv_linex = pv.line_segments_from_points(linex)
        pv_linex.point_data[key_added] = np.zeros(shape=(linex.shape[0],))
        grid.append(pv_linex)

        deform_linex = np.c_[deform_linex, np.zeros(shape=(deform_linex.shape[0], 1))]
        deform_linex = np.asarray(list(chain.from_iterable(zip(deform_linex[:-1, :], deform_linex[1:, :]))))
        pv_deform_linex = pv.line_segments_from_points(deform_linex)

        velocities = np.mean(np.abs(quary_velocities), axis=1).flatten()
        velocities = np.asarray(list(chain.from_iterable(zip(velocities[:-1], velocities[1:]))))
        pv_deform_linex.point_data[key_added] = velocities
        deformed_grid.append(pv_deform_linex)

    pv_grid = _merge_models(grid)
    pv_deformed_grid = _merge_models(deformed_grid)
    return pv_grid, pv_deformed_grid


"""
def check_in_hull(points: np.ndarray, grid: np.ndarray):

    hull = ConvexHull(points)
    hull = hull.points[hull.vertices, :]
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    grid_in_hull = hull.find_simplex(grid) >= 0
    grid_in_hull_index = np.argwhere(grid_in_hull == True).flatten()
    hull_grid = grid[grid_in_hull_index, :]
    return hull_grid, grid_in_hull_index


def grid_deformation_fail(
    model: AnnData,
    spatial_key: str = "spatial",
    vecfld_key: str = "VecFld_morpho",
    key_added: str = "deformation",
    grid_num: Optional[np.asarray] = None,
    expand_c: Union[float, int] = 0.2,
    dtype: str = "float64",
    device: str = "cpu",
):

    # Check aligned coordinates
    coords = model.obsm[spatial_key].copy()
    coords_dims = coords.shape[1]
    if coords_dims == 2:
        grid_num = np.asarray([50, 50]) if grid_num is None else grid_num
    else:
        grid_num = np.asarray([20, 20, 20]) if grid_num is None else grid_num

    # Generate grid
    min_vec, max_vec = coords.min(0), coords.max(0)
    min_vec = min_vec - expand_c * np.abs(max_vec - min_vec)
    max_vec = max_vec + expand_c * np.abs(max_vec - min_vec)
    grid_list = np.meshgrid(
        *[np.linspace(i, j, k) for i, j, k in zip(min_vec, max_vec, grid_num)]
    )
    grid = np.asarray([i.flatten() for i in grid_list]).T
    hull_grid, grid_in_hull_index = check_in_hull(
        points=coords.copy(), grid=grid.copy()
    )

    # Generate deformed grid
    vecfld = model.uns[vecfld_key].copy()
    deformed_hull_grid, quary_velocities = BA_transform(
        vecfld, quary_points=np.asarray(hull_grid), dtype=dtype, device=device
    )
    deformed_grid = grid.copy()
    deformed_grid[grid_in_hull_index, :] = deformed_hull_grid

    # from ..morphometrics import velocities
    # grid_velocities = velocities(vecfld, quary_points=np.asarray(hull_grid), dtype=dtype, device=device)
    # deformed_grid = grid.copy()
    # deformed_grid[grid_in_hull_index, :] = hull_grid + grid_velocities

    # Generate grid model using pyvista
    # Generate points
    grid = np.c_[grid, np.zeros(shape=(grid.shape[0], 1))]
    deformed_grid = np.c_[deformed_grid, np.zeros(shape=(deformed_grid.shape[0], 1))]

    # Generate faces
    if coords_dims == 2:
        dim_x, dim_y = grid_num[0], grid_num[1]
        points_ids = np.arange(0, dim_x * dim_y, 1).reshape(dim_y, dim_x)

        faces = np.zeros((dim_x * dim_y, 5)).astype(int)
        faces[:, 0] = 4
        for h in range(dim_y - 1):
            faces[h * dim_x : h * dim_x + (dim_x - 1), 1:] = np.stack(
                [
                    points_ids[h, :-1],
                    points_ids[h, 1:],
                    points_ids[h + 1, 1:],
                    points_ids[h + 1, :-1],
                ],
                axis=1,
            )
    else:
        dim_x, dim_y, dim_z = grid_num[0], grid_num[1], grid_num[2]
        points_ids = np.arange(0, dim_x * dim_y * dim_z, 1).reshape(dim_z, dim_y, dim_x)

        faces = np.zeros((dim_x * dim_y * dim_z, 9)).astype(int)
        faces[:, 0] = 8
        for w in range(dim_z - 1):
            w_start = w * dim_x * dim_y
            for h in range(dim_y - 1):
                faces[
                    h * dim_x + w_start : h * dim_x + (dim_x - 1) + w_start, 1:
                ] = np.stack(
                    [
                        points_ids[w, h, :-1],
                        points_ids[w, h, 1:],
                        points_ids[w, h + 1, 1:],
                        points_ids[w, h + 1, :-1],
                        points_ids[w + 1, h, :-1],
                        points_ids[w + 1, h, 1:],
                        points_ids[w + 1, h + 1, 1:],
                        points_ids[w + 1, h + 1, :-1],
                    ],
                    axis=1,
                )
    faces = faces[faces[:, -1] != 0]
    faces = np.hstack(faces.astype(int))

    # Generate models
    pv_grid = pv.PolyData(grid, faces=faces)
    pv_deformed_grid = pv.PolyData(deformed_grid, faces=faces)

    # Add deformed distance
    # distance_matrix = distance.cdist(deformed_grid, grid, "euclidean")
    # deformed_distance = np.diagonal(distance_matrix).flatten()
    # pv_deformed_grid.point_data[key_added] = deformed_distance
    velocities = np.zeros(shape=(deformed_grid.shape[0],))
    velocities[grid_in_hull_index] = np.mean(np.abs(quary_velocities), axis=1).flatten()
    pv_deformed_grid.point_data[key_added] = velocities

    return pv_grid, pv_deformed_grid
"""
