"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage import morphology

from ..configuration import SKM
from ..logging import logger_manager as lm


def fill_grid_label(
    adata,
    spatial_key,
    seg_grid_img,
    bdl_seg_coor_x,
    bdl_seg_coor_y,
    curr_layer,
    curr_sign,
    layer_label_key: str = "layer_label",
    column_label_key: str = "column_label",
    init: bool = False,
):

    # mask image should be 2 pixels wider and higher, according to cv2.floodFill
    layer_grid_img = seg_grid_img.copy()
    layer_mask = np.zeros((layer_grid_img.shape[0] + 2, layer_grid_img.shape[1] + 2), dtype=np.uint8)
    layer_mask[1:-1, 1:-1] = layer_grid_img
    column_grid_img = seg_grid_img.copy()
    column_mask = np.zeros((column_grid_img.shape[0] + 2, column_grid_img.shape[1] + 2), dtype=np.uint8)
    column_mask[1:-1, 1:-1] = column_grid_img

    for i in range(len(bdl_seg_coor_x) - 1):
        curr_column = i + 1
        fpx = int(
            np.mean([bdl_seg_coor_x[i][0], bdl_seg_coor_x[i + 1][0], bdl_seg_coor_y[i][0], bdl_seg_coor_y[i + 1][0]])
        )
        fpy = int(
            np.mean([bdl_seg_coor_x[i][1], bdl_seg_coor_x[i + 1][1], bdl_seg_coor_y[i][1], bdl_seg_coor_y[i + 1][1]])
        )
        cv2.floodFill(layer_grid_img, layer_mask, (fpx, fpy), curr_layer)
        cv2.floodFill(column_grid_img, column_mask, (fpx, fpy), curr_column)

    if init:
        adata.obs[layer_label_key] = 0
        adata.obs[column_label_key] = 0
    else:
        try:
            _ = adata.obs[layer_label_key]
        except:
            adata.obs[layer_label_key] = 0

        try:
            _ = adata.obs[column_label_key]
        except:
            adata.obs[column_label_key] = 0

    for i in range(len(adata)):
        if adata.obs[layer_label_key][i] == 0:
            adata.obs[layer_label_key][i] = (
                layer_grid_img[int(adata.obsm[spatial_key][i, 0]), int(adata.obsm[spatial_key][i, 1])] * curr_sign
            )
        if adata.obs[column_label_key][i] == 0:
            adata.obs[column_label_key][i] = column_grid_img[
                int(adata.obsm[spatial_key][i, 0]), int(adata.obsm[spatial_key][i, 1])
            ]
    adata.obs[layer_label_key][abs(adata.obs[layer_label_key]) == 255] = 0
    adata.obs[column_label_key][adata.obs[column_label_key] == 255] = 0

    return layer_grid_img, column_grid_img


def format_boundary_line(
    boundary_line_img,
    pt_start,
    pt_end,
):
    ctrs, _ = cv2.findContours(boundary_line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    formatted_bdl_img = np.zeros_like(boundary_line_img, dtype=np.uint8)

    ctrs_pt_list = []
    for pt in ctrs[0]:  # should only contain a single contour
        ctrs_pt_list.append((pt[0][0], pt[0][1]))

    start_idx = ctrs_pt_list.index(pt_start)
    end_idx = ctrs_pt_list.index(pt_end)
    formatted_bdl_list = ctrs_pt_list[min(start_idx, end_idx) : max(start_idx, end_idx) + 2]
    for i in range(len(formatted_bdl_list) - 1):
        cv2.line(formatted_bdl_img, formatted_bdl_list[i], formatted_bdl_list[i + 1], 255, 1)

    lm.main_info(f"Extracted boundary line length: {len(formatted_bdl_list)}.")

    return formatted_bdl_list, formatted_bdl_img


def draw_seg_grid(
    boundary_line_img,
    bdl_seg_coor_x,
    bdl_seg_coor_y,
    gridline_width=1,
    mode="grid",
):

    seg_grid_img = np.zeros_like(boundary_line_img, dtype=np.uint8)

    if len(bdl_seg_coor_x) != len(bdl_seg_coor_y):
        lm.main_info(f"Warning: segmentation does not match between two boundarys. Using the shorter boundary.")

    min_seg_num = min(len(bdl_seg_coor_x), len(bdl_seg_coor_y))
    for i in range(min_seg_num):
        cv2.line(seg_grid_img, bdl_seg_coor_x[i], bdl_seg_coor_y[i], 255, gridline_width)
        if i < min_seg_num - 1:
            cv2.line(seg_grid_img, bdl_seg_coor_x[i], bdl_seg_coor_x[i + 1], 255, gridline_width)
            cv2.line(seg_grid_img, bdl_seg_coor_y[i], bdl_seg_coor_y[i + 1], 255, gridline_width)

    if mode == "grid":  # gridding image
        return seg_grid_img
    elif mode == "gray":
        # TODO:Directly label each region in adata, function fill_grid_label can be merged.
        pass


def euclidean_dist(
    point_x: Tuple,  # geometric coordinate
    point_y: Tuple,
):
    return math.sqrt((point_x[0] - point_y[0]) ** 2 + (point_x[1] - point_y[1]) ** 2)


def segment_bd_line(  # Refactor not completed
    boundary_line_list,
    n_column,
):
    dist_ls = []  # dist between sequence points
    peri_ls = []  # accumulate dist
    dist_per = []  # length for each segmentation part
    slice_index = []  # index for segmentation points

    perimeter = 0
    for i in range(len(boundary_line_list) - 1):
        dist_ls.append(euclidean_dist(boundary_line_list[i + 1], boundary_line_list[i]))
        perimeter += dist_ls[i]
        peri_ls.append(perimeter)

    len_per_slice = perimeter / n_column
    lm.main_info(
        f"Line total length: {round(perimeter, 2)}. Segmenting into {n_column} columns, with {round(len_per_slice, 2)} each."
    )

    ls_ex_dist_add_ar = np.array(peri_ls)

    first = True
    for i in range(len(ls_ex_dist_add_ar)):  # per dist array add.
        if i == 0 or i == len(ls_ex_dist_add_ar) - 1:
            slice_index.append(i)
        else:
            if (ls_ex_dist_add_ar[i] >= len_per_slice) and first:  # first step
                error_dist = ls_ex_dist_add_ar[i] - len_per_slice
                slice_index.append(i)
                dist_per.append(ls_ex_dist_add_ar[i])
                ls_ex_dist_add_ar = ls_ex_dist_add_ar - ls_ex_dist_add_ar[i]
                first = False

            if (ls_ex_dist_add_ar[i] >= len_per_slice) and (error_dist > 0):
                error_dist = error_dist + ls_ex_dist_add_ar[i - 1] - len_per_slice
                slice_index.append(i - 1)
                dist_per.append(ls_ex_dist_add_ar[i - 1])
                ls_ex_dist_add_ar = ls_ex_dist_add_ar - ls_ex_dist_add_ar[i - 1]

            elif (ls_ex_dist_add_ar[i] >= len_per_slice) and (error_dist < 0):
                error_dist = error_dist + ls_ex_dist_add_ar[i] - len_per_slice
                slice_index.append(i)
                dist_per.append(ls_ex_dist_add_ar[i])
                ls_ex_dist_add_ar = ls_ex_dist_add_ar - ls_ex_dist_add_ar[i]

    ls_ar_slice = np.array(boundary_line_list)[slice_index]

    return ls_ar_slice  # segmentation point list


def extend_layer(
    boundary_line_img,
    boundary_line_list,
    extend_width=10,
):

    lm.main_info(f"Generating layer area.")
    extend_layer_mask = np.zeros_like(boundary_line_img, dtype=np.uint8)
    extend_layer_img = np.zeros_like(boundary_line_img, dtype=np.uint8)
    for pt in boundary_line_list:
        cv2.circle(extend_layer_mask, pt, extend_width, 255, -1)

    extend_layer_contour, _ = cv2.findContours(extend_layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(extend_layer_img, extend_layer_contour, -1, 255, 1)

    lm.main_info(f"Refining layer contour.")
    extend_layer_tmp = np.zeros_like(boundary_line_img, dtype=np.uint8)
    cv2.circle(extend_layer_tmp, boundary_line_list[0], extend_width, 255, -1)
    cv2.circle(extend_layer_tmp, boundary_line_list[-1], extend_width, 255, -1)
    contours_edge, _ = cv2.findContours(extend_layer_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    extend_layer_tmp = np.zeros_like(boundary_line_img, dtype=np.uint8)
    cv2.drawContours(extend_layer_tmp, contours_edge, -1, 255, 1)
    extend_layer_img = np.where(extend_layer_tmp != 0, 0, extend_layer_img)
    extend_layer_img = (
        morphology.remove_small_objects(extend_layer_img.astype(bool), min_size=5, connectivity=2).astype(np.uint8)
        * 255
    )

    extend_layer_bdl = []  # extended layer boundary line
    for pt in extend_layer_contour[0]:  # should be a single contour
        pt_x = pt[0][0]
        pt_y = pt[0][1]
        if extend_layer_img[pt_y, pt_x] != 0:
            extend_layer_bdl.append((pt_x, pt_y))

    return extend_layer_img, extend_layer_bdl


def field_contour_line(
    ctr_seq,
    pnt_pos,
    min_pnt,
    max_pnt,
):

    ctr_seq_rev = ctr_seq[::-1].copy()
    min_idx = ctr_seq.index(min_pnt)
    max_idx = ctr_seq.index(max_pnt) + 1
    if min_idx < max_idx:
        if sum(pnt_pos[min_idx + 1 : max_idx - 1]) == 0:
            line_seq = ctr_seq[min_idx:max_idx]
        else:
            min_idx = ctr_seq_rev.index(min_pnt)
            max_idx = ctr_seq_rev.index(max_pnt) + 1
            line_seq = ctr_seq_rev[min_idx:] + ctr_seq_rev[:max_idx]
    else:
        if sum(pnt_pos[min_idx + 1 :]) + sum(pnt_pos[: max_idx - 1]) == 0:
            line_seq = ctr_seq[min_idx:] + ctr_seq[:max_idx]
        else:
            min_idx = ctr_seq_rev.index(min_pnt)
            max_idx = ctr_seq_rev.index(max_pnt) + 1
            line_seq = ctr_seq_rev[min_idx:max_idx]

    return line_seq


def field_contours(
    contour,
    pnt_xy,
    pnt_Xy,
    pnt_xY,
    pnt_XY,
):
    """Identify four boundary lines according to given corner points.

    Args:
        contour (_type_): _description_
        pnt_xy (_type_): _description_
        pnt_Xy (_type_): _description_
        pnt_xY (_type_): _description_
        pnt_XY (_type_): _description_

    Returns:
        _type_: _description_
    """

    ctr_seq = [tuple(i) for i in contour[:, 0]]

    pnt_pos = np.zeros(len(ctr_seq))
    pnt_pos[ctr_seq.index(pnt_xy)] = 1
    pnt_pos[ctr_seq.index(pnt_Xy)] = 1
    pnt_pos[ctr_seq.index(pnt_xY)] = 1
    pnt_pos[ctr_seq.index(pnt_XY)] = 1

    min_line_l = field_contour_line(ctr_seq, pnt_pos, pnt_xy, pnt_Xy)
    max_line_l = field_contour_line(ctr_seq, pnt_pos, pnt_xY, pnt_XY)
    min_line_c = field_contour_line(ctr_seq, pnt_pos, pnt_xy, pnt_xY)
    max_line_c = field_contour_line(ctr_seq, pnt_pos, pnt_Xy, pnt_XY)

    return min_line_l, max_line_l, min_line_c, max_line_c


def add_ep_boundary(
    op_field,
    op_line,
    value,
):
    """Add equal weight boundary to op_field.

    Args:
        op_field (_type_): _description_
        op_line (_type_): _description_
        value (_type_): _description_

    Returns:
        _type_: _description_
    """

    for x, y in op_line:
        op_field[y, x] = value


def add_gp_boundary(
    op_field,
    op_line,
    value_s,
    value_e,
):
    """Add growing weight boundary to op_field.

    Args:
        op_field (_type_): _description_
        op_line (_type_): _description_
        value_s (_type_): _description_
        value_e (_type_): _description_

    Returns:
        _type_: _description_
    """

    gp_value = np.linspace(value_s, value_e, len(op_line))
    idx = 0
    for x, y in op_line:
        op_field[y, x] = gp_value[idx]
        idx += 1


def effective_L2_error(
    op_field_i,
    op_field_j,
    field_mask,
):
    """Calculate effective L2 error between two fields.

    Args:
        op_field_i (_type_): _description_
        op_field_j (_type_): _description_
        field_mask (_type_): _description_

    Returns:
        _type_: _description_
    """

    return np.sqrt(np.sum((op_field_j - op_field_i) ** 2 * field_mask) / np.sum(op_field_j**2 * field_mask))


def domain_heat_eqn_solver(
    op_field,
    min_line,
    max_line,
    edge_line_a,
    edge_line_b,
    field_border,
    field_mask,
    max_err: float = 1e-5,
    max_itr: float = 1e5,
    lh: float = 1,
    hh: float = 100,
):
    """Given the boundaries and boundary conditions of a close spatial domain, solve heat equation (a simple partial
    differential equation) to define the "heat" for each spatial pixel which can be used to digitize the
    spatial domain into different layers or columns. Diffusitivity is set to be 1/4, thus the update rule is defined as:

        grid_field[1:-1, 1:-1] = 0.25 * (
            grid_field_pre[1:-1, 2:] + grid_field_pre[1:-1, :-2] + grid_field_pre[2:, 1:-1] + grid_field_pre[:-2, 1:-1]
        )

    Args:
        op_field (_type_): _description_
        min_line (_type_): _description_
        max_line (_type_): _description_
        edge_line_a (_type_): _description_
        edge_line_b (_type_): _description_
        field_border (_type_): _description_
        field_mask (_type_): _description_
        max_err (_type_, optional): _description_. Defaults to 1e-5.
        max_itr (_type_, optional): _description_. Defaults to 1e5.
        lh: _description_. Defaults to 1.
        hh: _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """

    init_field = op_field.copy()
    add_ep_boundary(init_field, min_line, lh)
    add_ep_boundary(init_field, max_line, hh)
    add_gp_boundary(init_field, edge_line_a, lh, hh)
    add_gp_boundary(init_field, edge_line_b, lh, hh)

    err = 1
    itr = 0
    grid_field = init_field.copy()
    while (err > max_err) and (itr <= max_itr):
        grid_field_pre = grid_field.copy()
        grid_field[1:-1, 1:-1] = 0.25 * (
            grid_field_pre[1:-1, 2:] + grid_field_pre[1:-1, :-2] + grid_field_pre[2:, 1:-1] + grid_field_pre[:-2, 1:-1]
        )
        grid_field = np.where(field_border != 0, init_field, grid_field)
        err = effective_L2_error(grid_field, grid_field_pre, field_mask)
        if itr >= max_itr:
            lm.main_info("Max iteration reached, with L2 error at: " + str(err))
        itr = itr + 1
    lm.main_info("Total iteration: " + str(itr))
    grid_field = grid_field * field_mask

    return grid_field
