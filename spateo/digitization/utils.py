"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from nptyping import NDArray

from ..configuration import SKM
from ..logging import logger_manager as lm


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


def calc_op_field(
    op_field,
    min_line,
    max_line,
    edge_line_a,
    edge_line_b,
    field_border,
    field_mask,
    max_err: float = 1e-5,
    max_itr: float = 1e5,
    lp: int = 1,
    hp: int = 100,
):
    """Calculate op_field (weights) for given boundary weights.

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
        lp (int, optional): _description_. Defaults to 1.
        hp (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """

    init_field = op_field.copy()
    add_ep_boundary(init_field, min_line, lp)
    add_ep_boundary(init_field, max_line, hp)
    add_gp_boundary(init_field, edge_line_a, lp, hp)
    add_gp_boundary(init_field, edge_line_b, lp, hp)

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
