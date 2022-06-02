"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from skimage import morphology

from ..configuration import SKM
from ..logging import logger_manager as lm


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def fill_grid_label(
    adata: AnnData,
    spatial_key: str,
    seg_grid_img: np.ndarray,
    bdl_seg_coor_x: np.ndarray,
    bdl_seg_coor_y: np.ndarray,
    curr_layer: int,
    curr_sign: int,
    layer_label_key: str = "layer_label",
    column_label_key: str = "column_label",
    init: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign the interior/exterior layer, column and grid to each bucket.

    Args:
        adata: The adata object to be used for assign the interior/exterior layers, columns and grid.
        spatial_key: The key name in `adata.obsm` of the spatial coordinates. Default to "spatial". Passed to
            `fill_grid_label` function.
        seg_grid_img: The matrix that stores the image information of the borderline between the source and target
            cluster(s), as well as the i and i+1-th borderlines.
        bdl_seg_coor_x: The numpy array of the coordinates of the i-th borderline.
        bdl_seg_coor_y: The numpy array of the coordinates of the i+1-th borderline.
        curr_layer: The number of the current layer.
        curr_sign: The sign of the current layer.
        layer_label_key: The key in `.obs` that points to the key of the layer labels.
        column_label_key: The key in `.obs` that points to the key of the column labels.
        init: Whether to generate (and potentially overwrite) the `layer_label_key` and `column_label_key` in
            `fill_grid_label` function.
    Returns:
        layer_grid_img: A numpy array that store the image of the layers and layer grids.
        column_grid_img: A numpy array that store the image of the columns and column grids.
    """

    # mask image should be 2 pixels wider and higher, according to cv2.floodFill
    layer_grid_img = seg_grid_img.copy()
    layer_mask = np.zeros((layer_grid_img.shape[0] + 2, layer_grid_img.shape[1] + 2), dtype=np.uint8)
    layer_mask[1:-1, 1:-1] = layer_grid_img
    column_grid_img = seg_grid_img.copy()
    column_mask = np.zeros((column_grid_img.shape[0] + 2, column_grid_img.shape[1] + 2), dtype=np.uint8)
    column_mask[1:-1, 1:-1] = column_grid_img

    lm.main_info("Use cv2.floodFill to fill layer/column number.")
    for i in range(len(bdl_seg_coor_x) - 1):
        curr_column = i + 1
        # identify the middle point for each layer/column and use that as the seed for cv2.floodFill.
        fpx = int(
            np.mean([bdl_seg_coor_x[i][0], bdl_seg_coor_x[i + 1][0], bdl_seg_coor_y[i][0], bdl_seg_coor_y[i + 1][0]])
        )
        fpy = int(
            np.mean([bdl_seg_coor_x[i][1], bdl_seg_coor_x[i + 1][1], bdl_seg_coor_y[i][1], bdl_seg_coor_y[i + 1][1]])
        )
        # Fills a connected component with the given color.
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

    lm.main_info(
        f"Assign layer/column number for each bucket with the {layer_label_key} and {column_label_key}, "
        f"respectively."
    )
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


def order_borderline(
    borderline_img: np.ndarray,
    pt_start: Tuple[int, int],
    pt_end: Tuple[int, int],
) -> Tuple[List, np.ndarray]:
    """Retrieve the borderline segment given the start end end point with the coordinates ordered.

    Args:
        borderline_img: The matrix that stores the image of the borderline.
        pt_start: The coordinate tuple of the start point.
        pt_end: The coordinate tuple of the start point.

    Returns:
        ordered_bdl_list: List of points along the borderline segment.
        ordered_bdl_img: A numpy aray that stores the image of the borderline segment.

    """
    lm.main_info(
        f"Reorder the coordinates along the borderline with the givien start {pt_start} and end {pt_end} " f"points."
    )

    ctrs, _ = cv2.findContours(borderline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ordered_bdl_img = np.zeros_like(borderline_img, dtype=np.uint8)

    ctrs_pt_list = []
    for pt in ctrs[0]:  # should only contain a single contour
        ctrs_pt_list.append((pt[0][0], pt[0][1]))

    start_idx = ctrs_pt_list.index(pt_start)
    end_idx = ctrs_pt_list.index(pt_end)
    ordered_bdl_list = ctrs_pt_list[min(start_idx, end_idx) : max(start_idx, end_idx) + 2]
    for i in range(len(ordered_bdl_list) - 1):
        cv2.line(ordered_bdl_img, ordered_bdl_list[i], ordered_bdl_list[i + 1], 255, 1)

    lm.main_info(f"Extracted boundary line length: {len(ordered_bdl_list)}.")

    return ordered_bdl_list, ordered_bdl_img


def draw_seg_grid(
    borderline_img,
    bdl_seg_coor_x,
    bdl_seg_coor_y,
    gridline_width=1,
    mode="grid",
) -> Optional[np.ndarray]:
    """Draw the grid lines for each layer and column.

    Args:
        borderline_img: The matrix that stores the image information of the borderline between the source and target
            cluster(s).
        bdl_seg_coor_x: The coordinate of i-th layer.
        bdl_seg_coor_y: The coordinate of i+1-th (the consecutive) layer.
        gridline_width: Linewidth of the grid.
        mode: The mode to draw the grid line.

    Returns:
        When `mode` is set to be `grid`, a matrix with the gridlines is created.
    """

    seg_grid_img = np.zeros_like(borderline_img, dtype=np.uint8)

    if len(bdl_seg_coor_x) != len(bdl_seg_coor_y):
        lm.main_info(f"Warning: segmentation does not match between two borderlines. Using the shorter borderline.")

    min_seg_num = min(len(bdl_seg_coor_x), len(bdl_seg_coor_y))
    for i in range(min_seg_num):
        cv2.line(seg_grid_img, bdl_seg_coor_x[i], bdl_seg_coor_y[i], 255, gridline_width)
        if i < min_seg_num - 1:
            cv2.line(seg_grid_img, bdl_seg_coor_x[i], bdl_seg_coor_x[i + 1], 255, gridline_width)
            cv2.line(seg_grid_img, bdl_seg_coor_y[i], bdl_seg_coor_y[i + 1], 255, gridline_width)

    if mode == "grid":  # gridding image
        return seg_grid_img
    elif mode == "gray":
        # TODO: Directly label each region in adata, function fill_grid_label can be merged.
        pass


def euclidean_dist(
    point_x: Tuple,  # geometric coordinate
    point_y: Tuple,
) -> float:
    """Caluate the euclidean distance between two points."""
    return math.sqrt((point_x[0] - point_y[0]) ** 2 + (point_x[1] - point_y[1]) ** 2)


def segment_bd_line(
    borderline_list: List,
    column_num: int,
):
    """Segment the borderline into `column_num` even segments based on the arclength along the borderline.

    Args:
        borderline_list: An order list of np.arrays of coordinates of the borderlines.
        column_num: Number of columns to segment for each layer.

    Returns:
        seg_point_ls: The list of the segmentation points.
    """

    dist_ls = []  # dist between sequence points
    arclen_ls = []  # accumulative arclengths
    dist_seg = []  # length for each segmentation part
    seg_index = []  # index for segmentation points

    arclen = 0
    for i in range(len(borderline_list) - 1):
        dist_ls.append(euclidean_dist(borderline_list[i + 1], borderline_list[i]))
        arclen += dist_ls[i]
        arclen_ls.append(arclen)

    # length per line segment.
    len_per_seg = arclen / column_num
    lm.main_info(
        f"Line total length: {round(arclen, 2)}. Segmenting into {column_num} columns, with {round(len_per_seg, 2)} "
        f"each."
    )

    # The array that will keep the arclen of the line from the latest segment point dynamically.
    dynamic_arclen = np.array(arclen_ls)

    first = True
    for i in range(len(dynamic_arclen)):  # per dist array add.
        # add the start and end index
        if i == 0 or i == len(dynamic_arclen) - 1:
            seg_index.append(i)
        else:
            # When we find a point whose current accumative arclength is larger than required segment length, include
            # the index and subtract all arc_len by the current arc length.
            if (dynamic_arclen[i] >= len_per_seg) and first:  # first step
                error_dist = dynamic_arclen[i] - len_per_seg
                seg_index.append(i)
                dist_seg.append(dynamic_arclen[i])
                dynamic_arclen = dynamic_arclen - dynamic_arclen[i]
                first = False

            # compensate the extra length from the previous segment
            if (dynamic_arclen[i] >= len_per_seg) and (error_dist > 0):
                error_dist = error_dist + dynamic_arclen[i - 1] - len_per_seg
                seg_index.append(i - 1)
                dist_seg.append(dynamic_arclen[i - 1])
                dynamic_arclen = dynamic_arclen - dynamic_arclen[i - 1]

            # compensate the negative length from the previous segment
            elif (dynamic_arclen[i] >= len_per_seg) and (error_dist < 0):
                error_dist = error_dist + dynamic_arclen[i] - len_per_seg
                seg_index.append(i)
                dist_seg.append(dynamic_arclen[i])
                dynamic_arclen = dynamic_arclen - dynamic_arclen[i]

    seg_point_ls = np.array(borderline_list)[seg_index]

    return seg_point_ls  # segmentation point list


def extend_layer(
    borderline_img: np.ndarray,
    borderline_list: List,
    extend_width=10,
) -> Tuple[np.ndarray, List]:
    """Extend the layer to both interior ane exterior sides.

    Args:
        borderline_img: The matrix that stores the image information of the borderline between the source and target
            cluster(s).
        borderline_list: An order list of np.arrays of coordinates of the borderlines.
        extend_width: The layer width to extend.

    Returns:
        extend_layer_img: The matrix that stores the extended layer image.
        extend_layer_bdl: The list of extended layer borderline.
    """

    lm.main_info(f"Generating layer area.")
    extend_layer_mask = np.zeros_like(borderline_img, dtype=np.uint8)
    extend_layer_img = np.zeros_like(borderline_img, dtype=np.uint8)
    for pt in borderline_list:
        cv2.circle(extend_layer_mask, pt, extend_width, 255, -1)

    extend_layer_contour, _ = cv2.findContours(extend_layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(extend_layer_img, extend_layer_contour, -1, 255, 1)

    lm.main_info(f"Refining layer contour.")
    extend_layer_tmp = np.zeros_like(borderline_img, dtype=np.uint8)

    # extend only the start and end point of the border line.
    cv2.circle(extend_layer_tmp, borderline_list[0], extend_width, 255, -1)
    cv2.circle(extend_layer_tmp, borderline_list[-1], extend_width, 255, -1)
    contours_edge, _ = cv2.findContours(extend_layer_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    extend_layer_tmp = np.zeros_like(borderline_img, dtype=np.uint8)
    cv2.drawContours(extend_layer_tmp, contours_edge, -1, 255, 1)

    # remove the contour points formed by the start/end point extensions to keep only two borderlines.
    extend_layer_img = np.where(extend_layer_tmp != 0, 0, extend_layer_img)
    extend_layer_img = (
        morphology.remove_small_objects(extend_layer_img.astype(bool), min_size=5, connectivity=2).astype(np.uint8)
        * 255
    )

    # no start / end points region removed.
    extend_layer_bdl = []  # extended layer boundary line
    for pt in extend_layer_contour[0]:  # should be a single contour
        pt_x = pt[0][0]
        pt_y = pt[0][1]
        if extend_layer_img[pt_y, pt_x] != 0:
            extend_layer_bdl.append((pt_x, pt_y))

    return extend_layer_img, extend_layer_bdl


def field_contour_line(
    ctr_seq: np.ndarray,
    pnt_pos: np.ndarray,
    min_pnt: Tuple[int, int],
    max_pnt: Tuple[int, int],
) -> list:
    """Retrieve the field contour line give min and max values from an ordered set of contour points.

    Args:
        ctr_seq: The numpy array that stores the ordered list of points on the contour.
        pnt_pos: The array that tags the position of all four corner points.
        min_pnt: The point corresponds to the position with minimal heat value.
        max_pnt: The point corresponds to the position with maximal heat value.

    Returns:
        line_seq: The line segment that starts from the point with the minimal heat value to the point with maximal heat
        value.
    """
    ctr_seq_rev = ctr_seq[::-1].copy()
    min_idx = ctr_seq.index(min_pnt)
    max_idx = ctr_seq.index(max_pnt) + 1
    if min_idx < max_idx:
        # contour orientation is the same as the min_pnt to max_pnt orientation
        if sum(pnt_pos[min_idx + 1 : max_idx - 1]) == 0:
            line_seq = ctr_seq[min_idx:max_idx]
        else:
            # when there are other corner points.
            min_idx = ctr_seq_rev.index(min_pnt)
            max_idx = ctr_seq_rev.index(max_pnt) + 1
            # the beginning of the normal sequence and the end of reverse sequence
            line_seq = ctr_seq_rev[min_idx:] + ctr_seq_rev[:max_idx]
    else:
        # reverse
        if sum(pnt_pos[min_idx + 1 :]) + sum(pnt_pos[: max_idx - 1]) == 0:
            line_seq = ctr_seq[min_idx:] + ctr_seq[:max_idx]
        else:
            min_idx = ctr_seq_rev.index(min_pnt)
            max_idx = ctr_seq_rev.index(max_pnt) + 1
            line_seq = ctr_seq_rev[min_idx:max_idx]

    return line_seq


def field_contours(
    contour: np.ndarray,
    pnt_xy: Tuple[int, int],
    pnt_Xy: Tuple[int, int],
    pnt_xY: Tuple[int, int],
    pnt_XY: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify four boundary lines according to given corner points.

    Args:
        contour: Contours generated by `cv2.findContours`.
        pnt_xy: Corner point to define an area of interest. pnt_xy corresponds to the point with minimal layer and
            minimal column value.
        pnt_Xy: Corner point corresponds to the point with maximal column value but minimal layer value.
        pnt_xY: Corner point corresponds to the point with minimal column value but maximal layer value.
        pnt_XY: Corner point corresponds to the point with maximal layer and maximal columns value.

    Returns:
        min_line_l: The np array of the points on the layer with minimal layer heat values.
        max_line_l: The np array of the points on the layer with maximal layer heat values.
        min_line_c: The np array of the points on the layer with minimal column heat values.
        max_line_c: The np array of the points on the layer with maximal column heat values.
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


def add_eh_boundary(
    heat_field: np.ndarray,
    field_line: np.ndarray,
    value: float,
) -> None:
    """Set equal heat value to the boundary line on the heat field.

    Args:
        heat_field: The field of the spatial domain of interests.
        field_line: The isoline on the field of the spatial domain of interests.
        value: The value that will be assigned to the isoline.

    Returns:
        Nothing but set the provided isoline to the specific value.
    """

    for x, y in field_line:
        heat_field[y, x] = value


def add_gh_boundary(
    heat_field: np.ndarray,
    field_line: np.ndarray,
    value_s: float,
    value_e: float,
) -> None:
    """Increase heat value progressively along the boundary line on the heat field.

    Args:
        heat_field: the field of the spatial domain of interest
        field_line: The line on the field of the spatial domain of interests that should have increasing heat values.
        value_s: Source heat value.
        value_e: End heat value.

    Returns:
        Nothing but set the provided line to the growing heat value.
    """

    gp_value = np.linspace(value_s, value_e, len(field_line))
    idx = 0
    for x, y in field_line:
        heat_field[y, x] = gp_value[idx]
        idx += 1


def effective_L2_error(
    heat_field_i: np.ndarray,
    heat_field_j: np.ndarray,
    field_mask: np.ndarray,
) -> float:
    """Calculate effective L2 error between two fields.

    Args:
        heat_field_i: The target field used in solving the heat equation.
        heat_field_j: The source field used in solving the heat equation.
        field_mask: The domain of interests (1 if inside the domain and 0 otherwise).

    Returns:
        A float variable of the L2 difference between two fields, normalized by the source field.
    """

    return np.sqrt(np.sum((heat_field_j - heat_field_i) ** 2 * field_mask) / np.sum(heat_field_j**2 * field_mask))


def domain_heat_eqn_solver(
    heat_field: np.ndarray,
    min_line: np.ndarray,
    max_line: np.ndarray,
    edge_line_a: np.ndarray,
    edge_line_b: np.ndarray,
    field_border: np.ndarray,
    field_mask: np.ndarray,
    max_err: float = 1e-5,
    max_itr: float = 1e5,
    lh: float = 1,
    hh: float = 100,
) -> np.ndarray:
    """Given the boundaries and boundary conditions of a close spatial domain, solve heat equation (a simple partial
    differential equation) to define the "heat" for each spatial pixel which can be used to digitize the
    spatial domain into different layers or columns. Diffusitivity is set to be 1/4, thus the update rule is defined as:

        grid_field[1:-1, 1:-1] = 0.25 * (
            grid_field_pre[1:-1, 2:] + grid_field_pre[1:-1, :-2] + grid_field_pre[2:, 1:-1] + grid_field_pre[:-2, 1:-1]
        )

    Args:
        heat_field: The field of the spatial domain of interests.
        min_line: The np array of the isoline points with minimal heat values.
        max_line: The np array of the isoline points with maximal  heat values.
        edge_line_a: The np array of the points with increasing heat values, orthogonal to the isolines.
        edge_line_b: The np array of the points with increasing heat values, orthogonal to the isolines.
        field_border: The border of the field of the spatial domain of interests.
        field_mask: The field of the spatial domain of interests, used for masking.
        max_err: The maximal tolerated error. Default to 1e-5.
        max_itr: The maximal diffusion iteration error. Default to 1e5.
        lh: Lowest heat value. Defaults to 1.
        hh: Highest heat value. Defaults to 100.

    Returns:
        grid_field: The resultant field filled with final values after solving the heat equation.
    """

    init_field = heat_field.copy()
    add_eh_boundary(init_field, min_line, lh)
    add_eh_boundary(init_field, max_line, hh)
    add_gh_boundary(init_field, edge_line_a, lh, hh)
    add_gh_boundary(init_field, edge_line_b, lh, hh)

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
