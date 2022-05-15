"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from nptyping import NDArray

from ..configuration import SKM
from ..logging import logger_manager as lm
from .contour import *
from .utils import *


def identify_boundary(
    adata: AnnData,
    cluster_key,
    source_id,
    target_id,
    bin_size: int = 1,
    spatial_key: str = "spatial",
    boundary_key: str = "boundary_line",
    k_size=8,
    min_area=30,
    dilate_k_size: int = 3,
):

    lm.main_info(f"Setting up source and target area.")
    adata_tmp = adata.copy()
    adata_tmp.obs["tmp_boundary"] = 0
    adata_tmp.obs["tmp_boundary"][adata_tmp.obs[cluster_key].isin(source_id)] = 1
    adata_tmp.obs["tmp_boundary"][adata_tmp.obs[cluster_key].isin(target_id)] = 2

    lm.main_info(f"Identifying boundary.")
    boundary_img = gen_cluster_image(adata_tmp, bin_size, spatial_key, "tmp_boundary", show=False)
    source_label = np.unique(
        adata_tmp[
            np.isin(
                adata_tmp.obs["tmp_boundary"],
                [
                    1,
                ],
            ),
            :,
        ].obs["cluster_img_label"]
    )
    target_label = np.unique(
        adata_tmp[
            np.isin(
                adata_tmp.obs["tmp_boundary"],
                [
                    2,
                ],
            ),
            :,
        ].obs["cluster_img_label"]
    )
    _, _, ctr_img = extract_cluster_contours(
        boundary_img, source_label, bin_size=bin_size, k_size=k_size, min_area=min_area, show=False
    )
    _, tgt_img, _ = extract_cluster_contours(
        boundary_img, target_label, bin_size=bin_size, k_size=k_size, min_area=min_area, show=False
    )

    dilate_kernel = np.ones((dilate_k_size, dilate_k_size), np.uint8)
    tgt_img = cv2.dilate(tgt_img, dilate_kernel, iterations=1)

    lm.main_info(f"Generating boundary line image.")
    boundary_line_img = np.where(tgt_img != 0, ctr_img, 0)

    lm.main_info(f"Saving boundary into adata.obs['{boundary_key}'] for visualization.")
    adata.obs[boundary_key] = False
    for i in range(len(adata)):
        if boundary_line_img[int(adata.obsm[spatial_key][i, 0]), int(adata.obsm[spatial_key][i, 1])] != 0:
            adata.obs["boundary_line"][i] = True

    return boundary_line_img


def boundary_gridding(
    adata: AnnData,
    boundary_line_img,
    boundary_line_list,
    n_layer=3,
    n_column=25,
    layer_width=10,
    spatial_key: str = "spatial",
    init: bool = False,
):

    bdl_seg_inner_list = []
    bdl_seg_outer_list = []

    bdl_seg_ori = segment_bd_line(boundary_line_list, n_column)

    for i_layer in range(n_layer):
        curr_layer = i_layer + 1
        extend_width = layer_width * curr_layer

        img_ex, ext_bdl_list = extend_layer(boundary_line_img, boundary_line_list, extend_width=extend_width)
        # plt.imshow(img_ex)

        ext_bdl_list_tmp = ext_bdl_list.copy()
        ext_bdl_list_tmp.append(ext_bdl_list_tmp[0])
        edge_point_index = []
        for i in range(len(ext_bdl_list_tmp) - 1):
            max_manh_dist = max(
                abs(ext_bdl_list_tmp[i][0] - ext_bdl_list_tmp[i + 1][0]),
                abs(ext_bdl_list_tmp[i][1] - ext_bdl_list_tmp[i + 1][1]),
            )
            if max_manh_dist > 1:
                edge_point_index.append(i)

        ext_bdl_inner = ext_bdl_list[edge_point_index[0] + 1 : edge_point_index[1] + 1]
        ext_bdl_outer = ext_bdl_list[edge_point_index[1] + 1 :] + ext_bdl_list[: edge_point_index[0] + 1]
        ext_bdl_outer = ext_bdl_outer[::-1]

        ext_bdl_inner_seg = segment_bd_line(ext_bdl_inner, n_column)
        bdl_seg_inner_list.append(ext_bdl_inner_seg)

        ext_bdl_outer_seg = segment_bd_line(ext_bdl_outer, n_column)
        bdl_seg_outer_list.append(ext_bdl_outer_seg)

    bdl_seg_all_list = bdl_seg_inner_list[::-1] + [bdl_seg_ori] + bdl_seg_outer_list
    for i_layer in range(n_layer * 2):
        curr_layer_num = i_layer % 3 + 1
        curr_sign = (-1) ** (i_layer // 3 + 1)

        seg_grid_img = draw_seg_grid(boundary_line_img, bdl_seg_all_list[i_layer], bdl_seg_all_list[i_layer + 1])

        fill_grid_label(
            adata,
            spatial_key,
            seg_grid_img,
            bdl_seg_all_list[i_layer],
            bdl_seg_all_list[i_layer + 1],
            curr_layer_num,
            curr_sign,
            init=init and (i_layer == 0),
        )
