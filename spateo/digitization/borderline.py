"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

from typing import List

import cv2
import numpy as np
from anndata import AnnData

from ..configuration import SKM
from ..logging import logger_manager as lm
from .contour import *
from .utils import *


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def get_borderline(
    adata: AnnData,
    cluster_key: str,
    source_clusters: int,
    target_clusters: int,
    bin_size: int = 1,
    spatial_key: str = "spatial",
    borderline_key: str = "borderline",
    k_size: int = 8,
    min_area: int = 30,
    dilate_k_size: int = 3,
) -> np.ndarray:
    """Identify the borderline at the interface of the source and target cell clusters.

    The borderline will be identified by first retrieving the outline/contour formed by the source clusters, which will
    then be cleaned up to retrieve the borderline by masking with the expanded contours formed by the target clusters.

    Args:
        adata: The adata object to be used for identifying the borderline.
        cluster_key: The key name of the spatial cluster in `adata.obs`
        source_clusters: The source cluster(s) that will interface with the target clusters.
        target_clusters: The target cluster(s) that will interface with the source clusters.
        bin_size: The size of the binning.
        spatial_key: The key name of the spatial coordinates in `adata.obs`
        borderline_key: The key name in `adata.obs` that will be used to store the borderline.
        k_size: Kernel size of the elliptic structuring element.
        min_area: Minimal area threshold corresponding to the resulting contour(s).
        dilate_k_size: Kernel size of the cv2.dilate function.

    Returns:
        borderline_img: The matrix that stores the image information of the borderline between the source and target
            cluster(s). Note that the adata object will also be updated with the `boundary_line` key that stores the
            information about whether the bucket is on the borderline.
    """

    lm.main_info(f"Setting up source and target area.")
    adata_tmp = adata.copy()
    adata_tmp.obs["tmp_borderline"] = 0
    adata_tmp.obs["tmp_borderline"][adata_tmp.obs[cluster_key].isin(source_clusters)] = 1
    adata_tmp.obs["tmp_borderline"][adata_tmp.obs[cluster_key].isin(target_clusters)] = 2

    lm.main_info(f"Produce a joint cluster image formed by source and target clusters.")
    boundary_img = gen_cluster_image(adata_tmp, bin_size, spatial_key, "tmp_borderline", show=False)

    lm.main_info(f"Retrieve the cluster label belong to either the source or target clusters.")
    source_label = np.unique(
        adata_tmp[
            np.isin(
                adata_tmp.obs["tmp_borderline"],
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
                adata_tmp.obs["tmp_borderline"],
                [
                    2,
                ],
            ),
            :,
        ].obs["cluster_img_label"]
    )

    lm.main_info(f"Retrieve the contour belong to either the source or target clusters.")
    _, _, ctr_img = extract_cluster_contours(
        boundary_img, source_label, bin_size=bin_size, k_size=k_size, min_area=min_area, show=False
    )
    _, tgt_img, _ = extract_cluster_contours(
        boundary_img, target_label, bin_size=bin_size, k_size=k_size, min_area=min_area, show=False
    )

    lm.main_info(f"Dilate target filled contour image.")
    dilate_kernel = np.ones((dilate_k_size, dilate_k_size), np.uint8)
    tgt_img = cv2.dilate(tgt_img, dilate_kernel, iterations=1)

    lm.main_info(f"Get borderline by masking expanded target contour filling with source contour.")
    borderline_img = np.where(tgt_img != 0, ctr_img, 0)

    lm.main_info(f"Saving borderline into adata.obs['{borderline_key}'] for visualization.")
    adata.obs[borderline_key] = " "
    for i in range(len(adata)):
        if borderline_img[int(adata.obsm[spatial_key][i, 0]), int(adata.obsm[spatial_key][i, 1])] != 0:
            adata.obs[borderline_key][i] = "Borderline"

    return borderline_img.astype(np.uint8)


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def grid_borderline(
    adata: AnnData,
    borderline_img: np.ndarray,
    borderline_list: List,
    layer_num: int = 3,
    column_num: int = 25,
    layer_width: int = 10,
    spatial_key: str = "spatial",
    init: bool = False,
) -> None:
    """Extend the borderline to either interior or exterior side to each create `layer_num` layers, and segment such
        layers to `column_num` columns.

    Args:
        adata: The adata object to be used for identifying the interior/exterior layers and columns.
        borderline_img: The matrix that stores the image information of the borderline between the source and target
            cluster(s).
        borderline_list: An order list of np.arrays of coordinates of the borderlines.
        layer_num: Number of layers to extend on either interior or exterior side.
        column_num: Number of columns to segment for each layer.
        layer_width: Layer/column boundary width. This only affects grid_label.
        spatial_key: The key name in `adata.obsm` of the spatial coordinates. Default to "spatial". Passed to
            `fill_grid_label` function.
        init: Whether to generate (and potentially overwrite) the `layer_label_key` and `column_label_key` in
            `fill_grid_label` function.

    Returns:
        Nothing but update the adata object with following keys in `.obs`:
        1. layer_label_key: this key points to layer labels.
        2. column_label_key: this key points to column labels.
    """

    lm.main_info(f"Segment the initial borderline.")
    bdl_seg_ori = segment_bd_line(borderline_list, column_num)

    bdl_seg_inner_list = []
    bdl_seg_outer_list = []

    lm.main_info(f"Prepare lists of interior/exterior line segments.")
    for i_layer in range(layer_num):
        curr_layer = i_layer + 1
        extend_width = layer_width * curr_layer

        img_ex, ext_bdl_list = extend_layer(borderline_img, borderline_list, extend_width=extend_width)

        ext_bdl_list_tmp = ext_bdl_list.copy()
        ext_bdl_list_tmp.append(ext_bdl_list_tmp[0])  # because it is a loop

        # store the indices of four end points of the interior and exterior borderlines.
        end_points_indices = []
        for i in range(len(ext_bdl_list_tmp) - 1):
            max_bdl_dist = max(
                abs(ext_bdl_list_tmp[i][0] - ext_bdl_list_tmp[i + 1][0]),
                abs(ext_bdl_list_tmp[i][1] - ext_bdl_list_tmp[i + 1][1]),
            )
            # the consecutive points on the borderlines should be smaller than 1.
            if max_bdl_dist > 1:
                end_points_indices.append(i)

        # the interior and exterior borders are arrranged accordingly in the ext_bdl_list.
        ext_bdl_inner = ext_bdl_list[end_points_indices[0] + 1 : end_points_indices[1] + 1]
        ext_bdl_outer = ext_bdl_list[end_points_indices[1] + 1 :] + ext_bdl_list[: end_points_indices[0] + 1]

        # inverse ext_bdl_outer, so that interior and exterior are ordered in the same orientation.
        ext_bdl_outer = ext_bdl_outer[::-1]

        # segment and appended interior lines
        ext_bdl_inner_seg = segment_bd_line(ext_bdl_inner, column_num)
        bdl_seg_inner_list.append(ext_bdl_inner_seg)

        # segment and appended exterior line
        ext_bdl_outer_seg = segment_bd_line(ext_bdl_outer, column_num)
        bdl_seg_outer_list.append(ext_bdl_outer_seg)

    lm.main_info(f"Assign the interior/exterior layers and columns, and grid labels for each bucket.")

    # order borderlines from the most inside to most outside.
    bdl_seg_all_list = bdl_seg_inner_list[::-1] + [bdl_seg_ori] + bdl_seg_outer_list
    for i_layer in range(layer_num * 2):
        curr_layer_num = i_layer % layer_num + 1
        curr_sign = (-1) ** (i_layer // layer_num + 1)  # interior layers will have negative values.

        seg_grid_img = draw_seg_grid(borderline_img, bdl_seg_all_list[i_layer], bdl_seg_all_list[i_layer + 1])

        # this function returns layer_grid_img, column_grid_img
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
