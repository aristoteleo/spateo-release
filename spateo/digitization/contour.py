"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

import random
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from nptyping import NDArray
from skimage import morphology

from ..configuration import SKM
from ..logging import logger_manager as lm


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, optional=True)
def gen_cluster_images(
    adata: AnnData,
    bin_size: int,
    cluster_id: Union[int, List],
    spatial_key: str = "spatial",
    cluster_key: str = "scc",
    cmap: str = "tab20",
) -> Union[NDArray[np.uint16], List]:
    """Generate images with each cell cluster(s) a distinct color prepared from the designated cmap.

    Args:
        adata: The adata object used to create the image for cluster(s).
        bin_size: The size of the binning.
        cluster_id: The cluster id of interests.
        spatial_key: The key name of the spatial coordinates.
        cluster_key: The key name of the cell cluster.
        cmap: The colormap that will be used to draw colors for the resultant cluster image(s).

    Returns:
        color_cluster: A numpy array or a list of numpy arrays that store the image of each cluster, each with a
        distinct color.
    """

    lm.main_info(f"Set up the color for the clusters with the {cmap} colormap.")
    import matplotlib.pyplot as plt

    coord = adata.obsm[spatial_key]
    labels = adata.obs[cluster_key]

    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    color_ls = []
    for i in range(cmap.N):
        color_ls.append(tuple(np.array(colors[i][:3] * 255).astype(int)))

    # get cluster image
    lm.main_info(f"Prepare a mask image and assign each pixel to the corresponding cluster id.")
    x_ls = []
    y_ls = []
    for i in range(len(coord)):
        x = int(coord[i][0])
        y = int(coord[i][1])
        x_ls.append(x)
        y_ls.append(y)

    max_x, max_y = max(x_ls), max(y_ls)
    c, r = max_x, max_y

    mask = np.zeros((r, c), np.uint8)
    color_cluster = np.zeros((r, c, 3), np.uint16)
    for j in range(len(coord)):
        x = int(coord[j][0])
        y = int(coord[j][1])
        label = int(labels[j]) + 1  # background is 0, so cluster value start from 1

        # fill the image (mask) with the label
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(mask, (x, y), int(bin_size / 2), label, -1)

    random.seed(10000)
    color_ls_cut = random.sample(color_ls, len(np.unique(labels)))

    lm.main_info(f"Generate the cluster image with the color(s) in the color list.")
    if type(cluster_id) == int:
        color_cluster[mask == cluster_id + 1] = color_ls_cut[cluster_id]
        return color_cluster
    elif type(cluster_id) == List:
        color_cluster = []
        for cur_c in cluster_id:
            tmp = color_cluster.copy()
            tmp[mask == cur_c + 1] = color_ls_cut[cur_c]
            color_cluster.append(tmp)

        return color_cluster


def extract_cluster_contours(
    cluster_id_img: NDArray[np.uint8], ksize: float, min_area: float
) -> Tuple[NDArray, Tuple[NDArray]]:
    """Extract contour(s) for area(s) formed by buckets of the same identified cluster.

    Args:
        cluster_id_img: the image that sets the pixels of the cluster of interests as the front color (background is 0).
        ksize: kernel size of the elliptic structuring element.
        min_area: minimal area threshold corresponding to the resulting contour(s).

    Returns:
        close_img: The resultant image with the contour identified.
        contours: The coordinates of contors identified.
    """

    c, r = cluster_id_img.shape[:2]
    contours_1pixel = np.zeros((r, c), np.uint8)

    lm.main_info("Convert BGR cluster image to GRAY image.")
    cluster_img_gray = cv2.cvtColor(cluster_id_img, cv2.COLOR_BGR2GRAY)
    lm.main_info("Use MORPH_ELLIPSE to close cluster morphology.")
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    close = cv2.morphologyEx(cluster_img_gray, cv2.MORPH_CLOSE, kernal)

    lm.main_info("Remove small region.")
    close_img_bool = close.astype(bool)
    close_img = np.uint8(morphology.remove_small_objects(close_img_bool, min_area, connectivity=2).astype(int))

    lm.main_info("Extract contours.")
    contours, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lm.main_info("Draw contours.")
    # cv.drawContours(	image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]	) ->	image
    cv2.drawContours(contours_1pixel, contours, -1, 255, 1)
    return close_img, contours


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata_high_res", optional=True)
def set_domains(
    adata_high_res: AnnData,
    adata_low_res: Optional[AnnData] = None,
    spatial_key: str = "spatial",
    cluster_key: str = "scc",
    domain_key: str = "domain",
    bin_size: int = 60,
    cmap: str = "tab20",
    ksize: float = 110,
    min_area: float = 50000,
) -> None:
    """Set the domains for each bucket based on spatial clusters.

    Args:
        adata_high_res: The anndata object in high spatial resolution.
        adata_low_res: The anndata object in low spatial resolution.
        spatial_key: The key to the spatial coordinate of each bucket. Should be consistent in both `adata_high_res` and
            `adata_low_res`.
        cluster_key: The key in `.obs` to the cell cluster.
        domain_key: The key in `.obs` that will be used to store the spatial domain for each bucket.
        bin_size: The size of the binning.
        cmap: The colormap that will be used to draw colors for the resultant cluster image(s).
        ksize: kernel size of the elliptic structuring element.
        min_area: minimal area threshold corresponding to the resulting contour(s).

    Returns:
        Nothing but update the `adata_high_res` with the `domain` in `domain_key`.
    """

    if adata_low_res is None:
        adata_low_res = adata_high_res

    lm.main_info(f"Generate the cluster images with `gen_cluster_images`.")
    coord = adata_high_res.obsm[spatial_key]
    adata_high_res.obs[domain_key] = "-1"

    u, count = np.unique(adata_low_res.obs[cluster_key], return_counts=True)
    count_sort_ind = np.argsort(-count)

    cluster_ids = u[count_sort_ind]
    cluster_ids = [int(c) for c in cluster_ids]
    color_cluster_list = gen_cluster_images(adata_low_res, bin_size, cluster_ids, spatial_key, cluster_key, cmap)

    lm.main_info(f"Iterate through each cluster and identify contours with `extract_cluster_contours`.")
    for c, color_cluster in zip(cluster_ids, color_cluster_list):
        close_img, contours = extract_cluster_contours(np.uint8(color_cluster), ksize, min_area)

        for i in range(len(coord)):
            x = coord[i][0]
            y = coord[i][1]
            for j in range(len(contours)):
                if cv2.pointPolygonTest(contours[j], (x, y), False) >= 0:
                    adata_high_res.obs[domain_key][i] = c
