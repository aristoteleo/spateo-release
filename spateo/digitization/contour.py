"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

import random
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData
from skimage import morphology

from ..configuration import SKM
from ..logging import logger_manager as lm


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def gen_cluster_image(
    adata: AnnData,
    bin_size: Optional[int] = None,
    spatial_key: str = "spatial",
    cluster_key: str = "scc",
    label_mapping_key: str = "cluster_img_label",
    show: bool = True,
    cmap: str = "tab20",
) -> np.ndarray:
    """Generate images with spatial cluster(s) a distinct color prepared from the designated cmap.

    Args:
        adata: The adata object used to create the image for cluster(s).
        bin_size: The size of the binning.
        spatial_key: The key name of the spatial coordinates.
        cluster_key: The key name of the spatial cluster.
        label_mapping_key: The key name to store the mapping between cluster name and label index values.
        show: Visualize the cluster image.
        cmap: The colormap that will be used to draw colors for the resultant cluster image(s).

    Returns:
        cluster_image: A numpy array or a list of numpy arrays that store the image of each cluster, each with a
        distinct color.
    """

    import matplotlib.pyplot as plt

    if bin_size is None:
        bin_size = adata.uns["bin_size"]

    lm.main_info(f"Set up the color for the clusters with the {cmap} colormap.")

    # TODO: what if cluster number is larger than cmap.N?
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    color_ls = []
    for i in range(cmap.N):
        color_ls.append(tuple(np.array(colors[i][:3] * 255).astype(int)))

    random.seed(1)
    color_ls_cut = random.sample(color_ls, len(np.unique(adata.obs[cluster_key])))

    lm.main_info(f"Saving integer labels for clusters into adata.obs['{label_mapping_key}'].")

    # background is 0, so adata.obs[label_mapping] start from 1
    adata.obs[label_mapping_key] = 0
    cluster_list = np.unique(adata.obs[cluster_key])
    for i in range(len(cluster_list)):
        adata.obs[label_mapping_key][adata.obs[cluster_key] == cluster_list[i]] = i + 1

    # get cluster image
    lm.main_info(f"Prepare a mask image and assign each pixel to the corresponding cluster id.")

    max_coords = [int(np.max(adata.obsm[spatial_key][:, 0])) + 1, int(np.max(adata.obsm[spatial_key][:, 1])) + 1]

    cluster_label_image = np.zeros((max_coords[0], max_coords[1]), np.uint8)

    for i in range(len(adata)):
        # fill the image (mask) with the label
        cv2.circle(
            img=cluster_label_image,
            center=(int(adata.obsm[spatial_key][i, 1]), int(adata.obsm[spatial_key][i, 0])),
            radius=bin_size // 2,
            color=int(adata.obs[label_mapping_key][i]),
            thickness=-1,
        )

    if show:
        lm.main_info(f"Plot the cluster image with the color(s) in the color list.")
        cluster_rgb_image = np.zeros((max_coords[0], max_coords[1], 3), np.uint8)
        for i in np.unique(adata.obs[label_mapping_key]):
            cluster_rgb_image[cluster_label_image == i] = color_ls_cut[i - 1]
        plt.imshow(cluster_rgb_image)

    return cluster_label_image


def extract_cluster_contours(
    cluster_image: np.ndarray,
    cluster_labels: Union[int, List],
    bin_size: int,
    k_size: float = 2,
    min_area: float = 9,
    close_kernel: int = cv2.MORPH_ELLIPSE,
    show: bool = True,
) -> Tuple[Any, Any, np.ndarray]:
    """Extract contour(s) for area(s) formed by buckets of the same identified cluster.

    Args:
        cluster_image: the image that sets the pixels of the cluster of interests as the front color (background is 0).
        cluster_labels: The label values of clusters of interests.
        bin_size: The size of the binning.
        k_size: kernel size of the elliptic structuring element.
        min_area: minimal area threshold corresponding to the resulting contour(s).
        close_kernel:  The value to indicate the structuring element. By default, we use a circular structuring element.
        show: Visualize the result.

    Returns:
        contours: The coordinates of contors identified.
        cluster_image_close: The resultant image of the area of interest.
        cluster_image_contour: The resultant image of the contour.
    """

    import matplotlib.pyplot as plt

    k_size = int(k_size * bin_size)
    min_area = int(min_area * bin_size * bin_size)

    lm.main_info(f"Get selected areas with label id(s): {cluster_labels}.")
    cluster_image_close = cluster_image.copy()
    if type(cluster_labels) == int:
        cluster_image_close = np.where(cluster_image_close == cluster_labels, cluster_image_close, 0)
    else:
        cluster_image_close = np.where(np.isin(cluster_image_close, cluster_labels), cluster_image_close, 0)

    lm.main_info("Close cluster morphology.")
    kernal = cv2.getStructuringElement(close_kernel, (k_size, k_size))
    cluster_image_close = cv2.morphologyEx(cluster_image_close, cv2.MORPH_CLOSE, kernal)

    lm.main_info("Remove small region.")
    cluster_image_close = morphology.remove_small_objects(
        cluster_image_close.astype(bool),
        min_area,
        connectivity=2,
    ).astype(np.uint8)

    lm.main_info("Extract contours.")
    contours, _ = cv2.findContours(cluster_image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cluster_image_contour = np.zeros((cluster_image.shape[0], cluster_image.shape[1]))
    for i in range(len(contours)):
        cv2.drawContours(cluster_image_contour, contours, i, i + 1, bin_size)

    if show:
        lm.main_info("Showing extracted contours.")
        plt.imshow(cluster_image_contour)

    return contours, cluster_image_close, cluster_image_contour


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata_high_res")
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata_low_res", optional=True)
def set_domains(
    adata_high_res: AnnData,
    adata_low_res: Optional[AnnData] = None,
    spatial_key: str = "spatial",
    cluster_key: str = "scc",
    domain_key_prefix: str = "domain",
    bin_size_low: Optional[int] = None,
    k_size: float = 2,
    min_area: float = 9,
) -> None:
    """Set the domains for each bucket based on spatial clusters.

    Args:
        adata_high_res: The anndata object in high spatial resolution.
        adata_low_res: The anndata object in low spatial resolution.
        spatial_key: The key `.obsm` to the spatial coordinate of each bucket. Should be consistent in both
            `adata_high_res` and `adata_low_res`.
        cluster_key: The key in `.obs` to the spatial cluster.
        domain_key_prefix: The key prefix in `.obs` that will be used to store the spatial domain for each bucket.
        bin_size_low: The binning size of the adata_low_res object (when provided).
        k_size: kernel size of the elliptic structuring element.
        min_area: minimal area threshold corresponding to the resulting contour(s).

    Returns:
        Nothing but update the `adata_high_res` with the `domain` in `domain_key_prefix` + "_" + `cluster_key`.
    """

    domain_key = domain_key_prefix + "_" + cluster_key

    if adata_low_res is None:
        adata_low_res = adata_high_res

    if bin_size_low is None:
        bin_size_low = adata_low_res.uns["bin_size"]

    lm.main_info(f"Generate the cluster label image with `gen_cluster_image`.")
    cluster_label_image = gen_cluster_image(
        adata_low_res, bin_size=bin_size_low, spatial_key=spatial_key, cluster_key=cluster_key, show=False
    )

    lm.main_info(f"Iterate through each cluster and identify contours with `extract_cluster_contours`.")
    # TODO need a more stable mapping for ids and labels
    u, count = np.unique(adata_low_res.obs[cluster_key], return_counts=True)
    count_sort_ind = np.argsort(-count)
    cluster_ids = u[count_sort_ind]
    cluster_ids = [str(c) for c in cluster_ids]

    u, count = np.unique(adata_low_res.obs["cluster_img_label"], return_counts=True)
    # `cluster_img_label` is produced from  `cluster_key`, so use the same count_sort_ind
    cluster_labels = u[count_sort_ind]
    cluster_labels = [c for c in cluster_labels]

    adata_high_res.obs[domain_key] = "NA"

    for i in range(len(cluster_ids)):
        ctrs, _, _ = extract_cluster_contours(
            cluster_label_image, cluster_labels[i], bin_size=bin_size_low, k_size=k_size, min_area=min_area, show=False
        )
        for j in range(len(adata_high_res)):
            x = adata_high_res.obsm[spatial_key][j, 0]
            y = adata_high_res.obsm[spatial_key][j, 1]
            for k in range(len(ctrs)):
                if cv2.pointPolygonTest(ctrs[k], (y, x), False) >= 0:
                    adata_high_res.obs[domain_key][j] = cluster_ids[i]
                    # assume one bucket to one domain mapping
                    break
