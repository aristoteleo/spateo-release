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
    cmap: str = "tab20",
    show: bool = True,
) -> np.ndarray:
    """Generate matrix/image of spatial clusters with distinct labels/colors.

    Args:
        adata: The adata object used to create the matrix/image for clusters.
        bin_size: The size of the binning.
        spatial_key:  The key name of the spatial coordinates in `adata.obs`
        cluster_key: The key name of the spatial cluster in `adata.obs`
        label_mapping_key: The key name to store the label index values, mapped from the cluster names in `adata.obs`.
            Note that background is 0 so `label_mapping_key` starts from 1.
        cmap: The colormap that will be used to draw colors for the resultant cluster image.
        show: Whether to visualize the cluster image.

    Returns:
        cluster_label_image: A numpy array that stores the image of clusters, each with a distinct color. When `show`
            is True, `plt.imshow(cluster_rgb_image)` will be used to plot the clusters each with distinct labels
            prepared from the designated cmap.
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
    color_ls_sampled = random.sample(color_ls, len(np.unique(adata.obs[cluster_key])))

    lm.main_info(f"Saving integer labels for clusters into adata.obs['{label_mapping_key}'].")

    # background is 0, so adata.obs[label_mapping_key] starts from 1
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
            cluster_rgb_image[cluster_label_image == i] = color_ls_sampled[i - 1]
        plt.imshow(cluster_rgb_image)

    return cluster_label_image


def extract_cluster_contours(
    cluster_label_image: np.ndarray,
    cluster_labels: Union[int, List],
    bin_size: int,
    k_size: float = 2,
    min_area: float = 9,
    close_kernel: int = cv2.MORPH_ELLIPSE,
    show: bool = True,
) -> Tuple[Tuple, np.ndarray, np.ndarray]:
    """Extract contour(s) for area(s) formed by buckets of the same spatial cluster.

    Args:
        cluster_label_image: the image that sets the pixels of the cluster of interests as the front color (background
            is 0).
        cluster_labels: The label value(s) of clusters of interests.
        bin_size: The size of the binning.
        k_size: Kernel size of the elliptic structuring element.
        min_area: Minimal area threshold corresponding to the resulting contour(s).
        close_kernel: The value to indicate the structuring element. By default, we use a circular structuring element.
        show: Visualize the result.

    Returns:
        contours: The Tuple coordinates of contours identified.
        cluster_image_close: The resultant image of the area of interest with small area removed.
        cluster_image_contour: The resultant image of the contour, generated from `cluster_image_close`.
    """

    import matplotlib.pyplot as plt

    k_size = int(k_size * bin_size)
    min_area = int(min_area * bin_size * bin_size)

    lm.main_info(f"Get selected areas with label id(s): {cluster_labels}.")
    cluster_image_close = cluster_label_image.copy()
    if type(cluster_labels) == int:
        cluster_image_close = np.where(cluster_image_close == cluster_labels, cluster_image_close, 0)
    else:
        cluster_image_close = np.where(np.isin(cluster_image_close, cluster_labels), cluster_image_close, 0)

    lm.main_info(f"Close morphology of the area formed by cluster {cluster_labels}.")
    kernal = cv2.getStructuringElement(close_kernel, (k_size, k_size))
    cluster_image_close = cv2.morphologyEx(cluster_image_close, cv2.MORPH_CLOSE, kernal)

    lm.main_info("Remove small region(s).")
    cluster_image_close = morphology.remove_small_objects(
        cluster_image_close.astype(bool),
        min_area,
        connectivity=2,
    ).astype(np.uint8)

    lm.main_info("Extract contours.")
    contours, _ = cv2.findContours(cluster_image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cluster_image_contour = np.zeros((cluster_label_image.shape[0], cluster_label_image.shape[1]))
    for i in range(len(contours)):
        cv2.drawContours(cluster_image_contour, contours, i, i + 1, bin_size)

    if show:
        lm.main_info("Plotting extracted contours.")
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
    bin_size_high: Optional[int] = None,
    bin_size_low: Optional[int] = None,
    k_size: float = 2,
    min_area: float = 9,
) -> None:
    """Set the domains for each bucket based on spatial clusters. Use adata object of low resolution for contour
        identification but adata object of high resolution  for domain assignment.

    Args:
        adata_high_res: The anndata object in high spatial resolution. The adata with smaller binning (or single cell
            segmetnation) is more suitable to define more fine grained spatial domains.
        adata_low_res: The anndata object in low spatial resolution. When using data with big binning, it can often
            produce better spatial domain clustering results with the `scc` method and thus domain/domain contour
            identification.
        spatial_key: The key in `.obsm` of the spatial coordinate for each bucket. Should be same key in both
            `adata_high_res` and `adata_low_res`.
        cluster_key: The key in `.obs` (`adata_low_res`) to the spatial cluster.
        domain_key_prefix: The key prefix in `.obs` (in `adata_high_res`) that will be used to store the spatial domain
            for each bucket. The full key name will be set as: `domain_key_prefix` + "_" + `cluster_key`.
        bin_size_low: The binning size of the `adata_high_res` object.
        bin_size_low: The binning size of the `adata_low_res` object (only works when `adata_low_res` is provided).
        k_size: Kernel size of the elliptic structuring element.
        min_area: Minimal area threshold corresponding to the resulting contour(s).

    Returns:
        Nothing but update the `adata_high_res` with the `domain` in `domain_key_prefix` + "_" + `cluster_key`.
    """

    domain_key = domain_key_prefix + "_" + cluster_key

    if bin_size_high is None:
        bin_size_high = adata_high_res.uns["bin_size"]

    if adata_low_res is None:
        adata_low_res = adata_high_res
        bin_size_low = bin_size_high
    elif bin_size_low is None:
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
