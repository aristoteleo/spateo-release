"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from anndata import AnnData

from ...configuration import SKM
from .utils import save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def spatial_domains(
    adata: AnnData,
    bin_size: Optional[int] = None,
    spatial_key: str = "spatial",
    label_key: str = "cluster_img_label",
    plot_size=(3, 3),
    save_img="spatial_domains.png",
):
    """Generate an image with contours of each spatial domains.

    Args:
        adata: The adata object used to create the image.
        bin_size: The size of the binning. Default to None.
        spatial_key: The key name of the spatial coordinates. Default to "spatial".
        label_key: The key name of the image label values. Default to "cluster_img_label".
        plot_size: figsize for showing the image.
        save_img: path to saving image file.
    """
    import matplotlib.pyplot as plt
    from numpngw import write_png

    label_list = np.unique(adata.obs[label_key])
    labels = np.zeros(len(adata))
    for i in range(len(label_list)):
        labels[adata.obs[label_key] == label_list[i]] = i + 1

    if bin_size is None:
        bin_size = adata.uns["bin_size"]

    label_img = np.zeros(
        (
            int(max(adata.obsm[spatial_key][:, 0] // bin_size)) + 1,
            int(max(adata.obsm[spatial_key][:, 1] // bin_size)) + 1,
        )
    )
    for i in range(len(adata)):
        label_img[
            int(adata.obsm[spatial_key][i, 0] // bin_size), int(adata.obsm[spatial_key][i, 1] // bin_size)
        ] = labels[i]

    contour_img = label_img.copy()
    contour_img[:, :] = 255
    for i in np.unique(label_img):
        if i == 0:
            continue
        label_img_gray = np.where(label_img == i, 0, 1).astype("uint8")
        _, thresh = cv2.threshold(label_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contour, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour_img = cv2.drawContours(contour_img, contour[:], -1, 0.5, 1)

    fig = plt.figure()
    fig.set_size_inches(plot_size[0], plot_size[1])
    plt.imshow(contour_img, cmap="tab20", origin="lower")

    write_png(save_img, contour_img.astype("uint8"))
