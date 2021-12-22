from anndata import AnnData
import matplotlib.pyplot as plt
import cv2
from typing import Union, Optional

from ..tools.image import add_image_layer


def remove_background(
    adata: AnnData,
    threshold: Union[float, str] = "auto",
    slice: Optional[str] = None,
    used_img_layer: Optional[str] = None,
    return_img_layer: Optional[str] = None,
    inplace: bool = False,
    show: bool = True,
):
    """
    Preprocessing of an image. Remove background with the global threshold. Pixel intensity is set to 0, for all the
    pixels intensity, less than the threshold value. If the threshold is not provided, it will be calculated by
    OSTU's method.

    Parameters
    ----------
        adata: :class: `AnnData`
            AnnData object.
        threshold: `float` or auto
            Global threshold used. If the threshold is not provided, it will be calculated by OSTU's method.
        slice: `str` or None (default: None)
            Name of the slice.
        used_img_layer: `str` or None (default: None)
            Name of used image layer.
        return_img_layer: `str` or None (default: None)
            Name of output image layer.
        inplace: `bool` (default: `False`)
            Perform computation inplace or return result.
        show: `bool` (default: `True`)
            Show the preprocessed image or not.

    Returns
    -------
        adata: :class: `AnnData`
            :attr:`AnnData.uns`\\ `['spatial'][slice]['images'][return_img_layer]`
                The preprocessed image
            :attr:`AnnData.uns`\\ `['spatial'][slice]['scalefactors'][return_img_layer]`
                The scale factor for the spots
    """
    if not inplace:
        adata = adata.copy()

    img = adata.uns["spatial"][slice]["images"][used_img_layer].copy()
    scale_factor = adata.uns["spatial"][slice]["scalefactors"][used_img_layer]

    if threshold == "auto":
        threshold, _ = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_OTSU)
    print(f"Used Threshold: {threshold}")

    _, img = cv2.threshold(img.copy(), threshold, 255, cv2.THRESH_TOZERO)

    # add preprocessed img to AnnData object
    adata = add_image_layer(
        adata=adata,
        img=img,
        scale_factor=scale_factor,
        slice=slice,
        img_layer=return_img_layer,
    )

    if show:
        plt.figure(figsize=(16, 16))
        plt.imshow(img, "gray")

    return adata if not inplace else None
