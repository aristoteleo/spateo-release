"""Helper functions about image layer processing.
"""
from typing import Optional

import numpy as np
from anndata import AnnData


def add_image_layer(
    adata: AnnData,
    img: np.ndarray,
    scale_factor: float,
    slice: Optional[str] = None,
    img_layer: Optional[str] = None,
) -> AnnData:
    """
    A helper function that add an image layer to AnnData object.

    Args:
        adata: AnnData object.
        img: The image data.
        scale_factor: The scale factor of the image. Define: pixels/DNBs
        slice: Name of the slice. Will be used when displaying multiple slices.
        img_layer: Name of the image layer.

    Returns
    -------
        adata: `AnnData`
            :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['images'][img_layer]`
                The stored image
            :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['scalefactors'][img_layer]`
                The scale factor for the spots
    """
    # Create a new dictionary or add to the original slice
    if "spatial" not in adata.uns_keys():
        adata.uns["spatial"] = dict()
    if slice not in adata.uns["spatial"].keys():
        adata.uns["spatial"][slice] = dict()

    if "images" not in adata.uns["spatial"][slice]:
        adata.uns["spatial"][slice]["images"] = {img_layer: img}
    else:
        adata.uns["spatial"][slice]["images"][img_layer] = img

    if "scalefactors" not in adata.uns["spatial"][slice]:
        adata.uns["spatial"][slice]["scalefactors"] = {img_layer: scale_factor}
    else:
        adata.uns["spatial"][slice]["scalefactors"][img_layer] = scale_factor

    return adata
