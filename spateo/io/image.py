"""Image IO.
"""

from typing import Optional

import cv2
from anndata import AnnData

from .image_utils import add_image_layer


def read_image(
    adata: AnnData,
    filename: str,
    scale_factor: float,
    slice: Optional[str] = None,
    img_layer: Optional[str] = None,
) -> AnnData:
    """Load an image into the AnnData object.

    Args:
        adata: AnnData object
        filename: The path of the image
        scale_factor: The scale factor of the image. Define: pixels/DNBs
        slice: Name of the slice. Will be used when displaying multiple slices.
        img_layer: Name of the image layer.

    Returns
    -------
        :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['images'][img_layer]`
            The stored image
        :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['scalefactors'][img_layer]`
            The scale factor for the spots
    """
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Could not find '{filename}'")

    adata = add_image_layer(
        adata=adata,
        img=img,
        scale_factor=scale_factor,
        slice=slice,
        img_layer=img_layer,
    )

    # TODO: show image

    return adata
