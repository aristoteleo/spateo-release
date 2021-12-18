"""Helper functions about image layer processing.
"""
from anndata import AnnData


def add_img_layer(adata: AnnData,
                  img,
                  scale_factor: float,
                  slice: str = None,
                  img_layer: str = None
                  ):
    """
    A helper function that add an image layer to AnnData object.

    Parameters
    ----------
        adata: :class: `AnnData`
            AnnData object.
        img:
            The image data.
        scale_factor: `float`
            The scale factor of the image. Define: pixels/DNBs
        slice: `str` (default: None)
            Name of the slice. Will be used when displaying multiple slices.
        img_layer: `str` (default: None)
            Name of the image layer.

    Returns
    -------
        adata: `AnnData`
            :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['images'][img_layer]`
                The stored image
            :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['scalefactors'][img_layer]`
                The scale factor for the spots
    """
    # Create a new dictionary or add to the original slice
    if 'spatial' not in adata.uns_keys():
        adata.uns['spatial'] = dict()
    if slice not in adata.uns['spatial'].keys():
        adata.uns['spatial'][slice] = dict()

    if 'images' not in adata.uns['spatial'][slice]:
        adata.uns['spatial'][slice]['images'] = {img_layer: img}
    else:
        adata.uns['spatial'][slice]['images'][img_layer] = img

    if 'scalefactors' not in adata.uns['spatial'][slice]:
        adata.uns['spatial'][slice]['scalefactors'] = {img_layer: scale_factor}
    else:
        adata.uns['spatial'][slice]['scalefactors'][img_layer] = scale_factor

    return adata
