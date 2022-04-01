"""Use DeepCell for cell identification and labeling.
https://github.com/vanvalenlab/deepcell-tf

Greenwald, N.F., Miller, G., Moen, E. et al.
Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning.
Nat Biotechnol (2021). https://doi.org/10.1038/s41587-021-01094-0
"""
from typing import Optional, Union

import numpy as np
from anndata import AnnData
from deepcell.applications import Application, NuclearSegmentation

from ....configuration import SKM
from ....errors import PreprocessingError
from ....logging import logger_manager as lm


def _deepcell(
    img: np.ndarray,
    model: Application,
    **kwargs,
) -> np.ndarray:
    """Run DeepCell on the provided image.

    Args:
        img: Image as a Numpy array.
        model: DeepCell model to use
        **kwargs: Additional keyword arguments to :func:`Application.predict`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    img = np.expand_dims(np.expand_dims(img, 0), -1)
    return model.predict(img, **kwargs).squeeze(0).squeeze(-1)


def deepcell(
    adata: AnnData,
    model: Optional[Application] = None,
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
    **kwargs,
):
    """Run DeepCell to label cells from a staining image.

    Args:
        adata: Input Anndata
        model: DeepCell model to use
        layer: Layer that contains staining image. Defaults to `stain`.
        out_layer: Layer to put resulting labels. Defaults to `{layer}_labels`.
        **kwargs: Additional keyword arguments to :func:`Application.predict`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    if layer not in adata.layers:
        raise PreprocessingError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    img = SKM.select_layer_data(adata, layer, make_dense=True)
    if model is None:
        model = NuclearSegmentation()

    lm.main_info(f"Running DeepCell with model {model}.")
    labels = _deepcell(img, model, **kwargs)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)