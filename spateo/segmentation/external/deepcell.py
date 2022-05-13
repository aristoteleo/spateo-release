"""Use DeepCell for cell identification and labeling.
https://github.com/vanvalenlab/deepcell-tf

[Greenwald21]_
"""
from typing import Optional, Union

import numpy as np
from anndata import AnnData

try:
    from deepcell.applications import Application, NuclearSegmentation
except ModuleNotFoundError:
    Application = None
    NuclearSegmentation = None

from ...configuration import SKM
from ...errors import SegmentationError
from ...logging import logger_manager as lm
from ..utils import clahe


def _deepcell(
    img: np.ndarray,
    model: "Application",
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
    model: Optional["Application"] = None,
    equalize: float = 2.0,
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
    **kwargs,
):
    """Run DeepCell to label cells from a staining image.

    Args:
        adata: Input Anndata
        model: DeepCell model to use
        equalize: Controls the `clip_limit` argument to the :func:`clahe` function.
            Set this value to a non-positive value to turn off equalization.
        layer: Layer that contains staining image. Defaults to `stain`.
        out_layer: Layer to put resulting labels. Defaults to `{layer}_labels`.
        **kwargs: Additional keyword arguments to :func:`Application.predict`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    if Application is None or NuclearSegmentation is None:
        raise ModuleNotFoundError("Please install Cellpose by running `pip install deepcell`.")
    if layer not in adata.layers:
        raise SegmentationError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    img = SKM.select_layer_data(adata, layer, make_dense=True)
    if equalize:
        lm.main_info("Equalizing image with CLAHE.")
        img = clahe(img, equalize)
    if model is None:
        model = NuclearSegmentation()

    lm.main_info(f"Running DeepCell with model {model}.")
    labels = _deepcell(img, model, **kwargs)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)
