"""Use Cellpose for cell identification and labeling.
https://github.com/MouseLand/cellpose

[Stringer20]_
"""
from typing import Optional, Union

import numpy as np
from anndata import AnnData

try:
    from cellpose.models import Cellpose, CellposeModel
except ModuleNotFoundError:
    Cellpose = None
    CellposeModel = None
from typing_extensions import Literal

from ...configuration import SKM
from ...errors import SegmentationError
from ...logging import logger_manager as lm
from ..utils import clahe


def _cellpose(
    img: np.ndarray,
    model: Union[Literal["cyto", "nuclei"], "CellposeModel"] = "nuclei",
    **kwargs,
) -> np.ndarray:
    """Run Cellpose on the provided image.

    Args:
        img: Image as a Numpy array.
        model: Cellpose model to use. Can be one of the two pretrained models:
            * cyto: Labeled cytoplasm
            * nuclei: Labeled nuclei
            Or any generic CellposeModel model.
        **kwargs: Additional keyword arguments to :func:`Cellpose.eval`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    if isinstance(model, str):
        model = Cellpose(model_type=model, gpu=True)  # Use GPU if available

    masks, flows, styles, diams = model.eval(img, **kwargs)
    return masks


def cellpose(
    adata: AnnData,
    model: Union[Literal["cyto", "nuclei"], "CellposeModel"] = "nuclei",
    diameter: Optional[int] = None,
    normalize: bool = True,
    equalize: float = 2.0,
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
    **kwargs,
):
    """Run Cellpose to label cells from a staining image.

    Args:
        adata: Input Anndata
        model: Cellpose model to use. Can be one of the two pretrained models:
            * cyto: Labeled cytoplasm
            * nuclei: Labeled nuclei
            Or any generic CellposeModel model.
        diameter: Expected diameter of each segmentation (cells for `model="cyto"`,
            nuclei for `model="nuclei"`). Can be `None` to run automatic detection.
        normalize: Whether or not to percentile-normalize the image. This is an
            argument to :func:`Cellpose.eval`.
        equalize: Controls the `clip_limit` argument to the :func:`clahe` function.
            Set this value to a non-positive value to turn off equalization.
        layer: Layer that contains staining image. Defaults to `stain`.
        out_layer: Layer to put resulting labels. Defaults to `{layer}_labels`.
        **kwargs: Additional keyword arguments to :func:`Cellpose.eval`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    if Cellpose is None or CellposeModel is None:
        raise ModuleNotFoundError("Please install Cellpose by running `pip install cellpose`.")

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

    if diameter is None:
        lm.main_warning("`diameter` was not provided and will be estimated.")

    lm.main_info(f"Running Cellpose with model {model}.")
    labels = _cellpose(img, model, channels=[0, 0], normalize=normalize, **kwargs)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)
