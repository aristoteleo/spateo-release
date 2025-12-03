"""Use Cellpose for cell identification and labeling.
https://github.com/MouseLand/cellpose

[Stringer20]_
"""

from typing import Optional, Union

import numpy as np
from anndata import AnnData

# Cellpose API has changed across versions: some releases expose `Cellpose`,
# others only `CellposeModel`. Be permissive in imports and handle both cases.
try:
    from cellpose.models import Cellpose, CellposeModel
except ImportError:
    # Try importing only CellposeModel (newer versions)
    try:
        from cellpose.models import CellposeModel  # type: ignore

        Cellpose = None  # type: ignore
    except Exception:
        Cellpose = None  # type: ignore
        CellposeModel = None  # type: ignore
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
    # If a string model name was provided, instantiate the appropriate model class.
    if isinstance(model, str):
        # Prefer the legacy `Cellpose` class if available, otherwise use
        # `CellposeModel` which newer releases expose.
        if Cellpose is not None:
            model = Cellpose(model_type=model, gpu=True)  # Use GPU if available
        elif CellposeModel is not None:
            model = CellposeModel(model_type=model, gpu=True)  # Use GPU if available
        else:
            raise ModuleNotFoundError("cellpose is not available in the environment.")

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
    # Ensure at least one of the model classes is importable
    if Cellpose is None and CellposeModel is None:
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
