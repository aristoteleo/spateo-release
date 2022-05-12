"""Use StarDist for cell identification and labeling.
https://github.com/stardist/stardist

[Schmidt18]_ and [Weigert20]_
"""
import math
from typing import Optional, Union

import cv2
import numpy as np
from anndata import AnnData
from csbdeep.data import Normalizer, PercentileNormalizer
from skimage import measure

try:
    from stardist.models import StarDist2D
except ModuleNotFoundError:
    StarDist2D = None
from typing_extensions import Literal

from ...configuration import SKM
from ...errors import SegmentationError
from ...logging import logger_manager as lm
from ..utils import clahe


def _stardist(
    img: np.ndarray,
    model: Union[
        Literal["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"], "StarDist2D"
    ] = "2D_versatile_fluo",
    **kwargs,
) -> np.ndarray:
    """Run StarDist on the provided image.

    Args:
        img: Image as a Numpy array.
        model: Stardist model to use. Can be one of the three pretrained models
            from StarDist2D:
            1. '2D_versatile_fluo': 'Versatile (fluorescent nuclei)'
            2. '2D_versatile_he':  'Versatile (H&E nuclei)'
            3. '2D_paper_dsb2018': 'DSB 2018 (from StarDist 2D paper)'
            Or any generic Stardist2D model.
        **kwargs: Additional keyword arguments to :func:`StarDist2D.predict_instances`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    if isinstance(model, str):
        model = StarDist2D.from_pretrained(model)

    lm.main_debug(f"Running StarDist with kwargs {kwargs}")
    labels, _ = model.predict_instances(img, **kwargs)
    return labels


def _stardist_big(
    img: np.ndarray,
    model: Union[
        Literal["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"], "StarDist2D"
    ] = "2D_versatile_fluo",
    **kwargs,
) -> np.ndarray:
    """Run StarDist on the provided image.

    Args:
        img: Image as a Numpy array.
        model: Stardist model to use. Can be one of the three pretrained models
            from StarDist2D:
            1. '2D_versatile_fluo': 'Versatile (fluorescent nuclei)'
            2. '2D_versatile_he':  'Versatile (H&E nuclei)'
            3. '2D_paper_dsb2018': 'DSB 2018 (from StarDist 2D paper)'
            Or any generic Stardist2D model.
        **kwargs: Additional keyword arguments to :func:`StarDist2D.predict_instances_big`
            function.

    Returns:
        Numpy array containing cell labels.
    """
    if isinstance(model, str):
        model = StarDist2D.from_pretrained(model)

    lm.main_debug(f"Running StarDist BIG with kwargs {kwargs}")
    labels, _ = model.predict_instances_big(img, axes="YX", **kwargs)
    return labels


def _sanitize_labels(labels: np.ndarray) -> np.ndarray:
    """Sanitize labels obtained from StarDist.

    StarDist sometimes yields disconnected labels. This function removes
    these problems by selecting the largest area.

    Args:
        labels: Numpy array containing labels

    Returns:
        Sanitized labels.
    """

    def components(mtx):
        mtx = mtx.astype(np.uint8)
        return cv2.connectedComponentsWithStats(mtx)

    sanitized = labels.copy()
    label_props = measure.regionprops(labels, extra_properties=[components])
    for props in label_props:
        label = props.label
        comps = props.components
        if comps[0] > 2:
            lm.main_debug(f"Sanitizing label {label}.")
            largest_comp_label = comps[2][1:, cv2.CC_STAT_AREA].argmax() + 1
            min_row, min_col, max_row, max_col = props.bbox
            subset = (comps[1] > 0) & (comps[1] != largest_comp_label)
            sanitized[min_row:max_row, min_col:max_col][subset] = 0
    return sanitized


def stardist(
    adata: AnnData,
    model: Union[
        Literal["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"], "StarDist2D"
    ] = "2D_versatile_fluo",
    tilesize: int = 2000,
    min_overlap: Optional[int] = None,
    context: Optional[int] = None,
    normalizer: Optional[Normalizer] = PercentileNormalizer(),
    equalize: float = 2.0,
    sanitize: bool = True,
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
    **kwargs,
):
    """Run StarDist to label cells from a staining image.

    Note:
        When using `min_overlap`, the crucial assumption is that all predicted
        object instances are smaller than the provided `min_overlap`.
        Also, it must hold that: min_overlap + 2*context < tilesize.
        https://github.com/stardist/stardist/blob/858cae17cf17f979122000ad2294a156d0547135/stardist/models/base.py#L776

    Args:
        adata: Input Anndata
        img: Image as a Numpy array.
        model: Stardist model to use. Can be one of the three pretrained models
            from StarDist2D:
            1. '2D_versatile_fluo': 'Versatile (fluorescent nuclei)'
            2. '2D_versatile_he':  'Versatile (H&E nuclei)'
            3. '2D_paper_dsb2018': 'DSB 2018 (from StarDist 2D paper)'
            Or any generic Stardist2D model.
        tilesize: Run prediction separately on tiles of size `tilesize` x `tilesize`
            and merge them afterwards. Useful to avoid out-of-memory errors. Can be
            set to <= 0 to disable tiling. When `min_overlap` is also provided, this
            becomes the `block_size` parameter to :func:`StarDist2D.predict_instances_big`.
        min_overlap: Amount of guaranteed overlaps between tiles.
        context: Amount of image context on all sides of a tile, which is dicarded.
            Only used when `min_overlap` is not None. By default, an automatic
            estimate is used.
        normalizer: Normalizer to use to perform normalization prior to prediction.
            By default, percentile-based normalization is performed. `None` may
            be provided to disable normalization.
        equalize: Controls the `clip_limit` argument to the :func:`clahe` function.
            Set this value to a non-positive value to turn off equalization.
        sanitize: Whether to sanitize disconnected labels.
        layer: Layer that contains staining image. Defaults to `stain`.
        out_layer: Layer to put resulting labels. Defaults to `{layer}_labels`.
        **kwargs: Additional keyword arguments to pass to :func:`StarDist2D.predict_instances`.
    """
    if StarDist2D is None:
        raise ModuleNotFoundError("Please install StarDist by running `pip install stardist`.")
    if tilesize <= 0 and min_overlap:
        raise SegmentationError("Positive `tilesize` must be provided when `min_overlap` is used.")
    if layer not in adata.layers:
        raise SegmentationError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    img = SKM.select_layer_data(adata, layer, make_dense=True)
    if equalize > 0:
        lm.main_info("Equalizing image with CLAHE.")
        img = clahe(img, equalize)

    lm.main_info(f"Running StarDist with model {model}.")
    if not min_overlap:
        n_tiles = (math.ceil(img.shape[0] / tilesize), math.ceil(img.shape[1] / tilesize)) if tilesize > 0 else (1, 1)
        labels = _stardist(img, model, n_tiles=n_tiles, normalizer=normalizer, **kwargs)
    else:
        labels = _stardist_big(
            img,
            model,
            block_size=tilesize,
            min_overlap=min_overlap,
            context=context,
            n_tiles=(1, 1),
            normalizer=normalizer,
        )
    if sanitize:
        lm.main_info(f"Fixing disconnected labels.")
        labels = _sanitize_labels(labels)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)
