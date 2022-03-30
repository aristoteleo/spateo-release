"""Use StarDist for cell identification and labeling.
https://github.com/stardist/stardist

Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.
Cell Detection with Star-convex Polygons.
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.
Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy.
The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020
"""
import math
from typing import Optional, Union

import numpy as np
from anndata import AnnData
from csbdeep.data import Normalizer, PercentileNormalizer
from stardist.models import StarDist2D
from typing_extensions import Literal

from ...configuration import SKM
from ...errors import PreprocessingError
from ...logging import logger_manager as lm


def _stardist(
    img: np.ndarray,
    model: Union[Literal["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"], StarDist2D] = "2D_versatile_fluo",
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


def stardist(
    adata: AnnData,
    model: Union[Literal["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"], StarDist2D] = "2D_versatile_fluo",
    tilesize: int = 2000,
    normalizer: Optional[Normalizer] = PercentileNormalizer(),
    layer: str = SKM.STAIN_LAYER_KEY,
    out_layer: Optional[str] = None,
    **kwargs,
):
    """Run StarDist to label cells from a staining image.

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
            and merge them afterwards. Useful to avoid out-of-memory errors.
        normalizer: Normalizer to use to perform normalization prior to prediction.
            By default, percentile-based normalization is performed. `None` may
            be provided to disable normalization.
        layer: Layer that contains staining image. Defaults to `stain`.
        out_layer: Layer to put resulting labels. Defaults to `{layer}_labels`.
        **kwargs: Additional keyword arguments to pass to :func:`StarDist2D.predict_instances`.
    """
    if layer not in adata.layers:
        raise PreprocessingError(
            f'Layer "{layer}" does not exist in AnnData. '
            "Please import nuclei staining results either manually or "
            "with the `nuclei_path` argument to `st.io.read_bgi_agg`."
        )
    img = SKM.select_layer_data(adata, layer, make_dense=True)
    n_tiles = (math.ceil(img.shape[0] / tilesize), math.ceil(img.shape[1] / tilesize))

    lm.main_info(f"Running StarDist with model {model}.")
    labels = _stardist(img, model, n_tiles=n_tiles, normalizer=normalizer, **kwargs)
    out_layer = out_layer or SKM.gen_new_layer_key(layer, SKM.LABELS_SUFFIX)
    SKM.set_layer_data(adata, out_layer, labels)
