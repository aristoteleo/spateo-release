from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from ..configuration import SKM
from ..errors import SegmentationError
from ..logging import logger_manager as lm


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def select_qc_regions(
    adata: AnnData,
    regions: Union[List[Tuple[int, int]], List[Tuple[int, int, int, int]]] = None,
    n: int = 4,
    size: int = 2000,
    seed: Optional[int] = None,
    use_scale: bool = True,
    absolute: bool = False,
    weight_func: Optional[Callable[[AnnData], float]] = lambda adata: np.log1p(adata.X.sum()),
):
    """Select regions to use for segmentation quality control purposes.

    Note:
        All coordinates are in terms of "real" coordinates (i.e. the coordinates
        in `adata.obs_names` and `adata.var_names`) so that slicing the
        AnnData retains the regions correctly.

    Args:
        adata: Input AnnData
        regions: List of tuples in the form `(xmin, ymin)` or `(xmin, xmax, ymin, ymax)`.
            If the later, the `size` argument is used to compute the bounding box.
        n: Number of regions to select if `regions` is not provided.
        size: Width and height, in pixels, of each randomly selected region.
        seed: Random seed.
        use_scale: Whether or not the provided `regions` are in scale units.
            This option only has effect when `regions` are
            provided. `False` means the provided coordinates are in terms of
            pixels.
        absolute: Whether or not the provided `regions` are in terms of absolute
            X and Y coordinates. This option only has effect when `regions` are
            provided. `False` means the provided coordinates are relative with
            respect to the coordinates in the provided `adata`.
        weight_func: Weighting function when `regions` is not provided. The probability of
            selecting each `size x size` region will be weighted by this function, which
            accepts a single AnnData (the region) as its argument, and returns a single
            float weight, such that higher weights mean higher probability. By default,
            the log1p of the sum of the counts in the `X` layer is used. Set to `None`
            to weight each region equally.
    """
    if not regions:
        lm.main_info(f"Randomly selecting {n} regions of shape {(size, size)}.")
        _regions = np.zeros((n, 4), dtype=int)
        # Construct grid indices
        indices = np.dstack(
            np.meshgrid(np.arange(0, adata.n_obs - size, size), np.arange(0, adata.n_vars - size, size))
        ).reshape(-1, 2)

        if indices.shape[0] == 0:
            raise SegmentationError("No possible regions found. This may indicate the `size` argument is to big.")

        rng = np.random.default_rng(seed)
        if weight_func is None:
            idx = rng.choice(np.arange(indices.shape[0]), n, replace=False)
        else:
            p = np.zeros(indices.shape[0])
            for i, (x, y) in enumerate(indices):
                p[i] = weight_func(adata[x : x + size, y : y + size])
            idx = rng.choice(np.arange(indices.shape[0]), n, replace=False, p=p / p.sum())
        choices = indices[idx]

        for i, (x, y) in enumerate(choices):
            xmin = int(adata.obs_names[x])
            ymin = int(adata.var_names[y])
            _regions[i] = [xmin, xmin + size, ymin, ymin + size]
    else:
        lm.main_info(f"Using regions provided with `regions` argument.")
        _regions = np.zeros((len(regions), 4), dtype=int)
        adata_bounds = SKM.get_agg_bounds(adata)
        binsize = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
        scale = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY) * binsize
        unit = SKM.get_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY)
        for i, region in enumerate(regions):
            if len(region) == 4:
                xmin, xmax, ymin, ymax = region
            elif len(region) == 2:
                xmin, ymin = region
                xmax = xmin + size
                ymax = ymin + size
            else:
                raise SegmentationError("`regions` must be a list of 4-element or 2-element tuples.")

            if use_scale and unit is not None:
                xmin /= scale
                xmax /= scale
                ymin /= scale
                ymax /= scale

            # If absolute = False, adjust for anndata bounds
            if not absolute:
                xmin += adata_bounds[0]
                xmax += adata_bounds[0]
                ymin += adata_bounds[2]
                ymax += adata_bounds[2]

            if xmin < adata_bounds[0] or xmax >= adata_bounds[1] or ymin < adata_bounds[2] or ymax >= adata_bounds[3]:
                lm.main_warning(f"Region {region} is out of bounds. It will be clipped into bounds.")
            xmin = max(xmin, adata_bounds[0])
            xmax = min(xmax, adata_bounds[1])
            ymin = max(ymin, adata_bounds[2])
            ymax = min(ymax, adata_bounds[3])

            _regions[i] = (xmin, xmax, ymin, ymax)

    _regions = _regions.astype(int)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_QC_KEY, _regions)


def _generate_random_labels(shape: Tuple[int, int], areas: List[int], seed: Optional[int] = None) -> np.ndarray:
    n = np.prod(shape)
    if sum(areas) > n:
        raise SegmentationError("Sum of `areas` exceeds to total area")

    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    indices = np.arange(n)
    rng.shuffle(indices)
    for i, area in enumerate(areas):
        label = i + 1
        labels[indices[:area]] = label
        indices = indices[area:]
    return labels.reshape(shape)


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def generate_random_labels(
    adata: AnnData,
    areas: List[int],
    seed: Optional[int] = None,
    out_layer: str = "random_labels",
):
    """Create random labels, usually for benchmarking and QC purposes.

    Args:
        adata: Input Anndata
        areas: List of desired areas.
        seed: Random seed.
        out_layer: Layer to save results.
    """
    labels = _generate_random_labels(adata.shape, areas, seed)
    SKM.set_layer_data(adata, out_layer, labels)


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def generate_random_labels_like(
    adata: AnnData,
    layer: str,
    seed: Optional[int] = None,
    out_layer: str = "random_labels",
):
    """Create random labels, using another layer as a template.

    Args:
        adata: Input Anndata
        layer: Layer containing template labels
        seed: Random seed.
        out_layer: Layer to save results.
    """
    labels = SKM.select_layer_data(adata, layer)
    bincount = np.bincount(labels.flatten())
    generate_random_labels(adata, bincount[1:], seed=seed, out_layer=out_layer)
