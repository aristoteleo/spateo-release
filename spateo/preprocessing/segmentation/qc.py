import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from ...configuration import SKM
from ...errors import PreprocessingError
from ...warnings import PreprocessingWarning


def select_qc_regions(
    adata: AnnData,
    regions: Union[List[Tuple[int, int]], List[Tuple[int, int, int, int]]] = None,
    n: int = 4,
    size: int = 2000,
    seed: Optional[int] = None,
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
        size: Width and height of each randomly selected region.
        seed: Random seed.
    """
    _regions = np.zeros((n, 4), dtype=int)
    if not regions:
        # Construct grid indices
        indices = np.dstack(
            np.meshgrid(np.arange(0, adata.n_obs - size, size), np.arange(0, adata.n_vars - size, size))
        ).reshape(-1, 2)

        rng = np.random.default_rng(seed)
        choices = indices[rng.choice(np.arange(indices.shape[0]), n, replace=False)]

        for i, (x, y) in enumerate(choices):
            xmin = int(adata.obs_names[x])
            ymin = int(adata.var_names[y])
            _regions[i] = [xmin, xmin + size, ymin, ymin + size]
    else:
        adata_bounds = SKM.get_agg_bounds(adata)
        for i, region in enumerate(regions):
            if len(region) == 4:
                xmin, xmax, ymin, ymax = region
            elif len(region) == 2:
                xmin, ymin = region
                xmax = xmin + size
                ymax = ymin + size
            else:
                raise PreprocessingError("`regions` must be a list of 4-element or 2-element tuples.")

            if xmin < adata_bounds[0] or xmax >= adata_bounds[1] or ymin < adata_bounds[2] or ymax >= adata_bounds[3]:
                warnings.warn(
                    f"Region {region} is out of bounds. It will be clipped into bounds.", PreprocessingWarning
                )
            xmin = max(xmin, adata_bounds[0])
            xmax = min(xmax, adata_bounds[1])
            ymin = max(ymin, adata_bounds[2])
            ymax = min(ymax, adata_bounds[3])

            _regions[i] = (xmin, xmax, ymin, ymax)

    _regions = _regions.astype(int)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_QC_KEY, _regions)
