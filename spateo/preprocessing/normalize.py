"""
Functions to either scale single-cell data or normalize such that the row-wise sums are identical.
"""
from typing import Dict, Iterable, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import scipy
from anndata import AnnData
from sklearn.utils import sparsefuncs

from ..logging import logger_manager as lm


def _normalize_data(X, counts, after=None, copy=False, rows=True, round=False):
    """Row-wise or column-wise normalization of sparse data array.

    Args:
        X: Sparse data array to modify.
        counts: Array of shape [1, n], where n is the number of buckets or number of genes, containing the total
            counts in each cell or for each gene, respectively.
        after: Target sum total counts for each gene or each cell. Defaults to `None`, in which case each observation
            (cell) will have a total count equal to the median of total counts for observations (cells) before
            normalization.
        copy: Whether to operate on a copy of X.
        rows: Whether to perform normalization over rows (normalize each cell to have the same total count number) or
            over columns (normalize each gene to have the same total count number).
        round: Whether to round to three decimal places to more exactly match the desired number of total counts.
    """
    X = X.copy() if copy else X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)
    counts_greater_than_zero = counts[counts > 0]

    after = np.median(counts_greater_than_zero, axis=0) if after is None else after
    counts += counts == 0
    counts = counts / after
    if scipy.sparse.issparse(X):
        sparsefuncs.inplace_row_scale(X, 1 / counts)
    elif isinstance(counts, np.ndarray):
        if rows:
            np.divide(X, counts[:, None], out=X)
        else:
            np.divide(X, counts[None, :], out=X)
    else:
        if rows:
            X = np.divide(X, counts[:, None])
        else:
            X = np.divide(X, counts[None, :])

    if round:
        X = np.around(X, decimals=3)

    return X


# Normalization wrapper:
def normalize_total(
    adata: AnnData,
    target_sum: Optional[float] = 1e4,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: Optional[str] = None,
    layer: Optional[str] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Union[AnnData, Dict[str, np.ndarray]]:
    """\
    Normalize counts per cell.
    Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization.

    If `exclude_highly_expressed=True`, very highly expressed genes are excluded from the computation of the
    normalization factor (size factor) for each cell. This is meaningful as these can strongly influence the resulting
    normalized values for all other genes.

    Args:
        adata: The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        target_sum: Desired sum of counts for each gene post-normalization. If `None`, after normalization,
            each observation (cell) will have a total count equal to the median of total counts for observations (
            cells) before normalization.
        exclude_highly_expressed: Exclude (very) highly expressed genes for the computation of the normalization factor
            for each cell. A gene is considered highly expressed if it has more than `max_fraction` of the total counts
            in at least one cell.
        max_fraction: If `exclude_highly_expressed=True`, this is the cutoff threshold for excluding genes.
        key_added: Name of the field in `adata.obs` where the normalization factor is stored.
        layer: Layer to normalize instead of `X`. If `None`, `X` is normalized.
        inplace: Whether to update `adata` or return dictionary with normalized copies of `adata.X` and `adata.layers`.
        copy: Whether to modify copied input object. Not compatible with inplace=False.

    Returns:
        Returns dictionary with normalized copies of `adata.X` and `adata.layers` or updates `adata` with normalized
        version of the original `adata.X` and `adata.layers`, depending on `inplace`.
    """
    logger = lm.get_main_logger()

    if copy:
        if not inplace:
            logger.error("`copy=True` cannot be used with `inplace=False`.")
        adata = adata.copy()

    if max_fraction < 0 or max_fraction > 1:
        logger.error("Choose max_fraction between 0 and 1.")

    if adata.is_view:
        logger.warning("Received a view of an AnnData object; making a copy.")
        adata._init_as_actual(adata.copy())

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    gene_subset = None
    msg = "Normalizing counts per cell..."
    if exclude_highly_expressed:
        counts_per_cell = X.sum(1)  # original counts per cell
        counts_per_cell = np.ravel(counts_per_cell)

        gene_subset = (X > counts_per_cell[:, None] * max_fraction).sum(0)
        gene_subset = np.ravel(gene_subset) == 0

        msg += (
            " The following highly-expressed genes are not considered during "
            f"normalization factor computation:\n{adata.var_names[~gene_subset].tolist()}"
        )

        counts_per_cell = X[:, gene_subset].sum(1)
    else:
        counts_per_cell = X.sum(1)

    logger.info(msg)
    counts_per_cell = np.ravel(counts_per_cell)

    cell_subset = counts_per_cell > 0
    if not np.all(cell_subset):
        logger.warning("Some cells have zero counts")

    if inplace:
        if key_added is not None:
            adata.obs[key_added] = counts_per_cell
        X = _normalize_data(X, counts_per_cell, target_sum)
        if layer is not None:
            adata.layers[layer] = X
        else:
            adata.X = X
    else:
        dat = dict(
            X=_normalize_data(X, counts_per_cell, target_sum, copy=True),
            norm_factor=counts_per_cell,
        )

    if key_added is not None:
        logger.debug(f"and added {key_added!r}, counts per cell before normalization (adata.obs)")

    if copy:
        return adata
    elif not inplace:
        return dat
