"""Filter functions.
"""
from typing import Optional, Sequence

import numpy as np
from anndata import AnnData


def filter_cells(
    adata: AnnData,
    filter_bool: Optional[np.ndarray] = None,
    keep_filtered: bool = False,
    min_expr_genes: int = 50,
    max_expr_genes: float = np.inf,
    min_area: float = 0,
    max_area: float = np.inf,
    inplace: bool = False,
) -> Optional[AnnData]:
    """Select valid cells based on a collection of filters.
    This function is partially based on dynamo (https://github.com/aristoteleo/dynamo-release).

    TODO: What layers need to be considered? Argument `shared_count` ?

    Args:
        adata: AnnData object.
        filter_bool: A boolean array from the user to select cells for downstream analysis.
        keep_filtered: Whether to keep cells that don't pass the filtering in the adata object.
        min_expr_genes: Minimal number of genes with expression for a cell in the data from X.
        max_expr_genes: Maximal number of genes with expression for a cell in the data from X.
        min_area: Maximum area of a cell in the data from X.
        max_area: Maximum area of a cell in the data from X.
        inplace: Perform computation inplace or return result.
    Returns:
        An updated AnnData object with pass_basic_filter as a new column in obs to indicate the selection of cells for
        downstream analysis. adata will be subset with only the cells pass filtering if keep_filtered is set to
        be False.
    """
    if not inplace:
        adata = adata.copy()

    detected_bool = np.ones(adata.X.shape[0], dtype=bool)
    detected_bool = (detected_bool) & (
        ((adata.X > 0).sum(1) >= min_expr_genes) & ((adata.X > 0).sum(1) <= max_expr_genes)
    ).flatten()

    if (min_area != 0) or (max_area != np.inf):
        if "area" not in adata.obs.keys():
            # TODO: warning
            print("`area` is not in the adata.obs")
        else:
            detected_bool = (detected_bool) & (
                np.array((adata.obs["area"] >= min_area) & (adata.obs["area"] <= max_area)).flatten()
            )
            detected_bool = np.array(detected_bool).flatten()

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    filter_bool = np.array(filter_bool).flatten()
    if keep_filtered:
        adata.obs["pass_basic_filter"] = filter_bool
    else:
        adata._inplace_subset_obs(filter_bool)
        adata.obs["pass_basic_filter"] = True

    return adata if not inplace else None


def filter_genes(
    adata: AnnData,
    filter_bool: Optional[np.ndarray] = None,
    keep_filtered: bool = False,
    min_cells: int = 1,
    max_cells: float = np.inf,
    min_avg_exp: float = 0,
    max_avg_exp: float = np.inf,
    min_counts: float = 0,
    max_counts: float = np.inf,
    inplace: bool = False,
) -> Optional[AnnData]:
    """Select valid genes based on a collection of filters.
    This function is partially based on dynamo (https://github.com/aristoteleo/dynamo-release).

    Args:
        adata: filter_bool: :class:`~numpy.ndarray` (default: `None`)
            A boolean array from the user to select genes for downstream analysis.
        keep_filtered: Whether to keep genes that don't pass the filtering in the adata object.
        min_cells: Minimal number of cells with expression in the data from X.
        max_cells: Maximal number of cells with expression in the data from X.
        min_avg_exp: Minimal average expression across cells for the data.
        max_avg_exp: Maximal average expression across cells for the data.
        min_counts: Minimal number of counts (UMI/expression) for the data
        max_counts: Minimal number of counts (UMI/expression) for the data
        inplace: Perform computation inplace or return result.

    Returns:
        An updated AnnData object with pass_basic_filter as a new column in var to indicate the selection of genes for
        downstream analysis. adata will be subset with only the genes pass filtering if keep_filtered is set to
        be False.
    """
    if not inplace:
        adata = adata.copy()

    detected_bool = np.ones(adata.shape[1], dtype=bool)
    detected_bool = (detected_bool) & np.array(
        ((adata.X > 0).sum(0) >= min_cells)
        & ((adata.X > 0).sum(0) <= max_cells)
        & (adata.X.mean(0) >= min_avg_exp)
        & (adata.X.mean(0) <= max_avg_exp)
        & (adata.X.sum(0) >= min_counts)
        & (adata.X.sum(0) <= max_counts)
    ).flatten()

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    filter_bool = np.array(filter_bool).flatten()
    if keep_filtered:
        adata.var["pass_basic_filter"] = filter_bool
    else:
        adata._inplace_subset_var(filter_bool)
        adata.var["pass_basic_filter"] = True

    return adata if not inplace else None


def filter_by_coordinates(
    adata: AnnData,
    filter_bool: Optional[np.ndarray] = None,
    keep_filtered: bool = False,
    x_range: Sequence[float] = (-np.inf, np.inf),
    y_range: Sequence[float] = (-np.inf, np.inf),
    inplace: bool = False,
) -> Optional[AnnData]:
    """Select valid cells by coordinates.
    TODO: lasso tool

    Args:
        adata: AnnData object.
        filter_bool: A boolean array from the user to select cells for downstream analysis.
        keep_filtered: Whether to keep cells that don't pass the filtering in the adata object.
        x_range: The X-axis range of cell coordinates.
        y_range: The Y-axis range of cell coordinates.
        inplace: Perform computation inplace or return result.

    Returns:
        An updated AnnData object with pass_basic_filter as a new column in obs to indicate the selection of cells for
        downstream analysis. adata will be subset with only the cells pass filtering if keep_filtered is set to
        be False.
    """
    if not inplace:
        adata = adata.copy()

    detected_bool = np.ones(adata.X.shape[0], dtype=bool)
    detected_bool = (detected_bool) & (
        (adata.obsm["spatial"][:, 0] >= x_range[0])
        & (adata.obsm["spatial"][:, 0] <= x_range[1])
        & (adata.obsm["spatial"][:, 1] >= y_range[0])
        & (adata.obsm["spatial"][:, 1] <= y_range[1])
    ).flatten()

    filter_bool = filter_bool & detected_bool if filter_bool is not None else detected_bool

    filter_bool = np.array(filter_bool).flatten()
    if keep_filtered:
        adata.obs["pass_basic_filter"] = filter_bool
    else:
        adata._inplace_subset_obs(filter_bool)
        adata.obs["pass_basic_filter"] = True

    return adata if not inplace else None
