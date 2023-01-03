"""
Miscellaneous non-normalizing data transformations on AnnData objects
"""
from functools import singledispatch
from typing import Optional, Union

import numba
import numpy as np
import scipy
from anndata import AnnData
from sklearn.utils import check_array, sparsefuncs

from ..logging import logger_manager as lm


# ------------------------------------------ Log Transformation ------------------------------------------ #
@singledispatch
def log1p(
    X: Union[AnnData, np.ndarray, scipy.sparse.spmatrix],
    base: Optional[int] = None,
    copy: bool = False,
):
    """Computes the natural logarithm of the data matrix (unless different base is chosen using the `base` argument)

    Args:
        X: Either full AnnData object or .X. Rows correspond to cells and columns to genes.
        base: Natural log is used by default.
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned.
        layer: Layer to transform. If None, will transform .X. If given both argument to `layer` and `obsm`, argument to
            `layer` will take priority.
        obsm: Entry in .obsm to transform. If None, will transform .X.

    Returns:
        If `copy` is True or input is numpy array/sparse matrix, returns updated data array. Otherwise, returns updated
        AnnData object.
    """
    # if type(X) == AnnData:
    #     if layer is None:
    #         X = X.X.copy() if obsm is None else X.obsm[obsm].copy()
    #     else:
    #         X = X.layers[layer].copy()

    return log1p_array(X, copy=copy, base=base)


@log1p.register(scipy.sparse.spmatrix)
def log1p_sparse(X, *, base: Optional[int] = None, copy: bool = False):
    """Called if `log1p` is called with a sparse matrix input."""
    X = check_array(X, accept_sparse=("csr", "csc"), dtype=(np.float64, np.float32), copy=copy)
    # Perform computation on the non-sparse components
    X.data = log1p(X.data, copy=False, base=base)
    return X


@log1p.register(np.ndarray)
def log1p_array(X, *, base: Optional[int] = None, copy: bool = False):
    """
    Called if `log1p` is called with a numpy array input
    """
    if copy:
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(float)
        else:
            X = X.copy()
    elif not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, complex)):
        X = X.astype(float)
    np.log1p(X, out=X)
    if base is not None:
        np.divide(X, np.log(base), out=X)
    return X


@log1p.register(AnnData)
def log1p_anndata(
    adata,
    *,
    base: Optional[int] = None,
    copy: bool = False,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
) -> Optional[AnnData]:
    """
    Called if `log1p` is called with an AnnData input
    """
    logger = lm.get_main_logger()

    if "log1p" in adata.uns_keys():
        logger.warning("adata.X seems to be already log-transformed.")

    adata = adata.copy() if copy else adata
    if adata.is_view:
        logger.warning("Received a view of an AnnData. Making a copy.")
        adata._init_as_actual(adata.copy())

    if layer is not None:
        X = adata.layers[layer]
    elif obsm is not None:
        X = adata.obsm[obsm]
    else:
        X = adata.X

    X = log1p(X, copy=False, base=base)

    if layer is not None:
        adata.layers[layer] = X
    elif obsm is not None:
        adata.obsm[obsm] = X
    else:
        adata.X = X

    adata.uns["log1p"] = {"base": base}
    if copy:
        return adata


# ------------------------------------------ Standard Scaling ------------------------------------------ #
@singledispatch
def scale(
    X: Union[AnnData, scipy.sparse.spmatrix, np.ndarray],
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
    return_mean_std: bool = False,
):
    """Scale variables to unit variance and optionally zero mean. Variables that are constant across all observations
    will be set to 0.

    Args:
        X: Either full AnnData object or .X. Rows correspond to cells and columns to genes.
        zero_center: If False, will not center variables.
        max_value: Truncate to this value after scaling.
        copy: If an :class:`~anndata.AnnData` is passed, determines whether a copy is returned.
        layer: Layer to transform. If None, will transform .X. If given both argument to `layer` and `obsm`, argument to
            `layer` will take priority.
        obsm: Entry in .obsm to transform. If None, will transform .X.
        return_mean_std: Set True to return computed feature means and feature standard deviations.

    Returns:
        Depending on `copy` returns or updates `adata` with a scaled `adata.X`, annotated with `'mean'` and `'std'` in
        `adata.var`.
    """
    return scale_array(X, zero_center=zero_center, max_value=max_value, copy=copy)


@scale.register(scipy.sparse.spmatrix)
def scale_sparse(
    X,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    return_mean_std: bool = False,
):
    """Called if `log1p` is called with a sparse matrix input"""
    logger = lm.get_main_logger()

    # If `zero_center`, need to densify the array since centering cannot be done in sparse arrays:
    if zero_center:
        logger.info(
            "As `zero_center`=True, sparse input needs to be densified. Will result in increased memory " "consumption."
        )
        X = X.toarray()
        copy = False
    return scale_array(
        X,
        zero_center=zero_center,
        copy=copy,
        max_value=max_value,
        return_mean_std=return_mean_std,
    )


@scale.register(np.ndarray)
def scale_array(
    X,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    return_mean_std: bool = False,
):
    """Called if `log1p` is called with a numpy array input"""
    logger = lm.get_main_logger()

    if copy:
        X = X.copy()

    if np.issubdtype(X.dtype, np.integer):
        logger.info("As scaling leads to float results, integer input is cast to float, returning copy.")
        X = X.astype(float)

    # Mean and variance of the data:
    mean, var = _get_mean_var(X)
    std = np.sqrt(var)
    std[std == 0] = 1
    if scipy.sparse.issparse(X):
        if zero_center:
            logger.error("Cannot zero-center sparse matrix.")
        sparsefuncs.inplace_column_scale(X, 1 / std)
    else:
        if zero_center:
            X -= mean
        X /= std

    if max_value is not None:
        logger.info(f"Clipping to max value of {max_value}")
        X[X > max_value] = max_value

    if return_mean_std:
        return X, mean, std
    else:
        return X


@scale.register(AnnData)
def scale_anndata(
    adata: AnnData,
    *,
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False,
    layer: Optional[str] = None,
    obsm: Optional[str] = None,
) -> Optional[AnnData]:
    """
    Called if scale is called with an AnnData object
    """
    logger = lm.get_main_logger()

    adata = adata.copy() if copy else adata
    if adata.is_view:
        logger.warning("Received a view of an AnnData. Making a copy.")
        adata._init_as_actual(adata.copy())

    if layer is not None:
        X = adata.layers[layer]
    elif obsm is not None:
        X = adata.obsm[obsm]
    else:
        X = adata.X

    X, adata.var["mean"], adata.var["std"] = scale(
        X,
        zero_center=zero_center,
        max_value=max_value,
        copy=False,
        return_mean_std=True,
    )

    if layer is not None:
        adata.layers[layer] = X
    elif obsm is not None:
        adata.obsm[obsm] = X
    else:
        adata.X = X

    if copy:
        return adata


# ------------------------------------------ Mean/Standard Deviation ------------------------------------------ #
def _get_mean_var(X, *, axis=0):
    """Wrapper for computing row-wise or column-wise mean+variance on sparse array-likes."""
    if scipy.sparse.issparse(X):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var


def sparse_mean_variance_axis(mtx: scipy.sparse.spmatrix, axis: int):
    """
    Mean and variance of sparse array.

    Args:
        mtx : scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
        axis : int
            Either 0 or 1. Determines which axis mean and variance are computed along
    """
    logger = lm.get_main_logger()

    assert axis in (0, 1)
    if isinstance(mtx, scipy.sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, scipy.sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        logger.error("Input must be a csr matrix or csc matrix.")

    if axis == ax_minor:
        return sparse_mean_var_major_axis(mtx.data, mtx.indices, mtx.indptr, *shape, np.float64)
    else:
        return sparse_mean_var_minor_axis(mtx.data, mtx.indices, *shape, np.float64)


@numba.njit(cache=True)
def sparse_mean_var_minor_axis(data, indices, major_len, minor_len, dtype):
    """Given array for a csr matrix, returns means and variances for each column."""
    non_zero = indices.shape[0]

    means = np.zeros(minor_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    counts = np.zeros(minor_len, dtype=np.int64)

    for i in range(non_zero):
        col_ind = indices[i]
        means[col_ind] += data[i]

    for i in range(minor_len):
        means[i] /= major_len

    for i in range(non_zero):
        col_ind = indices[i]
        diff = data[i] - means[col_ind]
        variances[col_ind] += diff * diff
        counts[col_ind] += 1

    for i in range(minor_len):
        variances[i] += (major_len - counts[i]) * means[i] ** 2
        variances[i] /= major_len

    return means, variances


@numba.njit(cache=True)
def sparse_mean_var_major_axis(data, indices, indptr, major_len, minor_len, dtype):
    """Given array for a csc matrix, returns means and variances for each row."""
    means = np.zeros(major_len, dtype=dtype)
    variances = np.zeros_like(means, dtype=dtype)

    for i in range(major_len):
        startptr = indptr[i]
        endptr = indptr[i + 1]
        counts = endptr - startptr

        for j in range(startptr, endptr):
            means[i] += data[j]
        means[i] /= minor_len

        for j in range(startptr, endptr):
            diff = data[j] - means[i]
            variances[i] += diff * diff

        variances[i] += (minor_len - counts) * means[i] ** 2
        variances[i] /= minor_len

    return means, variances
