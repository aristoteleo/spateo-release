"""
Functions to either scale single-cell data or normalize such that the row-wise sums are identical.
"""
import inspect
import warnings
from typing import Dict, Iterable, Optional, Union

from ..configuration import SKM

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
    target_sum: float = 1e4,
    norm_factor: Optional[np.ndarray] = None,
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
        norm_factor: Optional array of shape `n_obs` × `1`, where `n_obs` is the number of observations (cells). Each
            entry contains a pre-computed normalization factor for that cell.
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

    if norm_factor is not None:
        counts_per_cell *= norm_factor

    # logger.info(msg)
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


# ---------------------------------------------------------------------------------------------------
# Scale factors
# ---------------------------------------------------------------------------------------------------
def calcFactorRLE(data: np.ndarray) -> np.ndarray:
    """
    Calculate scaling factors using the Relative Log Expression (RLE) method. Python implementation of the same-named
    function from edgeR:

    Robinson, M. D., McCarthy, D. J., & Smyth, G. K. (2010). edgeR: a Bioconductor package for
    differential expression analysis of digital gene expression data. Bioinformatics, 26(1), 139-140.

    Args:
        data: An array-like object representing the data matrix.

    Returns:
        factors: An array of scaling factors for each cell
    """
    gm = np.exp(np.mean(np.log(data), axis=0))
    factors = np.apply_along_axis(lambda u: np.median(u / gm[gm > 0]), axis=1, arr=data)
    return factors


def calcFactorQuantile(data: np.ndarray, lib_size: float, p: float = 0.75) -> np.ndarray:
    """
    Calculate scaling factors using the Quantile method. Python implementation of the same-named function from edgeR:

    Robinson, M. D., McCarthy, D. J., & Smyth, G. K. (2010). edgeR: a Bioconductor package for
    differential expression analysis of digital gene expression data. Bioinformatics, 26(1), 139-140.

    Args:
        data: An array-like object representing the data matrix.
        lib_size: The library size or total count to normalize against.
        p: The quantile value (default: 0.75).

    Returns:
        factors: An array of scaling factors for each cell
    """
    factors = np.percentile(data, p * 100, axis=1)
    if np.min(factors) == 0:
        print(
            f"Quantile method note: one or more quantiles are zero ({p * 100}th percentile for one or more cells "
            f"is zero."
        )
    factors /= lib_size
    return factors


def calcFactorTMM(
    obs: Union[float, np.ndarray],
    ref: Union[float, np.ndarray],
    libsize_obs: Optional[float] = None,
    libsize_ref: Optional[float] = None,
    logratioTrim: float = 0.3,
    sumTrim: float = 0.05,
    doWeighting: bool = True,
    Acutoff: float = -1e10,
) -> float:
    """
    Calculate scaling factors using the Trimmed Mean of M-values (TMM) method. Python implementation of the
    same-named function from edgeR:

    Robinson, M. D., McCarthy, D. J., & Smyth, G. K. (2010). edgeR: a Bioconductor package for
    differential expression analysis of digital gene expression data. Bioinformatics, 26(1), 139-140.

    Args:
        obs: An array-like object representing the observed library counts.
        ref: An array-like object representing the reference library counts.
        libsize_obs: The library size of the observed library (default: sum of observed counts).
        libsize_ref: The library size of the reference library (default: sum of reference counts).
        logratioTrim: The fraction of extreme log-ratios to be trimmed (default: 0.3).
        sumTrim: The fraction of extreme log-ratios to be trimmed based on the absolute expression (default: 0.05).
        doWeighting: Whether to perform weighted TMM estimation (default: True).
        Acutoff: The cutoff value for removing infinite values (default: -1e10).

    Returns:
        factor: floating point scaling factor
    """
    obs = np.asarray(obs, dtype=float)
    ref = np.asarray(ref, dtype=float)

    nO = np.sum(obs) if libsize_obs is None else libsize_obs
    nR = np.sum(ref) if libsize_ref is None else libsize_ref

    logR = np.log2((obs / nO) / (ref / nR))  # log ratio of expression, accounting for library size
    absE = (np.log2(obs / nO) + np.log2(ref / nR)) / 2  # absolute expression
    v = (nO - obs) / nO / obs + (nR - ref) / nR / ref  # estimated asymptotic variance

    # remove infinite values, cutoff based on A
    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > Acutoff)

    logR = logR[fin]
    absE = absE[fin]
    v = v[fin]

    if np.max(np.abs(logR)) < 1e-6:
        return 1

    n = len(logR)
    loL = int(n * logratioTrim) + 1
    loS = int(n * sumTrim) + 1

    keep = (np.argsort(logR).argsort() >= loL) & (np.argsort(absE).argsort() >= loS)

    if doWeighting:
        f = np.sum(logR[keep] / v[keep], axis=0, keepdims=True) / np.sum(1 / v[keep], axis=0, keepdims=True)
    else:
        f = np.mean(logR[keep], axis=0, keepdims=True)

    if np.isnan(f):
        f = 0
    factor = 2**f

    return factor


def calcFactorTMMwsp(
    obs: Union[float, np.ndarray],
    ref: Union[float, np.ndarray],
    libsize_obs: Optional[float] = None,
    libsize_ref: Optional[float] = None,
    logratioTrim: float = 0.3,
    sumTrim: float = 0.05,
    doWeighting: bool = True,
) -> float:
    """
    Calculate scaling factors using the Trimmed Mean of M-values with singleton pairing (TMMwsp) method. Python
    implementation of the same-named function from edgeR:

    Robinson, M. D., McCarthy, D. J., & Smyth, G. K. (2010). edgeR: a Bioconductor package for
    differential expression analysis of digital gene expression data. Bioinformatics, 26(1), 139-140.

    Args:
        obs: An array-like object representing the observed library counts.
        ref: An array-like object representing the reference library counts.
        libsize_obs: The library size of the observed library (default: sum of observed counts).
        libsize_ref: The library size of the reference library (default: sum of reference counts).
        logratioTrim: The fraction of extreme log-ratios to be trimmed (default: 0.3).
        sumTrim: The fraction of extreme log-ratios to be trimmed based on the absolute expression (default: 0.05).
        doWeighting: Whether to perform weighted TMM estimation (default: True).

    Returns:
        factor: floating point scale factor
    """
    obs = np.asarray(obs, dtype=float)
    ref = np.asarray(ref, dtype=float)

    eps = 1e-14

    pos_obs = obs > eps
    pos_ref = ref > eps
    npos = 2 * pos_obs + pos_ref

    i = np.where((npos == 0) | np.isnan(npos))[0]
    if len(i) > 0:
        obs = np.delete(obs, i)
        ref = np.delete(ref, i)
        npos = np.delete(npos, i)

    if libsize_obs is None:
        libsize_obs = np.sum(obs)
    if libsize_ref is None:
        libsize_ref = np.sum(ref)

    zero_obs = npos == 1
    zero_ref = npos == 2
    k = zero_obs | zero_ref
    n_eligible_singles = min(np.sum(zero_obs), np.sum(zero_ref))
    if n_eligible_singles > 0:
        refk = np.sort(ref[k])[::-1][:n_eligible_singles]
        obsk = np.sort(obs[k])[::-1][:n_eligible_singles]
        obs = np.concatenate([obs[~k], obsk])
        ref = np.concatenate([ref[~k], refk])
    else:
        obs = obs[~k]
        ref = ref[~k]

    n = len(obs)
    if n == 0:
        return 1

    obs_p = obs / libsize_obs
    ref_p = ref / libsize_ref
    M = np.log2(obs_p / ref_p)
    A = 0.5 * np.log2(obs_p * ref_p)

    if np.max(np.abs(M)) < 1e-6:
        return 1

    obs_p_shrunk = (obs + 0.5) / (libsize_obs + 0.5)
    ref_p_shrunk = (ref + 0.5) / (libsize_ref + 0.5)
    M_shrunk = np.log2(obs_p_shrunk / ref_p_shrunk)
    o_M = np.lexsort((M_shrunk, M))

    o_A = np.argsort(A)

    loM = int(n * logratioTrim) + 1
    hiM = n + 1 - loM
    keep_M = np.zeros(n, dtype=bool)
    keep_M[o_M[loM:hiM]] = True
    loA = int(n * sumTrim) + 1
    hiA = n + 1 - loA
    keep_A = np.zeros(n, dtype=bool)
    keep_A[o_A[loA:hiA]] = True
    keep = keep_M & keep_A
    M = M[keep]

    if doWeighting:
        obs_p = obs_p[keep]
        ref_p = ref_p[keep]
        v = (1 - obs_p) / obs_p / libsize_obs + (1 - ref_p) / ref_p / libsize_ref
        w = (1 + 1e-6) / (v + 1e-6)
        TMM = np.sum(w * M) / np.sum(w)
    else:
        TMM = np.mean(M)

    factor = 2**TMM
    return factor


def calcNormFactors(
    counts: Union[np.ndarray, scipy.sparse.spmatrix],
    lib_size: Optional[np.ndarray] = None,
    method: str = "TMM",
    refColumn: Optional[int] = None,
    logratioTrim: float = 0.3,
    sumTrim: float = 0.05,
    doWeighting: bool = True,
    Acutoff: float = -1e10,
    p: float = 0.75,
) -> np.ndarray:
    """
    Function to scale normalize RNA-Seq data for count matrices.
    This is a Python translation of an R function from edgeR package.

    Args:
        object: Array or sparse array of shape [n_samples, n_features] containing gene expression data. Note that a
            sparse array will be converted to dense before calculations.
        lib_size: The library sizes for each sample.
        method: The normalization method. Can be:
                -"TMM": trimmed mean of M-values,
                -"TMMwsp": trimmed mean of M-values with singleton pairings,
                -"RLE": relative log expression, or
                -"upperquartile": using the quantile method
            Defaults to "TMM".
        refColumn: Optional reference column for normalization
        logratioTrim: For TMM normalization, the fraction of extreme log-ratios to be trimmed (default: 0.3).
        sumTrim: For TMM normalization, the fraction of extreme log-ratios to be trimmed based on the absolute
            expression (default: 0.05).
        doWeighting: Whether to perform weighted TMM estimation (default: True).
        Acutoff: For TMM normalization, the cutoff value for removing infinite values (default: -1e10).
        p: Parameter for upper quartile normalization. Defaults to 0.75.

    Returns:
        factors: The normalization factors for each sample.
    """
    if isinstance(counts, scipy.sparse.spmatrix):
        counts = counts.toarray()

    if np.any(np.isnan(counts)):
        raise ValueError("NA counts not permitted")
    nsamples = counts.shape[0]

    # Check library size
    if lib_size is None:
        lib_size = np.sum(counts, axis=1)
    else:
        if np.any(np.isnan(lib_size)):
            raise ValueError("NA lib sizes not permitted")
        if len(lib_size) != nsamples:
            if len(lib_size) > 1:
                print("calcNormFactors: length (lib size) doesn't match number of samples")
            lib_size = np.repeat(lib_size, nsamples)

    # Remove all zero columns (all-zero genes)
    allzero = np.sum(counts > 0, axis=0) == 0
    if np.any(allzero):
        counts = counts[:, ~allzero]

    # Calculate factors
    if method == "TMM":
        if refColumn is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                f75 = calcFactorQuantile(data=counts, lib_size=lib_size, p=0.75)
                if np.median(f75) < 1e-20:
                    refColumn = np.argmax(np.sum(np.sqrt(counts), axis=1))
                else:
                    f75_mean_diff = np.abs(f75 - np.mean(f75))
                    refColumn = np.argmin(f75_mean_diff)

            factors = np.empty(nsamples)
            for i in range(nsamples):
                factors[i] = calcFactorTMM(
                    obs=counts[i, :],
                    ref=counts[refColumn, :],
                    libsize_obs=lib_size[i],
                    libsize_ref=lib_size[refColumn],
                    logratioTrim=logratioTrim,
                    sumTrim=sumTrim,
                    doWeighting=doWeighting,
                    Acutoff=Acutoff,
                )
            return factors

    elif method == "TMMwsp":
        if refColumn is None:
            refColumn = np.argmax(np.sum(np.sqrt(counts), axis=1))

        factors = np.empty(nsamples)
        for i in range(nsamples):
            factors[i] = calcFactorTMMwsp(
                obs=counts[i, :],
                ref=counts[refColumn, :],
                libsize_obs=lib_size[i],
                libsize_ref=lib_size[refColumn],
                logratioTrim=logratioTrim,
                sumTrim=sumTrim,
                doWeighting=doWeighting,
            )
        return factors

    elif method == "RLE":
        factors = calcFactorRLE(data=counts) / lib_size

    elif method == "upperquartile":
        factors = calcFactorQuantile(data=counts, lib_size=lib_size, p=p)

    else:
        raise ValueError("Invalid method: " + method)

    # Normalize factors:
    factors = factors / np.exp(np.mean(np.log(factors)))
    return factors


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def factor_normalization(adata: AnnData, norm_factors: Optional[np.ndarray] = None, **kwargs):
    """Wrapper to apply factor normalization to AnnData object.

    Args:
        adata: The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
        norm_factors: Array of shape (`n_obs`, ), the normalization factors for each sample. If not given,
            will compute using :func `calcNormFactors` and any arguments given to `kwargs`.
        **kwargs: Keyword arguments to pass to :func `calcNormFactors` or :func `normalize_total`. Options:
            lib_size: The library sizes for each sample.
            method: The normalization method. Can be:
                    -"TMM": trimmed mean of M-values,
                    -"TMMwsp": trimmed mean of M-values with singleton pairings,
                    -"RLE": relative log expression, or
                    -"upperquartile": using the quantile method
                Defaults to "TMM".
            refColumn: Optional reference column for normalization
            logratioTrim: For TMM normalization, the fraction of extreme log-ratios to be trimmed (default: 0.3).
            sumTrim: For TMM normalization, the fraction of extreme log-ratios to be trimmed based on the absolute
                expression (default: 0.05).
            doWeighting: Whether to perform weighted TMM estimation (default: True).
            Acutoff: For TMM normalization, the cutoff value for removing infinite values (default: -1e10).
            p: Parameter for upper quartile normalization. Defaults to 0.75.
            target_sum: Desired sum of counts for each gene post-normalization. If `None`, after normalization,
            each observation (cell) will have a total count equal to the median of total counts for observations (
            cells) before normalization.
            exclude_highly_expressed: Exclude (very) highly expressed genes for the computation of the normalization
                factor for each cell. A gene is considered highly expressed if it has more than `max_fraction` of the
                total counts in at least one cell.
            max_fraction: If `exclude_highly_expressed=True`, this is the cutoff threshold for excluding genes.
            key_added: Name of the field in `adata.obs` where the normalization factor is stored.
            layer: Layer to normalize instead of `X`. If `None`, `X` is normalized.
            inplace: Whether to update `adata` or return dictionary with normalized copies of `adata.X` and
                `adata.layers`.
            copy: Whether to modify copied input object. Not compatible with inplace=False.

    Returns:
        adata: The normalized AnnData object.
    """

    calc_norm_factors_signatures = inspect.signature(calcNormFactors)
    calc_norm_factors_valid_params = [
        p.name for p in calc_norm_factors_signatures.parameters.values() if p.name in kwargs
    ]
    calc_norm_factors_params = {k: kwargs.pop(k) for k in calc_norm_factors_valid_params}

    normalize_total_signatures = inspect.signature(normalize_total)
    normalize_total_valid_params = [p.name for p in normalize_total_signatures.parameters.values() if p.name in kwargs]
    normalize_total_params = {k: kwargs.pop(k) for k in normalize_total_valid_params}

    if norm_factors is None:
        norm_factors = calcNormFactors(adata.X, **calc_norm_factors_params)

    # If 'inplace' is False or 'copy' is True, get appropriate return from :func `normalize_total`
    if not kwargs.get("inplace", True) or kwargs.get("copy", False):
        norm_return = normalize_total(adata, norm_factor=norm_factors, **normalize_total_params)
        return norm_return
    else:
        normalize_total(adata, norm_factor=norm_factors, **normalize_total_params)
        return adata
