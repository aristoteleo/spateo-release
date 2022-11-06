"""
Companion functions for cell-cell communication inference analyses
"""
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd


def fdr_correct(
    pvals: pd.DataFrame,
    corr_method: str,
    corr_axis: Literal["interactions", "clusters"] = "clusters",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Correct p-values for FDR along specific axis in `pvals`.

    Args:
        pvals : pd.DataFrame
        corr_method : str
            Correction method, should be one of the options in :func `statsmodels.stats.multitest.multipletests`
            (listed below for reference):
                - Named methods:
                    - bonferroni
                    - sidak
                    - holm-sidak
                    - holm
                    - simes-hochberg
                    - hommel
                - Abbreviated methods:
                    - fdr_bh: Benjamini-Hochberg correction
                    - fdr_by: Benjamini-Yekutieli correction
                    - fdr_tsbh: Two-stage Benjamini-Hochberg
                    - fdr_tsbky: Two-stage Benjamini-Krieger-Yekutieli method
        corr_axis : str
            Either "interactions" or "clusters"- specifies whether the dataframe given to 'pvals' corresponds to
            samples or interactions. If 'interactions', will need to transpose the array first before performing
            multiple hypothesis correction.

    """
    from pandas.core.arrays.sparse import SparseArray
    from statsmodels.stats.multitest import multipletests

    def fdr(pvals: pd.Series) -> SparseArray:
        _, qvals, _, _ = multipletests(
            np.nan_to_num(pvals.values, copy=True, nan=1.0),
            method=corr_method,
            alpha=alpha,
            is_sorted=False,
            returnsorted=False,
        )
        qvals[np.isnan(pvals.values)] = np.nan

        return SparseArray(qvals, dtype=qvals.dtype, fill_value=np.nan)

    if corr_axis == "clusters":
        # clusters are in columns
        pvals = pvals.apply(fdr)
    elif corr_axis == "interactions":
        pvals = pvals.T.apply(fdr).T
    else:
        raise NotImplementedError(f"FDR correction for `{corr_axis}` is not implemented.")

    return pvals
