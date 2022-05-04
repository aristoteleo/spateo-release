"""Spatial DEGs
"""
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from pysal import explore, lib
from scipy.sparse import issparse
from statsmodels.sandbox.stats.multicomp import multipletests

from ..configuration import SKM


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, optional=True)
def moran_i(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
    k: int = 5,
    weighted: Optional[List[str]] = None,
    permutations: int = 199,
    n_jobs: int = 40,
) -> pd.DataFrame:
    """Identify genes with strong spatial autocorrelation with Moran's I test.
    This can be used to identify genes that are
    potentially related to cluster.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for Moran's I calculation
            directly.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        x: 'list' or None(default: `None`)
            x-coordinates of all buckets.
        y: 'list' or None(default: `None`)
            y-coordinates of all buckets.
        k: 'int' (defult=20)
            Number of neighbors to use by default for kneighbors queries.
        weighted : 'str'(defult='kernel')
            Spatial weights, defult is None, 'kernel' is based on kernel functions.
        permutations: `int` (default=999)
            Number of random permutations for calculation of pseudo-p_values.
        n_cores: `int` (default=30)
            The maximum number of concurrently running jobs, If -1 all CPUs are used.
            If 1 is given, no parallel computing code is used at all.
    Returns
    -------
        A pandas DataFrame of the Moran' I test results.
    """
    if X_data is None:
        X_data = adata.X
    else:
        X_data = X_data
    if genes is None:
        genes = adata.var_names
    else:
        genes = genes
    if x is None:
        x = adata.obsm["spatial"][:, 0].tolist()
    else:
        x = x
    if y is None:
        y = adata.obsm["spatial"][:, 1].tolist()
    else:
        y = y
    xymap = pd.DataFrame({"x": x, "y": y})
    if weighted is not None:
        # weighted matrix (kernel distance)
        kw = lib.weights.Kernel(xymap, k, function="gaussian")
        W = lib.weights.W(kw.neighbors, kw.weights)
    else:
        # weighted matrix (in:1,out:0)
        kd = lib.cg.KDTree(np.array(xymap))
        nw = lib.weights.KNN(kd, k)
        W = lib.weights.W(nw.neighbors, nw.weights)

    # computing the moran_i for a single gene, and then used the joblib.Parallel to compute all genes in adata object.
    def _single(gene, X_data, W, adata, permutations):
        cur_X = X_data[:, adata.var.index == gene].A if issparse(X_data) else X_data[:, adata.var.index == gene]
        mbi = explore.esda.moran.Moran(cur_X, W, permutations=permutations, two_tailed=False)
        Moran_I = mbi.I
        p_value = mbi.p_sim
        statistics = mbi.z_sim
        return [gene, Moran_I, p_value, statistics]

    # parallel computing
    res = Parallel(n_jobs)(delayed(_single)(gene, X_data, W, adata, permutations) for gene in adata.var_names)
    res = pd.DataFrame(res, index=adata.var_names)
    res = res.drop(columns=0)
    res.columns = ["moran_i", "moran_p_val", "moran_z"]
    res["moran_q_val"] = multipletests(res["moran_p_val"], method="fdr_bh")[1]
    return res
