"""Spatial DEGs
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from anndata import AnnDat
from pysal import explore, lib
from scipy.sparse import issparse
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm


def moran_i(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
    k: int = 5,
    weighted: str = "kernel",
    assumption: str = "permutation",
):
    """Identify genes with strong spatial autocorrelation with Moran's I test.
    This can be used to identify genes that are
    potentially related to cluster.

    TODO: `assumption` argument is never used.
    TODO: `weighted` argument should be Literal type.

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
            Spatial weights, defult is based on kernel functions.
        assumption: `str` (default: `permutation`)
            Monte Carlo approach(a permutation bootstrap test) to estimating
            significance.
    Returns
    -------
        Returns an updated `~anndata.AnnData` with a new key `'Moran_' + type`
        in the .uns attribute, storing the Moran' I test results.
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
        x = adata.obs["x_array"].tolist()
    else:
        x = x
    if y is None:
        y = adata.obs["y_array"].tolist()
    else:
        y = y
    gene_num = len(genes)
    sparse = issparse(X_data)
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
    Moran_I, p_value, statistics = (
        [0] * gene_num,
        [0] * gene_num,
        [0] * gene_num,
    )
    for i_gene, gene in tqdm(
        enumerate(genes), desc="Moran's I Global Autocorrelation Statistic"
    ):
        cur_X = (
            X_data[:, adata.var.index == gene].A
            if sparse
            else X_data[:, adata.var.index == gene]
        )
        mbi = explore.esda.moran.Moran(cur_X, W, permutations=999, two_tailed=False)
        Moran_I[i_gene] = mbi.I
        p_value[i_gene] = mbi.p_sim
        statistics[i_gene] = mbi.z_sim
    Moran_res = pd.DataFrame(
        {
            "moran_i": Moran_I,
            "moran_p_val": p_value,
            "moran_q_val": multipletests(p_value, method="fdr_bh")[1],
            "moran_z": statistics,
        },
        index=genes,
    )
    return Moran_res
