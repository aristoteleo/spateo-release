"""Spatial DEGs
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from pysal import explore, lib
from scipy.sparse import issparse
from statsmodels.sandbox.stats.multicomp import multipletests

from ..logging import logger_manager as lm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..configuration import SKM


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def moran_i(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    spatial_key: str = "spatial",
    model: Literal["2d", "3d"] = "2d",
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
    z: Optional[List[int]] = None,
    k: int = 5,
    weighted: Optional[List[str]] = None,
    permutations: int = 199,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Identify genes with strong spatial autocorrelation with Moran's I test.
    This can be used to identify genes that are
    potentially related to cluster.

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each cell.
        x: 'list' or None(default: `None`)
            x-coordinates of all buckets.
        y: 'list' or None(default: `None`)
            y-coordinates of all buckets.
        z: 'list' or None(default: `None`)
            z-coordinates of all buckets.
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
    if layer is None:
        X_data = adata.X
    else:
        X_data = adata.layers[layer]
    if genes is None:
        genes = adata.var_names
    else:
        genes = genes
    if x is None:
        x = adata.obsm[spatial_key][:, 0].tolist()
    else:
        x = x
    if y is None:
        y = adata.obsm[spatial_key][:, 1].tolist()
    else:
        y = y
    if model == "3d":
        if z is None:
            z = adata.obsm[spatial_key][:, 2].tolist()
        else:
            z = z
        xymap = pd.DataFrame({"x": x, "y": y, "z": z})
    else:
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


def cellbin_morani(
    adata_cellbin: AnnData,
    binsize: int,
    cluster_key: str = "Celltype",
) -> pd.DataFrame:
    """Calculate Moran's I score for each celltype (in segmented cell adata).
    Since the presentation of cells are boolean values, this function first
    summarizes the number of each celltype using a given binsize, creating a
    spatial 2D matrix with cell counts. Then calculates Moran's I score on the
    matrix for spatial score for each celltype.

    Parameters
    ----------
        adata_cellbin: :class:`~anndata.AnnData`
            An Annodata object for segmented cells.
        binsize: int
            The binsize used to summarize cell counts for each celltype.
        cluster_key: `str` (default="Celltype")
            The key in adata.obs including celltype labels.
    Returns
    -------
        A pandas DataFrame containing the Moran' I score for celltypes.
    """
    from esda.moran import Moran
    from libpysal.weights import lat2W

    lm.main_info("Calculating cell counts in each bin, using binsize " + str(binsize))
    spatial_cellbin = np.zeros(
        (
            int(max(adata_cellbin.obsm["X_spatial"][:, 0] // binsize)) + 1,
            int(max(adata_cellbin.obsm["X_spatial"][:, 1] // binsize)) + 1,
        )
    )
    w = lat2W(spatial_cellbin.shape[0], spatial_cellbin.shape[1])
    mi = []
    mi_norm = []
    lm.main_info("Calculating Moran's I score for each celltype")
    for i in np.unique(adata_cellbin.obs[cluster_key]):
        spatial_cellbin[:, :] = 0
        subset_adata_cellbin = adata_cellbin[adata_cellbin.obs[cluster_key] == i, :].copy()
        for j in subset_adata_cellbin.obsm["spatial"] // binsize:
            spatial_cellbin[int(j[0]), int(j[1])] = spatial_cellbin[int(j[0]), int(j[1])] + 1
        mi_tmp = Moran(spatial_cellbin, w)
        mi.append(mi_tmp.I)
        mi_norm.append(mi_tmp.p_norm)
    mi_df = pd.DataFrame(
        {"cluster": np.unique(adata_cellbin.obs[cluster_key]), "moran_i": mi, "moran_i_p_norm": mi_norm}
    )
    mi_df = mi_df.sort_values(by="moran_i", ascending=False)
    return mi_df
