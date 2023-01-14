from typing import List, Optional, Union

import dynamo as dyn

# import ot
import numpy as np
import ot
import pandas as pd
from anndata import AnnData

# import multiprocessing
from scipy.sparse import csr_matrix, issparse

# import sys
# from tqdm import tqdm
# from functools import partial
# import scipy.stats
# import statsmodels
from scipy.sparse.csgraph import floyd_warshall

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..logging import logger_manager as lm


def bin_adata(
    adata: AnnData,
    bin_size: int = 1,
    layer: str = "spatial",
) -> AnnData:
    """Aggregate cell-based adata by bin size. Cells within a bin would be
    aggregated together as one cell.

    Args:
        adata: the input adata.
        bin_size: the size of square to bin adata.

    Returns:
        Aggreated adata.
    """
    adata = adata.copy()
    adata.obsm[layer] = (adata.obsm[layer] // bin_size).astype(np.int32)
    df = (
        pd.DataFrame(adata.X.A, columns=adata.var_names)
        if issparse(adata.X)
        else pd.DataFrame(adata.X, columns=adata.var_names)
    )
    df[["x", "y"]] = adata.obsm[layer]
    df2 = df.groupby(by=["x", "y"]).sum()
    a = AnnData(df2)
    a.uns["__type"] = "UMI"
    a.obs_names = [str(i[0]) + "_" + str(i[1]) for i in df2.index.to_list()]
    a.obsm[layer] = np.array([list(i) for i in df2.index.to_list()], dtype=np.float64)
    return a


def shuffle_adata(
    adata: AnnData,
    seed: int = 0,
    replace: bool = False,
):
    """Shuffle X in anndata object randomly.

    Args:
        adata: AnnData object
        seed: seed for randomly shuffling

    Returns:
        adata: AnnData object
    """
    adata = adata.copy()
    if seed == 0:
        return adata
    np.random.seed(seed)
    if issparse(adata.X):
        tmp = adata.X.A
        if replace:
            tmp = tmp[np.random.randint(len(tmp), size=len(tmp))]
        else:
            np.random.shuffle(tmp)
        adata.X.A = tmp
    else:
        tmp = adata.X
        if replace:
            tmp = tmp[np.random.randint(len(tmp), size=len(tmp))]
        else:
            np.random.shuffle(tmp)
        adata.X = tmp
    return adata


def filter_adata_by_pos_ratio(
    adata,
    pos_ratio,
):
    genes, adata = get_genes_by_pos_ratio(adata, pos_ratio)
    return adata[:, genes].copy()


def get_genes_by_pos_ratio(
    adata: AnnData,
    pos_ratio: float = 0.1,
) -> list:
    adata = adata.copy()
    adata.var["nCells"] = np.sum(adata.X.A > 0, axis=0) if issparse(adata.X) else np.sum(adata.X > 0, axis=0)
    adata.var["raw_pos_rate"] = adata.var["nCells"] / adata.n_obs
    return adata.var_names[adata.var["nCells"] / adata.n_obs > pos_ratio].to_list(), adata


def add_pos_ratio_to_adata(adata, layer=None, var_name="raw_pos_rate"):
    if layer:
        adata.var[var_name] = (
            np.sum(adata.layers[layer].A > 0, axis=0)
            if issparse(adata.layers[layer])
            else np.sum(adata.layers[layer] > 0, axis=0)
        )
    else:
        adata.var[var_name] = np.sum(adata.X.A > 0, axis=0) if issparse(adata.X) else np.sum(adata.X > 0, axis=0)
    adata.var[var_name] = adata.var[var_name] / adata.n_obs


def cal_geodesic_distance(
    adata: AnnData,
    layer: str = "spatial",
    n_neighbors: int = 30,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 4.0,
) -> AnnData:

    dyn.tl.neighbors(
        adata,
        X_data=adata.obsm[layer],
        n_neighbors=n_neighbors,
        result_prefix="spatial",
    )
    # remove islated one cell
    b = adata[
        np.min(
            adata.obsp["spatial_distances"].A,
            axis=1,
            initial=1e10,
            where=np.array(adata.obsp["spatial_distances"].A > 0),
        )
        <= min_dis_cutoff
    ]
    lm.main_info(f"The cell/buckets number after filtering by min_dis_cutoff is {len(b)}")

    # remove sparse cells
    b = b[np.max(b.obsp["spatial_distances"].A, axis=1) <= max_dis_cutoff]
    lm.main_info(f"The cell/buckets number after filtering by max_dis_cutoff is {len(b)}")

    dyn.tl.neighbors(
        b,
        X_data=b.obsm[layer],
        n_neighbors=n_neighbors,
        result_prefix="spatial",
    )
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import floyd_warshall

    conn = b.obsp["spatial_distances"].toarray()
    conn[conn == np.inf] = 0
    conn = csr_matrix(conn)
    dist_matrix, predecessors = floyd_warshall(csgraph=conn, directed=False, return_predecessors=True)
    b.obsp["distance"] = dist_matrix
    return b


def cal_euclidean_distance(
    adata: AnnData,
    layer: str = "spatial",
    min_dis_cutoff: float = np.inf,
    max_dis_cutoff: float = np.inf,
) -> AnnData:

    dyn.tl.neighbors(
        adata,
        X_data=adata.obsm[layer],
        n_neighbors=adata.n_obs,
        result_prefix="spatial",
    )
    # remove islated one cell
    b = adata[
        np.min(
            adata.obsp["spatial_distances"].A,
            axis=1,
            initial=1e10,
            where=np.array(adata.obsp["spatial_distances"].A > 0),
        )
        <= min_dis_cutoff
    ]

    # remove sparse cells
    b = b[np.max(b.obsp["spatial_distances"].A, axis=1) <= max_dis_cutoff]

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import floyd_warshall

    conn = b.obsp["spatial_distances"].toarray()
    conn[conn == np.inf] = 0
    conn = csr_matrix(conn)
    dist_matrix, predecessors = floyd_warshall(csgraph=conn, directed=False, return_predecessors=True)
    b.obsp["distance"] = dist_matrix
    return b


def scale_to(
    adata: AnnData,
    to_median: bool = True,
    N: int = 10000,
) -> AnnData:
    adata = adata.copy()
    if issparse(adata.X):
        adata.X.A = adata.X.A.astype(np.float64)
    else:
        adata.X = adata.X.astype(np.float64)

    if to_median:
        if issparse(adata.X):
            N = np.median(np.sum(adata.X.A, axis=1))
        else:
            N = np.median(np.sum(adata.X, axis=1))

    if issparse(adata.X):
        adata.X.A = (adata.X.A.T / (np.sum(adata.X.A, axis=1) / N)).T
    else:
        adata.X = (adata.X.T / (np.sum(adata.X, axis=1) / N)).T
    return adata


def cal_wass_dis(M, a, b=[], numItermax=1000000):
    """Computing Wasserstein distance.

    Args:
        M:  (ns,nt) array-like, float – Loss matrix (c-order array in numpy with type float64)
        a: (ns,) array-like, float – Source histogram (uniform weight if empty list)
        b: (nt,) array-like, float – Target histogram (uniform weight if empty list)

    Returns:
        W: (float, array-like) – Optimal transportation loss for the given parameters

    """

    W = ot.emd2(a, b, M, numItermax=numItermax)

    return W


def cal_rank_p(genes, ws, w_df, bin_num=100):
    ws_dict = {}
    for g, w in zip(genes, ws):
        if g not in ws_dict:
            ws_dict[g] = []
        ws_dict[g].append(w)

    sorted_genes = w_df["mean"].sort_values().index.to_list()
    each_bin_gene_num = int(len(sorted_genes) / bin_num) + 1
    each_bin_ws = {}
    bin_of_gene = {}
    for i in range(bin_num):
        each_bin_ws[i] = []
        for g in sorted_genes[i * each_bin_gene_num : (i + 1) * each_bin_gene_num]:
            if np.sum(np.array(ws_dict[g])) > 0:
                each_bin_ws[i].append(ws_dict[g])
            bin_of_gene[g] = i
        each_bin_ws[i] = np.array(each_bin_ws[i])
    rank_p = []
    for g in w_df.index:
        t = each_bin_ws[bin_of_gene[g]].flatten()
        rank_p.append((np.sum(t >= w_df.loc[g, "Wasserstein_distance"]) + 1) / len(t))
    return rank_p, each_bin_ws


def loess_reg(
    adata: AnnData,
    layers: str = "X",
) -> AnnData:
    adata = adata.copy()
    if issparse(adata.X):
        adata.X.A = adata.X.A.astype(np.float64)
    else:
        adata.X = adata.X.astype(np.float64)

    if issparse(adata.X):
        adata.X.A = (adata.X.A.T / (np.sum(adata.X.A, axis=1) / N)).T
    else:
        adata.X = (adata.X.T / (np.sum(adata.X, axis=1) / N)).T
    return adata
