"""Wasserstein distance would be calculated by ot python package, see following:

Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer,
POT Python Optimal Transport library,
Journal of Machine Learning Research, 22(78):1−8, 2021.
Website: https://pythonot.github.io/
"""
import multiprocessing
import sys
from functools import partial
from typing import List, Optional, Union

import dynamo as dyn
import numpy as np
import ot
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm
from typing_extensions import Literal


def _cal_dis(adata, x1):
    """Compute distance between samples in x1.

    Args:
        adata:

    Returns:
        adata:
    """
    adata = adata.copy()
    x1 = np.array(adata.obsm["spatial"], dtype=np.float64)
    M = ot.dist(x1)
    adata.obsp["geodesic_distance"] = M
    return adata


def _cal_wass_dis(M, a, b=[], numItermax=1000000):
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


def bin_adata(adata, bin_size=1):
    adata = adata.copy()
    adata.obsm["spatial"] = (adata.obsm["spatial"] // bin_size).astype(np.int32)
    df = (
        pd.DataFrame(adata.X.A, columns=adata.var_names)
        if issparse(adata.X)
        else pd.DataFrame(adata.X, columns=adata.var_names)
    )
    df[["x", "y"]] = adata.obsm["spatial"]
    df2 = df.groupby(by=["x", "y"]).sum()
    a = AnnData(df2)
    a.uns["__type"] = "UMI"
    a.obs_names = [str(i[0]) + "_" + str(i[1]) for i in df2.index.to_list()]
    a.obsm["spatial"] = np.array([list(i) for i in df2.index.to_list()], dtype=np.float64)

    return a


def _cal_geodesic_distance(adata, n_neighbors=30, min_dis_cutoff=2.0, max_dis_cutoff=4.0):
    dyn.tl.neighbors(
        adata,
        X_data=adata.obsm["spatial"],
        n_neighbors=n_neighbors,
        result_prefix="spatial",
    )

    b = adata[
        np.min(
            adata.obsp["spatial_distances"].A,
            axis=1,
            initial=1e10,
            where=np.array(adata.obsp["spatial_distances"].A > 0),
        )
        <= min_dis_cutoff
    ]
    b = b[np.max(b.obsp["spatial_distances"].A, axis=1) <= max_dis_cutoff]

    dyn.tl.neighbors(
        b,
        X_data=b.obsm["spatial"],
        n_neighbors=n_neighbors,
        result_prefix="spatial",
    )
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import floyd_warshall

    conn = b.obsp["spatial_distances"].toarray()
    conn[conn == np.inf] = 0
    conn = csr_matrix(conn)
    dist_matrix, predecessors = floyd_warshall(csgraph=conn, directed=False, return_predecessors=True)
    b.obsp["geodesic_distance"] = dist_matrix

    return b


def _cal_wass_dis_on_genes(M, inp):  # adata, gene_ids, b=[], numItermax=1000000):
    adata, gene_ids, b, numItermax = inp
    ws = []
    pos_rs = []
    if issparse(adata.X):
        df = pd.DataFrame(adata.X.A, columns=adata.var_names)
    else:
        df = pd.DataFrame(adata.X, columns=adata.var_names)

    for gene_id in gene_ids:
        a = np.array(df.loc[:, gene_id], dtype=(np.float64)) / df.loc[:, gene_id].sum()
        w = _cal_wass_dis(M, a, b, numItermax=numItermax)
        pos_r = np.sum(a > 0) / len(a)
        ws.append(w)
        pos_rs.append(pos_r)
    return gene_ids, ws, pos_rs


def shuffle_adata(adata: AnnData, seed: int = 0):
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
        np.random.shuffle(tmp)
        adata.X.A = tmp
    else:
        tmp = adata.X
        np.random.shuffle(tmp)
        adata.X = tmp

    return adata


def cal_wass_dis_bs(
    adata: AnnData,
    bin_size: int = 1,
    numItermax: int = 1000000,
    gene_set: Union[List, np.ndarray] = None,
    compare_to: Literal["uniform", "allUMI"] = "allUMI",
    processes: int = 1,
    bootstrap: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
) -> pd.DataFrame:
    """Computing Wasserstein distance for a AnnData to identify spatially variable genes.

    Args:
        adata: AnnData object
        bin_size: bin size for mergeing cells.
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged
        gene_set: Gene set for computing, default is for all genes.
        compare_to: compare distance to uniform distribution or allUMI distribution.
        processes: process number for parallelly running
        bootstrap: bootstrap number for permutation to calculate p-value
        min_dis_cutoff: Cells/Bins whose min distance to 30 neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30 neighbors are larger than this cutoff would be filtered.
    Returns:
        w_df: a dataframe
        adata0: binned AnnData object
    """

    adata0 = bin_adata(adata, bin_size)
    adata0 = _cal_geodesic_distance(adata0, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff)
    M = adata0.obsp["geodesic_distance"]
    if np.sum(~np.isfinite(M)) > 0:
        print("geodesic_distance has inf value")
        sys.exit()

    if gene_set is None:
        gene_set = adata0.var_names

    adatas, bs = [], []

    adata = shuffle_adata(adata0, 0)
    if compare_to == "uniform":
        b = []
    elif compare_to == "allUMI":
        if issparse(adata.X):
            b = np.array(adata.X.A.sum(axis=1) / adata.X.A.sum(), dtype=np.float64)
        else:
            b = np.array(adata.X.sum(axis=1) / adata.X.sum(), dtype=np.float64)

    # pbar=tqdm(total=bootstrap+1)
    genes, ws, pos_rs = _cal_wass_dis_on_genes(M, (adata, gene_set, b, numItermax))
    w_df0 = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})
    for i in range(1, bootstrap + 1):
        adata = shuffle_adata(adata0, i)
        adatas.append(adata)

        if compare_to == "uniform":
            b = []
        elif compare_to == "allUMI":
            if issparse(adata.X):
                b = np.array(adata.X.A.sum(axis=1) / adata.X.A.sum(), dtype=np.float64)
            else:
                b = np.array(adata.X.sum(axis=1) / adata.X.sum(), dtype=np.float64)

        bs.append(b)

    pool = multiprocessing.Pool(processes=processes)

    # res = pool.starmap(_cal_wass_dis_on_genes, [(M, adatas[i], gene_set, bs[i], numItermax) for i in range(len(adatas))])
    inputs = [(adatas[i], gene_set, bs[i], numItermax) for i in range(len(adatas))]
    res = []
    for result in tqdm(pool.imap_unordered(partial(_cal_wass_dis_on_genes, M), inputs), total=len(inputs)):
        res.append(result)

    genes, ws, pos_rs = zip(*res)
    genes = [g for i in genes for g in i]
    ws = [g for i in ws for g in i]
    pos_rs = [g for i in pos_rs for g in i]
    w_df = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})
    mean_std_df = pd.DataFrame(
        {
            "mean": w_df.groupby("gene_id")["Wasserstein_distance"].mean().to_list(),
            "std": w_df.groupby("gene_id")["Wasserstein_distance"].std().to_list(),
        },
        index=w_df.groupby("gene_id")["Wasserstein_distance"].mean().index,
    )
    w_df = pd.concat([w_df0.set_index("gene_id"), mean_std_df], axis=1)
    w_df["zscore"] = (w_df["Wasserstein_distance"] - w_df["mean"]) / w_df["std"]
    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)
    import scipy.stats

    # find p-value for two-tailed test
    w_df["pvalue"] = scipy.stats.norm.sf(abs(w_df["zscore"])) * 2
    import statsmodels

    w_df["adj_pvalue"] = statsmodels.stats.multitest.multipletests(w_df["pvalue"])[1]

    w_df["fc"] = w_df["Wasserstein_distance"] / w_df["mean"]
    w_df["log2fc"] = np.log2(w_df["fc"])
    w_df["-log10adjp"] = -np.log10(w_df["adj_pvalue"])
    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)
    return w_df, adata0
