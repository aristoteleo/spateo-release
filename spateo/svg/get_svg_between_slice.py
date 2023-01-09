import multiprocessing
import sys
from functools import partial
from typing import List, Optional, Union

import dynamo as dyn
import numpy as np
import ot
import pandas as pd
import scipy.stats
import statsmodels
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm
from typing_extensions import Literal

from .get_svg import bin_scale_adata_get_distance
from .utils import *


def cal_gro_wass_bs(
    adata1: AnnData,
    adata2: AnnData,
    bin_size1: int = 1,
    bin_size2: int = 1,
    bin_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    distance_layer: str = "spatial",
    n_neighbors: int = 30,
    gene_set: Union[List, np.ndarray] = None,
    processes: int = 1,
    bootstrap: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
    larger_or_small: str = "larger",
):
    adata1, C1 = bin_scale_adata_get_distance(
        adata1,
        bin_size=bin_size1,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )
    # print("read one finish")
    adata2, C2 = bin_scale_adata_get_distance(
        adata2,
        bin_size=bin_size2,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )
    # return(adata1, adata2)
    if gene_set is None:
        print("Please provide gene set")
        sys.exit()

    gene_set_ov = np.intersect1d(adata1.var_names, adata2.var_names)
    if np.isin(gene_set, gene_set_ov, invert=True).any():
        print("gene_set is not all in intersection of two adata")
        sys.exit()
    # print("get genes finish")
    genes, gws, pos_r1s, pos_r2s = cal_gw_dis_on_genes((C1, C2, adata1, adata2), (0, gene_set))
    # print('one')

    gw_df0 = pd.DataFrame(
        {
            "gene_id": gene_set,
            "Gromov-wasserstein_distance": gws,
            "positive_ratio1": pos_r1s,
            "positive_ratio2": pos_r2s,
        }
    )

    pool = multiprocessing.Pool(processes=processes)

    inputs = [(i, gene_set) for i in range(1, bootstrap + 1)]
    res = []
    for result in tqdm(
        pool.imap_unordered(partial(cal_gw_dis_on_genes, (C1, C2, adata1, adata2)), inputs), total=len(inputs)
    ):
        res.append(result)
    genes, gws, pos_r1s, pos_r2s = zip(*res)
    genes = [g for i in genes for g in i]
    gws = [g for i in gws for g in i]
    pos_r1s = [g for i in pos_r1s for g in i]
    pos_r2s = [g for i in pos_r2s for g in i]

    gw_df = pd.DataFrame(
        {"gene_id": genes, "Gromov-wasserstein_distance": gws, "positive_ratio1": pos_r1s, "positive_ratio2": pos_r2s}
    )
    mean_std_df = pd.DataFrame(
        {
            "mean": gw_df.groupby("gene_id")["Gromov-wasserstein_distance"].mean().to_list(),
            "std": gw_df.groupby("gene_id")["Gromov-wasserstein_distance"].std().to_list(),
        },
        index=gw_df.groupby("gene_id")["Gromov-wasserstein_distance"].mean().index,
    )
    gw_df = pd.concat([gw_df0.set_index("gene_id"), mean_std_df], axis=1)
    gw_df["zscore"] = (gw_df["Gromov-wasserstein_distance"] - gw_df["mean"]) / gw_df["std"]

    gw_df = gw_df.replace(np.inf, 0).replace(np.nan, 0)

    # find p-value
    if larger_or_small == "larger":
        gw_df["pvalue"] = scipy.stats.norm.sf(gw_df["zscore"])
    elif larger_or_small == "small":
        gw_df["pvalue"] = 1 - scipy.stats.norm.sf(gw_df["zscore"])

    gw_df["adj_pvalue"] = statsmodels.stats.multitest.multipletests(gw_df["pvalue"])[1]

    gw_df["fc"] = gw_df["Gromov-wasserstein_distance"] / gw_df["mean"]
    gw_df["log2fc"] = np.log2(gw_df["fc"])
    gw_df["-log10adjp"] = -np.log10(gw_df["adj_pvalue"])

    gw_df = gw_df.replace(np.inf, 0).replace(np.nan, 0)
    return gw_df, adata1, adata2


def cal_gw_dis_on_genes(inp1, inp2):
    C1, C2, adata1, adata2 = inp1
    seed, gene_set = inp2

    # adata1 = shuffle_adata(adata1, seed)
    adata2 = shuffle_adata(adata2, seed)

    gws = []
    pos_r1s = []
    pos_r2s = []
    if issparse(adata1.X):
        df1 = pd.DataFrame(adata1.X.A, columns=adata1.var_names)
    else:
        df1 = pd.DataFrame(adata1.X, columns=adata1.var_names)

    if issparse(adata2.X):
        df2 = pd.DataFrame(adata2.X.A, columns=adata2.var_names)
    else:
        df2 = pd.DataFrame(adata2.X, columns=adata2.var_names)

    for gene_id in gene_set:
        p = np.array(df1.loc[:, gene_id], dtype=np.float64) / np.array(df1.loc[:, gene_id], dtype=np.float64).sum()
        q = np.array(df2.loc[:, gene_id], dtype=np.float64) / np.array(df2.loc[:, gene_id], dtype=np.float64).sum()
        gw = ot.gromov_wasserstein2(C1, C2, p, q)
        gws.append(gw)
        pos_r1s.append(np.sum(p > 0) / len(p))
        pos_r2s.append(np.sum(q > 0) / len(q))
    return gene_set, gws, pos_r1s, pos_r2s
