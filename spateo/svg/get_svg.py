import multiprocessing
import sys
from functools import partial
from typing import List, Optional, Union

import dynamo as dyn
import numpy as np
import ot
import pandas as pd

# sys.path.insert(0, "/home/panhailin/software/source/git_hub/spateo-release_daniel/")
import scipy
import scipy.stats
import statsmodels
from anndata import AnnData
from dynamo.tools.sampling import sample
from loess import loess_1d
from scipy.sparse import issparse
from tqdm import tqdm
from typing_extensions import Literal

import spateo as st

from .utils import *


def svg_iden_reg(
    adata: AnnData,
    bin_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    distance_layer: str = "spatial",
    n_neighbors: int = 8,
    numItermax: int = 1000000,
    gene_set: Union[List, np.ndarray] = None,
    target: Union[List, np.ndarray, str] = [],
    min_dis_cutoff: float = 500,
    max_dis_cutoff: float = 1000,
    n_nei_for_std: int = 30,
):
    """Identify SVGs compared to uniform distribution.

    Args:
        adata: AnnData

    Returns:
        Anndata, df
    """
    w0 = cal_wass_dis_nobs(
        adata,
        bin_size=1,
        bin_layer=bin_layer,
        cell_distance_method=cell_distance_method,
        distance_layer=distance_layer,
        n_neighbors=n_neighbors,
        numItermax=numItermax,
        gene_set=gene_set,
        target=target,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
    )

    w0["pos_ratio_raw"] = adata.var["pos_ratio_raw"][w0.index]
    w0.sort_values(by="pos_ratio_raw", inplace=True)

    xout, yout, wout = loess_1d.loess_1d(x=w0["pos_ratio_raw"], y=w0["Wasserstein_distance"])
    w0["mean_reg"] = yout
    w0["std"] = get_std(w0["Wasserstein_distance"], n_nei=n_nei_for_std)

    std_xout, std_yout, _ = loess_1d.loess_1d(x=w0["pos_ratio_raw"], y=w0["std"])
    w0["std_reg"] = std_yout
    w0["zscore"] = (w0["Wasserstein_distance"] - w0["mean_reg"]) / w0["std_reg"]
    w0["pvalue"] = scipy.stats.norm.sf(w0["zscore"])
    w0["adj_pvalue"] = statsmodels.stats.multitest.multipletests(w0["pvalue"])[1]

    return w0


def get_std(l, n_nei=30):
    std = l.copy()
    left = int(n_nei / 2)
    right = n_nei - left
    for i in range(0, left):
        std[i] = np.std(l[0 : n_nei + 2])

    for i in range(left, len(l) - right + 1):
        std[i] = np.std(l[i - left : i + right + 2])

    for i in range(len(l) - right, len(l)):
        std[i] = np.std(l[len(l) - n_nei - 1 : len(l)])

    return std


def imputataion_and_sampling(
    adata: AnnData,
    positive_ratio_cutoff: float = 0.1,
    imputation: bool = True,
    downsampling: int = 400,
) -> AnnData:
    adata.X = adata.X.astype("int64")
    adata = filter_adata_by_pos_ratio(adata, positive_ratio_cutoff)

    # imputation
    if imputation:
        adata = st.tl.run_denoise_impute(adata)
    adata_im = adata.copy()

    # downsampling
    ind = sample(arr=np.array(range(adata.X.shape[0])), n=downsampling, method="trn", X=adata.obsm["spatial"])
    adata = adata[ind].copy()

    return adata, adata_im


def cal_wass_dis_on_genes(inp0, inp1):
    M, adata = inp0
    seed, gene_ids, b, numItermax = inp1
    adata = shuffle_adata(adata, seed)
    ws = []
    pos_rs = []
    if issparse(adata.X):
        df = pd.DataFrame(adata.X.A, columns=adata.var_names)
    else:
        df = pd.DataFrame(adata.X, columns=adata.var_names)

    for gene_id in gene_ids:
        a = np.array(df.loc[:, gene_id], dtype=np.float64) / np.array(df.loc[:, gene_id], dtype=np.float64).sum()
        w = cal_wass_dis(M, a, b, numItermax=numItermax)
        pos_r = np.sum(a > 0) / len(a)
        ws.append(w)
        pos_rs.append(pos_r)
    return gene_ids, ws, pos_rs


# within slice
def cal_wass_dis_bs(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    distance_layer: str = "spatial",
    n_neighbors: int = 30,
    numItermax: int = 1000000,
    gene_set: Union[List, np.ndarray] = None,
    target: Union[List, np.ndarray, str] = [],
    processes: int = 1,
    bootstrap: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
    rank_p: bool = True,
    bin_num: int = 100,
    larger_or_small: str = "larger",
):
    """Computing Wasserstein distance for a AnnData to identify spatially variable genes.

    Args:
        adata: AnnData object
        bin_size: bin size for mergeing cells.
        bin_layer: data for this layer would be used to bin data
        cell_distance_method: the method for calculating distance of two cells. geodesic or spatial
        distance_layer: the data of this layer would be used to calculate distance
        n_neighbors: the number of neighbors for calculating geodesic distance
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged
        gene_set: Gene set for computing, default is for all genes.
        target: the target distribution or the target gene name.
        processes: process number for parallelly running
        bootstrap: bootstrap number for permutation to calculate p-value
        min_dis_cutoff: Cells/Bins whose min distance to 30 neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30 neighbors are larger than this cutoff would be filtered.
        rank_p: whether to calculate p value in ranking manner.
        bin_num: classy genes into bin_num groups acording to mean Wasserstein distance from bootstrap.
        larger_or_small: in what direction to get p value. Larger means the right tail area of the null distribution.
    Returns:
        w_df: a dataframe
        adata0: binned AnnData object
    """

    # adata0 = bin_adata(adata, bin_size, layer=bin_layer)
    # adata0 = adata0[:,np.sum(adata0.X,axis=0) > 0]
    # adata0 = scale_to(adata0)
    # print(adata0)
    # if cell_distance_method == 'geodesic':
    #     adata0 = cal_geodesic_distance(adata0, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer)
    # elif cell_distance_method == 'spatial':
    #     adata0 = cal_spatial_distance(adata0, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer)
    # M = adata0.obsp['distance']
    # if np.sum(~np.isfinite(M)) > 0:
    #     print("distance has inf value")
    #     sys.exit()

    adata0, M = bin_scale_adata_get_distance(
        adata,
        bin_size=bin_size,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )

    if gene_set is None:
        gene_set = adata0.var_names

    if isinstance(target, (list, np.ndarray)):
        b = target
    if isinstance(target, str):
        b = adata0[:, target].X.A.flatten() if issparse(adata0[:, target].X) else adata0[:, target].X.flatten()
        b = np.array(b, dtype=np.float64)
        b = b / np.sum(b)

    genes, ws, pos_rs = cal_wass_dis_on_genes((M, adata0), (0, gene_set, b, numItermax))
    w_df0 = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})

    pool = multiprocessing.Pool(processes=processes)

    inputs = [(i, gene_set, b, numItermax) for i in range(1, bootstrap + 1)]
    res = []
    for result in tqdm(pool.imap_unordered(partial(cal_wass_dis_on_genes, (M, adata0)), inputs), total=len(inputs)):
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

    # find p-value
    if larger_or_small == "larger":
        w_df["pvalue"] = scipy.stats.norm.sf(w_df["zscore"])
    elif larger_or_small == "small":
        w_df["pvalue"] = 1 - scipy.stats.norm.sf(w_df["zscore"])

    w_df["adj_pvalue"] = statsmodels.stats.multitest.multipletests(w_df["pvalue"])[1]

    w_df["fc"] = w_df["Wasserstein_distance"] / w_df["mean"]
    w_df["log2fc"] = np.log2(w_df["fc"])
    w_df["-log10adjp"] = -np.log10(w_df["adj_pvalue"])

    # rank p
    if rank_p:
        w_df["rank_p"], each_bin_ws = cal_rank_p(genes, ws, w_df, bin_num=bin_num)
        w_df.loc[w_df["positive_ratio"] == 0, "rank_p"] = 1.0
        w_df["adj_rank_p"] = statsmodels.stats.multitest.multipletests(w_df["rank_p"])[1]

    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)
    return w_df, adata0  # , each_bin_ws


# within slice
def cal_wass_dis_nobs(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    distance_layer: str = "spatial",
    n_neighbors: int = 30,
    numItermax: int = 1000000,
    gene_set: Union[List, np.ndarray] = None,
    target: Union[List, np.ndarray, str] = [],
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
):
    """Computing Wasserstein distance for a AnnData to identify spatially variable genes.

    Args:
        adata: AnnData object
        bin_size: bin size for mergeing cells.
        bin_layer: data for this layer would be used to bin data
        cell_distance_method: the method for calculating distance of two cells. geodesic or spatial
        distance_layer: the data of this layer would be used to calculate distance
        n_neighbors: the number of neighbors for calculation geodesic distance
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged
        gene_set: Gene set for computing, default is for all genes.
        target: the target distribution or the target gene name.
        min_dis_cutoff: Cells/Bins whose min distance to 30 neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30 neighbors are larger than this cutoff would be filtered.
    Returns:
        w_df0: a dataframe
    """

    # adata0 = bin_adata(adata, bin_size, layer=bin_layer)
    # adata0 = adata0[:,np.sum(adata0.X,axis=0) > 0]
    # adata0 = scale_to(adata0)
    # #print(adata0)
    # if cell_distance_method == 'geodesic':
    #     adata0 = cal_geodesic_distance(adata0, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer)
    # elif cell_distance_method == 'spatial':
    #     adata0 = cal_spatial_distance(adata0, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer)
    # M = adata0.obsp['distance']
    # if np.sum(~np.isfinite(M)) > 0:
    #     print("distance has inf value")
    #     sys.exit()

    adata0, M = bin_scale_adata_get_distance(
        adata,
        bin_size=bin_size,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )

    if gene_set is None:
        gene_set = adata0.var_names

    if isinstance(target, (list, np.ndarray)):
        b = target
    if isinstance(target, str):
        b = adata0[:, target].X.A.flatten() if issparse(adata0[:, target].X) else adata0[:, target].X.flatten()
        b = np.array(b, dtype=np.float64)
        b = b / np.sum(b)

    genes, ws, pos_rs = cal_wass_dis_on_genes((M, adata0), (0, gene_set, b, numItermax))
    w_df0 = pd.DataFrame({"Wasserstein_distance": ws, "positive_ratio": pos_rs}, index=genes)
    return w_df0


def bin_scale_adata_get_distance(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    distance_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
    n_neighbors: int = 30,
):
    adata0 = bin_adata(adata, bin_size, layer=bin_layer)
    adata0 = adata0[:, np.sum(adata0.X, axis=0) > 0]
    adata0 = scale_to(adata0)
    if cell_distance_method == "geodesic":
        adata0 = cal_geodesic_distance(
            adata0,
            min_dis_cutoff=min_dis_cutoff,
            max_dis_cutoff=max_dis_cutoff,
            layer=distance_layer,
            n_neighbors=n_neighbors,
        )
    elif cell_distance_method == "spatial":
        adata0 = cal_spatial_distance(
            adata0, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer
        )
    M = adata0.obsp["distance"]
    if np.sum(~np.isfinite(M)) > 0:
        print("distance has inf value")
        sys.exit()
    return adata0, M


def cal_wass_dis_target_on_genes(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    distance_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    n_neighbors: int = 30,
    numItermax: int = 1000000,
    target_genes: Union[List, np.ndarray] = None,
    gene_set: Union[List, np.ndarray] = None,
    processes: int = 1,
    bootstrap: int = 0,
    top: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
) -> pd.DataFrame:
    """Find genes in gene_set that have similar distribution to each target_genes."""
    adata0, M = bin_scale_adata_get_distance(
        adata,
        bin_size=bin_size,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )

    # print(adata0.shape)
    # print(M.shape)

    if gene_set is None:
        gene_set = adata0.var_names

    if issparse(adata0.X):
        df = pd.DataFrame(adata0.X.A, columns=adata0.var_names)
    else:
        df = pd.DataFrame(adata0.X, columns=adata0.var_names)

    w_genes = {}
    for gene in target_genes:
        b = np.array(df.loc[:, gene], dtype=(np.float64)) / np.array(df.loc[:, gene], dtype=(np.float64)).sum()
        # print(b.shape)
        genes, ws, pos_rs = cal_wass_dis_on_genes((M, adata0), (0, gene_set, b, numItermax))
        w_genes[gene] = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})

    if bootstrap == 0:
        return w_genes, adata0

    for gene in target_genes:
        tmp = w_genes[gene]
        gene_set = tmp[tmp["positive_ratio"] > 0].sort_values(by="Wasserstein_distance")["gene_id"].head(top + 1)
        w_df, _ = cal_wass_dis_bs(
            adata,
            gene_set=gene_set,
            target=gene,
            bin_size=bin_size,
            bin_layer=bin_layer,
            distance_layer=distance_layer,
            min_dis_cutoff=min_dis_cutoff,
            max_dis_cutoff=max_dis_cutoff,
            cell_distance_method=cell_distance_method,
            bootstrap=bootstrap,
            processes=processes,
            larger_or_small="small",
            rank_p=False,
        )

        w_genes[gene] = w_df
    return w_genes, adata0
