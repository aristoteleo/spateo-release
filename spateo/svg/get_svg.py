import multiprocessing
import sys
from functools import partial
from typing import List, Optional, Tuple, Union

import dynamo as dyn
import numpy as np
import ot
import pandas as pd
import scipy
import scipy.stats
from anndata import AnnData
from dynamo.tools.sampling import sample
from loess.loess_1d import loess_1d
from scipy.sparse import issparse
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..logging import logger_manager as lm
from ..tools.spatial_smooth.run_smoothing import smooth_and_downsample
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
    n_neighbors_for_std: int = 30,
) -> pd.DataFrame:
    """Identifying SVGs using a spatial uniform distribution as the reference.

    Args:
        adata: AnnData object
        bin_layer: Data in this layer will be binned according to the spatial information.
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        distance_layer: Data in this layer will be used to calculate the spatial distance.
        n_neighbors: The number of nearest neighbors that will be considered for calculating spatial distance.
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged.
        gene_set: Gene set that will be used to identified spatial variable genes, default is for all genes.
        target: The target gene expression distribution or the target gene name.
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.
        n_neighbors_for_std: Number of neighbors that will be used to calculate the standard deviation of the
            Wasserstein distances.

    Returns:
        w0: a pandas data frame that stores the information of spatial variable genes results. It includes the following
        columns:
             "raw_pos_rate": The raw positive ratio (the fraction of cells that have non-zero expression ) of the gene
                across all cells.
             "Wasserstein_distance": The computed Wasserstein distance of each gene to the reference uniform
                distribution.
             "expectation_reg": The predicted Wasserstein distance after fitting a loess regression using the gene
                positive rate as the predictor.
             "std": Standard deviation of the Wasserstein distance.
             "std_reg": The predicted standard deviation of the Wasserstein distance after fitting a loess regression
                using the gene positive rate as the predictor.
             "zscore": The z-score of the Wasserstein distance.
             "pvalue": The p-value based on the z-score.
             "adj_pvalue": Adjusted p-value.

        In addition, the input adata object has updated with the following information:
            adata.var["raw_pos_rate"]: The positive rate of each gene.

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

    w0["raw_pos_rate"] = adata.var["raw_pos_rate"][w0.index]
    w0.sort_values(by="raw_pos_rate", inplace=True)

    xout, yout, _ = loess_1d(x=w0["raw_pos_rate"], y=w0["Wasserstein_distance"])
    w0["expectation_reg"] = yout
    w0["std"] = get_std_wasserstein(w0["Wasserstein_distance"], n_neighbors=n_neighbors_for_std)

    std_xout, std_yout, _ = loess_1d(x=w0["raw_pos_rate"], y=w0["std"])
    w0["std_reg"] = std_yout
    w0["zscore"] = (w0["Wasserstein_distance"] - w0["expectation_reg"]) / w0["std_reg"]
    w0["pvalue"] = norm.sf(w0["zscore"].abs())
    w0["adj_pvalue"] = multipletests(w0["pvalue"])[1]

    return w0


def get_std_wasserstein(l: Union[np.ndarray, pd.DataFrame], n_neighbors: int = 30) -> np.ndarray:
    """Calculate the standard deviation of the Wasserstein distance.

    Args:
        l: The vector of the Wasserstein distance.
        n_neighbors: number of nearest neighbors.

    Returns:
        std: The standard deviation of the Wasserstein distance.
    """

    std = l.copy()
    left = int(n_neighbors / 2)
    right = n_neighbors - left
    for i in range(0, left):
        std[i] = np.std(l[0 : n_neighbors + 2])

    for i in range(left, len(l) - right + 1):
        std[i] = np.std(l[i - left : i + right + 2])

    for i in range(len(l) - right, len(l)):
        std[i] = np.std(l[len(l) - n_neighbors - 1 : len(l)])

    return std


def smoothing_and_sampling(
    adata: AnnData,
    smoothing: bool = True,
    downsampling: int = 400,
    device: str = "cpu",
) -> Tuple[AnnData, AnnData]:
    """Smoothing the gene expression using a graph neural network and downsampling the cells from the adata object.

    Args:
        adata: The input AnnData object.
        smoothing: Whether to do smooth the gene expression.
        downsampling: The number of cells to down sample.
        device: The device to run the deep learning smoothing model. Can be either "cpu" or proper "cuda" related
            devices, such as: "cuda:0".

    Returns:
        adata: The adata after smoothing and downsampling.
        adata_smoothed: The adata after smoothing but not downsampling.
    """
    # smoothing
    adata = adata.copy()
    if smoothing:
        adata.X = adata.X.astype("int64")
        adata, _ = smooth_and_downsample(adata, device=device, positive_ratio_cutoff=0.0, n_ds=downsampling)
    adata_smoothed = adata.copy()

    # downsampling
    ind = sample(arr=np.array(range(adata.X.shape[0])), n=downsampling, method="trn", X=adata.obsm["spatial"])
    adata = adata[ind].copy()

    return adata, adata_smoothed


def smoothing(
    adata: AnnData,
    device: str = "cpu",
) -> AnnData:
    """Smoothing the gene expression using a graph neural network.

    Args:
        adata: The input AnnData object.
        device: The device to run the deep learning smoothing model. Can be either "cpu" or proper "cuda" related
            devices, such as: "cuda:0".

    Returns:
        adata_smoothed: imputation result
    """
    adata = adata.copy()
    adata.X = adata.X.astype("int64")
    adata_smoothed, _ = smooth_and_downsample(adata, device=device, positive_ratio_cutoff=0.0, n_ds=400)
    return adata_smoothed


def downsampling(
    adata: AnnData,
    downsampling: int = 400,
) -> AnnData:
    """Downsampling the cells from the adata object.

    Args:
        adata: The input AnnData object.
        downsampling: The number of cells to down sample.

    Returns:
        adata: adata after the downsampling.
    """
    adata = adata.copy()
    ind = sample(arr=np.array(range(adata.X.shape[0])), n=downsampling, method="trn", X=adata.obsm["spatial"])
    adata = adata[ind].copy()
    return adata


def cal_wass_dis_for_genes(
    inp0: Tuple[scipy.sparse._csr.csr_matrix, AnnData], inp1: Tuple[int, List, np.ndarray, int]
) -> Tuple[List, np.ndarray, np.ndarray]:
    """Calculate Wasserstein distances for a list of genes.

    Args:
        inp0: A tuple of the sparse matrix of spatial distance between nearest neighbors, and the adata object.
        inp1: A tuple of the seed, the list of genes, the target gene expression vector (need to be normalized to have a
                sum of 1), and the maximal number of iterations.

    Returns:
        gene_ids: The gene list that is used to calculate the Wasserstein distribution.
        ws: The Wasserstein distances from each gene to the target gene.
        pos_rs: The expression positive rate vector related to the gene list.
    """

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
def cal_wass_dist_bs(
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
) -> Tuple[pd.DataFrame, AnnData]:
    """Computing Wasserstein distance for an AnnData to identify spatially variable genes.

    Args:
        adata: AnnData object.
        bin_size: Bin size for mergeing cells.
        bin_layer: Data in this layer will be binned according to the spatial information.
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        distance_layer: The data of this layer would be used to calculate distance
        n_neighbors: The number of neighbors for calculating spatial distance.
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged.
        gene_set: Gene set that will be used to compute Wasserstein distances, default is for all genes.
        target: The target gene expression distribution or the target gene name.
        processes: The process number for parallel computing
        bootstrap: Bootstrap number for permutation to calculate p-value
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.
        rank_p: Whether to calculate p value in ranking manner.
        bin_num: Classy genes into bin_num groups according to mean Wasserstein distance from bootstrap.
        larger_or_small: In what direction to get p value. Larger means the right tail area of the null distribution.

    Returns:
        w_df: A dataframe storing information related to the Wasserstein distances.
        bin_scale_adata: Binned AnnData object
    """

    bin_scale_adata, M = bin_scale_adata_get_distance(
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
        gene_set = bin_scale_adata.var_names

    if isinstance(target, (list, np.ndarray)):
        b = target
    if isinstance(target, str):
        b = (
            bin_scale_adata[:, target].X.A.flatten()
            if issparse(bin_scale_adata[:, target].X)
            else bin_scale_adata[:, target].X.flatten()
        )
        b = np.array(b, dtype=np.float64)
        b = b / np.sum(b)

    genes, ws, pos_rs = cal_wass_dis_for_genes((M, bin_scale_adata), (0, gene_set, b, numItermax))
    w_df_ori = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})

    pool = multiprocessing.Pool(processes=processes)

    inputs = [(i, gene_set, b, numItermax) for i in range(1, bootstrap + 1)]
    res = []
    for result in tqdm(
        pool.imap_unordered(partial(cal_wass_dis_for_genes, (M, bin_scale_adata)), inputs), total=len(inputs)
    ):
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
    w_df = pd.concat([w_df_ori.set_index("gene_id"), mean_std_df], axis=1)
    w_df["zscore"] = (w_df["Wasserstein_distance"] - w_df["mean"]) / w_df["std"]

    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)

    # find p-value
    if larger_or_small == "larger":
        w_df["pvalue"] = norm.sf(w_df["zscore"].abs())
    elif larger_or_small == "small":
        w_df["pvalue"] = 1 - norm.sf(w_df["zscore"].abs())

    w_df["adj_pvalue"] = multipletests(w_df["pvalue"])[1]

    w_df["fc"] = w_df["Wasserstein_distance"] / w_df["mean"]
    w_df["log2fc"] = np.log2(w_df["fc"])
    w_df["-log10adjp"] = -np.log10(w_df["adj_pvalue"])

    # rank p
    if rank_p:
        w_df["rank_p"], each_bin_ws = cal_rank_p(genes, ws, w_df, bin_num=bin_num)
        w_df.loc[w_df["positive_ratio"] == 0, "rank_p"] = 1.0
        w_df["adj_rank_p"] = multipletests(w_df["rank_p"])[1]

    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)
    return w_df, bin_scale_adata  # , each_bin_ws


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
) -> Tuple[pd.DataFrame, AnnData]:
    """Computing Wasserstein distance for a AnnData to identify spatially variable genes.

    Args:
        adata: AnnData object
        bin_size: bin size for mergeing cells.
        bin_layer: data in this layer will be binned according to spatial information.
        cell_distance_method: the method for calculating distance of two cells. geodesic or euclidean
        distance_layer: the data of this layer would be used to calculate distance
        n_neighbors: the number of neighbors for calculation geodesic distance
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged
        gene_set: Gene set for computing, default is for all genes.
        target: the target distribution or the target gene name.
        min_dis_cutoff: Cells/Bins whose min distance to 30 neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30 neighbors are larger than this cutoff would be filtered.

    Returns:
        w_df: A dataframe storing information related to the Wasserstein distances.
    """
    bin_scale_adata, M = bin_scale_adata_get_distance(
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
        gene_set = bin_scale_adata.var_names

    if isinstance(target, (list, np.ndarray)):
        b = target
    if isinstance(target, str):
        b = (
            bin_scale_adata[:, target].X.A.flatten()
            if issparse(bin_scale_adata[:, target].X)
            else bin_scale_adata[:, target].X.flatten()
        )
        b = np.array(b, dtype=np.float64)
        b = b / np.sum(b)

    genes, ws, pos_rs = cal_wass_dis_for_genes((M, bin_scale_adata), (0, gene_set, b, numItermax))
    w_df = pd.DataFrame({"Wasserstein_distance": ws, "positive_ratio": pos_rs}, index=genes)
    return w_df


def bin_scale_adata_get_distance(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    distance_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
    n_neighbors: int = 30,
) -> Tuple[AnnData, scipy.sparse._csr.csr_matrix]:
    """Bin (based on spatial information), scale adata object and calculate the distance matrix based on the specified
    method (either geodesic or euclidean).

    Args:
        adata: AnnData object.
        bin_size: Bin size for mergeing cells.
        bin_layer: Data in this layer will be binned according to the spatial information.
        distance_layer: The data of this layer would be used to calculate distance
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.
        n_neighbors: The number of nearest neighbors that will be considered for calculating spatial distance.

    Returns:
        bin_scale_adata: Bin, scaled anndata object.
        M: The scipy sparse matrix of the calculated distance of nearest neighbors.
    """
    bin_scale_adata = bin_adata(adata, bin_size, layer=bin_layer)
    bin_scale_adata = bin_scale_adata[:, np.sum(bin_scale_adata.X, axis=0) > 0]
    bin_scale_adata = scale_to(bin_scale_adata)
    if cell_distance_method == "geodesic":
        bin_scale_adata = cal_geodesic_distance(
            bin_scale_adata,
            min_dis_cutoff=min_dis_cutoff,
            max_dis_cutoff=max_dis_cutoff,
            layer=distance_layer,
            n_neighbors=n_neighbors,
        )
    elif cell_distance_method == "euclidean":
        bin_scale_adata = cal_euclidean_distance(
            bin_scale_adata, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer
        )

    M = bin_scale_adata.obsp["distance"]
    if np.sum(~np.isfinite(M)) > 0:
        lm.main_exception("distance has inf values.")
        sys.exit()
    return bin_scale_adata, M


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
    top_n: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
) -> Tuple[dict, AnnData]:
    """Find genes in gene_set that have similar distribution to each target_genes.

    Args:
        adata: AnnData object.
        bin_size: Bin size for mergeing cells.
        bin_layer: Data in this layer will be binned according to the spatial information.
        distance_layer: The data of this layer would be used to calculate distance
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        n_neighbors: The number of neighbors for calculating spatial distance.
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged.
        target_genes: The list of the target genes.
        gene_set: Gene set that will be used to compute Wasserstein distances, default is for all genes.
        processes: The process number for parallel computing.
        bootstrap: Number of bootstraps.
        top_n: Number of top genes to select.
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.

    Returns:
        w_genes: The dictionary of the Wasserstein distance. Each key corresponds to a gene name while the corresponding
            value the pandas DataFrame of the Wasserstein distance related information.
        bin_scale_adata: binned, scaled anndata object.
    """
    bin_scale_adata, M = bin_scale_adata_get_distance(
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
        gene_set = bin_scale_adata.var_names

    if issparse(bin_scale_adata.X):
        df = pd.DataFrame(bin_scale_adata.X.A, columns=bin_scale_adata.var_names)
    else:
        df = pd.DataFrame(bin_scale_adata.X, columns=bin_scale_adata.var_names)

    w_genes = {}
    for gene in target_genes:
        b = np.array(df.loc[:, gene], dtype=np.float64) / np.array(df.loc[:, gene], dtype=np.float64).sum(0)
        genes, ws, pos_rs = cal_wass_dis_for_genes((M, bin_scale_adata), (0, gene_set, b, numItermax))
        w_genes[gene] = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})

    if bootstrap == 0:
        return w_genes, bin_scale_adata

    for gene in target_genes:
        tmp = w_genes[gene]
        gene_set = tmp[tmp["positive_ratio"] > 0].sort_values(by="Wasserstein_distance")["gene_id"].head(top_n + 1)
        w_df, _ = cal_wass_dist_bs(
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
    return w_genes, bin_scale_adata
