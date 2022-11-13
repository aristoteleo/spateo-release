"""
Gene Expression Cartography
M Nitzan*, N Karaiskos*, N Friedman†, N Rajewsky†
Nature (2019)

code adapted from: https://github.com/rajewsky-lab/novosparc
"""

from typing import List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
from tqdm import tqdm

from ..configuration import SKM
from ..logging import logger_manager as lm


def find_spatial_archetypes(
    num_clusters: int,
    exp_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clusters the expression data and finds gene archetypes. Current implementation is based on hierarchical
    clustering with the Ward method. The archetypes are simply the average of genes belong to the same cell cluster.

    Args:
        num_clusters: number of gene clusters or archetypes.
        exp_mat: expression matrix. Rows are genes and columns are buckets.

    Returns:
        Returns the archetypes, the gene sets (clusters) and the Pearson correlations of every gene with respect to
        each archetype.
    """
    clusters = hierarchy.fcluster(hierarchy.ward(exp_mat), num_clusters, criterion="maxclust")
    arch_comp = lambda x: np.mean(exp_mat[np.where(clusters == x)[0], :], axis=0)
    archetypes = np.array([arch_comp(xi) for xi in range(1, num_clusters + 1)])
    gene_corrs = np.array([])

    for gene in lm.main_tqdm(range(len(exp_mat)), "Finding gene archetypes"):
        gene_corrs = np.append(gene_corrs, pearsonr(exp_mat[gene, :], archetypes[clusters[gene] - 1, :])[0])

    lm.main_info("done!")

    return archetypes, clusters, gene_corrs


def get_genes_from_spatial_archetype(
    exp_mat: np.ndarray,
    gene_names: Union[np.ndarray, list],
    archetypes: np.ndarray,
    archetype: int,
    pval_threshold: float = 0,
) -> Union[np.ndarray, list]:
    """Get a list of genes which are the best representatives of the archetype.

    Args:
        exp_mat: expression matrix.
        gene_names: the gene names list that associates with the rows of expression matrix
        archetypes: the archetypes output of find_spatial_archetypes
        archetype: a number denoting the archetype
        pval_threshold: the pvalue returned from the pearsonr function

    Returns:
        a list of genes which are the best representatives of the archetype
    """

    # Classify all genes and return the most significant ones
    all_corrs = np.array([])
    all_corrs_p = np.array([])

    for g in range(len(exp_mat)):
        all_corrs = np.append(all_corrs, pearsonr(exp_mat[g, :], archetypes[archetype, :])[0])
        all_corrs_p = np.append(all_corrs_p, pearsonr(exp_mat[g, :], archetypes[archetype, :])[1])

    indices = np.where(all_corrs_p[all_corrs > 0] <= pval_threshold)[0]

    if len(indices) == 0:
        lm.main_warning("No genes with significant correlation were found at the current p-value threshold.")
        return None

    genes = gene_names[all_corrs > 0][indices]

    return genes


def find_spatially_related_genes(
    exp_mat: np.ndarray,
    gene_names: Union[np.ndarray, list],
    archetypes: np.ndarray,
    gene: int,
    pval_threshold: float = 0,
):
    """Given a gene, find other genes which correlate well spatially.

    Args:
        exp_mat: expression matrix.
        gene_names: gene name list that associates with the rows of expression matrix.
        archetypes: the archetypes output of find_spatial_archetypes
        gene: the index of the gene to be queried
        pval_threshold: the pvalue returned from the pearsonr function

    Returns:
        a list of genes which are the best representatives of the archetype
    """
    # First find the archetype of the gene
    arch_corrs = np.array([])

    for archetype in range(len(archetypes)):
        arch_corrs = np.append(arch_corrs, pearsonr(exp_mat[gene, :], archetypes[archetype, :])[0])

    if np.max(arch_corrs) < 0.7:
        lm.main_warning("No significant correlation between the gene and the spatial archetypes was found.")
        return None

    archetype = np.argmax(arch_corrs)

    return get_genes_from_spatial_archetype(exp_mat, gene_names, archetypes, archetype, pval_threshold=pval_threshold)


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def archetypes(
    adata: anndata.AnnData,
    moran_i_genes: Union[np.ndarray, list],
    num_clusters: int = 5,
    layer: Union[str, None] = None,
) -> np.ndarray:

    """Identify archetypes from the anndata object.

    Args:
        adata: Anndata object of interests.
        moran_i_genes: genes that are identified as singificant autocorrelation genes in space based on Moran's I.
        num_clusters: number of archetypes.
        layers: the layer for the gene expression, can be None which corresponds to adata.X.

    Returns:
        archetypes: the archetypes within the genes with high moran I scores.

    Examples:
        >>> archetypes = st.tl.archetypes(adata)
        >>> adata.obs = pd.concat((adata.obs, df), 1)
        >> arch_cols = adata.obs.columns
        >>> st.pl.space(adata, basis="spatial", color=arch_cols, pointsize=0.1, alpha=1)
    """

    if layer is None:
        exp = adata[:, moran_i_genes].X.A
    else:
        exp = adata[:, moran_i_genes].layers[layer].A

    archetypes, clusters, gene_corrs = find_spatial_archetypes(num_clusters, exp.T)
    arch_cols = ["archetype %d" % i for i in np.arange(num_clusters)]

    df = pd.DataFrame(archetypes.T, columns=arch_cols)
    df.index = adata.obs_names

    archetypes = pd.concat((adata.obs, df), 1)

    return archetypes


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def archetypes_genes(
    adata: anndata.AnnData,
    archetypes: np.ndarray,
    num_clusters: int,
    moran_i_genes: Union[np.ndarray, list],
    layer: Union[str, None] = None,
) -> dict:
    """Identify genes that belong to each expression archetype.

    Args:
        adata: Anndata object of interests.
        archetypes: the archetypes output of find_spatial_archetypes
        num_clusters: number of archetypes.
        moran_i_genes: genes that are identified as singificant autocorrelation genes in space based on Moran's I.
        layer: the layer for the gene expression, can be None which corresponds to adata.X.


    Returns:
        archetypes_dict: a dictionary where the key is the index of the archetype and the values are the top genes for
        that particular archetype.

    Examples:
         >>> st.tl.archetypes_genes(adata)
         >>> dyn.pl.scatters(subset_adata,
         >>>     basis="spatial",
         >>>     color=['archetype %d'% i] + typical_genes.to_list(),
         >>>     pointsize=0.03,
         >>>     alpha=1,
         >>>     figsize=(3, ptp_vec[1]/ptp_vec[0] * 3)
         >>> )
    """

    if layer is None:
        exp = adata[:, moran_i_genes].X.A
    else:
        exp = adata[:, moran_i_genes].layers[layer].A

    archetypes_dict = {}

    for i in np.arange(num_clusters):
        # lm.main_info("current archetype is, ", str(i))

        typical_genes = get_genes_from_spatial_archetype(
            exp.T, moran_i_genes, archetypes, archetype=i, pval_threshold=0
        )

        # lm.main_info("typical gene for the current archetype include, ", typical_genes)
        archetypes_dict[i] = typical_genes

    return archetypes_dict
