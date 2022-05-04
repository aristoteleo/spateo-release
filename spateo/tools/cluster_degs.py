from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from scipy import stats
from scipy.sparse import issparse
from scipy.spatial import distance
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from statsmodels.sandbox.stats.multicomp import multipletests
from tqdm import tqdm
from typing_extensions import Literal

from ..configuration import SKM

@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, optional=True)
def find_spatial_cluster_degs(
    adata: AnnData,
    test_group: str,
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
    group: Optional[str] = None,
    genes: Optional[List[str]] = None,
    k: int = 10,
    ratio_thresh: float = 0.5,
) -> pd.DataFrame:
    """Function to search nearest neighbor groups in spatial space
    for the given test group.

    Args:
        test_group: The group name from `group` for which neighbors has to be found.
        adata: an Annodata object.
        x: x-coordinates of all buckets.
        y: y-coordinates of all buckets.
        group: The column key/name that identifies the grouping information
            (for example, clusters that correspond to different cell types)
            of buckets.
        k: Number of neighbors to use by default for kneighbors queries.
        ratio_thresh: For each non-test group, if more than 50% (default) of its buckets
            are in the neighboring set, this group is then selected as a neigh
            -boring group.

    Returns:
        A pandas DataFrame of the differential expression analysis result
        between the test group and neighbor groups.

    """
    # get x,y
    if x is not None:
        x = x
    else:
        x = adata.obsm["spatial"][:, 0].tolist()
    if y is not None:
        y = y
    else:
        y = adata.obsm["spatial"][:, 1].tolist()
    group_list = adata.obs[group].tolist()

    df = pd.DataFrame({"x": x, "y": y, "group": group_list})
    test_df = df[df["group"] == test_group]

    # KNN
    xymap = pd.DataFrame({"x": x, "y": y})
    xynbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean").fit(xymap)
    _, xyindices = xynbrs.kneighbors(xymap)
    nbr_id = xyindices[test_df.index]
    # neighbor count
    nbr_id_unique = np.unique(nbr_id)
    group_id = []
    for x in np.nditer(nbr_id_unique):
        group_id.append(df.loc[x, "group"])
    nbr_group = Counter(group_id)
    nbr_group
    # ratio
    groups = sorted(adata.obs[group].drop_duplicates())
    group_num = dict()
    ratio = dict()
    for i in groups:
        group_num[i] = df["group"].value_counts()[i]
        ratio[i] = nbr_group[i] / group_num[i]
    nbr_groups = [i for i, e in ratio.items() if e > ratio_thresh]
    nbr_groups.remove(test_group)
    res = find_cluster_degs(
        adata,
        group=group,
        genes=genes,
        test_group=test_group,
        control_groups=nbr_groups,
    )
    return res


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, optional=True)
def find_cluster_degs(
    adata: AnnData,
    test_group: str,
    control_groups: List[str],
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    X_data: Optional[np.ndarray] = None,
    group: Optional[str] = None,
    qval_thresh: float = 0.05,
    ratio_expr_thresh: float = 0.1,
    diff_ratio_expr_thresh: float = 0,
    log2fc_thresh: float = 0,
    method: Literal["multiple", "pairwise"] = "multiple",
) -> pd.DataFrame:
    """Find marker genes between one group to other groups based on gene
    expression.Test each gene for differential expression between buckets in
    one group and the other groups via Mann-Whitney U test. we calcute the
    percentage of buckets expressing the gene in the test group(ratio_expr),
    the difference between the percentages of buckets expressing the gene in
    the test group and control groups(diff_ratio_expr),the expression fold
    change between the test and control groups(log2fc), qval is calculated using
    Benjamini-Hochberg,in addition,the 1 - Jessen-Shannon distance between the
    distribution of percentage of cells with expression across all groups to the
    hypothetical perfect distribution in which only the test group of cells has
    expression(jsd_adj_score),and Pearson's correlation coefficient between gene
    vector which actually detected expression in all cells and an ideal marker
    gene which is only expressed in test_group cells(ppc_score),as well as consin_score.


    Args:
        adata: an Annodata object
        test_group: The group name from `group` for which markers has to be found.
        control_groups: The list of group name(s) from `group` for which markers has to be
            tested against.
        genes: The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        group: The column key/name that identifies the grouping information (for
            example, clusters that correspond to different cell types) of buckets.
            This will be used for calculating group-specific genes.
        X_data: The user supplied data that will be used for marker gene detection
            directly.
        qval_thresh: The maximal threshold of qval to be considered as significant
            genes.
        expr_thresh: The minimum percentage of buckets expressing the gene in the test
            group.
        diff_ratio_expr_thresh: The minimum of the difference between two groups.
        log2fc: The minimum expression log2 fold change.
        method: This method is to choose the difference expression genes between
            test group and other groups one by one or combine them together
            (default: 'multiple'). Valid values are "multiple" and "pairwise".

    Returns:
        A pandas DataFrame of the differential expression analysis result between
        the two groups.

    Raises:
        ValueError: If the `method` is not one of "pairwise" or "multiple".
    """
    if X_data is not None:
        X_data = X_data
    else:
        X_data = adata.X
    if genes is not None:
        genes = genes
    else:
        genes = adata.var_names
    sparse = issparse(X_data)
    if type(control_groups) == str:
        control_groups = [control_groups]
    num_groups = len(control_groups)
    test_cells = adata.obs[group] == test_group
    control_cells = adata.obs[group].isin(control_groups)
    num_test_cells = test_cells.sum()
    num_control_cells = control_cells.sum()
    num_cells = X_data.shape[0]
    de = []
    for i_gene, gene in tqdm(enumerate(genes), desc="identifying top markers for each group"):
        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        test_vals = all_vals[test_cells]
        control_vals = all_vals[control_cells]
        test_mean = test_vals.mean() + 1e-9
        # ratio_expr
        ratio_expr = len(test_vals.nonzero()[0]) / num_test_cells
        if ratio_expr < ratio_expr_thresh:
            continue
        # jsd_adj_score
        perc = [len(test_vals.nonzero()[0]) / num_cells]
        perc.extend([len(all_vals[adata.obs[group] == x].nonzero()[0]) / num_cells for x in control_groups])
        perc_spec = np.repeat(0.0, num_groups + 1)
        perc_spec[0] = 1.0
        M = (perc + perc_spec) / 2
        js_divergence = 0.5 * stats.entropy(perc, M) + 0.5 * stats.entropy(perc_spec, M)
        jsd_adj_score = 1 - js_divergence
        # pearson_test_score
        test_group_spec = np.repeat(0, num_cells)
        test_group_spec[test_cells] = 1
        person_test_score = 1 - distance.correlation(all_vals, test_group_spec)
        # consin_test_score
        cosine_test_score = 1 - distance.cosine(all_vals, test_group_spec)

        if method == "multiple":
            # log2fc
            control_mean = control_vals.mean() + 1e-9
            log2fc = np.log2(test_mean / control_mean + 10e-5)
            # pvals
            if len(control_vals.nonzero()[0]) > 0:
                pvals = mannwhitneyu(test_vals, control_vals)[1][0]
            else:
                pvals = 1
            # diff_ratio_expr
            diff_ratio_expr = ratio_expr - len(control_vals.nonzero()[0]) / num_control_cells
            # person_score
            control_group_spec = np.repeat(0, num_cells)
            control_group_spec[control_cells] = 1
            person_control_score = 1 - distance.correlation(all_vals, control_group_spec)
            person_score = np.power(person_test_score, 3) / (
                np.power(person_control_score, 2) + np.power(person_test_score, 2)
            )
            # cosine_score
            cosine_control_score = 1 - distance.cosine(all_vals, control_group_spec)
            cosine_score = np.power(cosine_test_score, 3) / (
                np.power(cosine_control_score, 2) + np.power(cosine_test_score, 2)
            )

            de.append(
                (
                    gene,
                    control_groups,
                    log2fc,
                    pvals,
                    ratio_expr,
                    diff_ratio_expr,
                    person_score,
                    cosine_score,
                    jsd_adj_score,
                )
            )
        elif method == "pairwise":
            for i in range(num_groups):
                control_cells = adata.obs[group] == control_groups[i]
                control_vals = all_vals[control_cells]
                # log2fc
                control_mean = np.mean(control_vals, axis=0) + 1e-9
                log2fc = np.log2(test_mean / control_mean + 10e-5)[0]
                # pvals
                if len(control_vals.nonzero()[0]) > 0:
                    pvals = mannwhitneyu(test_vals, control_vals)[1][0]
                else:
                    pvals = 1
                # diff_ratio_expr
                diff_ratio_expr = ratio_expr - len(control_vals.nonzero()[0]) / len(control_vals)
                # person_score
                control_group_spec = np.repeat(0, num_cells)
                control_group_spec[control_cells] = 1
                person_control_score = 1 - distance.correlation(all_vals, control_group_spec)
                person_score = np.power(person_test_score, 3) / (
                    np.power(person_control_score, 2) + np.power(person_test_score, 2)
                )
                # cosine_score
                cosine_control_score = 1 - distance.cosine(all_vals, control_group_spec)
                cosine_score = np.power(cosine_test_score, 3) / (
                    np.power(cosine_control_score, 2) + np.power(cosine_test_score, 2)
                )

                de.append(
                    (
                        gene,
                        control_groups[i],
                        log2fc,
                        pvals,
                        ratio_expr,
                        diff_ratio_expr,
                        person_score,
                        cosine_score,
                        jsd_adj_score,
                    )
                )
        else:
            raise ValueError(f'`method` must be one of "multiple" or "pairwise"')
    de = pd.DataFrame(
        de,
        columns=[
            "gene",
            "control_group",
            "log2fc",
            "pval",
            "ratio_expr",
            "diff_ratio_expr",
            "person_score",
            "cosine_score",
            "jsd_adj_score",
        ],
    )
    if de.shape[0] > 1:
        de["qval"] = multipletests(de["pval"].values, method="fdr_bh")[1]
    else:
        de["qval"] = [np.nan for _ in range(de.shape[0])]
    de["test_group"] = [test_group for _ in range(de.shape[0])]
    out_order = [
        "gene",
        "test_group",
        "control_group",
        "ratio_expr",
        "diff_ratio_expr",
        "person_score",
        "cosine_score",
        "jsd_adj_score",
        "log2fc",
        "pval",
        "qval",
    ]
    de = de[out_order].sort_values(by="qval")
    de = de[
        (de.qval < qval_thresh) & (de.diff_ratio_expr > diff_ratio_expr_thresh) & (de.log2fc > log2fc_thresh)
    ].reset_index(drop=True)
    return de


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, optional=True)
def find_all_cluster_degs(
    adata: AnnData,
    group: str,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    X_data: Optional[np.ndarray] = None,
    copy: bool = True,
    n_jobs: int = -1,
) -> AnnData:
    """Find marker genes for each group of buckets based on gene expression.

    Args:
        adata: an Annodata object
        genes: The list of genes that will be used to subset the data for dimen-
            sion reduction and clustering. If `None`, all genes will be used.
        layer: The layer that will be used to retrieve data for dimension reduc-
            tion and clustering. If `None`, .X is used.
        group: The column key/name that identifies the grouping information (for
            example, clusters that correspond to different cell types) of
            buckets.This will be used for calculating group-specific genes.
        test_group: The group name from `group` for which markers has to be found.
        control_groups: The list of group name(s) from `group` for which markers
            has to be tested against.
        X_data: The user supplied data that will be used for marker gene detection
            directly.
        copy: If True (default) a new copy of the adata object will be returned,
            otherwise if False, the adata will be updated inplace.
        n_cores: `int` (default=-1)
            The maximum number of concurrently running jobs, If -1 all CPUs are used.
            If 1 is given, no parallel computing code is used at all.

    Returns:
        An `~anndata.AnnData` with a new property `cluster_markers` in
        the .uns attribute, which includes a concated pandas DataFrame
        of the differential expression analysis result for all groups and a
        dictionary where keys are cluster numbers and values are lists of
        marker genes for the corresponding clusters.
    """
    X_data = adata.X
    if genes is not None:
        genes = genes
    else:
        genes = adata.var_names
    if group not in adata.obs.keys():
        raise ValueError(f"group {group} is not a valid key for .obs in your adata object.")
    else:
        adata.obs[group] = adata.obs[group].astype("str")
        cluster_set = np.sort(adata.obs[group].unique())
    if len(cluster_set) < 2:
        raise ValueError(f"the number of groups for the argument {group} must be at least two.")
    de_tables = [None] * len(cluster_set)
    de_genes = {}
    if len(cluster_set) > 2:

        def single_group(i, test_group, cluster_set, adata, genes, X_data, group, de_tables, de_genes):
            control_groups = sorted(set(cluster_set).difference([test_group]))
            de = find_cluster_degs(
                adata,
                test_group,
                control_groups,
                genes=genes,
                X_data=X_data,
                group=group,
            )
            de_tables[i] = de.copy()
            de_genes[i] = [k for k, v in Counter(de["gene"]).items() if v >= 1]
            return de_tables, de_genes

        de_tables, de_genes = zip(
            *Parallel(n_jobs)(
                delayed(single_group)(i, test_group, cluster_set, adata, genes, X_data, group, de_tables, de_genes)
                for i, test_group in enumerate(cluster_set)
            )
        )
    else:
        de = find_cluster_degs(
            adata,
            cluster_set[0],
            cluster_set[1],
            genes=genes,
            X_data=X_data,
            group=group,
        )
        de_tables[0] = de.copy()
        de_genes[0] = [k for k, v in Counter(de["gene"]).items() if v >= 1]
    if copy:
        adata_1 = adata.copy()
        adata_1.uns["cluster_markers"] = {"deg_tables": de_tables, "de_genes": de_genes}
        return adata_1
    else:
        adata.uns["cluster_markers"] = {"deg_tables": de_tables, "de_genes": de_genes}
        return adata
