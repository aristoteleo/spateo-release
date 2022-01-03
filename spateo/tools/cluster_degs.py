import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.neighbors import NearestNeighbors
from collections import Counter

def find_spatial_cluster_degs(
    test_group,
    adata,
    x,
    y,
    group,
    genes=None,
    k=20,
    ratio_thresh=0.5,
):
    '''Function to search nearest neighbor groups in spatial space
    for the given test group.
    Parameters
    ----------
        test_group: `str`  
            The group name from `group` for which neighbors has to be found.
        adata: class:`~anndata.AnnData`
            an Annodata object. 
        x: 'list' or None(default: `None`)
            x-coordinates of all buckets.
        y: 'list' or None(default: `None`)
            y-coordinates of all buckets.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information 
            (for example, clusters that correspond to different cell types) 
            of buckets.
        k: 'int' (defult=20)
            Number of neighbors to use by default for kneighbors queries.
        ratio_thresh:'float'(defult=0.5)
            For each non-test group, if more than 50% (default) of its buckets
            are in the neighboring set, this group is then selected as a neigh
            -boring group.
    Returns
    -------
        A pandas DataFrame of the differential expression analysis result 
        between the test group and neighbor groups.

    '''
    # get x,y
    if x is not None:
        x = x
    else:
        x = adata.obs['x_array'].tolist()
    if y is not None:
        y = y
    else:
        y = adata.obs['y_array'].tolist()
    group = adata.obs[group].tolist()

    df = pd.DataFrame({'x': x, 'y': y, 'group': group})
    test_df = df[df["group"] == test_group]

    # KNN    
    xymap = pd.DataFrame({'x': x, "y": y})
    xynbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(xymap)
    _, xyindices = xynbrs.kneighbors(xymap)
    nbr_id = xyindices[test_df.index]
    # neighbor count
    nbr_id_unique = np.unique(nbr_id)
    group_id = []
    for x in np.nditer(nbr_id_unique):
        group_id.append(df.loc[x, 'group'])
    nbr_group = Counter(group_id)
    nbr_group
    # ratio
    groups = sorted(adata.obs['group'].drop_duplicates())
    group_num = dict()
    ratio = dict()
    for i in groups:
        group_num[i] = df['group'].value_counts()[i]
        ratio[i] = nbr_group[i]/group_num[i]
    nbr_groups = [i for i, e in enumerate(ratio.values()) if e > ratio_thresh]
    nbr_groups.remove(test_group)
    res = find_cluster_degs(adata, 
                        group='group', 
                        genes=genes,
                        test_group=test_group, 
                        control_groups=nbr_groups)
    return res

def find_cluster_degs(
    adata,
    test_group,  #given test_group,difference in find_all_cluster_degs 
    control_groups,
    genes=None,
    layer=None,
    X_data=None,
    group=None,
    qval_thresh=0.05,
    ratio_expr_thresh=0.1,
    diff_ratio_expr_thresh=0,
    log2fc_thresh=0,
    method='all',
):
    """Find marker genes between one group to other groups based on gene 
    expression.Test each gene for differential expression between buckets in 
    one group and the other groups via Mann-Whitney U test. we calcute the 
    percentage of buckets expressing the gene in the test group(ratio_expr), 
    the difference between the percentages of buckets expressing the gene in 
    the test group and control groups,the expression fold change between the 
    test and control groups(log2fc),in addition, qval is calculated using 
    Benjamini-Hochberg.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        test_group: `str` or None (default: `None`)
            The group name from `group` for which markers has to be found.
        control_groups: `list`
            The list of group name(s) from `group` for which markers has to be 
            tested against.
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduction
            and clustering. If `None`, .X is used.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for 
            example, clusters that correspond to different cell types) of buckets. 
            This will be used for calculating group-specific genes.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for marker gene detection 
            directly.
        qval_thresh: `float` (default: 0.05)
            The maximal threshold of qval to be considered as significant 
            genes.
        expr_thresh: `float` (default: 0.1)
            The minimum percentage of buckets expressing the gene in the test 
            group.
        diff_ratio_expr_thresh: `float` (default: 0)
            The minimum of the difference between two groups.
        log2fc: `float` (default: 0)
            The minimum expression log2 fold change.
        method: 'str'(default:'all')
            This method is to choose the difference expression genes between 
            test group and other groups one by one or combine them together
            (default:'all')

    Returns
    -------
        A pandas DataFrame of the differential expression analysis result between
        the two groups.
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
    de = []
    for i_gene, gene in tqdm(enumerate(genes), desc="identifying top markers for each group"): 
        all_vals = X_data[:, i_gene].A if sparse else X_data[:, i_gene]
        test_vals = all_vals[test_cells]
        control_vals = all_vals[control_cells]
        test_mean = test_vals.mean() + 1e-9
        # ratio_expr
        ratio_expr = len(test_vals.nonzero()[0])/num_test_cells
        if ratio_expr < ratio_expr_thresh:
            continue
        if method == 'all':
            # log2fc
            control_mean = control_vals.mean() + 1e-9
            log2fc = np.log2(test_mean/control_mean + 10e-5)
            # pvals
            if len(control_vals.nonzero()[0]) > 0:
                pvals = mannwhitneyu(test_vals, control_vals)[1][0]
            else: 
                pvals = 1
            # diff_ratio_expr
            diff_ratio_expr = ratio_expr - len(control_vals.nonzero()[0])/num_control_cells
            de.append(
                (   
                    gene,
                    control_groups,
                    log2fc,
                    pvals,
                    ratio_expr,
                    diff_ratio_expr,
                )
            )
        else:
            for i in range(num_groups):
                control_vals = all_vals[adata.obs[group] == control_groups[i]]
                # log2fc
                control_mean = np.mean(control_vals, axis=0) + 1e-9
                log2fc = np.log2(test_mean/control_mean + 10e-5)[0]
                # pvals
                if len(control_vals.nonzero()[0]) > 0:
                    pvals = mannwhitneyu(test_vals, control_vals)[1][0]
                else: 
                    pvals = 1
                # diff_ratio_expr
                diff_ratio_expr = ratio_expr - len(control_vals.nonzero()[0])/len(control_vals)
                de.append(
                    (   
                        gene,
                        control_groups[i],
                        log2fc,
                        pvals,
                        ratio_expr,
                        diff_ratio_expr,
                    )
                )
    de = pd.DataFrame(
        de,
        columns=[
            "gene",
            "control_group",
            "log2fc",
            "pval",
            "ratio_expr",
            "diff_ratio_expr",
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
        "log2fc",
        "pval",
        "qval",
    ]
    de = de[out_order].sort_values(by="qval")
    de = de[(de.qval < qval_thresh) &
                (de.diff_ratio_expr > diff_ratio_expr_thresh) &
                (de.log2fc > log2fc_thresh)].reset_index(drop=True)
    return de

def find_all_cluster_degs(
    adata,
    group,
    genes=None,
    layer=None,
    X_data=None,
):
    """Find marker genes for each group of buckets based on gene expression.    
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimen-
            sion reduction and clustering. If `None`, all genes will be used.
        layer: `str` or None (default: `None`)
            The layer that will be used to retrieve data for dimension reduc-
            tion and clustering. If `None`, .X is used.
        group: `str` or None (default: `None`)
            The column key/name that identifies the grouping information (for 
            example, clusters that correspond to different cell types) of 
            buckets.This will be used for calculating group-specific genes.
        test_group: `str` or None (default: `None`)
            The group name from `group` for which markers has to be found.
        control_groups: `list`
            The list of group name(s) from `group` for which markers has to be 
            tested against.
        X_data: `np.ndarray` (default: `None`)
            The user supplied data that will be used for marker gene detection 
            directly.
    -------
        Returns an updated `~anndata.AnnData` with a new property `cluster_mar-
        kers`in the .uns attribute, which includes a concated pandas DataFrame 
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
        for i, test_group in enumerate(cluster_set):
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
    de_table = pd.concat(de_tables).reset_index().drop(columns=["index"])
    adata.uns["cluster_markers"] = {"deg_table": de_table, "de_genes": de_genes}
    return adata


