"""
Characterizing cell-to-cell variability within spatial domains
"""
from collections import OrderedDict
from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from matplotlib import rcParams
from tqdm import tqdm

from ...configuration import SKM, config_spateo_rcParams
from ...logging import logger_manager as lm
from ...plotting.static.utils import save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def compute_variance_decomposition(
        adata: AnnData,
        spatial_label_id: str,
        celltype_label_id: str,
        visualize: bool = False,
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {}
):
    """
    Computes and then optionally visualizes the variance decomposition for an AnnData object.

    Within spatial regions, determines the proportion of the total variation that occurs within the same cell type,
    the proportion of the variation that occurs between cell types in the region, and the proportion of the variation
    that comes from baseline differences in the expression levels of the genes in the data. The within-cell type
    variation could potentially come from differences in cell-cell communication.

    Args:
        adata: class `anndata.AnnData`
        spatial_label_id : str
            Key in .obs containing spatial domain labels
        celltype_label_id : str
            Key in .obs containing cell type labels
        visualize : bool, default False
            If True, generates a plot for the results
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show"
            Whether to save, show or return the figure. Only used if 'visualize' is True
            If "both", it will save and plot the figure at the same time. If "all", the figure will be saved, displayed
            and the associated axis and other object will be return.
        save_kwargs : optional dict
            A dictionary that will passed to the save_fig function. Only used if 'visualize' is True
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.

    Returns:
        var_decomposition : pd.DataFrame
            Dataframe containing four columns, for the category label, celltype variation,
    """
    adata_copy = adata.copy()

    # Dataframe containing gene expression, cell type labels and spatial domain labels:
    data = adata_copy.X.toarray() if scipy.sparse.issparse(adata_copy.X) else adata_copy.X
    df = pd.DataFrame(data, columns=adata_copy.var_names)
    df["Spatial Domain"] = pd.Series(list(adata.obs[spatial_label_id]), dtype="category")
    df["Cell Type"] = pd.Series(list(adata.obs[celltype_label_id]), dtype="category")
    domains = np.unique(df["Spatial Domain"])
    var_decomposition_list = []

    with tqdm(total=len(domains)) as pbar:
        for domain in domains:
            # For each gene, compute mean within the domain:
            mean_domain_genes = np.mean(df[df["Spatial Domain"] == domain][::-2], axis=0)
            # Compute average for all genes:
            mean_domain_global = np.mean(mean_domain_genes)

            intra_ct_var = []
            inter_ct_var = []
            gene_var = []
            for celltype in np.unique(df["Cell Type"]):
                # Gene expression (take all but last two columns) for each cell type within each spatial domain
                domain_celltype = np.array(df[(df["Spatial Domain"] == domain) & (df["Cell Type"] == celltype)])[:, :-2]
                if domain_celltype.shape[0] == 0:
                    continue
                # For each cell type, compute the mean gene expression for each gene
                mean_domain_celltype = np.mean(domain_celltype, axis=0)

                for i in range(domain_celltype.shape[0]):
                    # Within the cell type, variance for each gene from the mean of the cell type
                    intra_ct_var.append((domain_celltype[i, :] - mean_domain_celltype) ** 2)
                    # For each cell type, the difference in mean expression within the cell type as compared to the
                    # mean of the domain
                    inter_ct_var.append((mean_domain_celltype - mean_domain_genes) ** 2)
                    # Within each domain, variance for the domain from the mean of the domain
                    gene_var.append((mean_domain_genes - mean_domain_global) ** 2)

            intra_ct_var = np.sum(intra_ct_var)
            inter_ct_var = np.sum(inter_ct_var)
            gene_var = np.sum(gene_var)
            var_decomposition_list.append(np.array([domain, intra_ct_var, inter_ct_var, gene_var]))
            pbar.update(1)

    df = (
        pd.DataFrame(var_decomposition_list, columns=["Domain", "intra_celltype_var", "inter_celltype_var", "gene_var"])
        .astype(
            {
                "Domain": str,
                "intra_celltype_var": "float32",
                "inter_celltype_var": "float32",
                "gene_var": "float32",
            }
        )
        .set_index("Domain")
    )

    df["Total variance"] = df.intra_celltype_var + df.inter_celltype_var + df.gene_var
    # Normalize to sum to 1:
    df["Intra-cell type variance"] = df.intra_celltype_var / df["Total variance"]
    df["Inter-cell type variance"] = df.inter_celltype_var / df["Total variance"]
    df["Gene variance"] = df.gene_var / df["Total variance"]

    # Optionally plot with default plotting parameters:
    multiindex = True if isinstance(df.index, pd.MultiIndex) else False
    if visualize:
        plot_variance_decomposition(df, save_show_or_return=save_show_or_return, save_kwargs=save_kwargs)

    return df


def plot_variance_decomposition(
    var_df: pd.DataFrame,
    figsize: Tuple[float, float] = (6, 2),
    cmap: str = "Blues_r",
    multiindex: bool = False,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = {}
):
    """
    Visualization of the parts-wise intra-cell type variation, cell type-independent gene variation to the total
    variation within the data.

    Args:
        var_df : pd.DataFrame
            Output from :func `compute_variance_decomposition`
        figsize : tuple of form (float, float)
            (width, height) of the figure window
        cmap : str, default "Blues_r"
            Name of the matplotlib colormap to use
        multiindex : bool, default False
            Specifies whether to set labels to record multi-level index information. Should only be used if var_df
            has a multi-index.
        save_dir : optional str
            If provided, sets the directory to save the figure to. Only used if "save_show_or_return" is True
        file_name : str
            Name (without the extension) of the file to save. Will only be
        save_show_or_return: Literal["save", "show", "return", "both", "all"], default "show"
            Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If "all", the figure will be saved, displayed
            and the associated axis and other object will be return.
        save_kwargs : optional dict
            A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.
    """
    logger = lm.get_main_logger()
    if not isinstance(var_df.index, pd.MultiIndex) and multiindex:
        logger.error("'var_df' index is not a multi-level index. 'Multiindex' cannot be set True.")

    config_spateo_rcParams()
    figsize = rcParams.get("figure.figsize") if figsize is None else figsize

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    var_df.plot(
        y=["Intra-cell type variance", "Inter-cell type variance", "Gene variance"],
        kind="bar",
        stacked=True,
        edgecolor='black',
        width=0.5,
        figsize=figsize,
        ax=ax,
        colormap=cmap,
    )

    if multiindex:
        def process_index(k):
            return tuple(k.split("_"))

        var_df["index1"], var_df["index2"] = zip(*map(process_index, var_df.index))
        var_df = var_df.set_index(["index1", "index2"])

        ax.set_xlabel("")
        xlabel_mapping = OrderedDict()
        for index1, index2 in var_df.index:
            xlabel_mapping.setdefault(index1, [])
            xlabel_mapping[index1].append(index2)

        hline = []
        new_xlabels = []
        for _index1, index2_list in xlabel_mapping.items():
            index2_list[0] = "{}".format(index2_list[0])
            new_xlabels.extend(index2_list)

            if hline:
                hline.append(len(index2_list) + hline[-1])
            else:
                hline.append(len(index2_list))
        ax.set_xticklabels(new_xlabels)

    # Configuring plot:
    ax.set_xlabel("")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title("Variance Decomposition for Spatial Domains")
    ax.set_ylabel("Proportion of variance")
    plt.tight_layout()

    save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=True,
        background="white",
        prefix="variance_decomposition",
        save_kwargs=save_kwargs,
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )