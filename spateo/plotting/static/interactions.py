"""
Plots to visualize results from cell-cell colocalization based analyses, as well as cell-cell communication
inference-based analyses. Makes use of dotplot-generating functions
"""
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from inspect import signature

import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import rcParams
from scipy.cluster import hierarchy as sch

from ...configuration import SKM, config_spateo_rcParams, set_pub_style
from ...logging import logger_manager as lm
from ...plotting.static.dotplot import CCDotplot
from .utils import _dendrogram_sig, save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def ligrec(
    adata: AnnData,
    dict_key: str,
    source_groups: Union[None, str, List[str]] = None,
    target_groups: Union[None, str, List[str]] = None,
    means_range: Tuple[float, float] = (-np.inf, np.inf),
    pvalue_threshold: float = 1.0,
    remove_empty_interactions: bool = True,
    remove_nonsig_interactions: bool = False,
    dendrogram: Union[None, str] = None,
    alpha: float = 0.001,
    swap_axes: bool = False,
    title: Union[None, str] = None,
    figsize: Union[None, Tuple[float, float]] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = {},
    **kwargs,
):
    """
    Dotplot for visualizing results of ligand-receptor interaction analysis

    For each L:R pair on the dotplot, molecule 1 is sent from the cluster(s) labeled on the top of the plot (or on the
    right, if 'swap_axes' is True), whereas molecule 2 is the receptor on the cluster(s) labeled on the bottom.

    Args:
        adata: Object of :class `anndata.AnnData`
        dict_key: Key in .uns to dictionary containing cell-cell communication information. Should contain keys labeled
            "means" and "pvalues", with values being dataframes for the mean cell type-cell type L:R product and
            significance values.
        source_groups: Source interaction clusters. If `None`, select all clusters.
        target_groups: Target interaction clusters. If `None`, select all clusters.
        means_range: Only show interactions whose means are within this **closed** interval
        pvalue_threshold: Only show interactions with p-value <= `pvalue_threshold`
        remove_empty_interactions: Remove rows and columns that contain NaN values
        remove_nonsig_interactions: Remove rows and columns that only contain interactions that are larger than `alpha`
        dendrogram: How to cluster based on the p-values. Valid options are:
                -  None (no input) - do not perform clustering.
                - `'interacting_molecules'` - cluster the interacting molecules.
                - `'interacting_clusters'` - cluster the interacting clusters.
                - `'both'` - cluster both rows and columns. Note that in this case, the dendrogram is not shown.
        alpha: Significance threshold. All elements with p-values <= `alpha` will be marked by tori instead of dots.
        swap_axes: Whether to show the cluster combinations as rows and the interacting pairs as columns
        title: Title of the plot
        figsize: The width and height of a figure
        save_show_or_return: Options: "save", "show", "return", "both", "all"
                - "both" for save and show
        save_kwargs: A dictionary that will passed to the save_fig function. By default it is an empty dictionary
            and the save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. But to change any of these
            parameters, this dictionary can be used to do so.
        kwargs :
            Keyword arguments for :func `style` or :func `legend` of :class `Dotplot`
    """
    logger = lm.get_main_logger()

    config_spateo_rcParams()
    set_pub_style()

    if figsize is None:
        figsize = rcParams.get("figure.figsize")

    if title is None:
        title = "Ligand-Receptor Inference"

    dict = adata.uns[dict_key]

    def filter_values(
        pvals: pd.DataFrame, means: pd.DataFrame, *, mask: pd.DataFrame, kind: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mask_rows = mask.any(axis=1)
        pvals = pvals.loc[mask_rows]
        means = means.loc[mask_rows]

        if pvals.empty:
            raise ValueError(f"After removing rows with only {kind} interactions, none remain.")

        mask_cols = mask.any(axis=0)
        pvals = pvals.loc[:, mask_cols]
        means = means.loc[:, mask_cols]

        if pvals.empty:
            raise ValueError(f"After removing columns with only {kind} interactions, none remain.")

        return pvals, means

    def get_dendrogram(adata: AnnData, linkage: str = "complete") -> Mapping[str, Any]:
        z_var = sch.linkage(
            adata.X,
            metric="correlation",
            method=linkage,
            # Unlikely to ever be profiling this many LR pairings, but cap at 1500
            optimal_ordering=adata.n_obs <= 1500,
        )
        dendro_info = sch.dendrogram(z_var, labels=adata.obs_names.values, no_plot=True)
        # this is what the DotPlot requires
        return {
            "linkage": z_var,
            "cat_key": ["groups"],
            "cor_method": "pearson",
            "use_rep": None,
            "linkage_method": linkage,
            "categories_ordered": dendro_info["ivl"],
            "categories_idx_ordered": dendro_info["leaves"],
            "dendrogram_info": dendro_info,
        }

    if len(means_range) != 2:
        logger.error(f"Expected `means_range` to be a sequence of size `2`, found `{len(means_range)}`.")
    means_range = tuple(sorted(means_range))

    if alpha is not None and not (0 <= alpha <= 1):
        logger.error(f"Expected `alpha` to be in range `[0, 1]`, found `{alpha}`.")

    if source_groups is None:
        source_groups = dict["pvalues"].columns.get_level_values(0)
    elif isinstance(source_groups, str):
        source_groups = (source_groups,)

    if target_groups is None:
        target_groups = dict["pvalues"].columns.get_level_values(1)
    if isinstance(target_groups, str):
        target_groups = (target_groups,)

    # Get specified source and target groups from the dictionary:
    pvals: pd.DataFrame = dict["pvalues"].loc[:, (source_groups, target_groups)]
    means: pd.DataFrame = dict["means"].loc[:, (source_groups, target_groups)]

    if pvals.empty:
        raise ValueError("No valid clusters have been selected.")

    means = means[(means >= means_range[0]) & (means <= means_range[1])]
    pvals = pvals[pvals <= pvalue_threshold]

    if remove_empty_interactions:
        pvals, means = filter_values(pvals, means, mask=~(pd.isnull(means) | pd.isnull(pvals)), kind="NaN")
    if remove_nonsig_interactions and alpha is not None:
        pvals, means = filter_values(pvals, means, mask=pvals <= alpha, kind="non-significant")

    start, label_ranges = 0, {}

    if dendrogram == "interacting_clusters":
        # Set rows to be cluster combinations, not LR pairs:
        pvals = pvals.T
        means = means.T

    for cls, size in (pvals.groupby(level=0, axis=1)).size().to_dict().items():
        label_ranges[cls] = (start, start + size - 1)
        start += size
    label_ranges = {k: label_ranges[k] for k in sorted(label_ranges.keys())}

    pvals = pvals[label_ranges.keys()].astype("float")
    # Add minimum value to p-values to avoid value error- 3.0 will be the largest possible value:
    pvals = -np.log10(pvals + min(1e-3, alpha if alpha is not None else 1e-3)).fillna(0)

    pvals.columns = map(" | ".join, pvals.columns.to_flat_index())
    pvals.index = map(" | ".join, pvals.index.to_flat_index())

    means = means[label_ranges.keys()].fillna(0)
    means.columns = map(" | ".join, means.columns.to_flat_index())
    means.index = map(" | ".join, means.index.to_flat_index())
    means = np.log2(means + 1)

    var = pd.DataFrame(pvals.columns)
    var = var.set_index(var.columns[0])

    # Instantiate new AnnData object containing plot values:
    adata = AnnData(pvals.values, obs={"groups": pd.Categorical(pvals.index)}, var=var, dtype=pvals.values.dtype)
    adata.obs_names = pvals.index
    minn = np.nanmin(adata.X)
    delta = np.nanmax(adata.X) - minn
    adata.X = (adata.X - minn) / delta
    # To satisfy conditional check that happens on instantiating dotplot:
    adata.uns["__type"] = "UMI"

    try:
        if dendrogram == "both":
            row_order, col_order, _, _ = _dendrogram_sig(
                adata.X, method="complete", metric="correlation", optimal_ordering=adata.n_obs <= 1500
            )
            adata = adata[row_order, :][:, col_order]
            pvals = pvals.iloc[row_order, :].iloc[:, col_order]
            means = means.iloc[row_order, :].iloc[:, col_order]
        elif dendrogram is not None:
            adata.uns["dendrogram"] = get_dendrogram(adata)
    except Exception as e:
        logger.warning(f"Unable to create a dendrogram. Reason: `{e}`. Will display without one.")
        dendrogram = None

    kwargs["dot_edge_lw"] = 0
    kwargs.setdefault("cmap", "magma")
    kwargs.setdefault("grid", True)
    kwargs.pop("color_on", None)

    # Set style and legend kwargs:
    dotplot_style_params = {k for k in signature(CCDotplot.style).parameters.keys()}
    dotplot_style_kwargs = {k: v for k, v in kwargs.items() if k in dotplot_style_params}
    dotplot_legend_params = {k for k in signature(CCDotplot.legend).parameters.keys()}
    dotplot_legend_kwargs = {k: v for k, v in kwargs.items() if k in dotplot_legend_params}

    dp = (
        CCDotplot(
            delta=delta,
            minn=minn,
            alpha=alpha,
            adata=adata,
            var_names=adata.var_names,
            cat_key="groups",
            dot_color_df=means,
            dot_size_df=pvals,
            title=title,
            var_group_labels=None if dendrogram == "both" else list(label_ranges.keys()),
            var_group_positions=None if dendrogram == "both" else list(label_ranges.values()),
            standard_scale=None,
            figsize=figsize,
        )
        .style(**dotplot_style_kwargs)
        .legend(
            size_title=r"$-\log_{10} ~ P$",
            colorbar_title=r"$log_2(molecule_1 * molecule_2 + 1)$",
            **dotplot_legend_kwargs,
        )
    )
    if dendrogram in ["interacting_molecules", "interacting_clusters"]:
        dp.add_dendrogram(size=1.6, dendrogram_key="dendrogram")
    if swap_axes:
        dp.swap_axes()

    dp.make_figure()

    if dendrogram != "both":
        # Remove the target part in: source | target
        labs = dp.ax_dict["mainplot_ax"].get_yticklabels() if swap_axes else dp.ax_dict["mainplot_ax"].get_xticklabels()
        for text in labs:
            text.set_text(text.get_text().split(" | ")[1])
        if swap_axes:
            dp.ax_dict["mainplot_ax"].set_yticklabels(labs)
        else:
            dp.ax_dict["mainplot_ax"].set_xticklabels(labs)

    if alpha is not None:
        yy, xx = np.where((pvals.values + alpha) >= -np.log10(alpha))
        if len(xx) and len(yy):
            # for dendrogram='both', they are already re-ordered
            mapper = (
                np.argsort(adata.uns["dendrogram"]["categories_idx_ordered"])
                if "dendrogram" in adata.uns
                else np.arange(len(pvals))
            )
            logger.info(f"Found `{len(yy)}` significant interactions at level `{alpha}`")
            ss = 0.33 * (adata.X[yy, xx] * (dp.largest_dot - dp.smallest_dot) + dp.smallest_dot)

            yy = np.array([mapper[y] for y in yy])
            if swap_axes:
                xx, yy = yy, xx
            dp.ax_dict["mainplot_ax"].scatter(
                xx + 0.5,
                yy + 0.5,
                color="white",
                edgecolor=kwargs["dot_edge_color"],
                linewidth=kwargs["dot_edge_lw"],
                s=ss,
                lw=0,
            )

    # Save, show or return figures:
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        # Doesn't matter what show_legend is for this plotting function
        show_legend=False,
        background="white",
        prefix="dotplot",
        save_kwargs=save_kwargs,
        total_panels=1,
        fig=dp.fig,
        axes=dp.ax_dict,
        # Return all parameters are for returning multiple values for 'axes', but this function uses a single dictionary
        return_all=False,
        return_all_list=None,
    )
