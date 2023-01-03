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
import scipy
import seaborn as sns
from anndata import AnnData
from matplotlib import rcParams
from matplotlib.collections import PolyCollection
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster import hierarchy as sch

from ...configuration import SKM, config_spateo_rcParams, set_pub_style
from ...logging import logger_manager as lm
from ...plotting.static.dotplot import CCDotplot
from ...tools.find_neighbors import generate_spatial_weights_fixed_nbrs
from ...tools.labels import Label, interlabel_connections
from .utils import _dendrogram_sig, save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def plot_connections(
    adata: AnnData,
    cat_key: str,
    spatial_key: str = "spatial",
    n_spatial_neighbors: Union[None, int] = 6,
    spatial_weights_matrix: Union[None, scipy.sparse.csr_matrix, np.ndarray] = None,
    expr_weights_matrix: Union[None, scipy.sparse.csr_matrix, np.ndarray] = None,
    reverse_expr_plot_orientation: bool = False,
    ax: Union[None, mpl.axes.Axes] = None,
    figsize: tuple = (3, 3),
    zero_self_connections: bool = True,
    normalize_by_self_connections: bool = False,
    shapes_style: bool = True,
    label_outline: bool = False,
    max_scale: float = 0.46,
    colormap: Union[str, dict, "mpl.colormap"] = "Spectral",
    title_str: Union[None, str] = None,
    title_fontsize: Union[None, float] = None,
    label_fontsize: Union[None, float] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = {},
):
    """Plot spatial_connections between labels- visualization of how closely labels are colocalized

    Args:
        adata: AnnData object
        cat_key: Key in .obs containing categorical grouping labels. Colocalization will be assessed
            for pairwise combinations of these labels.
        spatial_key: Key in .obsm containing coordinates in the physical space. Not used unless
            'spatial_weights_matrix' is None, in which case this is required. Defaults to "spatial".
        n_spatial_neighbors: Optional, number of neighbors in the physical space for each cell. Not used unless
            'spatial_weights_matrix' is None.
        spatial_weights_matrix: Spatial distance matrix, weighted by distance between spots. If not given,
            will compute at runtime.
        expr_weights_matrix: Gene expression distance matrix, weighted by distance in transcriptomic or PCA space.
            If not given, only the spatial distance matrix will be plotted. If given, will plot the spatial distance
            matrix in the left plot and the gene expression distance matrix in the right plot.
        reverse_expr_plot_orientation: If True, plot the gene expression connections in the form of a lower right
            triangle. If False, gene expression connections will be an upper left triangle just like the spatial
            connections.
        ax: Existing axes object, if applicable
        figsize: Width x height of desired figure window in inches
        zero_self_connections: If True, ignores intra-label interactions
        normalize_by_self_connections: Only used if 'zero_self_connections' is False. If True, normalize intra-label
            connections by the number of spots of that label
        shapes_style: If True plots squares, if False plots heatmap
        label_outline: If True, gives dark outline to axis tick label text
        max_scale: Only used for the case that 'shape_style' is True, gives maximum size of square
        colormap: Specifies colors to use for plotting. If dictionary, keys should be numerical labels corresponding
            to those of the Label object.
        title_str: Optionally used to give plot a title
        title_fontsize: Size of plot title- only used if 'title_str' is given.
        label_fontsize: Size of labels along the axes of the graph
        save_show_or_return: Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If "all", the figure will be saved, displayed
            and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.

    Returns:
        (fig, ax): Returns plot and axis object if 'save_show_or_return' is "all"
    """
    from ...plotting.static.utils import save_fig
    from ...tools.utils import update_dict

    logger = lm.get_main_logger()
    config_spateo_rcParams()
    title_fontsize = rcParams.get("axes.titlesize") if title_fontsize is None else title_fontsize
    label_fontsize = rcParams.get("axes.labelsize") if label_fontsize is None else label_fontsize

    if ax is None:
        if expr_weights_matrix is not None:
            figsize = (figsize[0] * 2.25, figsize[1])
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax_sp, ax_expr = axes[0], axes[1]

            if reverse_expr_plot_orientation:
                # Allow subplot boundaries to technically be partially overlapping (for better visual)
                box = ax_expr.get_position()
                box.x0 = box.x0 - 0.3
                box.x1 = box.x1 - 0.3
                ax_expr.set_position(box)
        else:
            fig, ax_sp = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = ax
        if len(ax) > 1:
            ax_sp, ax_expr = ax[0], ax[1]
        else:
            ax_sp = ax
        fig = ax.get_figure()

    # Convert cell type labels to numerical using Label object:
    categories_str_cat = np.unique(adata.obs[cat_key].values)
    categories_num_cat = range(len(categories_str_cat))
    map_dict = dict(zip(categories_num_cat, categories_str_cat))
    categories_str = adata.obs[cat_key]
    categories_num = adata.obs[cat_key].replace(categories_str_cat, categories_num_cat)

    label = Label(categories_num.to_numpy(), str_map=map_dict)

    # If spatial weights matrix is not given, compute it. 'spatial_key' needs to be present in the AnnData object:
    if spatial_weights_matrix is None:
        if spatial_key not in adata.obsm_keys():
            logger.error(
                f"Given 'spatial_key' {spatial_key} does not exist as key in adata.obsm. Options: "
                f"{adata.obsm_keys()}."
            )
        spatial_weights_matrix, _, _ = generate_spatial_weights_fixed_nbrs(
            adata, spatial_key=spatial_key, num_neighbors=n_spatial_neighbors, decay_type="reciprocal"
        )

    # Compute spatial connections array:
    spatial_connections = interlabel_connections(label, spatial_weights_matrix)

    if zero_self_connections:
        np.fill_diagonal(spatial_connections, 0)
    elif normalize_by_self_connections:
        spatial_connections /= spatial_connections.diagonal()[:, np.newaxis]

    spatial_connections_max = np.amax(spatial_connections)

    # Optionally, compute gene expression connections array:
    if expr_weights_matrix is not None:
        expr_connections = interlabel_connections(label, expr_weights_matrix)

        if zero_self_connections:
            np.fill_diagonal(expr_connections, 0)
        elif normalize_by_self_connections:
            expr_connections /= expr_connections.diagonal()[:, np.newaxis]

        expr_connections_max = np.amax(expr_connections)

    # Set label colors:
    if isinstance(colormap, str):
        cmap = mpl.cm.get_cmap(colormap)
    else:
        cmap = colormap

    # If colormap is given, map label ID to points along the colormap. If dictionary is given, instead map each label
    # to a color using the dictionary keys as guides.
    if isinstance(cmap, dict):
        if type(list(cmap.keys())[0]) == str:
            id_colors = {n_id: cmap[id] for n_id, id in zip(label.ids, label.str_ids)}
        else:
            id_colors = {id: cmap[id] for id in label.ids}
    else:
        id_colors = {id: cmap(id / label.max_id) for id in label.ids}

    # -------------------------------- Spatial Connections Plot- Setup -------------------------------- #
    if shapes_style:
        # Cell types/labels will be represented using triangles:
        left_triangle = np.array(
            (
                (-1.0, 1.0),
                # (1., 1.),
                (1.0, -1.0),
                (-1.0, -1.0),
            )
        )

        right_triangle = np.array(
            (
                (-1.0, 1.0),
                (1.0, 1.0),
                (1.0, -1.0),
                # (-1., -1.)
            )
        )

        polygon_list = []
        color_list = []

        ax_sp.set_ylim(-0.55, label.num_labels - 0.45)
        ax_sp.set_xlim(-0.55, label.num_labels - 0.45)

        for label_1 in range(spatial_connections.shape[0]):
            for label_2 in range(spatial_connections.shape[1]):

                if label_1 <= label_2:

                    for triangle in [left_triangle, right_triangle]:
                        center = np.array((label_1, label_2))[np.newaxis, :]
                        scale_factor = spatial_connections[label_1, label_2] / spatial_connections_max
                        offsets = triangle * max_scale * scale_factor
                        polygon_list.append(center + offsets)

                    color_list += (id_colors[label.ids[label_2]], id_colors[label.ids[label_1]])

        collection = PolyCollection(polygon_list, facecolors=color_list, edgecolors="face", linewidths=0)

        ax_sp.add_collection(collection)

        # Remove ticks
        ax_sp.tick_params(labelbottom=False, labeltop=True, top=False, bottom=False, left=False)
        ax_sp.xaxis.set_tick_params(pad=-2)
    else:
        # Heatmap of connection strengths
        heatmap = ax_sp.imshow(spatial_connections, cmap=colormap, interpolation="nearest")

        divider = make_axes_locatable(ax_sp)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        fig.colorbar(heatmap, cax=cax)
        cax.tick_params(axis="both", which="major", labelsize=6, rotation=-45)

        # Change formatting if values too small
        if spatial_connections_max < 0.001:
            cax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1e}"))

    # Formatting adjustments
    ax_sp.set_aspect("equal")

    ax_sp.set_xticks(
        np.arange(label.num_labels),
    )
    text_outline = [PathEffects.Stroke(linewidth=0.5, foreground="black", alpha=0.8)] if label_outline else None

    # If label has categorical labels associated, use those to label the axes instead:
    if label.str_map is not None:
        ax_sp.set_xticklabels(
            label.str_ids,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=90,
            path_effects=text_outline,
        )
    else:
        ax_sp.set_xticklabels(
            label.ids,
            fontsize=label_fontsize,
            fontweight="bold",
            rotation=0,
            path_effects=text_outline,
        )

    ax_sp.set_yticks(np.arange(label.num_labels))
    if label.str_map is not None:
        ax_sp.set_yticklabels(
            label.str_ids,
            fontsize=label_fontsize,
            fontweight="bold",
            path_effects=text_outline,
        )
    else:
        ax_sp.set_yticklabels(
            label.ids,
            fontsize=label_fontsize,
            fontweight="bold",
            path_effects=text_outline,
        )

    for ticklabels in [ax_sp.get_xticklabels(), ax_sp.get_yticklabels()]:
        for n, id in enumerate(label.ids):
            ticklabels[n].set_color(id_colors[id])

    title_str_sp = "Spatial Connections" if title_str is None else title_str
    ax_sp.set_title(title_str_sp, fontsize=title_fontsize, fontweight="bold")

    # ------------------------------ Optional Gene Expression Connections Plot- Setup ------------------------------ #
    if expr_weights_matrix is not None:
        if shapes_style:
            polygon_list = []
            color_list = []

            ax_expr.set_ylim(-0.55, label.num_labels - 0.45)
            ax_expr.set_xlim(-0.55, label.num_labels - 0.45)

            for label_1 in range(expr_connections.shape[0]):
                for label_2 in range(expr_connections.shape[1]):

                    if label_1 <= label_2:
                        for triangle in [left_triangle, right_triangle]:
                            center = np.array((label_1, label_2))[np.newaxis, :]
                            scale_factor = expr_connections[label_1, label_2] / expr_connections_max
                            offsets = triangle * max_scale * scale_factor
                            polygon_list.append(center + offsets)

                        color_list += (id_colors[label.ids[label_2]], id_colors[label.ids[label_1]])

                # Remove ticks
                if reverse_expr_plot_orientation:
                    ax_expr.tick_params(
                        labelbottom=True,
                        labeltop=False,
                        labelleft=False,
                        labelright=True,
                        top=False,
                        bottom=False,
                        left=False,
                    )
                    # Flip x- and y-axes of the expression plot:
                    ax_expr.invert_xaxis()
                    ax_expr.invert_yaxis()
                else:
                    ax_expr.tick_params(labelbottom=False, labeltop=True, top=False, bottom=False, left=False)
                ax_expr.xaxis.set_tick_params(pad=-2)

            collection = PolyCollection(polygon_list, facecolors=color_list, edgecolors="face", linewidths=0)

            ax_expr.add_collection(collection)
        else:
            # Heatmap of connection strengths
            heatmap = ax_expr.imshow(expr_connections, cmap=colormap, interpolation="nearest")

            divider = make_axes_locatable(ax_expr)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            fig.colorbar(heatmap, cax=cax)
            cax.tick_params(axis="both", which="major", labelsize=6, rotation=-45)

            # Change formatting if values too small
            if spatial_connections_max < 0.001:
                cax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1e}"))

        # Formatting adjustments
        ax_expr.set_aspect("equal")

        ax_expr.set_xticks(
            np.arange(label.num_labels),
        )
        if reverse_expr_plot_orientation:
            # Despine both spatial connections & gene expression connections plots:
            ax_sp.spines["right"].set_visible(False)
            ax_sp.spines["top"].set_visible(False)
            ax_sp.spines["left"].set_visible(False)
            ax_sp.spines["bottom"].set_visible(False)

            ax_expr.spines["right"].set_visible(False)
            ax_expr.spines["top"].set_visible(False)
            ax_expr.spines["left"].set_visible(False)
            ax_expr.spines["bottom"].set_visible(False)

        text_outline = [PathEffects.Stroke(linewidth=0.5, foreground="black", alpha=0.8)] if label_outline else None

        # If label has categorical labels associated, use those to label the axes instead:
        if label.str_map is not None:
            ax_expr.set_xticklabels(
                label.str_ids,
                fontsize=label_fontsize,
                fontweight="bold",
                rotation=90,
                path_effects=text_outline,
            )
        else:
            ax_expr.set_xticklabels(
                label.ids,
                fontsize=label_fontsize,
                fontweight="bold",
                rotation=0,
                path_effects=text_outline,
            )

        ax_expr.set_yticks(np.arange(label.num_labels))
        if label.str_map is not None:
            ax_expr.set_yticklabels(
                label.str_ids,
                fontsize=label_fontsize,
                fontweight="bold",
                path_effects=text_outline,
            )
        else:
            ax_expr.set_yticklabels(
                label.ids,
                fontsize=label_fontsize,
                fontweight="bold",
                path_effects=text_outline,
            )

        for ticklabels in [ax_expr.get_xticklabels(), ax_expr.get_yticklabels()]:
            for n, id in enumerate(label.ids):
                ticklabels[n].set_color(id_colors[id])

        title_str_expr = "Gene Expression Similarity" if title_str is None else title_str
        if reverse_expr_plot_orientation:
            if label_fontsize <= 8:
                y = -0.35
            elif label_fontsize > 8:
                y = -0.45
        else:
            y = None
        ax_expr.set_title(title_str_expr, fontsize=title_fontsize, fontweight="bold", y=y)

    prefix = "spatial_connections" if expr_weights_matrix is None else "spatial_and_expr_connections"
    if save_show_or_return in ["save", "both", "all"]:
        s_kwargs = {
            "path": None,
            "prefix": prefix,
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)

    elif save_show_or_return in ["show", "both", "all"]:
        plt.show()
    elif save_show_or_return in ["return", "all"]:
        if expr_weights_matrix is not None:
            ax = axes
        else:
            ax = ax_sp
        return (fig, ax)


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
