"""
Dotplot class adapted from https://github.com/scverse/scanpy with modifications for suitability to cell-cell
communication and interaction analyses

Development notes: some of the methods mention dendrograms/other extra plots and there is currently no capability to
generate those- coming in future update...additions that will have to be made: functions for plot_dendrogram,
plot_totals, additional if condition in make_figure()...
"""
import collections.abc as cabc
from collections import namedtuple
from itertools import product

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import gridspec, rcParams
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.path import Path
from pandas.api.types import is_numeric_dtype

from ...configuration import SKM, config_spateo_rcParams, set_pub_style
from ...logging import logger_manager as lm
from .utils import (
    _get_array_values,
    check_colornorm,
    deduplicate_kwargs,
    plot_dendrogram,
    save_return_show_fig_utils,
)


# --------------------------------------- Data conversion for plotting --------------------------------------- #
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def adata_to_frame(
    adata: AnnData,
    var_names: Sequence[str],
    cat_key: Union[str, Sequence[str]],
    num_categories: int = 7,
    layer: Union[None, str] = None,
    gene_symbols_key: Union[None, str] = None,
):
    """
    For the purposes of dot plotting, converts the information given in AnnData object to a dataframe in which the
    row names are categories defined by groups and column names correspond to variable names.

    Args:
        adata: class `anndata.AnnData`
        var_names: Should be a subset of adata.var_names
        cat_key: The key(s) in .obs of the grouping to consider. Should be a categorical observation; if not,
            will be subdivided into 'num_categories'.
        num_categories: Only used if groupby observation is not categorical. This value determines the number of groups into
            which the groupby observation should be subdivided.
        layer: Key in .layers specifying layer to use. If not given, will use .X.
        gene_symbols_key: Key in .var containing gene symbols
    """

    logger = lm.get_main_logger()
    if isinstance(var_names, str):
        var_names = [var_names]

    # Can group by either .obs key or index. Set this flag to group by adata index
    cat_index = None
    if cat_key is not None:
        if isinstance(cat_key, str):
            cat_key = [cat_key]
        for group in cat_key:
            if group not in list(adata.obs_keys()) + [adata.obs.index.name]:
                if adata.obs.index.name is not None:
                    msg = f' or index name "{adata.obs.index.name}"'
                else:
                    msg = ""
                logger.error(
                    f"Grouping key cannot be found. Given: {group}, could not be found in {adata.obs_keys()}" + msg
                )
            if group in adata.obs.keys() and group == adata.obs.index.name:
                logger.error(f"Given group {group} is both and index and a column level, which is ambiguous.")
            if group == adata.obs.index.name:
                cat_index = group

    if cat_index is not None:
        # Downstream operations will already cover the index, so it does not need to be given:
        cat_key = cat_key.copy()  # copy to not modify user passed parameter
        cat_key.remove(cat_index)

    keys = list(cat_key) + list(np.unique(var_names))
    # Convert chosen .obs entries to dataframe:
    if gene_symbols_key is not None:
        alias_index = pd.Index(adata.var[gene_symbols_key])
    else:
        alias_index = None

    # Check indices and return warnings in the case of duplicate names:
    if alias_index is not None:
        # Map from current var_names to gene symbols:
        alt_names = pd.Series(adata.var.index, index=alias_index)
        alias_name = alias_index.name
        alt_search_repr = f"var['{alias_name}']"
    else:
        alt_names = pd.Series(adata.var.index, index=adata.var.index)
        alt_search_repr = "var_names"

    # Looking for keys within AnnData- store based on found location:
    obs_cols = []
    var_idx_keys = []
    var_symbols = []
    not_found = []

    # Check that adata.obs does not contain duplicated columns.
    # (if duplicated columns names are present, they will be further duplicated when selecting them)
    if not adata.obs.columns.is_unique:
        dup_cols = adata.obs.columns[adata.obs.columns.duplicated()].tolist()
        logger.error(
            f"adata.obs contains duplicated columns. Please rename or remove these columns first.\n`"
            f"Duplicated columns: {dup_cols}"
        )

    if not adata.var.index.is_unique:
        logger.error(
            f"adata.var_names contains duplicated items. \n"
            f"Rename variable names first for example using `adata.var_names_make_unique()`."
        )

    # Use only unique keys:
    for key in np.unique(keys):
        if key in adata.obs.columns:
            obs_cols.append(key)
            if key in alt_names.index:
                logger.error(f"The key '{key}' is found in both adata.obs and adata.{alt_search_repr}.")
        elif key in alt_names.index:
            val = alt_names[key]
            if isinstance(val, pd.Series):
                # while var_names must be unique, adata.var[gene_symbols] does not
                # It's still ambiguous to refer to a duplicated entry though.
                assert alias_index is not None
                raise KeyError(f"Found duplicate entries for '{key}' in adata.{alt_search_repr}.")
            var_idx_keys.append(val)
            var_symbols.append(key)
        else:
            not_found.append(key)
    if len(not_found) > 0:
        raise KeyError(f"Could not find keys '{not_found}' in columns of `adata.obs` or in adata.{alt_search_repr}.")

    adata_tidy_df = pd.DataFrame(index=adata.obs_names)

    # Adding var values to DataFrame:
    if len(var_idx_keys) > 0:
        adata_arr = adata.X if layer is None else adata.layers[layer]

        matrix = _get_array_values(
            adata_arr, dim_names=adata.var.index, keys=var_idx_keys, axis=1, backed=adata.isbacked
        )
        adata_tidy_df = pd.concat(
            [adata_tidy_df, pd.DataFrame(matrix, columns=var_symbols, index=adata.obs_names)],
            axis=1,
        )

    # Adding obs values to DataFrame:
    if len(obs_cols) > 0:
        adata_tidy_df = pd.concat([adata_tidy_df, adata.obs[obs_cols]], axis=1)

    # Reorder columns to given order (including duplicates keys if present)
    if keys:
        adata_tidy_df = adata_tidy_df[keys]

    assert np.all(np.array(keys) == np.array(adata_tidy_df.columns))

    if cat_index is not None:
        adata_tidy_df.reset_index(inplace=True)
        cat_key.append(cat_index)

    if cat_key is None:
        categorical = pd.Series(np.repeat("", len(adata_tidy_df))).astype("category")
    elif len(cat_key) == 1 and is_numeric_dtype(adata_tidy_df[cat_key[0]]):
        # If category column is not categorical, turn it into one by subdividing ranges of values into 'num_categories'
        # categories:
        categorical = pd.cut(adata_tidy_df[cat_key[0]], num_categories)
    elif len(cat_key) == 1:
        categorical = adata_tidy_df[cat_key[0]].astype("category")
        categorical.name = cat_key[0]
    else:
        # Join the category values  using "_" to make a new 'category' consisting of both categorical columns:
        categorical = adata_tidy_df[cat_key].apply("_".join, axis=1).astype("category")
        categorical.name = "_".join(cat_key)

        # Preserve category order as it appears in adata_tidy_df:
        order = {"_".join(k): idx for idx, k in enumerate(product(*(adata_tidy_df[g].cat.categories for g in cat_key)))}
        categorical = categorical.cat.reorder_categories(sorted(categorical.cat.categories, key=lambda x: order[x]))
    adata_tidy_df = adata_tidy_df[var_names].set_index(categorical)
    categories = adata_tidy_df.index.categories

    return categories, adata_tidy_df


# --------------------------------------- Initialize plotting grid --------------------------------------- #
# For multi-component plots to plot within the same plotting window- will use multiple Axes objects, but not define
# separate subplots
def make_grid_spec(
    ax_or_figsize: Union[Tuple[int, int], mpl.axes.Axes],
    nrows: int,
    ncols: int,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
) -> Tuple[Figure, gridspec.GridSpecBase]:
    """
    Initialize grid layout to place subplots within a figure environment

    Args:
        ax_or_figsize: Either already-existing ax object or the width and height to create a figure window
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
        wspace: The amount of width reserved for space between subplots, expressed as a fraction of the average axis width
        hspace: The amount of height reserved for space between subplots, expressed as a fraction of the average axis height
        width_ratios: Defines the relative widths of the columns. Each column gets a relative width of width_ratios[i] / sum(
            width_ratios). If not given, all columns will have the same width.
        height_ratios: Defines the relative heights of the rows. Each row gets a relative width of height_ratios[i] / sum(
            height_ratios). If not given, all columns will have the same width.

    Returns:
        fig: Instantiated Figure object
        gs: Instantiated gridspec object
    """
    kw = dict(
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(ax_or_figsize, tuple):
        fig = plt.figure(figsize=ax_or_figsize)
        gs = gridspec.GridSpec(nrows, ncols, **kw)
        return fig, gs
    else:
        ax = ax_or_figsize
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(nrows, ncols, **kw)
        return fig, gs


# --------------------------------------- Dotplot class --------------------------------------- #
class Dotplot:
    """
    Simultaneous visualization of two variates that are encoded by the dot size and the dot color. Size usually
    represents the fraction of samples that have non-zero values, and color usually represents the magnitude of the
    value.

    Args:
        adata: class `anndata.AnnData`
        var_names: Should be a subset of adata.var_names
        cat_key: The key(s) in .obs of the grouping to consider. Should be a categorical observation; if not,
            will be subdivided into 'num_categories'.
        num_categories: Only used if groupby observation is not categorical. This value determines the number of
            groups into which the groupby observation should be subdivided.
        categories_order: Sets order of categories given by 'cat_key' along the plotting axis
        title: Sets title for figure window
        figsize: The width and height of a figure
        gene_symbols_key: Key in .var containing gene symbols
        var_group_positions: Each item in the list should contain the start and end position that the bracket
            should cover. Eg. [(0, 4), (5, 8)] means that there are two brackets, one for the var_names in positions 0-4
            and other for positions 5-8.
        var_group_labels:  List of group labels for the variable names (e.g. can group var_names in positions 0-4
            as being "group A")
        var_group_rotation: Rotation in degrees of the variable name labels. If not given, small labels (<4
            characters) are not rotated, but otherwise labels are rotated 90 degrees.
        layer: Key in .layers specifying layer to use. If not given, will use .X.
        expression_cutoff: Used for binarizing feature expression- feature is considered to be expressed only if
            the expression value is greater than this threshold
        mean_only_expressed: If True, gene expression is averaged only over the cells expressing the given features
        standard_scale: Whether or not to standardize that dimension between 0 and 1, meaning for each variable or
            group, subtract the minimum and divide each by its maximum. 'val' or 'group' is used to specify whether this
            should be done over variables or groups.
        dot_color_df: Pre-prepared dataframe with features as indices, categories as columns, and indices
            corresponding to color intensities
        dot_size_df: Pre-prepared dataframe with features as indices, categories as columns, and indices
            corresponding to dot sizes
        ax: Can be used to provide pre-existing plotting axis
        vmin: The data value that defines 0.0 in the normalization. Defaults to the min value of the dataset.
        vmax: The data value that defines 1.0 in the normalization. Defaults to the the max value of the dataset.
        vcenter: The data value that defines 0.5 in the normalization
        norm: Optional already-initialized normalizing object that scales data, typically into the interval [0, 1],
            for the purposes of mapping to color intensities for plotting. Do not pass both 'norm' and
            'vmin'/'vmax', etc.
        **kwargs:
            Additional arguments passed to `matplotlib.pyplot.scatter()`
    """

    # Default parameters- visualization:
    default_colormap = "winter"
    default_color_on = "dot"
    default_dot_max = None
    default_dot_min = None
    default_smallest_dot = 0.0
    default_largest_dot = 200.0
    default_dot_edgecolor = "black"
    default_dot_edgelw = 0.2
    default_size_exponent = 1.5

    default_size_legend_title = "Fraction of cells\nin group (%)"
    default_color_legend_title = "Mean expression\nin group"
    default_base = 10
    default_num_colorbar_ticks = 5
    default_num_size_legend_dots = 5
    default_legends_width = 1.5  # inches
    default_plot_x_padding = 0.8  # a unit is equivalent to the distance between two x-axis ticks
    default_plot_y_padding = 1.0  # a unit is equivalent to the distance between two y-axis ticks

    # Default parameters- spacing:
    default_category_height = 0.35
    default_category_width = 0.37
    min_figure_height = 2.5
    # Space between main plot, dendrogram and legend:
    default_wspace = 0

    @SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
    def __init__(
        self,
        adata: AnnData,
        var_names: Sequence[str],
        cat_key: Union[str, Sequence[str]],
        num_categories: int = 7,
        prescale_adata: bool = False,
        categories_order: Union[None, Sequence[str]] = None,
        title: Union[None, str] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        gene_symbols_key: Union[None, str] = None,
        var_group_positions: Union[None, Sequence[Tuple[int, int]]] = None,
        var_group_labels: Union[None, Sequence[str]] = None,
        var_group_rotation: Union[None, float] = None,
        layer: Union[None, str] = None,
        expression_cutoff: float = 0.0,
        mean_only_expressed: bool = False,
        standard_scale: Literal["var", "group"] = None,
        dot_color_df: Optional[pd.DataFrame] = None,
        dot_size_df: Optional[pd.DataFrame] = None,
        ax: Union[None, mpl.axes.Axes] = None,
        vmin: Union[None, float] = None,
        vmax: Union[None, float] = None,
        vcenter: Union[None, float] = None,
        norm: Optional[Normalize] = None,
        **kwargs,
    ):

        # Default plotting parameters:
        config_spateo_rcParams()
        set_pub_style()

        self.logger = lm.get_main_logger()
        self.adata = adata
        self.cat_key = [cat_key] if isinstance(cat_key, str) else cat_key
        self.kwargs = kwargs

        self.var_names = var_names
        self.var_group_labels = var_group_labels
        self.var_group_positions = var_group_positions
        self.var_group_rotation = var_group_rotation

        self.has_var_groups = True if var_group_positions is not None and len(var_group_positions) > 0 else False

        # Update variable names if given as a dictionary:
        self._update_var_groups()

        # Figure formatting:
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize
        self.width, self.height = self.figsize
        self.fig_title = title

        # Limit for the number of categories that are allowed to be plotted:
        self.max_num_categories = 100

        self.categories, self.adata_tidy_df = adata_to_frame(
            adata,
            self.var_names,
            cat_key=cat_key,
            num_categories=num_categories,
            layer=layer,
            gene_symbols_key=gene_symbols_key,
        )

        # Check categories:
        if len(self.categories) > self.max_num_categories:
            self.logger.warning(f"Over {self.max_num_categories} categories found. Plot would be very large.")

        if categories_order is not None:
            if set(self.adata_tidy_df.index.categories) != set(categories_order):
                self.logger.error(
                    "Please check that the categories given by the `order` parameter match the categories to be "
                    "reordered. \n\n"
                    "Mismatch: "
                    f"{set(self.adata_tidy_df.index.categories).difference(categories_order)}\n\n"
                    f"Given order categories: {categories_order}\n\n"
                    f"{cat_key} categories: {list(self.adata_tidy_df.index.categories)}\n"
                )
                return

        # Compute fraction of cells having value > chosen expression cutoff, and transform into Boolean matrix using
        # the expression cutoff:
        obs_bool = self.adata_tidy_df > expression_cutoff

        # If dot size-specifying dataframe is not provided:
        if dot_size_df is None:
            # Compute the sum per group (for the Boolean matrix, the number of values > expression cutoff), divide the
            # result by the total number of cells in the group:
            dot_size_df = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()

        # If dot color-specifying dataframe is not provided:
        if dot_color_df is None:
            # Compute mean expression value, either only of cells that are expressing or of all cells:
            if mean_only_expressed:
                dot_color_df = self.adata_tidy_df.mask(~obs_bool).groupby(level=0).mean().fillna(0)
            else:
                dot_color_df = self.adata_tidy_df.groupby(level=0).mean()

            if standard_scale == "group":
                dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
                dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
            elif standard_scale == "var":
                dot_color_df -= dot_color_df.min(0)
                dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)
            elif standard_scale is None:
                pass
            else:
                self.logger.warning(
                    "Unknown input given for 'standard_scale', proceeding without further " "processing array"
                )
        else:
            # check that both matrices have the same shape
            if dot_color_df.shape != dot_size_df.shape:
                self.logger.error(
                    "The given dot_color_df data frame has a different shape than the data frame used for the dot "
                    "size. Both data frames need to have the same index and columns."
                )

        # Remove duplicated features (can occur e.g. if the same gene is a marker for two groups)
        unique_var_names, unique_idx = np.unique(dot_color_df.columns, return_index=True)

        if len(unique_var_names) != len(self.var_names):
            dot_color_df = dot_color_df.iloc[:, unique_idx]

        # Use the same order for rows and columns in the color and size dataframes:
        dot_color_df = dot_color_df.loc[dot_size_df.index][dot_size_df.columns]

        self.dot_color_df = dot_color_df
        self.dot_size_df = dot_size_df

        # Initialize all style parameters to the default:
        self.cmap = self.default_colormap
        self.dot_max = self.default_dot_max
        self.dot_min = self.default_dot_min
        self.smallest_dot = self.default_smallest_dot
        self.largest_dot = self.default_largest_dot
        self.color_on = self.default_color_on
        self.size_exponent = self.default_size_exponent
        self.grid = False
        self.plot_x_padding = self.default_plot_x_padding
        self.plot_y_padding = self.default_plot_y_padding

        self.dot_edge_color = self.default_dot_edgecolor
        self.dot_edge_lw = self.default_dot_edgelw

        # Set legend defaults:
        self.color_legend_title = self.default_color_legend_title
        self.size_title = self.default_size_legend_title
        self.num_colorbar_ticks = self.default_num_colorbar_ticks
        self.num_size_legend_dots = self.default_num_size_legend_dots
        self.base = self.default_base
        self.legends_width = self.default_legends_width
        self.show_size_legend = True
        self.show_colorbar = True
        self.wspace = self.default_wspace

        # For plotting:
        VBoundNorm = namedtuple("VBoundNorm", ["vmin", "vmax", "vcenter", "norm"])
        self.vboundnorm = VBoundNorm(vmin=vmin, vmax=vmax, vcenter=vcenter, norm=norm)

        # Label order:
        self.are_axes_swapped = False
        self.categories_order = categories_order
        self.var_names_idx_order = None

        # For creating extra plots:
        self.group_extra_size = 0
        self.plot_group_extra = None
        # Instantiate plotting variables- ax_dict will contain a dictionary of axes used in the plot:
        self.fig = None
        self.ax_dict = None
        self.ax = ax

    def swap_axes(self):
        """
        Modifies variables to flip x- and y-axes of dotplot.

        By default, the x axis contains 'var_names' (e.g. genes) and the y axis the groupby categories. By setting
        'swap_axes' the x-axis becomes the categories and the y-axis becomes the variable names.
        """
        self.default_category_height, self.default_category_width = (
            self.default_category_width,
            self.default_category_height,
        )

        self.are_axes_swapped = True
        return self

    def add_dendrogram(self, show: bool = True, dendrogram_key: Union[None, str] = None, size: float = 0.8):
        """
        Show dendrogram based on the hierarchical clustering between the `cat_key` categories. Categories are
        reordered to match the dendrogram order.

        The dendrogram information is computed using :func:`utils.dendrogram` within Spateo.
        If `utils.dendrogram` has not been called previously the function is called with default parameters here.

        The dendrogram is by default shown on the right side of the plot or on top if the axes are swapped.

        Args:
            show: Boolean to turn on (True) or off (False) 'add_dendrogram'
            dendrogram_key: Needed if :func `utils.dendrogram` saved the dendrogram using a key different than the
                default name.
            size: Size of the dendrogram. Corresponds to width when dendrogram shown on the right of the plot,
            or height when shown on top. The unit is the same as in matplotlib (inches).
        """
        if not show:
            self.plot_group_extra = None
            return self

        if self.cat_key is None or len(self.categories) <= 2:
            # dendrogram can only be computed  between groupby categories
            self.logger.warning(
                "Too few categories for dendrogram. Dendrogram is added only when the number of categories to plot > 2"
            )
            return self

        self.group_extra_size = size

        # To correctly plot dataframe, categories need to be reordered according to the dendrogram ordering:
        self.reorder_categories_after_dendrogram(dendrogram_key)

        # So that dendrogram "spines" are aligned with dotplot labels:
        dendro_ticks = np.arange(len(self.categories)) + 0.5

        self.group_extra_size = size
        self.plot_group_extra = {
            "kind": "dendrogram",
            "width": size,
            "dendrogram_key": dendrogram_key,
            "dendrogram_ticks": dendro_ticks,
        }
        return self

    # To modify the style of the plot:
    def style(
        self,
        cmap: str = default_colormap,
        color_on: Optional[Literal["dot", "square"]] = default_color_on,
        dot_max: Optional[float] = default_dot_max,
        dot_min: Optional[float] = default_dot_min,
        smallest_dot: Optional[float] = default_smallest_dot,
        largest_dot: Optional[float] = default_largest_dot,
        dot_edge_color: Optional[float] = default_dot_edgecolor,
        dot_edge_lw: Optional[float] = default_dot_edgelw,
        size_exponent: Optional[float] = default_size_exponent,
        grid: Optional[float] = False,
        x_padding: Optional[float] = default_plot_x_padding,
        y_padding: Optional[float] = default_plot_y_padding,
    ):
        """
        Modifies visual aspects of the dot plot

        Args:
        cmap: Name of Matplotlib color map to use
        color_on: Options are 'dot' or 'square'. By default the colormap is applied to the color of the dot,
            but 'square' changes this to be applied to a square region behind the dot, in which case the dot becomes
            transparent with only the edge of the dot shown.
        dot_max: If none, the maximum dot size is set to the maximum fraction value found (e.g. 0.6). If given,
            the value should be a number between 0 and 1. All fractions larger than dot_max are clipped to this value.
        dot_min: If none, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value.
        smallest_dot: If none, the smallest dot has size 0. All expression fractions with `dot_min` are plotted with this size.
        largest_dot: If none, the largest dot has size 200. All expression fractions with `dot_max` are plotted with this size.
        dot_edge_color: Only used if 'color_on' is 'square'. Sets dot edge color
        dot_edge_lw: Only used if 'color_on' is 'square'. Sets dot edge line width
        size_exponent: Dot size is computed as:
                fraction  ** size exponent
            and is afterwards scaled to match the 'smallest_dot' and 'largest_dot' size parameters.
            Using a different size exponent changes the relative sizes of the dots to each other.
        grid: Set to true to show grid lines. By default grid lines are not shown. Further configuration of the grid
            lines can be achieved directly on the returned ax.
        x_padding: Space between the plot left/right borders and the dots center. A unit is the distance between the x
            ticks. Only applied when 'color_on' = 'dot'
        y_padding: Space between the plot top/bottom borders and the dots center. A unit is the distance between the x
            ticks. Only applied when 'color_on' = 'dot'

        Returns:
             self (instance of class DotPlot)

        Example:
            Creating a modified dot plot (w/ a loaded AnnData object given name 'adata'):
            markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
            st.pl.DotPlot(adata, var_names=markers, cat_key='Celltype').style(cmap='RdBu_r', color_on='square').show()
        """

        # All variables initialized to their default value, check if any of them were selected to change by the user:
        if cmap != self.cmap:
            self.cmap = cmap
        if dot_max != self.dot_max:
            self.dot_max = dot_max
        if dot_min != self.dot_min:
            self.dot_min = dot_min
        if smallest_dot != self.smallest_dot:
            self.smallest_dot = smallest_dot
        if largest_dot != self.largest_dot:
            self.largest_dot = largest_dot
        if color_on != self.color_on:
            self.color_on = color_on
        if size_exponent != self.size_exponent:
            self.size_exponent = size_exponent
        if dot_edge_color != self.dot_edge_color:
            self.dot_edge_color = dot_edge_color
        if dot_edge_lw != self.dot_edge_lw:
            self.dot_edge_lw = dot_edge_lw
        if grid != self.grid:
            self.grid = grid
        if x_padding != self.plot_x_padding:
            self.plot_x_padding = x_padding
        if y_padding != self.plot_y_padding:
            self.plot_y_padding = y_padding

        return self

    # Working with the plot legends:
    def legend(
        self,
        show: bool = True,
        show_size_legend: bool = True,
        show_colorbar: bool = True,
        size_title: Optional[str] = default_size_legend_title,
        colorbar_title: Optional[str] = default_color_legend_title,
        base: Optional[int] = default_base,
        num_colorbar_ticks: Optional[int] = default_num_colorbar_ticks,
        num_size_legend_dots: Optional[int] = default_num_size_legend_dots,
        width: Optional[float] = default_legends_width,
    ):
        """
        Configures colorbar and other legends for dotplot

        Args:
            show: Set to `False` to hide the default plot of the legends. This sets the legend width to zero,
                which will result in a wider main plot.
            show_size_legend: Set to `False` to hide the dot size legend
            show_colorbar: Set to `False` to hide the colorbar legend
            size_title:  Title for the dot size legend. Use '\\n' to add line breaks. Will be shown at the top of
                the dot size legend box
            colorbar_title: Title for the color bar. Use '\\n' to add line breaks. Will be shown at the top of the
                color bar.
            base: To determine the size of each "benchmark" dot in the size legend, will use a logscale; this parameter
                sets the base of that scale.
            num_colorbar_ticks: Number of ticks for the colorbar
            num_size_legend_dots: Number of "benchmark" dots to include in the dot size legend
            width: Width of the legends area. The unit is the same as in matplotlib (inches)

        Returns:
             self (instance of class DotPlot)

        Example:
            Setting the colorbar title (w/ a loaded AnnData object given name 'adata'):
            markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
            dp = st.pl.DotPlot(adata, markers, groupby='Celltype')
            dp.legend(colorbar_title='log(UMI counts + 1)').show()
        """

        if not show:
            # Turn off legends by setting width to 0
            self.legends_width = 0
        else:
            self.color_legend_title = colorbar_title
            self.size_title = size_title
            self.base = base
            self.num_colorbar_ticks = num_colorbar_ticks
            self.num_size_legend_dots = num_size_legend_dots
            self.legends_width = width
            self.show_size_legend = show_size_legend
            self.show_colorbar = show_colorbar

        return self

    def get_axes(self):
        if self.ax_dict is None:
            self.make_figure()
        return self.ax_dict

    def _plot_size_legend(self, size_legend_ax: mpl.axes.Axes):
        """
        Given axis object, generates dot size legend and displays on plot

        For the dot size "benchmarks" on the legend, adjust the difference in size between consecutive benchmarks
        based on how different 'self.dot_max' and 'self.dot_min' are.
        """
        diff = self.dot_max - self.dot_min
        if 0.3 < diff <= 0.6:
            step = 0.1
        elif diff <= 0.3:
            step = 0.05
        else:
            step = 0.2

        # Want the max size to be part of the legend- min size doesn't matter as much (and it's often going to be
        # zero anyways)- so set size scale to be inverted:
        size_range = np.arange(self.dot_max, self.dot_min, step * -1)[::-1]
        if self.dot_min != 0 or self.dot_max != 1:
            dot_range = self.dot_max - self.dot_min
            size_values = (size_range - self.dot_min) / dot_range
        else:
            size_values = size_range

        size = size_values**self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # Plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = ["{}".format(np.round((x * 100), decimals=0).astype(int)) for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize=8)

        # Remove y ticks and labels
        size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

        # Remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size=6)

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    def _plot_colorbar(self, color_legend_ax: mpl.axes.Axes, normalize: Union[None, mpl.colors.Normalize] = None):
        """
        Given axis object, plots a horizontal colorbar

        Args:
            color_legend_ax : `mpl.axes.Axes` object
                Matplotlib axis object to plot onto
            normalize : `mpl.colors.Normalize` object
                The normalizing object that scales data, typically into the interval [0, 1], for the purposes of
                mapping to color intensities for plotting. If None, norm defaults to a colors.Normalize object and
                automatically scales based on min/max values in the data.
        """

        cmap = plt.get_cmap(self.cmap)

        ColorbarBase(
            color_legend_ax,
            orientation="horizontal",
            cmap=cmap,
            norm=normalize,
            ticks=np.linspace(
                np.nanmin(self.dot_color_df.values),
                np.nanmax(self.dot_color_df.values),
                self.default_num_colorbar_ticks,
            ),
            format="%.2f",
        )

        color_legend_ax.set_title(self.color_legend_title, fontsize=7)
        color_legend_ax.xaxis.set_tick_params(labelsize=7)

    def _plot_legend(
        self, legend_ax: mpl.axes.Axes, return_ax_dict: dict, normalize: Union[None, mpl.colors.Normalize] = None
    ):
        """
        Organizes the size legend and color legend.

        The structure for the legends is:
        First row: Empty space of variable size to control the size of the other rows
        Second row: Dot size legend
        Third row: Spacer to prevent titles/labels of the color and dot size legends overlapping
        Fourth row: Colorbar

        Args:
            legend_ax : mpl.axes.Axes
                Matplotlib axis object to plot onto
            return_ax_dict :
        """

        cbar_legend_height = self.min_figure_height * 0.08
        size_legend_height = self.min_figure_height * 0.27
        spacer_height = self.min_figure_height * 0.3

        height_ratios = [
            self.height - size_legend_height - cbar_legend_height - spacer_height,
            size_legend_height,
            spacer_height,
            cbar_legend_height,
        ]
        fig, legend_gs = make_grid_spec(legend_ax, nrows=4, ncols=1, height_ratios=height_ratios)

        if self.show_size_legend:
            size_legend_ax = fig.add_subplot(legend_gs[1])
            self._plot_size_legend(size_legend_ax)
            return_ax_dict["size_legend_ax"] = size_legend_ax

        if self.show_colorbar:
            color_legend_ax = fig.add_subplot(legend_gs[3])

            self._plot_colorbar(color_legend_ax, normalize)
            return_ax_dict["color_legend_ax"] = color_legend_ax

    # Working with the main body of the plot:
    def _mainplot(self, ax: mpl.axes.Axes):
        # Work on a copy of the dataframes. This is to avoid changes on the original data frames after repetitive
        # calls to the DotPlot object.
        _color_df = self.dot_color_df.copy()
        _size_df = self.dot_size_df.copy()

        if self.var_names_idx_order is not None:
            _color_df = _color_df.iloc[:, self.var_names_idx_order]
            _size_df = _size_df.iloc[:, self.var_names_idx_order]

        if self.categories_order is not None:
            _color_df = _color_df.loc[self.categories_order, :]
            _size_df = _size_df.loc[self.categories_order, :]

        if self.are_axes_swapped:
            _size_df = _size_df.T
            _color_df = _color_df.T
        self.cmap = self.kwargs.get("cmap", self.cmap)
        if "cmap" in self.kwargs:
            del self.kwargs["cmap"]

        # Put dotplot together!
        normalize, dot_min, dot_max = self._dotplot(
            _size_df,
            _color_df,
            ax,
            cmap=self.cmap,
            dot_max=self.dot_max,
            dot_min=self.dot_min,
            color_on=self.color_on,
            edge_color=self.dot_edge_color,
            edge_lw=self.dot_edge_lw,
            smallest_dot=self.smallest_dot,
            largest_dot=self.largest_dot,
            size_exponent=self.size_exponent,
            grid=self.grid,
            x_padding=self.plot_x_padding,
            y_padding=self.plot_y_padding,
            vmin=self.vboundnorm.vmin,
            vmax=self.vboundnorm.vmax,
            vcenter=self.vboundnorm.vcenter,
            norm=self.vboundnorm.norm,
            **self.kwargs,
        )

        self.dot_min, self.dot_max = dot_min, dot_max
        return normalize

    @staticmethod
    def _dotplot(
        dot_size: pd.DataFrame,
        dot_color: pd.DataFrame,
        dot_ax: mpl.axes.Axes,
        cmap: str = "Reds",
        color_on: str = "dot",
        y_label: Union[None, str] = None,
        dot_max: Union[None, float] = None,
        dot_min: Union[None, float] = None,
        standard_scale: Union[None, Literal["var", "group"]] = None,
        smallest_dot: float = 0.0,
        largest_dot: float = 200,
        size_exponent: float = 2,
        edge_color: Union[None, str] = None,
        edge_lw: Union[None, float] = None,
        grid: bool = False,
        x_padding: float = 0.8,
        y_padding: float = 1.0,
        vmin: Union[None, float] = None,
        vmax: Union[None, float] = None,
        vcenter: Union[None, float] = None,
        norm: Union[None, Normalize] = None,
        **kwargs,
    ):
        """
        Generate a dotplot given the axis object and two dataframes containing the dot size and dot color. The
        indices and columns of the dataframes are used to label the resultant image.

        The dots are plotted using :func:`matplotlib.pyplot.scatter()`. Thus, additional
        arguments can be passed.

        Args:
            dot_size : pd.DataFrame
                Data frame containing the dot_size.
            dot_color : pd.DataFrame
                Data frame containing the dot_color, should have the same shape, columns and indices as dot_size.
            dot_ax : matplotlib Axes object
                Axis to plot figure onto
            cmap : str, default 'Reds'
                String denoting matplotlib color map
            color_on : str, default 'dot'
                Options: 'dot' or 'square'. By default the colormap is applied to the color of the dot. Optionally,
                the colormap can be applied to an square behind the dot, in which case the dot is transparent and only
                the edge is shown.
            y_label : optional str
                Label for y-axis
            dot_max : optional float
                If none, the maximum dot size is set to the maximum fraction value found (e.g. 0.6). If given,
                the value should be a number between 0 and 1. All fractions larger than dot_max are clipped to this value.
            dot_min : optional float
                If none, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1.
                All fractions smaller than dot_min are clipped to this value.
            standard_scale : 'None', 'val', or 'group'
                Whether or not to standardize that dimension between 0 and 1, meaning for each variable or group,
                subtract the minimum and divide each by its maximum. 'val' or 'group' is used to specify whether this
                should be done over variables or groups.
            smallest_dot : optional float
                If none, the smallest dot has size 0. All expression fractions with `dot_min` are plotted with this size.
            largest_dot : optional float
                If none, the largest dot has size 200. All expression fractions with `dot_max` are plotted with this size.
            size_exponent : float, default 1.5
                Dot size is computed as:
                    fraction  ** size exponent
                and is afterwards scaled to match the 'smallest_dot' and 'largest_dot' size parameters.
                Using a different size exponent changes the relative sizes of the dots to each other.
            edge_color : str, default 'black'
                Only used if 'color_on' is 'square'. Sets dot edge color
            edge_lw : float, default 0.2
                Only used if 'color_on' is 'square'. Sets dot edge line width
            grid : bool, default False
                Set to true to show grid lines. By default grid lines are not shown. Further configuration of the grid
                lines can be achieved directly on the returned ax.
            x_padding : float, default 0.8
                Space between the plot left/right borders and the dots center. A unit is the distance between the x
                ticks. Only applied when 'color_on' = 'dot'
            y_padding : float, default 1.0
                Space between the plot top/bottom borders and the dots center. A unit is the distance between the x
                ticks. Only applied when 'color_on' = 'dot'
            vmin : optional float
                The data value that defines 0.0 in the normalization. Defaults to the min value of the dataset.
            vmax : optional float
                The data value that defines 1.0 in the normalization. Defaults to the the max value of the dataset.
            vcenter : optional float
                The data value that defines 0.5 in the normalization
            norm : optional `matplotlib.colors.Normalize` object
                Optional already-initialized normalizing object that scales data, typically into the interval [0, 1],
                for the purposes of mapping to color intensities for plotting. Do not pass both 'norm' and
                'vmin'/'vmax', etc.
            **kwargs :
                Additional arguments passed to `matplotlib.pyplot.scatter`

        Returns:
            normalize : `matplotlib.colors.Normalize` object
                The normalizing object that scales data, typically into the interval [0, 1], for the purposes of
                mapping to color intensities for plotting.
            dot_min : float
                The minimum dot size represented on the plot, given as a fration of the maximum value in the data
            dot_max : float
                The maximum dot size represented on the plot, given as a fraction of the maximum value in the data
        """

        logger = lm.get_main_logger()

        if dot_size.shape != dot_color.shape:
            logger.error("Dot size and dot color dataframes are not the same size.")

        if list(dot_size.index) != list(dot_color.index):
            logger.error("Dot size and dot color dataframes do not have the same features.")

        if list(dot_size.columns) != list(dot_color.columns):
            logger.error("Dot size and dot color dataframes do not have the same categories.")

        if standard_scale == "group":
            dot_color = dot_color.sub(dot_color.min(1), axis=0)
            dot_color = dot_color.div(dot_color.max(1), axis=0).fillna(0)
        elif standard_scale == "var":
            dot_color -= dot_color.min(0)
            dot_color = (dot_color / dot_color.max(0)).fillna(0)
        elif standard_scale is None:
            pass

        # Set the center of each first dot at 0.5 to more easily line up dotplot w/ possible dendrograms:
        y, x = np.indices(dot_color.shape)
        y = y.flatten() + 0.5
        x = x.flatten() + 0.5
        frac = dot_size.values.flatten()
        mean_flat = dot_color.values.flatten()

        cmap = plt.get_cmap(kwargs.get("cmap", cmap))
        if "cmap" in kwargs:
            del kwargs["cmap"]
        if dot_max is None:
            dot_max = np.ceil(max(frac) * 10) / 10
        else:
            if dot_max < 0 or dot_max > 1:
                raise ValueError("`dot_max` value has to be between 0 and 1")
        if dot_min is None:
            dot_min = 0
        else:
            if dot_min < 0 or dot_min > 1:
                raise ValueError("`dot_min` value has to be between 0 and 1")

        if dot_min != 0 or dot_max != 1:
            # clip frac between dot_min and  dot_max
            frac = np.clip(frac, dot_min, dot_max)
            old_range = dot_max - dot_min
            # re-scale frac between 0 and 1
            frac = (frac - dot_min) / old_range

        size = frac**size_exponent
        # rescale size to match smallest_dot and largest_dot
        size = size * (largest_dot - smallest_dot) + smallest_dot
        normalize = check_colornorm(vmin, vmax, vcenter, norm)

        if color_on == "square":
            if edge_color is None:
                from seaborn.utils import relative_luminance

                # Use either black or white for the edge color depending on the luminance of the background
                # square color
                edge_color = []
                for color_value in cmap(normalize(mean_flat)):
                    lum = relative_luminance(color_value)
                    edge_color.append(".15" if lum > 0.408 else "w")

            edge_lw = 1.5 if edge_lw is None else edge_lw

            # Create heatmap with squares, then create circles and plot them over the top:
            dot_ax.pcolor(dot_color.values, cmap=cmap, norm=normalize)
            for axis in ["top", "bottom", "left", "right"]:
                dot_ax.spines[axis].set_linewidth(1.5)
            # A few created variables will be used as keyword args to ax.scatter...ensure that they aren't already
            # given as keyword args to this function:
            kwargs = deduplicate_kwargs(
                kwargs,
                s=size,
                cmap=cmap,
                linewidth=edge_lw,
                facecolor="none",
                edgecolor=edge_color,
                norm=normalize,
            )
            dot_ax.scatter(x, y, **kwargs)

        else:
            edge_color = "none" if edge_color is None else edge_color
            edge_lw = 0.0 if edge_lw is None else edge_lw

            color = cmap(normalize(mean_flat))
            # A few created variables will be used as keyword args to ax.scatter...ensure that they aren't already
            # given as keyword args to this function:
            kwargs = deduplicate_kwargs(
                kwargs,
                s=size,
                cmap=cmap,
                color=color,
                linewidth=edge_lw,
                edgecolor=edge_color,
                norm=normalize,
            )
            dot_ax.scatter(x, y, **kwargs)

        y_ticks = np.arange(dot_color.shape[0]) + 0.5
        dot_ax.set_yticks(y_ticks)
        dot_ax.set_yticklabels([dot_color.index[idx] for idx, _ in enumerate(y_ticks)], minor=False)

        x_ticks = np.arange(dot_color.shape[1]) + 0.5
        dot_ax.set_xticks(x_ticks)
        dot_ax.set_xticklabels(
            [dot_color.columns[idx] for idx, _ in enumerate(x_ticks)],
            rotation=90,
            ha="center",
            minor=False,
        )
        dot_ax.tick_params(axis="both", labelsize=6)
        dot_ax.grid(False)
        dot_ax.set_ylabel(y_label)

        # To be consistent with the heatmap plot, is better to invert the order of the y-axis, such that the first
        # group is on top
        dot_ax.set_ylim(dot_color.shape[0], 0)
        dot_ax.set_xlim(0, dot_color.shape[1])

        if color_on == "dot":
            # Add padding to the x and y lims when the color is not in the square
            # Default y range goes from 0.5 to num cols + 0.5 and default x range goes from 0.5 to num rows + 0.5
            x_padding = x_padding - 0.5
            y_padding = y_padding - 0.5
            dot_ax.set_ylim(dot_color.shape[0] + y_padding, -y_padding)

            dot_ax.set_xlim(-x_padding, dot_color.shape[1] + x_padding)

        if grid:
            dot_ax.grid(True, color="lightgray", linewidth=0.1)
            dot_ax.set_axisbelow(True)

        return normalize, dot_min, dot_max

    def reorder_categories_after_dendrogram(self, dendrogram_key):
        """
        Reorders categorical observations along plot axis based on dendrogram results.

        The function checks if a dendrogram has already been precomputed. If not, `utils.dendrogram` is run with
        default parameters.

        The results found in `.uns[dendrogram_key]` are used to reorder `var_group_labels` and `var_group_positions`.
        """

        def _format_first_three_categories(_categories):
            """used to clean up warning message"""
            _categories = list(_categories)
            if len(_categories) > 3:
                _categories = _categories[:3] + ["etc."]
            return ", ".join(_categories)

        # Get dendrogram key:
        if not isinstance(dendrogram_key, str):
            if isinstance(self.cat_key, str):
                dendrogram_key = f"dendrogram_{self.cat_key}"
            elif isinstance(self.cat_key, list):
                dendrogram_key = f'dendrogram_{"_".join(self.cat_key)}'

        if dendrogram_key not in self.adata.uns:
            from .utils import dendrogram

            self.logger.warning(
                f"Dendrogram data not found (using key={dendrogram_key}). Running :func `st.pl.dendrogram` with "
                f"default parameters. For fine tuning it is recommended to run `st.pl.dendrogram` independently."
            )
            dendrogram(self.adata, self.cat_key, key_added=dendrogram_key)

        if "dendrogram_info" not in self.adata.uns[dendrogram_key]:
            raise ValueError(
                f"The given dendrogram key ({dendrogram_key!r}) does not contain valid dendrogram information."
            )

        dendro_info = self.adata.uns[dendrogram_key]
        if self.cat_key != dendro_info["cat_key"]:
            raise ValueError(
                "Incompatible observations. The precomputed dendrogram contains information for the "
                f"observation: '{self.cat_key}' while the plot is made for the observation: '{dendro_info['cat_key']}. "
                "Please run :func `st.pl.dendrogram` using the right observation.'"
            )

        # Category order:
        categories_idx_ordered = dendro_info["categories_idx_ordered"]
        categories_ordered = dendro_info["categories_ordered"]

        if len(self.categories) != len(categories_idx_ordered):
            raise ValueError(
                f"Incompatible observations. Dendrogram data has {len(categories_idx_ordered)} categories but current "
                f"groupby observation {self.cat_key} contains {len(self.categories)} categories. Most likely the "
                "underlying groupby observation changed after the initial computation of :func `st.pl.dendrogram`. "
                "Please run `st.pl.dendrogram` again.'"
            )

        # Reorder var_groups (if any)
        if self.var_names is not None:
            var_names_idx_ordered = list(range(len(self.var_names)))

        if self.has_var_groups:
            if set(self.var_group_labels) == set(self.categories):
                positions_ordered = []
                labels_ordered = []
                position_start = 0
                var_names_idx_ordered = []
                for cat_name in categories_ordered:
                    idx = self.var_group_labels.index(cat_name)
                    position = self.var_group_positions[idx]
                    _var_names = self.var_names[position[0] : position[1] + 1]
                    var_names_idx_ordered.extend(range(position[0], position[1] + 1))
                    positions_ordered.append((position_start, position_start + len(_var_names) - 1))
                    position_start += len(_var_names)
                    labels_ordered.append(self.var_group_labels[idx])
                self.var_group_labels = labels_ordered
                self.var_group_positions = positions_ordered
            else:
                self.logger.warning(
                    "Groups are not reordered because the `groupby` categories and the `var_group_labels` are "
                    f"different.\n"
                    f"categories: {_format_first_three_categories(self.categories)}\n"
                    "var_group_labels: "
                    f"{_format_first_three_categories(self.var_group_labels)}"
                )

        if var_names_idx_ordered is not None:
            var_names_ordered = [self.var_names[x] for x in var_names_idx_ordered]
        else:
            var_names_ordered = None

        self.categories_idx_ordered = categories_idx_ordered
        self.categories_order = dendro_info["categories_ordered"]
        self.var_names_idx_order = var_names_idx_ordered
        self.var_names_ordered = var_names_ordered

    @staticmethod
    def _plot_var_groups_brackets(
        gene_groups_ax: mpl.axes.Axes,
        group_positions: Iterable[Tuple[int, int]],
        group_labels: Sequence[str],
        left_adjustment: float = -0.3,
        right_adjustment: float = 0.3,
        rotation: Optional[float] = None,
        orientation: Literal["top", "right"] = "top",
    ):
        """
        Draws brackets that represent groups of features on the given axis.

        The 'gene_groups_ax' Axes object should share the x-axis/y-axis (depending on the axis along which the
        features are plotted) with the main plot axis. For example, in instantiation:
        gene_groups_ax = fig.add_subplot(axs[0,0], sharex=dot_ax)

        Args:
            gene_groups_ax : `matplotlib.axes.Axes` object
                Axis to plot on, should correspond to the axis of the main plot on which the feature names/feature
                ticks are drawn
            group_positions : list of tuples of form (int, int)
                Each item in the list, should contain the start and end position that the bracket should cover.
                Eg. [(0, 4), (5, 8)] means that there are two brackets, one for the var_names (eg genes) in
                positions 0-4 and the second for positions 5-8.
            group_labels : list of str
                List of labels for the feature groups
            left_adjustment : float, default -0.3
                Adjustment to plot the bracket start slightly before or after the first feature position.
                If the value is negative the start is moved before.
            right_adjustment : float, default 0.3
                Adjustment to plot the bracket end slightly before or after the first feature position.
                If the value is negative the end is moved before, if positive the end is moved after.
            rotation : optional float
                In degrees, angle of rotation for the labels. If not given, small labels (<4 characters) are not
                rotated, otherwise, they are rotated 90 degrees
            orientation : str
                Options: 'top' or 'right' to set the location of the brackets
        """

        # Get the 'brackets' coordinates as lists of start and end positions
        left = [x[0] + left_adjustment for x in group_positions]
        right = [x[1] + right_adjustment for x in group_positions]

        # verts and codes are used by PathPatch to make the brackets
        verts = []
        codes = []
        # If brackets are specified to be placed along the top of the figure:
        if orientation == "top":
            # If custom rotation is not specified, rotate labels if any of them is longer than 4 characters
            if rotation is None and group_labels:
                if max([len(x) for x in group_labels]) > 4:
                    rotation = 90
                else:
                    rotation = 0
            for idx, (left_coor, right_coor) in enumerate(zip(left, right)):
                verts.append((left_coor, 0))  # lower-left
                verts.append((left_coor, 0.6))  # upper-left
                verts.append((right_coor, 0.6))  # upper-right
                verts.append((right_coor, 0))  # lower-right

                codes.append(Path.MOVETO)
                codes.append(Path.LINETO)
                codes.append(Path.LINETO)
                codes.append(Path.LINETO)

                group_x_center = left[idx] + float(right[idx] - left[idx]) / 2
                gene_groups_ax.text(
                    group_x_center,
                    1.1,
                    group_labels[idx],
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                )
        # Else, the brackets will be placed to the right of the figure:
        else:
            top = left
            bottom = right
            for idx, (top_coor, bottom_coor) in enumerate(zip(top, bottom)):
                verts.append((0, top_coor))  # upper-left
                verts.append((0.4, top_coor))  # upper-right
                verts.append((0.4, bottom_coor))  # lower-right
                verts.append((0, bottom_coor))  # lower-left

                codes.append(Path.MOVETO)
                codes.append(Path.LINETO)
                codes.append(Path.LINETO)
                codes.append(Path.LINETO)

                diff = bottom[idx] - top[idx]
                group_y_center = top[idx] + float(diff) / 2
                if diff * 2 < len(group_labels[idx]):
                    # cut label to fit available space
                    group_labels[idx] = group_labels[idx][: int(diff * 2)] + "."
                gene_groups_ax.text(
                    1.1,
                    group_y_center,
                    group_labels[idx],
                    ha="right",
                    va="center",
                    rotation=270,
                    fontsize=8,
                )

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", lw=1.5)

        gene_groups_ax.add_patch(patch)
        gene_groups_ax.grid(False)
        gene_groups_ax.axis("off")
        # Remove all ticks from the bracket plot Axes object:
        gene_groups_ax.tick_params(axis="y", left=False, labelleft=False)
        gene_groups_ax.tick_params(axis="x", bottom=False, labelbottom=False, labeltop=False)

    def _update_var_groups(self):
        """
        Checks if var_names is a dict. Is this is the cases, then set the
        correct values for var_group_labels and var_group_positions

        Updates var_names, var_group_labels, var_group_positions
        """
        if isinstance(self.var_names, cabc.Mapping):
            if self.has_var_groups:
                self.logger.warning(
                    "Given `var_names` is a dictionary. This will reset the current values of `var_group_labels` "
                    "and `var_group_positions`."
                )
            var_group_labels = []
            _var_names = []
            var_group_positions = []

            start = 0
            for label, vars_list in self.var_names.items():
                if isinstance(vars_list, str):
                    vars_list = [vars_list]
                # use list() in case var_list is a numpy array or pandas series
                _var_names.extend(list(vars_list))
                var_group_labels.append(label)
                var_group_positions.append((start, start + len(vars_list) - 1))
                start += len(vars_list)
            self.var_names = _var_names
            self.var_group_labels = var_group_labels
            self.var_group_positions = var_group_positions
            self.has_var_groups = True

        elif isinstance(self.var_names, str):
            self.var_names = [self.var_names]

    def make_figure(self):
        """Renders the image, but does not call :func:`matplotlib.pyplot.show`."""
        category_height = self.default_category_height
        category_width = self.default_category_width

        if self.height is None:
            mainplot_height = len(self.categories) * category_height
            mainplot_width = len(self.var_names) * category_width + self.group_extra_size
            if self.are_axes_swapped:
                mainplot_height, mainplot_width = mainplot_width, mainplot_height

            height = mainplot_height + 1  # +1 to make room for labels

            # If the number of categories is small use a larger height, otherwise the legends do not fit
            self.height = max([self.min_figure_height, height])
            self.width = mainplot_width + self.legends_width
        else:
            self.min_figure_height = self.height
            mainplot_height = self.height

            mainplot_width = self.width - (self.legends_width + self.group_extra_size)

        return_ax_dict = {}
        # Define a layout of 1 rows x 2 columns:
        # First ax is for the main figure.
        # Second ax is to plot legends
        legends_width_spacer = 0.7 / self.width

        self.fig, gs = make_grid_spec(
            self.ax or (self.width, self.height),
            nrows=1,
            ncols=2,
            wspace=legends_width_spacer,
            width_ratios=[mainplot_width + self.group_extra_size, self.legends_width],
        )

        # Add some space in case brackets will be used to group categories:
        if self.has_var_groups:
            if self.are_axes_swapped:
                var_groups_height = category_height
            else:
                var_groups_height = category_height / 2

        else:
            var_groups_height = 0

        mainplot_width = mainplot_width - self.group_extra_size
        spacer_height = self.height - var_groups_height - mainplot_height
        if not self.are_axes_swapped:
            height_ratios = [spacer_height, var_groups_height, mainplot_height]
            width_ratios = [mainplot_width, self.group_extra_size]
        else:
            height_ratios = [spacer_height, self.group_extra_size, mainplot_height]
            width_ratios = [mainplot_width, var_groups_height]

        if self.fig_title is not None and self.fig_title.strip() != "":
            # For the figure title use the ax that contains all the main graphical elements (main plot, dendrogram etc);
            # title will thus be centered over the main plot:
            _ax = self.fig.add_subplot(gs[0, 0])
            _ax.axis("off")
            ymax = _ax.get_ylim()[1]
            if self.figsize[1] > 4 and self.figsize[1] < 8:
                offset = 0.15
            elif self.figsize[1] <= 4:
                offset = 0.35
            elif self.figsize[1] >= 8:
                offset = 0.1
            _ax.set_title(self.fig_title, y=ymax + offset)

        # The main plot is divided into three rows and two columns:
        # The first row is a spacer that is adjusted in case the legends need more height than the main plot.
        # The second row is for potential brackets
        # The third row is for the mainplot along with any potential extra plots (dendrogram/totals)
        mainplot_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=3,
            ncols=2,
            wspace=self.wspace,
            hspace=0.0,
            subplot_spec=gs[0, 0],
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        main_ax = self.fig.add_subplot(mainplot_gs[2, 0])
        return_ax_dict["mainplot_ax"] = main_ax

        if not self.are_axes_swapped:
            if self.plot_group_extra is not None:
                group_extra_ax = self.fig.add_subplot(mainplot_gs[2, 1], sharey=main_ax)
                group_extra_orientation = "right"
            if self.has_var_groups:
                gene_groups_ax = self.fig.add_subplot(mainplot_gs[1, 0], sharex=main_ax)
                var_group_orientation = "top"
        else:
            if self.plot_group_extra:
                group_extra_ax = self.fig.add_subplot(mainplot_gs[1, 0], sharex=main_ax)
                group_extra_orientation = "top"
            if self.has_var_groups:
                gene_groups_ax = self.fig.add_subplot(mainplot_gs[2, 1], sharey=main_ax)
                var_group_orientation = "right"

        if self.plot_group_extra is not None:
            if self.plot_group_extra["kind"] == "dendrogram":
                plot_dendrogram(
                    group_extra_ax,
                    self.adata,
                    self.cat_key,
                    dendrogram_key=self.plot_group_extra["dendrogram_key"],
                    ticks=self.plot_group_extra["dendrogram_ticks"],
                    orientation=group_extra_orientation,
                )

            return_ax_dict["group_extra_ax"] = group_extra_ax

        # Plot category group brackets atop the main ax (if given):
        if self.has_var_groups:
            self._plot_var_groups_brackets(
                gene_groups_ax,
                group_positions=self.var_group_positions,
                group_labels=self.var_group_labels,
                rotation=self.var_group_rotation,
                left_adjustment=0.2,
                right_adjustment=0.7,
                orientation=var_group_orientation,
            )
            return_ax_dict["gene_group_ax"] = gene_groups_ax

        # Create the dot plot:
        normalize = self._mainplot(ax=main_ax)

        # In case minor tick labels are present, delete them:
        main_ax.yaxis.set_tick_params(which="minor", left=False, right=False)
        main_ax.xaxis.set_tick_params(which="minor", top=False, bottom=False, length=0)
        main_ax.set_zorder(100)
        if self.legends_width > 0:
            legend_ax = self.fig.add_subplot(gs[0, 1])
            self._plot_legend(legend_ax, return_ax_dict, normalize)

        self.ax_dict = return_ax_dict


class CCDotplot(Dotplot):
    """
    Because of the often much smaller values dealt with in cell-cell communication inference, this class creates a
    modified legend.

    Args:
        delta : optional float
            Distance between the largest value to consider and the smallest value to consider (see 'minn'
            parameter below)
        minn : optional float
            For the dot size legend, sets the value corresponding to the smallest dot on the legend
        alpha : optional float
            Significance threshold. If given, all elements w/ p-values <= 'alpha' will be marked by rings instead of
            dots.
        *args :
            Positional arguments to initialize :class `Dotplot`
        **kwargs :
            Keyword arguments to initialize :class `Dotplot`
    """

    base = 10
    default_largest_dot = 50.0

    def __init__(self, minn: float, delta: float, alpha: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.minn = minn
        self.alpha = alpha
        self.largest_dot = self.default_largest_dot

    def _plot_size_legend(self, size_legend_ax: mpl.axes.Axes):
        """
        Given axis object, generates dot size legend and displays on plot

        Overwrites the default :func `plot_size_legend` for :class `Dotplot`
        """

        # Ending point:
        y = self.base ** -((self.dot_max * self.delta) + self.minn)
        # Starting point:
        x = self.base ** -((self.dot_min * self.delta) + self.minn)
        size_range = -(np.logspace(x, y, self.num_size_legend_dots + 1, base=10).astype(np.float64))
        size_range = (size_range - np.min(size_range)) / (np.max(size_range) - np.min(size_range))
        # no point in showing dot of size 0
        size_range = size_range[1:]

        # See documentation for 'style()'- matching the methodology for plotting the actual dots
        size = size_range**self.size_exponent
        mult = (self.largest_dot - self.smallest_dot) + self.smallest_dot
        size = size * mult

        # Plot size legend
        ymin = -self.largest_dot * 0.003
        ymax = 0.65

        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color="gray",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{(x * self.delta) + self.minn:.1f}" for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize=9)

        # Remove y ticks and labels
        size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

        # Remove surrounding lines
        size_legend_ax.spines["right"].set_visible(False)
        size_legend_ax.spines["top"].set_visible(False)
        size_legend_ax.spines["left"].set_visible(False)
        size_legend_ax.spines["bottom"].set_visible(False)
        size_legend_ax.grid(False)

        size_legend_ax.set_ylim(ymin, ymax)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.05, size=9)

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

        # If significance check is involved, a separate legend panel will be used to indicate significance w/
        # closed/open circles:
        if self.alpha is not None:
            # Attribute will be created/set to not-None upon calling 'make_figure()' during the process of creating the
            # outer plotting class
            ax = self.fig.add_subplot()
            ax.scatter(
                [0.35, 0.65],
                [0, 0],
                s=size[-1],
                color="black",
                edgecolor="black",
                linewidth=self.dot_edge_lw,
                zorder=100,
            )
            ax.scatter(
                [0.65], [0], s=0.33 * mult, color="white", edgecolor="black", linewidth=self.dot_edge_lw, zorder=100
            )
            ax.set_xlim([0, 1])
            ax.set_xticks([0.35, 0.65])
            ax.set_xticklabels(["false", "true"])
            ax.set_yticks([])
            ax.set_title(f"significant\n$p={self.alpha}$", y=ymax + 0.05, size=9)
            ax.set(frame_on=False)

            l, b, w, h = size_legend_ax.get_position().bounds
            ax.set_position([l, b + h + 0.2, w, h])


# --------------------------------------- Dotplot wrapper --------------------------------------- #
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def dotplot(
    adata: AnnData,
    var_names: Sequence[str],
    cat_key: Union[str, Sequence[str]],
    num_categories: int = 7,
    cell_cell_dp: bool = False,
    delta: Union[None, float] = None,
    minn: Union[None, float] = None,
    alpha: Union[None, float] = None,
    prescale_adata: bool = False,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    cmap: str = "Reds",
    dot_max: float = Dotplot.default_dot_max,
    dot_min: float = Dotplot.default_dot_min,
    standard_scale: Literal["var", "group"] = None,
    smallest_dot: float = Dotplot.default_smallest_dot,
    largest_dot: float = Dotplot.default_largest_dot,
    title: str = None,
    colorbar_title: str = Dotplot.default_color_legend_title,
    size_title: str = Dotplot.default_size_legend_title,
    figsize: Union[None, Tuple[float, float]] = None,
    dendrogram: Union[bool, str] = False,
    gene_symbols_key: Union[None, str] = None,
    var_group_positions: Union[None, Sequence[Tuple[int, int]]] = None,
    var_group_labels: Union[None, Sequence[str]] = None,
    var_group_rotation: Union[None, float] = None,
    layer: Union[None, str] = None,
    swap_axes: bool = False,
    dot_color_df: Union[None, pd.DataFrame] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
    save_kwargs: dict = {},
    ax: Union[None, mpl.axes.Axes] = None,
    vmin: Union[None, float] = None,
    vmax: Union[None, float] = None,
    vcenter: Union[None, float] = None,
    norm: Union[None, Normalize] = None,
    **kwargs,
):
    """
    Makes a dot plot of the expression values of `var_names`. For each var_name and each `groupby` category a dot
    is plotted.
    Each dot represents two values: mean expression within each category (visualized by color) and fraction of cells
    expressing the `var_name` in the category (visualized by the size of the dot). If `groupby` is not given,
    the dotplot assumes that all data belongs to a single category.

    Args:
        adata: object of class `anndata.AnnData`
        var_names: Should be a subset of adata.var_names
        cat_key: The key(s) in .obs of the grouping to consider. Should be a categorical observation; if not,
            will be subdivided into 'num_categories'.
        num_categories: Only used if groupby observation is not categorical. This value determines the number of
            groups into which the groupby observation should be subdivided.
        cell_cell_dp: Set True to initialize specialized cell-cell dotplot instead of gene expression dotplot
        delta: Only used if 'cell_cell_dp' is True- distance between the largest value to consider and the smallest
            value to consider (see 'minn' parameter below)
        minn: Only used if 'cell_cell_dp' is True- for the dot size legend, sets the value corresponding to the
            smallest dot on the legend
        alpha: Only used if 'cell_cell_dp' is True- significance threshold. If given, all elements w/ p-values <=
            'alpha' will be marked by rings instead of dots.
        prescale_adata: Set True to indicate that AnnData object should be scaled- if so, will use 'delta' and
            'minn' to do so. If False, will proceed as though adata has already been processed as needed.
        expression_cutoff: Used for binarizing feature expression- feature is considered to be expressed only if
            the expression value is greater than this threshold
        mean_only_expressed: If True, gene expression is averaged only over the cells expressing the given features
        cmap: Name of Matplotlib color map to use
        dot_max: If none, the maximum dot size is set to the maximum fraction value found (e.g. 0.6). If given,
            the value should be a number between 0 and 1. All fractions larger than dot_max are clipped to this value.
        dot_min: If none, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value.
        standard_scale: Whether or not to standardize that dimension between 0 and 1, meaning for each variable or
            group, subtract the minimum and divide each by its maximum. 'val' or 'group' is used to specify whether this
            should be done over variables or groups.
        smallest_dot: If None, the smallest dot has size 0. All expression fractions with `dot_min` are plotted with
            this size.
        largest_dot: If None, the largest dot has size 200. All expression fractions with `dot_max` are plotted with
            this size.
        title: Title for the entire plot
        colorbar_title: Title for the color legend. If None will use generic default title
        size_title: Title for the dot size legend. If None will use generic default title
        figsize: Sets width and height of figure window
        dendrogram: If True, adds dendrogram to plot. Will do the same thing if string is given here,
            but will recompute dendrogram and save using this argument to set key in .uns.
        gene_symbols_key: Key in .var containing gene symbols
        var_group_positions:  Each item in the list should contain the start and end position that the bracket
            should cover. Eg. [(0, 4), (5, 8)] means that there are two brackets, one for the var_names in positions
            0-4 and other for positions 5-8
        var_group_labels: List of group labels for the variable names (e.g. can group var_names in positions 0-4 as
            being "group A")
        var_group_rotation: Rotation in degrees of the variable name labels. If not given, small labels (<4
            characters) are not rotated, but otherwise labels are rotated 90 degrees.
        layer: Key in .layers specifying layer to use. If not given, will use .X.
        swap_axes: Set True to switch what is plotted on the x- and y-axes
        dot_color_df: Pre-prepared dataframe with features as indices, categories as columns, and indices
            corresponding to color intensities
        save_show_or_return: Options: "save", "show", "return", "both", "all"
                - "both" for save and show
        save_kwargs:  A dictionary that will passed to the save_fig function. By default it is an empty dictionary
            and the save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
            "transparent": True, "close": True, "verbose": True} as its parameters. But to change any of these
            parameters, this dictionary can be used to do so.
        ax: Pre-initialized axis object to plot on
        vmin: The data value that defines 0.0 in the normalization. Defaults to the min value of the dataset.
        vmax: The data value that defines 1.0 in the normalization. Defaults to the the max value of the dataset.
        vcenter: The data value that defines 0.5 in the normalization
        norm: Optional already-initialized normalizing object that scales data, typically into the interval [0, 1],
            for the purposes of mapping to color intensities for plotting. Do not pass both 'norm' and
            'vmin'/'vmax', etc.
        kwargs: Additional keyword arguments passed to :func:`matplotlib.pyplot.scatter`

    Returns:
        fig: Instantiated Figure object- only if 'return' is True
        axes: Instantiated Axes object- only if 'return' is True
    """

    if cell_cell_dp:
        dp = CCDotplot(
            adata,
            var_names,
            cat_key,
            delta=delta,
            minn=minn,
            alpha=alpha,
            prescale_adata=prescale_adata,
            num_categories=num_categories,
            expression_cutoff=expression_cutoff,
            mean_only_expressed=mean_only_expressed,
            standard_scale=standard_scale,
            title=title,
            figsize=figsize,
            gene_symbols_key=gene_symbols_key,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=var_group_rotation,
            layer=layer,
            dot_color_df=dot_color_df,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            norm=norm,
            **kwargs,
        )
    else:
        dp = Dotplot(
            adata,
            var_names,
            cat_key,
            prescale_adata=prescale_adata,
            num_categories=num_categories,
            expression_cutoff=expression_cutoff,
            mean_only_expressed=mean_only_expressed,
            standard_scale=standard_scale,
            title=title,
            figsize=figsize,
            gene_symbols_key=gene_symbols_key,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=var_group_rotation,
            layer=layer,
            dot_color_df=dot_color_df,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            norm=norm,
            **kwargs,
        )

    if dendrogram or isinstance(dendrogram, str):
        dp.add_dendrogram(dendrogram_key=dendrogram)
    if swap_axes:
        dp.swap_axes()

    dp = dp.style(
        cmap=cmap,
        dot_max=dot_max,
        dot_min=dot_min,
        smallest_dot=smallest_dot,
        largest_dot=largest_dot,
        dot_edge_lw=kwargs.pop("linewidth", Dotplot.default_dot_edgelw),
    ).legend(colorbar_title=colorbar_title, size_title=size_title)

    dp.make_figure()

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
