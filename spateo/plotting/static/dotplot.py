"""
Dotplot class adapted from https://github.com/scverse/scanpy
"""
from typing import List, Literal, Mapping, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from itertools import product
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pandas.api.types import is_numeric_dtype

from ...configuration import SKM, config_spateo_rcParams, set_pub_style
from ...logging import logger_manager as lm
from .utils import _get_array_values


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def adata_to_frame(
        adata: AnnData,
        var_names: Sequence[str],
        cat_key: Union[str, Sequence[str]],
        num_categories: int = 7,
        layer: Union[None, str] = None,
        gene_symbols_key: Union[None, str] = None
):
    """
    For the purposes of dot plotting, converts the information given in AnnData object to a dataframe in which the
    row names are categories defined by groups and column names correspond to variable names.

    Args:
        adata : class `anndata.AnnData`
        var_names : sequence of str
            Should be a subset of adata.var_names
        cat_key : str or sequence of str
            The key(s) in .obs of the grouping to consider. Should be a categorical observation; if not,
            will be subdivided into 'num_categories'.
        num_categories : int, default 7
            Only used if groupby observation is not categorical. This value determines the number of groups into
            which the groupby observation should be subdivided.
        layer : optional str
            Key in .layers specifying layer to use. If not given, will use .X.
        gene_symbols_key: optional str
            Key in .var containing gene symbols
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
                    msg = ''
                logger.error(
                    f"Grouping key cannot be found. Given: {group}, could not be found in {adata.obs_keys()}" + msg
                )
            if group in adata.obs.keys() and group == adata.obs.index.name:
                logger.error(
                    f"Given group {group} is both and index and a column level, which is ambiguous."
                )
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
        logger.error(f"adata.obs contains duplicated columns. Please rename or remove these columns first.\n`"
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
                logger.error(
                    f"The key '{key}' is found in both adata.obs and adata.{alt_search_repr}."
                )
            elif key in alt_names.index:
                val = alt_names[key]
                if isinstance(val, pd.Series):
                    # while var_names must be unique, adata.var[gene_symbols] does not
                    # It's still ambiguous to refer to a duplicated entry though.
                    assert alias_index is not None
                    raise KeyError(
                        f"Found duplicate entries for '{key}' in adata.{alt_search_repr}."
                    )
                var_idx_keys.append(val)
                var_symbols.append(key)
            else:
                not_found.append(key)
        if len(not_found) > 0:
            raise KeyError(
                f"Could not find keys '{not_found}' in columns of `adata.obs` or in adata.{alt_search_repr}."
            )

    adata_tidy_df = pd.DataFrame(index=adata.obs_names)

    # Adding var values to DataFrame:
    if len(var_idx_keys) > 0:
        adata_arr = adata.X if layer is None else adata.layers[layer]

        matrix = _get_array_values(
            adata_arr,
            dim_names=adata.var.index,
            keys=var_idx_keys,
            axis=1,
            backed=adata.isbacked
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
        categorical = pd.Series(np.repeat('', len(adata_tidy_df))).astype('category')
    elif len(cat_key) == 1 and is_numeric_dtype(adata_tidy_df[cat_key[0]]):
        # If category column is not categorical, turn it into one by subdividing ranges of values into 'num_categories'
        # categories:
        categorical = pd.cut(adata_tidy_df[cat_key[0]], num_categories)
    elif len(cat_key) == 1:
        categorical = adata_tidy_df[cat_key[0]].astype('category')
        categorical.name = cat_key[0]
    else:
        # Join the category values  using "_" to make a new 'category' consisting of both categorical columns:
        categorical = adata_tidy_df[cat_key].apply('_'.join, axis=1).astype('category')
        categorical.name = "_".join(cat_key)

        # Preserve category order as it appears in adata_tidy_df:
        order = {
            "_".join(k): idx
            for idx, k in enumerate(
                product(*(adata_tidy_df[g].cat.categories for g in cat_key))
            )
        }
        categorical = categorical.cat.reorder_categories(
            sorted(categorical.cat.categories, key=lambda x: order[x])
        )
    adata_tidy_df = adata_tidy_df[var_names].set_index(categorical)
    categories = adata_tidy_df.index.categories

    return categories, adata_tidy_df



class Dotplot:
    """
    Simultaneous visualization of two variates that are encoded by the dot size and the dot color. Size usually
    represents the fraction of samples that have non-zero values, and color usually represents the magnitude of the
    value.

    Args:
        adata : class `anndata.AnnData`
        var_names : sequence of str
            Should be a subset of adata.var_names
        cat_key : str or sequence of str
            The key(s) in .obs of the grouping to consider. Should be a categorical observation; if not,
            will be subdivided into 'num_categories'.
        num_categories : int, default 7
            Only used if groupby observation is not categorical. This value determines the number of groups into
            which the groupby observation should be subdivided.
        delta : optional float
            Distance between the largest value to consider and the smallest value to consider (see 'minn'
            parameter below)
        minn : optional float
            For the dot size legend, sets the value corresponding to the smallest dot on the legend
        alpha : optional float
            Significance threshold. If given, all elements w/ p-values <= 'alpha' will be marked by rings instead of
            dots.
        prescale_adata : bool, default False
            Set True to indicate that AnnData object should be scaled- if so, will use 'delta' and 'minn' to do so.
            If False, will proceed as though adata has already been processed as needed.
        categories_order : sequence of str
            Sets order of categories given by 'cat_key' along the plotting axis
        title : optional str
            Sets title for figure window
        figsize: None or tuple of form (float, float) (default: None)
            The width and height of a figure
        gene_symbols_key: optional str
            Key in .var containing gene symbols
        var_group_positions : optional sequence of tuples of form (int, int)
            Each item in the list should contain the start and end position that the bracket should cover.
            Eg. [(0, 4), (5, 8)] means that there are two brackets, one for the var_names in positions 0-4 and other for
            positions 5-8
        var_group_labels : optional sequence of str
            List of group labels for the variable names (e.g. can group var_names in positions 0-4 as being "group A")
        var_group_rotation : optional float
            Rotation in degrees of the variable name labels. If not given, small labels (<4 characters) are not
            rotated, but otherwise labels are rotated 90 degrees.
        layer : optional str
            Key in .layers specifying layer to use. If not given, will use .X.
        expression_cutoff : float, default 0.0
            Used for binarizing feature expression- feature is considered to be expressed only if the expression
            value is greater than this threshold
        mean_only_expressed : bool, default False
            If True, gene expression is averaged only over the cells expressing the given features
        standard_scale : 'None', 'val', or 'group'
            Whether or not to standardize that dimension between 0 and 1, meaning for each variable or group,
            subtract the minimum and divide each by its maximum. 'val' or 'group' is used to specify whether this
            should be done over variables or groups.
        ax : optional `matplotlib.Axes` object
            Can be used to provide pre-existing plotting axis
        vmin : optional float
            The data value that defines 0.0 in the normalization. Defaults to the min value of the dataset.
        vmax : optional float
            The data value that defines 1.0 in the normalization. Defaults to the the max value of the dataset.
        vcenter : optional float
            The data value that defines 0.5 in the normalization
        **kwargs :
            Additional arguments passed to `matplotlib.pyplot.scatter`
    """
    # Default parameters- visualization:
    default_colormap = 'winter'
    default_color_on = 'dot'
    default_dot_max = None
    default_dot_min = None
    default_smallest_dot = 0.0
    default_largest_dot = 200.0
    default_dot_edgecolor = 'black'
    default_dot_edgelw = 0.2
    default_size_exponent = 1.5

    default_size_legend_title = 'Fraction of cells\nin group (%)'
    default_color_legend_title = 'Mean expression\nin group'
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
            delta: Union[None, float] = None,
            minn: Union[None, float] = None,
            alpha: Union[None, float] = None,
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
            standard_scale: Literal['var', 'group'] = None,
            ax: Union[None, mpl.axes.Axes] = None,
            vmin: Union[None, float] = None,
            vmax: Union[None, float] = None,
            vcenter: Union[None, float] = None,
            **kwargs,
    ):

        # Default plotting parameters:
        config_spateo_rcParams()
        set_pub_style()

        self.logger = lm.get_main_logger()

        self.var_names = var_names
        self.var_group_labels = var_group_labels
        self.var_group_positions = var_group_positions
        self.var_group_rotation = var_group_rotation

        self.has_var_groups = (
            True
            if var_group_positions is not None and len(var_group_positions) > 0
            else False
        )

        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize
        self.width, self.height = self.figsize

        # If min_n and delta are not provided and 'prescale_adata' is True, preprocess adata first:
        self.minn = minn if minn is not None else np.nanmin(adata.X)
        self.delta = delta if delta is not None else np.nanmax(adata.X) - minn
        self.alpha = alpha
        if prescale_adata:
            adata.X = (adata.X - minn) / delta

        self.categories, self.adata_tidy_df = adata_to_frame(
            adata,
            self.var_names,
            cat_key=cat_key,
            num_categories=num_categories,
            layer=layer,
            gene_symbols_key=gene_symbols_key
        )

        # Compute fraction of cells having value > chosen expression cutoff, and transform into Boolean matrix using
        # the expression cutoff:
        obs_bool = self.adata_tidy_df > expression_cutoff

        # Compute the sum per group (for the Boolean matrix, the number of values > expression cutoff), divide the
        # result by the total number of cells in the group:
        dot_size_df = (
                obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
        )

        # Compute mean expression value, either only of cells that are expressing or of all cells:
        if mean_only_expressed:
            dot_color_df = (
                self.adata_tidy_df.mask(~obs_bool).groupby(level=0).mean().fillna(0)
            )
        else:
            dot_color_df = self.adata_tidy_df.groupby(level=0).mean()

        if standard_scale == 'group':
            dot_color_df = dot_color_df.sub(dot_color_df.min(1), axis=0)
            dot_color_df = dot_color_df.div(dot_color_df.max(1), axis=0).fillna(0)
        elif standard_scale == 'var':
            dot_color_df -= dot_color_df.min(0)
            dot_color_df = (dot_color_df / dot_color_df.max(0)).fillna(0)
        elif standard_scale is None:
            pass
        else:
            self.logger.warning("Unknown input given for 'standard_scale', proceeding without further processing array")

        # Remove duplicated features (can occur e.g. if the same gene is a marker for two groups)
        unique_var_names, unique_idx = np.unique(
            dot_color_df.columns, return_index=True
        )

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

        # set legend defaults
        self.color_legend_title = self.default_color_legend_title
        self.size_title = self.default_size_legend_title
        self.num_colorbar_ticks = self.default_num_colorbar_ticks
        self.num_size_legend_dots = self.default_num_size_legend_dots
        self.base = self.default_base
        self.legends_width = self.default_legends_width
        self.show_size_legend = True
        self.show_colorbar = True

    # To modify the style of the plot:
    def style(
            self,
            cmap: str = default_colormap,
            color_on: Optional[Literal['dot', 'square']] = default_color_on,
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
        cmap : str
            Name of Matplotlib color map to use
        color_on : str, default 'dot'
            Options are 'dot' or 'square'. By default the colormap is applied to the color of the dot, but 'square'
            changes this to be applied to a square region behind the dot, in which case the dot becomes transparent
            with only the edge of the dot shown.
        dot_max : optional float
            If none, the maximum dot size is set to the maximum fraction value found (e.g. 0.6). If given,
            the value should be a number between 0 and 1. All fractions larger than dot_max are clipped to this value.
        dot_min : optional float
            If none, the minimum dot size is set to 0. If given, the value should be a number between 0 and 1.
            All fractions smaller than dot_min are clipped to this value.
        smallest_dot : optional float
            If none, the smallest dot has size 0. All expression fractions with `dot_min` are plotted with this size.
        largest_dot : optional float
            If none, the largest dot has size 200. All expression fractions with `dot_max` are plotted with this size.
        dot_edge_color : str, default 'black'
            Only used if 'color_on' is 'square'. Sets dot edge color
        dot_edge_lw : float, default 0.2
            Only used if 'color_on' is 'square'. Sets dot edge line width
        size_exponent : float, default 1.5
            Dot size is computed as:
                fraction  ** size exponent
            and is afterwards scaled to match the 'smallest_dot' and 'largest_dot' size parameters.
            Using a different size exponent changes the relative sizes of the dots to each other.
        grid : bool, default False
            Set to true to show grid lines. By default grid lines are not shown. Further configuration of the grid
            lines can be achieved directly on the returned ax.
        x_padding : float, default 0.8
            Space between the plot left/right borders and the dots center. A unit is the distance between the x
            ticks. Only applied when 'color_on' = 'dot'
        y_padding : float, default 1.0
            Space between the plot top/bottom borders and the dots center. A unit is the distance between the x
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

    # To modify plot legends:
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
            show : bool, default True
                Set to `False` to hide the default plot of the legends. This sets the legend width to zero,
                which will result in a wider main plot.
            show_size_legend : bool, default True
                Set to `False` to hide the dot size legend
            show_colorbar : bool, default True
                Set to `False` to hide the colorbar legend
            size_title : str
                Title for the dot size legend. Use '\\n' to add line breaks. Will be shown at the top of the dot size.
                legend box
            colorbar_title : str
                Title for the color bar. Use '\\n' to add line breaks. Will be shown at the top of the color bar.
            base : int
                To determine the size of each "benchmark" dot in the size legend, will use a logscale; this parameter
                sets the base of that scale.
            num_colorbar_ticks : int
                Number of ticks for the colorbar
            num_size_legend_dots : int
                Number of "benchmark" dots to include in the dot size legend
            width : float, default 1.5
                Width of the legends area. The unit is the same as in matplotlib (inches)

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

    def _plot_size_legend(
            self,
            size_legend_ax: mpl.axes.Axes
    ):
        """
        Given axis object, generates dot size legend and displays on plot

        For the dot size "benchmarks" on the legend, adjust the difference in size between consecutive benchmarks
        based on how different 'self.dot_max' and 'self.dot_min' are.
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
        size = size_range ** self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # Plot size legend
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(0, len(size)),
            s=size,
            color='gray',
            edgecolor='black',
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [
            "{}".format(np.round((x * 100), decimals=0).astype(int)) for x in size_range
        ]
        size_legend_ax.set_xticklabels(labels, fontsize='small')

        # Remove y ticks and labels
        size_legend_ax.tick_params(
            axis='y', left=False, labelleft=False, labelright=False
        )

        # Remove surrounding lines
        size_legend_ax.spines['right'].set_visible(False)
        size_legend_ax.spines['top'].set_visible(False)
        size_legend_ax.spines['left'].set_visible(False)
        size_legend_ax.spines['bottom'].set_visible(False)
        size_legend_ax.grid(False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.45, size='small')

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

        if self.alpha is not None:
            ax = self.fig.add_subplot()

    def _plot_colorbar(
            self,
            color_legend_ax: mpl.axes.Axes,
            normalize: Union[None, mpl.colors.Normalize] = None
    ):
        """
        Given axis object, plots a horizontal colorbar

        Args:
            color_legend_ax : mpl.axes.Axes
                Matplotlib axis object to plot onto
            normalize : mpl.colors.Normalize
                The normalizing object that scales data, typically into the interval [0, 1]. If None, norm defaults to
                a colors.Normalize object which initializes its scaling based on the first data processed.
        """
        cmap = plt.get_cmap(self.cmap)

        mappable = ScalarMappable(norm=normalize, cmap=cmap)

        matplotlib.colorbar.Colorbar(
            color_legend_ax, mappable=mappable, orientation='horizontal'
        )

        color_legend_ax.set_title(self.color_legend_title, fontsize='small')

        color_legend_ax.xaxis.set_tick_params(labelsize='small')

    def _plot_legend(
            self,
            legend_ax: mpl.axes.Axes,
            return_ax_dict: dict,
            normalize: Union[None, mpl.colors.Normalize] = None
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