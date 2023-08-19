from typing import List, Optional, Tuple, Union, Literal

import anndata
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

from ...configuration import SKM
from ...tools.utils import compute_smallest_distance
from .scatters import scatters
from .utils import _convert_to_geo_dataframe

# from .scatters import (
#     scatters,
#     docstrings,
# )


# from ..dynamo_logger import main_critical, main_info, main_finish_progress, main_log_time, main_warning

# docstrings.delete_params("scatters.parameters", "adata", "basis", "figsize")


# @docstrings.with_indent(4)
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def space(
    adata: anndata.AnnData,
    color: Optional[Union[List[str], str, None]] = None,
    genes: List[str] = [],
    gene_cmaps=None,
    space: str = "spatial",
    width: float = 6,
    marker: str = ".",
    pointsize: Optional[float] = None,
    dpi: int = 100,
    ps_sample_num: int = 1000,
    alpha: float = 0.8,
    stack_genes: bool = False,
    stack_genes_threshold: float = 0.01,
    stack_colors_legend_size: int = 10,
    figsize=None,
    *args,
    **kwargs,
):
    """\
    Scatter plot for physical coordinates of each cell.

    Args:
        adata: An AnnData object that contains the physical coordinates for each bin/cell, etc.
        genes: The gene list that will be used to plot the gene expression on the same scatter plot. Each gene will
            have a different color.
        color: `string` (default: `ntr`)
            Any or any list of column names or gene name, etc. that will be used for coloring cells. If `color` is not
            None, stack_genes will be disabled automatically because `color` can contain non numerical values.
        space: The key to space coordinates.
        stack_genes: Whether to show all gene plots on the same plot
        stack_genes_threshold: Lower bound of gene values that will be drawn on the plot
        stack_colors_legend_size: Control the size of legend when stacking genes
        alpha: The alpha value of the scatter points
        width: Width of the figure
        marker: A string representing some marker from matplotlib
            https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        pointsize: The size of the points on the scatter plot
        dpi: `float`, (default: 100.0)
            The resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
            comprises. dpi is different from ppi or points per inches. Note that most elements like lines, markers,
            texts have a size given in points so you can convert the points to inches. Matplotlib figures use Points per
            inch (ppi) of 72. A line with thickness 1 point will be 1./72. inch wide. A text with fontsize 12 points
            will be 12./72. inch heigh. Of course if you change the figure size in inches, points will not change, so a
            larger figure in inches still has the same size of the elements.Changing the figure size is thus like taking
            a piece of paper of a different size. Doing so, would of course not change the width of the line drawn with
            the same pen. On the other hand, changing the dpi scales those elements. At 72 dpi, a line of 1 point size
            is one pixel strong. At 144 dpi, this line is 2 pixels strong. A larger dpi will therefore act like a
            magnifying glass. All elements are scaled by the magnifying power of the lens. see more details at answer 2
            by @ImportanceOfBeingErnest:
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        ps_sample_num: The number of bins / cells that will be sampled to estimate the distance between different
            bin / cells
        %(scatters.parameters.no_adata|basis|figsize)s

    Returns:
        plots gene or cell feature of the adata object on the physical spatial coordinates.
    """
    if color is not None and stack_genes:
        stack_genes = False

    genes = [genes] if type(genes) is str else list(genes)
    # concatenate genes and colors for scatters plot
    if color is not None and genes is not None:
        color = [color] if type(color) is str else list(color)
        genes.extend(color)

    show_colorbar = True
    if stack_genes:
        # main_warning("disable side colorbar due to colorbar scale (numeric tick) related issue.")
        show_colorbar = False

    if genes is None or (len(genes) == 0):
        # main_critical("No genes provided. Please check your argument passed in.")
        return
    if "X_" + space in adata.obsm_keys():
        space_key = space
    elif space in adata.obsm_keys():
        if space.startswith("X_"):
            space_key = space.split("X_")[1]
        else:
            # scatters currently will append "X_" to the basis, so we need to create the `X_{space}` key.
            # In future, extend scatters to directly plot coordinates in space key without append "X_"
            if "X_" + space not in adata.obsm_keys():
                adata.obsm["X_" + space] = adata.obsm[space]
                space_key = space

    ptp_vec = adata.obsm["X_" + space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    if figsize is None:
        figsize = (width, ptp_vec[1] / ptp_vec[0] * width + 0.3)

    # calculate point size based on minimum radius
    if pointsize is None:
        pointsize = compute_smallest_distance(adata.obsm["X_" + space_key], sample_num=ps_sample_num)
        # here we will scale the point size by the dpi and the figure size in inch.
        pointsize *= figsize[0] / ptp_vec[0] * dpi
        # meaning of s in scatters:
        # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size/47403507#47403507
        # Note that np.sqrt(adata.shape[0]) / 16000.0 is used in pl.scatters
        pointsize = pointsize**2 * np.sqrt(adata.shape[0]) / 16000.0

        # main_info("estimated point size for plotting each cell in space: %f" % (pointsize))

    # here we should pass different point size, type (square or hexogon, etc), etc.
    res = scatters(
        adata,
        marker=marker,
        basis=space_key,
        color=genes,
        figsize=figsize,
        pointsize=pointsize,
        dpi=dpi,
        alpha=alpha,
        stack_colors=stack_genes,
        stack_colors_threshold=stack_genes_threshold,
        stack_colors_title="stacked spatial genes",
        show_colorbar=show_colorbar,
        stack_colors_legend_size=stack_colors_legend_size,
        stack_colors_cmaps=gene_cmaps,
        *args,
        **kwargs,
    )

    # main_finish_progress("space plot")
    return res


def plot_cell_signaling(
    adata: anndata.AnnData,
    vf_key: str,
    geo: bool = False,
    color: Optional[Union[List[str], str, None]] = None,
    arrow_color: str = "tab:blue",
    edgewidth: float = 0.2,
    genes: List[str] = [],
    gene_cmaps=None,
    space: str = "spatial",
    width: float = 6,
    marker: str = ".",
    basis: str = "contour",
    boundary_width: float = 0.2,
    boundary_color: str = "black",
    pointsize: Optional[float] = None,
    dpi: int = 100,
    ps_sample_num: int = 1000,
    alpha: float = 0.8,
    plot_method: Literal["cell", "grid", "stream"] = "cell",
    scale: Optional[float] = None,
    scale_units: Optional[Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy']] = None,
    grid_density: float = 1,
    grid_knn: Optional[int] = None,
    grid_scale: float = 1.0,
    grid_threshold: float = 1.0,
    grid_width: Optional[float] = None,
    stream_density: Optional[float] = None,
    stream_linewidth: Optional[float] = None,
    stream_cutoff_percentile: float = 5,
    figsize: Optional[Tuple[float, float]] = None,
    *args,
    **kwargs,
):
    """After inferring directionality of effects for models that consider ligand expression (:attr `mod_type` is
    'ligand' or 'lr', this can be used to visualize the inferred directionality in the form of a vector field plot.
    Note: currently incompatible with datashader.

    Parts of this function are inspired by 'plot_cell_signaling' from COMMOT: https://github.com/zcang/COMMOT

    Args:
        adata: An AnnData object that contains the physical coordinates for each bin/cell, etc.
        vf_key: Key in .obsm where the vector field is stored
        geo: Whether to plot the vector field on top of a geometry plot rather than a scatter plot of cells. Note
            that none of the pointsize arguments (e.g. 'pointsize', 'ps_sample_num') will be used if this is True.
        color: `string` (default: `ntr`)
            Any or any list of column names or gene name, etc. that will be used for coloring cells. If `color` is not
            None, stack_genes will be disabled automatically because `color` can contain non numerical values.
        arrow_color: Sets color of vector field arrows
        edgewidth: Sets width of vector field arrows. Recommended 0.1-0.3.
        genes: The gene list that will be used to plot the gene expression on the same scatter plot. Each gene will
            have a different color.
        genes_cmaps: Optional color map for each gene.
        space: The key to spatial coordinates
        alpha: The alpha value of the scatter points. Recommended 0.5-0.7.
        width: Width of the figure
        marker: A string representing some marker from matplotlib
            https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers


        basis: Only used if `geo` is True. The key to the column in adata.obs from which the contour of the cell
            segmentation will be generated.
        boundary_width: Only used if `geo` is True. The width of the contour lines.
        boundary_color: Only used if `geo` is True. The color of the contour lines.


        pointsize: The size of the points on the scatter plot. Not used if 'geo' is True.
        dpi: `float`, (default: 100.0)
            The resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
            comprises. dpi is different from ppi or points per inches. Note that most elements like lines, markers,
            texts have a size given in points so you can convert the points to inches. Matplotlib figures use Points per
            inch (ppi) of 72. A line with thickness 1 point will be 1./72. inch wide. A text with fontsize 12 points
            will be 12./72. inch heigh. Of course if you change the figure size in inches, points will not change, so a
            larger figure in inches still has the same size of the elements.Changing the figure size is thus like taking
            a piece of paper of a different size. Doing so, would of course not change the width of the line drawn with
            the same pen. On the other hand, changing the dpi scales those elements. At 72 dpi, a line of 1 point size
            is one pixel strong. At 144 dpi, this line is 2 pixels strong. A larger dpi will therefore act like a
            magnifying glass. All elements are scaled by the magnifying power of the lens. see more details at answer 2
            by @ImportanceOfBeingErnest:
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        ps_sample_num: The number of bins / cells that will be sampled to estimate the distance between different
            bin / cells


        plot_method: The method used to plot the vector field. Can be one of the following:
            'cell': Plot the vector field at the center of each cell.
            'grid': Plot the vector field on a grid.
            'stream': Plot the vector field as stream lines.
        scale: The scale parameter passed to :func `matplotlib.pyplot.quiver` for vector field plots. The smaller the
            value, the longer the arrows. Recommended ~0.01.
        grid_density: Only used if 'to_plot' is 'grid'. The density of the grid points used to estimate the vector
            field. The larger the value, the more grid points there will be.
        grid_knn: Only used if 'to_plot' is 'grid'. The number of nearest neighbors used to interpolate the signaling
            directions from spots to grid points.
        grid_scale: Only used if 'to_plot' is 'grid'. The scale parameter for the kernel function used to map
            directions of spots to grid points.
        grid_threshold: Only used if 'to_plot' is 'grid'. The threshold forinterpolation weights, used to determine
            whether to include a grid point. Smaller values give tighter coverage of the tissue by the grid points.
        grid_width: Only used if 'to_plot' is 'grid'. The width of the vector lines. Recommended on the order of 0.005.
        stream_density: Only used if 'to_plot' is 'stream'. The density of the stream lines passed to :func
            `matplotlib.pyplot.streamplot`.
        stream_linewidth: Only used if 'to_plot' is 'stream'. The width of the stream lines passed to :func
            `matplotlib.pyplot.streamplot`.
        stream_cutoff_percentile: Only used if 'to_plot' is 'stream'. The percentile of the vector field magnitude
            used to determine if/which vectors are plotted or not. Defaults to 5, meaning that vectors shorter than the
            5% quantile will not be plotted.
    """

    genes = [genes] if type(genes) is str else list(genes)
    # concatenate genes and colors for scatters plot
    if color is not None and genes is not None:
        color = [color] if type(color) is str else list(color)
        genes.extend(color)

    if genes is None or (len(genes) == 0):
        # main_critical("No genes provided. Please check your argument passed in.")
        return
    if "X_" + space in adata.obsm_keys():
        space_key = space
    elif space in adata.obsm_keys():
        if space.startswith("X_"):
            space_key = space.split("X_")[1]
        else:
            # scatters currently will append "X_" to the basis, so we need to create the `X_{space}` key.
            # In future, extend scatters to directly plot coordinates in space key without append "X_"
            if "X_" + space not in adata.obsm_keys():
                adata.obsm["X_" + space] = adata.obsm[space]
                space_key = space

    ptp_vec = adata.obsm["X_" + space_key].ptp(0)
    # calculate the figure size based on the width and the ratio between width and height
    # from the physical coordinate.
    if figsize is None:
        figsize = (width, ptp_vec[1] / ptp_vec[0] * width + 0.3)

    # calculate point size based on minimum radius
    if pointsize is None:
        pointsize = compute_smallest_distance(adata.obsm["X_" + space_key], sample_num=ps_sample_num)
        # here we will scale the point size by the dpi and the figure size in inch.
        pointsize *= figsize[0] / ptp_vec[0] * dpi
        # meaning of s in scatters:
        # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size/47403507#47403507
        # Note that np.sqrt(adata.shape[0]) / 16000.0 is used in pl.scatters
        pointsize = pointsize**2 * np.sqrt(adata.shape[0]) / 16000.0

    # Configure vector field and define additional variables for vector field plotting:
    vf = adata.obsm[vf_key]
    X = adata.obsm[space]

    if plot_method == "cell":
        vf_cell = vf.copy()
        vf_cell_sum = np.sum(vf_cell, axis=1)
        vf_cell[np.where(vf_cell_sum == 0)[0], :] = np.nan
        X_grid = X

    elif plot_method == "grid" or plot_method == "stream":
        # Define rectangular grid to serve as stopping and starting points for vectors
        xl, xr = np.min(X[:, 0]), np.max(X[:, 0])
        epsilon = 0.02 * (xr - xl)
        xl -= epsilon
        xr += epsilon
        yl, yr = np.min(X[:, 1]), np.max(X[:, 1])
        epsilon = 0.02 * (yr - yl)
        yl -= epsilon
        yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr - xl) / float(ngrid_x)
        ngrid_y = int((yr - yl) / gridsize)
        meshgrid = np.meshgrid(np.linspace(xl, xr, ngrid_x), np.linspace(yl, yr, ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1, 1), meshgrid[1].reshape(-1, 1)), axis=1)

        if grid_knn is None:
            grid_knn = int(X.shape[0] / 50)
        nn_mdl = NearestNeighbors()
        nn_mdl.fit(X)
        distances, neighbors = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        w = norm.pdf(x=distances, scale=gridsize * grid_scale)
        w_sum = w.sum(axis=1)

        vf_grid = (vf[neighbors] * w[:, :, None]).sum(axis=1)
        vf_grid /= np.maximum(1, w_sum)[:, None]

        if plot_method == "grid":
            grid_threshold *= np.percentile(w_sum, 99) / 100
            grid_pts, vf_grid = grid_pts[w_sum > grid_threshold], vf_grid[w_sum > grid_threshold]
            X_grid = grid_pts
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            X_grid = np.vstack((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)))
            vf_grid = vf_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((vf_grid**2).sum(0))
            grid_thresh = 10 ** (grid_threshold - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(vf_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(vf[neighbors]), axis=1), axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_percentile)

            vf_grid[0][cutoff] = np.nan

            lengths = np.sqrt((vf_grid**2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()

    else:
        raise ValueError(f"plot_method must be one of 'cell', 'grid', or 'stream'. Got {plot_method}.")

    # scatters call with vector field kwargs
    vf_kwargs = {
        "scale": scale,
        "scale_units": scale_units,
        "width": grid_width,
        "color": arrow_color,
        "edgecolor": "black",
        "density": stream_density,
        "linewidth": stream_linewidth if stream_linewidth is not None else edgewidth,
    }

    V = vf_cell if plot_method == "cell" else vf_grid

    if geo:
        adata = adata.copy()  # make sure don't modify the original data.
        adata = _convert_to_geo_dataframe(adata, basis)

    res = scatters(
        adata,
        vf_key=vf_key,
        X_grid=X_grid,
        V=V,
        marker=marker,
        basis=space_key,
        color=genes,
        figsize=figsize,
        pointsize=pointsize,
        dpi=dpi,
        alpha=alpha,
        stack_colors_cmaps=gene_cmaps,
        geo=geo,
        boundary_width=boundary_width,
        boundary_color=boundary_color,
        vf_plot_method=plot_method,
        vf_kwargs=vf_kwargs,
        *args,
        **kwargs,
    )

    return res
