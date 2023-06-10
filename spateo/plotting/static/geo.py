"""Plotting functions for spatial geometry plots.
"""

from typing import List, Optional, Tuple, Union

import anndata
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.wkb import dumps

from ...configuration import SKM
from .scatters import scatters
from .utils import _convert_to_geo_dataframe, save_return_show_fig_utils


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def geo(
    adata: anndata.AnnData,
    basis: str = "contour",
    color: Union[list, str, None] = None,
    genes: Union[list, None] = [],
    gene_cmaps: Optional[str] = None,
    dpi: int = 100,
    alpha: float = 0.8,
    boundary_width: float = 0.2,
    boundary_color="black",
    stack_genes: bool = False,
    stack_genes_threshold: float = 0.01,
    stack_colors_legend_size: int = 10,
    figsize: Tuple[float, float] = (6, 6),
    aspect: str = "equal",
    slices: Optional[int] = None,
    img_layers: Optional[int] = None,
    *args,
    **kwargs,
):
    """
    Geometry plot for physical coordinates of each cell.

    Args:
        adata: an Annodata object that contain the physical coordinates for each bin/cell, etc.
        basis: The key to the column in the adata.obs, from which the contour of the cell segmentation will be
            generated.
        color: Any or any list of column names or gene name, etc. that will be used for coloring cells.
            If `color` is not None, stack_genes will be disabled automatically because `color` can contain non numerical
            values.
        genes: The gene list that will be used to plot the gene expression on the same scatter plot. Each gene will have
            a different color. Can be a single gene name string and we will convert it to a list.
        gene_cmaps:
            The colormap used to stack different genes in a single plot.
        dpi: The resolution of the figure in dots-per-inch. Dots per inches (dpi) determines how many pixels the figure
            comprises. dpi is different from ppi or points per inches. Note that most elements like lines, markers,
            texts have a size given in points so you can convert the points to inches. Matplotlib figures use Points per
            inch (ppi) of 72. A line with thickness 1 point will be 1./72. inch wide. A text with fontsize 12 points
            will be 12./72. inch height. Of course if you change the figure size in inches, points will not change, so a
            larger figure in inches still has the same size of the elements.Changing the figure size is thus like taking
            a piece of paper of a different size. Doing so, would of course not change the width of the line drawn with
            the same pen. On the other hand, changing the dpi scales those elements. At 72 dpi, a line of 1 point size
            is one pixel strong. At 144 dpi, this line is 2 pixels strong. A larger dpi will therefore act like a
            magnifying glass. All elements are scaled by the magnifying power of the lens. see more details at answer 2
            by @ImportanceOfBeingErnest:
            https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
        alpha: The alpha value of the cells.
        boundary_width: The line width of boundary.
        boundary_color: The color value of boundary.
        stack_genes: whether to show all gene plots on the same plot
        stack_genes_threshold: lower bound of gene values that will be drawn on the plot.
        stack_colors_legend_size: control the size of legend when stacking genes
        figuresize: size of the figure.
        aspect: Set the aspect of the axis scaling, i.e. the ratio of y-unit to x-unit. In physical spatial plot, the
            default is 'equal'. See more details at:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_aspect.html
        slices: The index to the tissue slice, will used in adata.uns["spatial"][slices].
        img_layers: The index to the (staining) image of a tissue slice, will be used in
            adata.uns["spatial"][slices]["images"].

    Returns
    -------
        plots gene or cell feature of the adata object on the physical spatial coordinates.
    """
    # main_info("Plotting geometry info on adata")
    # main_log_time()

    if color is not None and stack_genes:
        # main_warning(
        #     "Set `stack_genes` to False because `color` argument cannot be used with stack_genes. If you would like to stack genes (or other numeical values), please pass gene expression like column names into `gene` argument."
        # )
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

    adata = adata.copy()  # make sure don't modify the original data.
    adata = _convert_to_geo_dataframe(adata, basis)

    res = scatters(
        adata,
        color=genes,
        figsize=figsize,
        dpi=dpi,
        alpha=alpha,
        stack_colors=stack_genes,
        stack_colors_threshold=stack_genes_threshold,
        stack_colors_title="stacked spatial genes",
        show_colorbar=show_colorbar,
        stack_colors_legend_size=stack_colors_legend_size,
        stack_colors_cmaps=gene_cmaps,
        geo=True,
        boundary_width=boundary_width,
        boundary_color=boundary_color,
        aspect=aspect,
        slices=slices,
        img_layers=img_layers,
        *args,
        **kwargs,
    )

    # main_finish_progress("space plot")
    return res


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def space_polygons(
    polygons_path: str,
    adata: anndata.AnnData,
    color: Optional[Union[List[str], str]] = None,
    fov: Optional[Union[int, str]] = None,
    **kwargs,
):
    """
    Plot polygons on spatial coordinates.

    Args:
        polygons_path: The path to the file containing polygon coordinates
        adata: The AnnData object that contains the spatial coordinates, gene expression, cell type labels, etc.
        color: The column name in the adata.obs that will be used to color the polygons.
        fov: The fov name that will be used to select the polygons. If None, all polygons will be plotted.
        kwargs: Additional arguments passed to :func ~`spateo.pl.geo.geo()`.
    """

    # Search for global spatial coordinates if fov is not provided:
    if fov is None:
        try:
            adata.obsm["global_spatial"] = np.hstack(
                (
                    adata.obs["CenterX_global_px"].values.reshape(-1, 1),
                    adata.obs["CenterY_global_px"].values.reshape(-1, 1),
                )
            )
        except:
            raise ValueError(
                "Global spatial coordinates could not be found in AnnData object. Conventional Nanostring "
                "data stores "
                "these in 'CenterX_global_px' and 'CenterY_global_px', and so the search was done for "
                "these columns."
            )

    polygons = pd.read_csv(polygons_path)

    # It is assumed that each row in the polygons dataframe corresponds to the cell at the same position in the
    # AnnData object
    polygons["cellID_fov"] = polygons.apply(lambda row: f"{int(row['cellID'])}_{int(row['fov'])}", axis=1)
    polygons["cellID_fov"] = polygons["cellID_fov"].astype(str)

    if fov is not None:
        polygons_fov = polygons[polygons["fov"] == fov]
        adata = adata[adata.obs["fov"] == fov]
    else:
        polygons_fov = polygons

    props = create_polygon_object_nanostring(polygons_fov)
    adata.obs["area"] = props["area"].values.astype(float)
    adata.obsm["spatial"] = props.filter(regex="centroid-").values.astype(float)
    adata.obsm["contour"] = props["contour"].values.astype(str)
    adata.obsm["bbox"] = props.filter(regex="bbox-").values.astype(int)
    adata.obsm["scaled_spatial"] = adata.obsm["spatial"] // 100

    geo(adata, color=color, **kwargs)


def create_polygon_object_nanostring(polygon_df: pd.DataFrame):
    """Process cell data to construct contours and calculate area and centroid.

    Args:
        polygon_df: Input DataFrame containing pixel-to-cell correspondence data with columns 'cellID_fov',
            'x_local_px', and 'y_local_px'.

    Returns:
        processed_polygon_df: Dataframe containing polygon information. Contains columns 'label', 'area', 'bbox-0',
            'bbox-1', 'bbox-2', 'bbox-3', 'centroid-0', 'centroid-1', and 'contour'. Indexed by the 'label' column.
    """
    rows = []  # Initialize the rows list

    # Loop over the cell IDs to construct the contour for each cell
    for cell_id in polygon_df["cellID_fov"].unique():
        # Filter the DataFrame for the current cell ID
        cell_df = polygon_df[polygon_df["cellID_fov"] == cell_id]

        # Retrieve the relevant coordinates for constructing the contour
        coordinates = cell_df[["x_local_px", "y_local_px"]].values.astype(int)

        min_offset = coordinates.min(axis=0)
        max_offset = coordinates.max(axis=0)
        min0, min1 = min_offset
        max0, max1 = max_offset

        # Construct the polygon/contour for the cell using the coordinates
        polygon = Polygon(coordinates)

        poly = dumps(polygon, hex=True)  # geometry object to hex

        # Convert the polygon to a numpy array representing the contour
        contour = coordinates

        # Calculate the moments using cv2.moments()
        moments = cv2.moments(contour)

        # Calculate the area and centroid using the moments
        area = moments["m00"]
        if area > 0:
            centroid0 = moments["m10"] / area
            centroid1 = moments["m01"] / area
        elif contour.shape[0] == 2:
            line = contour - min_offset
            mask = cv2.line(np.zeros((max_offset - min_offset + 1)[::-1], dtype=np.uint8), line[0], line[1], color=1).T
            area = mask.sum()
            centroid0, centroid1 = contour.mean(axis=0)
        elif contour.shape[0] == 1:
            area = 1
            centroid0, centroid1 = contour[0] + 0.5
        else:
            raise IOError(f"Contour contains {contour.shape[0]} points.")

        rows.append([str(cell_id), area, min0, min1, max0 + 1, max1 + 1, centroid0, centroid1, poly])

    # Construct the DataFrame from the rows list:
    processed_polygon_df = pd.DataFrame(
        rows, columns=["label", "area", "bbox-0", "bbox-1", "bbox-2", "bbox-3", "centroid-0", "centroid-1", "contour"]
    ).set_index("label")

    return processed_polygon_df
