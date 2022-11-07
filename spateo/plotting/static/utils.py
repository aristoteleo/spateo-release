# code adapted from https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/plot/utils.py
import copy
import math
import os
import warnings
from typing import Any, Collection, Dict, List, Optional, Tuple, Union

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geopandas as gpd

from inspect import signature

import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.api.types import is_categorical_dtype
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance
from shapely.wkb import loads
from sklearn.decomposition import PCA
from typing_extensions import Literal

from ...configuration import SKM, _themes
from ...logging import logger_manager as lm


# ---------------------------------------------------------------------------------------------------
# variable checking utilities
def is_gene_name(adata, var):
    if type(var) in [str, np.str_]:
        return var in adata.var.index
    else:
        return False


def is_cell_anno_column(adata, var):
    if type(var) in [str, np.str_]:
        return var in adata.obs.columns
    else:
        return False


def is_layer_keys(adata, var):
    if type(var) in [str, np.str_]:
        return var in adata.layers.keys()
    else:
        return False


def is_list_of_lists(list_of_lists):
    all(isinstance(elem, list) for elem in list_of_lists)


def _get_adata_color_vec(adata, layer, col):
    if layer in ["protein", "X_protein"]:
        _color = adata.obsm[layer].loc[col, :]
    elif layer == "X":
        _color = adata.obs_vector(col, layer=None)
    else:
        _color = adata.obs_vector(col, layer=layer)
    return np.array(_color).flatten()


# ---------------------------------------------------------------------------------------------------
# plotting utilities that borrowed from umap
# link: https://github.com/lmcinnes/umap/blob/7e051d8f3c4adca90ca81eb45f6a9d1372c076cf/umap/plot.py


def map2color(val, min=None, max=None, cmap="viridis"):
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    minima = np.min(val) if min is None else min
    maxima = np.max(val) if max is None else max

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

    cols = [mapper.to_rgba(v) for v in val]

    return cols


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


# https://stackoverflow.com/questions/8468855/convert-a-rgb-colour-value-to-decimal
"""
Convert RGB color to decimal RGB integers are typically treated as three distinct bytes where \
the left-most (highest-order) byte is red, the middle byte is green and the right-most (lowest-order) byte is blue. \
"""


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _red(x):
    return (x & 0xFF0000) >> 16


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _green(x):
    return (x & 0x00FF00) >> 8


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _blue(x):
    return x & 0x0000FF


def _embed_datashader_in_an_axis(datashader_image, ax):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev), _green(img_rev), _red(img_rev)])
    ax.imshow(mpl_img)
    return ax


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def _select_font_color(background):
    if background in ["k", "black"]:
        font_color = "white"
    elif background in ["w", "white"]:
        font_color = "black"
    elif background.startswith("#"):
        mean_val = np.mean(
            # specify 0 as the base in order to invoke this prefix-guessing behavior;
            # omitting it means to assume base-10
            [int("0x" + c, 0) for c in (background[1:3], background[3:5], background[5:7])]
        )
        if mean_val > 126:
            font_color = "black"
        else:
            font_color = "white"

    else:
        font_color = "black"

    return font_color


def _scatter_projection(ax, points, projection, **kwargs):
    if projection == "3d":
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], **kwargs)


def _geo_projection(ax, points, **kwargs):
    linecolor = kwargs.pop("linecolor")
    if "values" in kwargs:
        # using value
        gdf = gpd.GeoDataFrame(data={"values": kwargs.pop("values"), "points": points}, geometry="points")
        ax = gdf.plot("values", ax=ax, **kwargs)
    else:
        # using color
        gdf = gpd.GeoDataFrame(geometry=points)
        ax = gdf.plot(ax=ax, **kwargs)

    # clean args for boundary plotting
    if "color" in kwargs:
        kwargs.pop("color")
    if "cmap" in kwargs:
        kwargs.pop("cmap")
    gdf.boundary.plot(ax=ax, color=linecolor, **kwargs)


def _matplotlib_points(
    points,
    ax=None,
    labels=None,
    values=None,
    highlights=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=7,
    height=5,
    show_legend=True,
    vmin=2,
    vmax=98,
    sort="raw",
    frontier=False,
    contour=False,
    ccmap=None,
    calpha=0.4,
    sym_c=False,
    inset_dict={},
    show_colorbar=True,
    projection=None,  # default in matplotlib
    geo=False,
    **kwargs,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    dpi = plt.rcParams["figure.dpi"]
    width, height = width * dpi, height * dpi
    rasterized = kwargs["rasterized"] if "rasterized" in kwargs.keys() else None
    # """Use matplotlib to plot points"""
    # point_size = 500.0 / np.sqrt(points.shape[0])

    legend_elements = None

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111, projection=projection)

    ax.set_facecolor(background)

    # Color by labels
    unique_labels = []

    if labels is not None:
        # main_debug("labels are not None, drawing by labels")
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(labels.shape[0], points.shape[0])
            )
        if color_key is None:
            # main_debug("color_key is None")
            cmap = copy.copy(matplotlib.cm.get_cmap(color_key_cmap))
            cmap.set_bad("lightgray")
            colors = None

            if highlights is None:
                unique_labels = np.unique(labels)
                num_labels = unique_labels.shape[0]
                color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            else:
                if type(highlights) is str:
                    highlights = [highlights]
                highlights.append("other")
                unique_labels = np.array(highlights)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
                color_key[-1] = "#bdbdbd"  # lightgray hex code https://www.color-hex.com/color/d3d3d3

                labels[[i not in highlights[:-1] for i in labels]] = "other"
                points = pd.DataFrame(points)
                points["label"] = pd.Categorical(labels)

                # reorder data so that highlighting points will be on top of background points
                highlight_ids, background_ids = (
                    points["label"] != "other",
                    points["label"] == "other",
                )
                # reorder_data = points.copy(deep=True)
                # (
                #     reorder_data.loc[:(sum(background_ids) - 1), :],
                #     reorder_data.loc[sum(background_ids):, :],
                # ) = (points.loc[background_ids, :].values, points.loc[highlight_ids, :].values)
                points = pd.concat(
                    (
                        points.loc[background_ids, :],
                        points.loc[highlight_ids, :],
                    )
                ).values
                # labels = points[:, 2]
                labels = points["label"]

        # WARNING: do not change the following line to "elif" during refactor
        # This if-else branch is not logically parallel to the previous one. The following branch sets `colors`.
        if isinstance(color_key, dict):
            # main_debug("color_key is a dict")
            colors = pd.Series(labels).map(color_key).values
            unique_labels = np.unique(labels)
            legend_elements = [
                # Patch(facecolor=color_key[k], label=k) for k in unique_labels
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color_key[k],
                    label=k,
                    linestyle="None",
                )
                for k in unique_labels
            ]
        else:
            # main_debug("color_key is not None and not a dict")
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError("Color key must have enough colors for the number of labels")

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            legend_elements = [
                # Patch(facecolor=color_key[i], label=k)
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color_key[i],
                    label=k,
                    linestyle="None",
                )
                for i, k in enumerate(unique_labels)
            ]
            colors = pd.Series(labels).map(new_color_key)

        if frontier:
            # main_debug("drawing frontier")
            _scatter_projection(
                ax,
                points,
                projection,
                s=kwargs["s"] * 2,
                c="0.0",
                lw=2,
                rasterized=rasterized,
            )
            _scatter_projection(
                ax,
                points,
                projection,
                s=kwargs["s"] * 2,
                c="1.0",
                lw=0,
                rasterized=rasterized,
            )
            _scatter_projection(
                ax,
                points,
                projection,
                c=colors,
                plotnonfinite=True,
                **kwargs,
            )
        elif contour:
            # main_debug("drawing contour")
            # try:
            #     from shapely.geometry import Polygon, MultiPoint, Point
            # except ImportError:
            #     raise ImportError(
            #         "If you want to use the tricontourf in plotting function, you need to install `shapely` "
            #         "package via `pip install shapely` see more details at https://pypi.org/project/Shapely/,"
            #     )
            #
            # x, y = points[:, :2].T
            # triang = tri.Triangulation(x, y)
            # concave_hull, edge_points = alpha_shape(x, y, alpha=calpha)
            # ax = plot_polygon(concave_hull, ax=ax)
            #
            # # Use the mean distance between the triangulated x & y poitns
            # x2 = x[triang.triangles].mean(axis=1)
            # y2 = y[triang.triangles].mean(axis=1)
            # ##note the very obscure mean command, which, if not present causes an error.
            # ##now we need some masking condition.
            #
            # # Create an empty set to fill with zeros and ones
            # cond = np.empty(len(x2))
            # # iterate through points checking if the point lies within the polygon
            # for i in range(len(x2)):
            #     cond[i] = concave_hull.contains(Point(x2[i], y2[i]))
            #
            # mask = np.where(cond, 0, 1)
            # # apply masking
            # triang.set_mask(mask)
            #
            # # ax.tricontourf(triang, values, cmap=ccmap)
            import seaborn as sns

            ccmap = "viridis" if ccmap is None else ccmap
            df = pd.DataFrame(points, columns=["x", "y", "z"][: points.shape[1]])
            ax = sns.kdeplot(
                data=df.iloc[:, :2],
                x="x",
                y="y",
                fill=True,
                alpha=calpha,
                palette=ccmap,
                ax=ax,
                thresh=0,
                levels=100,
            )
            x, y = points[:, :2].T
            _scatter_projection(
                ax,
                points,
                projection,
                c=colors,
                plotnonfinite=True,
                zorder=21,
                **kwargs,
            )
        else:
            # main_debug("drawing without frontiers and contour")
            if geo:
                _geo_projection(
                    ax,
                    points,
                    color=colors,
                    **kwargs,
                )
            else:
                _scatter_projection(
                    ax,
                    points,
                    projection,
                    c=colors,
                    plotnonfinite=True,
                    **kwargs,
                )

    # Color by values
    elif values is not None:
        # main_debug("drawing points by values")
        cmap_ = copy.copy(matplotlib.cm.get_cmap(cmap))
        cmap_.set_bad("lightgray")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.cm.register_cmap(name=cmap_.name, cmap=cmap_, override_builtin=True)

        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(values.shape[0], points.shape[0])
            )
        # reorder data so that high values points will be on top of background points
        sorted_id = (
            np.argsort(abs(values)) if sort == "abs" else np.argsort(-values) if sort == "neg" else np.argsort(values)
        )
        values, points = values[sorted_id], points[sorted_id]

        # if there are very few cells have expression, set the vmin/vmax only based on positive values to
        # get rid of outliers
        if np.nanmin(values) == 0:
            n_pos_cells = sum(values > 0)
            if 0 < n_pos_cells / len(values) < 0.02:
                vmin = 0 if n_pos_cells == 1 else np.percentile(values[values > 0], 2)
                vmax = np.nanmax(values) if n_pos_cells == 1 else np.percentile(values[values > 0], 98)
                if vmin + vmax in [1, 100]:
                    vmin += 1e-12
                    vmax += 1e-12

        # if None: min/max from data
        # if positive and sum up to 1, take fraction
        # if positive and sum up to 100, take percentage
        # otherwise take the data
        _vmin = (
            np.nanmin(values)
            if vmin is None
            else np.nanpercentile(values, vmin * 100)
            if (vmin + vmax == 1 and 0 <= vmin < vmax)
            else np.nanpercentile(values, vmin)
            if (vmin + vmax == 100 and 0 <= vmin < vmax)
            else vmin
        )
        _vmax = (
            np.nanmax(values)
            if vmax is None
            else np.nanpercentile(values, vmax * 100)
            if (vmin + vmax == 1 and 0 <= vmin < vmax)
            else np.nanpercentile(values, vmax)
            if (vmin + vmax == 100 and 0 <= vmin < vmax)
            else vmax
        )

        if sym_c and _vmin < 0 and _vmax > 0:
            bounds = np.nanmax([np.abs(_vmin), _vmax])
            bounds = bounds * np.array([-1, 1])
            _vmin, _vmax = bounds

        if frontier:
            # main_debug("drawing frontier")
            _scatter_projection(
                ax,
                points,
                projection,
                s=kwargs["s"] * 2,
                c="0.0",
                lw=2,
                rasterized=rasterized,
            )
            _scatter_projection(
                ax,
                points,
                projection,
                s=kwargs["s"] * 2,
                c="1.0",
                lw=0,
                rasterized=rasterized,
            )
            _scatter_projection(
                ax,
                points,
                projection,
                c=values,
                cmap=cmap,
                vmin=_vmin,
                vmax=_vmax,
                plotnonfinite=True,
                **kwargs,
            )
        elif contour:
            # main_debug("drawing contour")
            # try:
            #     from shapely.geometry import Polygon, MultiPoint, Point
            # except ImportError:
            #     raise ImportError(
            #         "If you want to use the tricontourf in plotting function, you need to install `shapely` "
            #         "package via `pip install shapely` see more details at https://pypi.org/project/Shapely/,"
            #     )
            #
            # x, y = points[:, :2].T
            # triang = tri.Triangulation(x, y)
            # concave_hull, edge_points = alpha_shape(x, y, alpha=calpha)
            # ax = plot_polygon(concave_hull, ax=ax)
            #
            # # Use the mean distance between the triangulated x & y poitns
            # x2 = x[triang.triangles].mean(axis=1)
            # y2 = y[triang.triangles].mean(axis=1)
            # ##note the very obscure mean command, which, if not present causes an error.
            # ##now we need some masking condition.
            #
            # # Create an empty set to fill with zeros and ones
            # cond = np.empty(len(x2))
            # # iterate through points checking if the point lies within the polygon
            # for i in range(len(x2)):
            #     cond[i] = concave_hull.contains(Point(x2[i], y2[i]))
            #
            # mask = np.where(cond, 0, 1)
            # # apply masking
            # triang.set_mask(mask)

            ccmap = "viridis" if ccmap is None else ccmap
            # # ax.tricontourf(triang, values, cmap=ccmap)
            # _scatter_projection(x, y,
            #            c=values,
            #            cmap=cmap,
            #            plotnonfinite=True,
            #            **kwargs, )
            import seaborn as sns

            df = pd.DataFrame(points, columns=["x", "y", "z"][: points.shape[1]])
            ax = sns.kdeplot(
                data=df.iloc[:, :2],
                x="x",
                y="y",
                fill=True,
                alpha=calpha,
                palette=ccmap,
                ax=ax,
                thresh=0,
                levels=100,
            )
            _scatter_projection(
                ax,
                points,
                projection,
                c=values,
                cmap=cmap,
                vmin=_vmin,
                vmax=_vmax,
                plotnonfinite=True,
                **kwargs,
            )
        else:
            # main_debug("drawing without frontiers and contour")
            # main_debug("using cmap: %s" % (str(cmap)))
            if geo:
                _geo_projection(
                    ax,
                    points,
                    values=values,
                    cmap=cmap,
                    vmin=_vmin,
                    vmax=_vmax,
                    **kwargs,
                )
            else:
                _scatter_projection(
                    ax,
                    points,
                    projection,
                    c=values,
                    cmap=cmap,
                    vmin=_vmin,
                    vmax=_vmax,
                    **kwargs,
                )

        if "norm" in kwargs:
            norm = kwargs["norm"]
        else:
            norm = matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)

        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(values)
        if show_colorbar:
            cb = plt.colorbar(mappable, cax=set_colorbar(ax, inset_dict), ax=ax)
            cb.set_alpha(1)
            cb.draw_all()
            cb.locator = MaxNLocator(nbins=3, integer=True)
            cb.update_ticks()

        cmap = matplotlib.cm.get_cmap(cmap)
        colors = cmap(values)
    # No color (just pick the midpoint of the cmap)
    else:
        # main_debug("drawing points without color passed in args, using midpoint of the cmap")
        colors = plt.get_cmap(cmap)(0.5)
        if geo:
            _geo_projection(ax, points, color=colors, **kwargs)
        else:
            _scatter_projection(ax, points, projection, c=colors, **kwargs)

    if show_legend and legend_elements is not None:
        if len(unique_labels) == 1 and show_legend == "on data":
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 1),
                loc=matplotlib.rcParams["legend.loc"],
                ncol=len(unique_labels) // 20 + 1,
                prop=dict(size=8),
            )
        elif len(unique_labels) > 1 and show_legend == "on data":
            font_color = "white" if background in ["black", "#ffffff"] else "black"
            for i in unique_labels:
                if i == "other":
                    continue
                if not geo:
                    color_cnt_x, color_cnt_y = np.nanmedian(points[np.where(labels == i)[0], :2].astype("float"), 0)
                else:
                    color_cnt_x = np.nanmedian(points[np.where(labels == i)[0]].centroid.x.astype("float"), 0)
                    color_cnt_y = np.nanmedian(points[np.where(labels == i)[0]].centroid.x.astype("float"), 0)
                txt = plt.text(
                    color_cnt_x,
                    color_cnt_y,
                    str(i),
                    color=_select_font_color(font_color),
                    zorder=1000,
                    verticalalignment="center",
                    horizontalalignment="center",
                    weight="bold",
                )  #
                txt.set_path_effects(
                    [
                        PathEffects.Stroke(linewidth=1.5, foreground=font_color, alpha=0.8),
                        PathEffects.Normal(),
                    ]
                )
        else:
            show_legend = "best" if show_legend == "on data" else show_legend
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 1),
                loc=show_legend,
                ncol=len(unique_labels) // 20 + 1,
            )
    else:
        # main_debug("hiding legend")
        pass

    return ax, colors


def _datashade_points(
    points,
    ax=None,
    labels=None,
    values=None,
    highlights=None,
    cmap="blue",
    color_key=None,
    color_key_cmap="Spectral",
    background="black",
    width=7,
    height=5,
    show_legend=True,
    vmin=2,
    vmax=98,
    sort="raw",
    projection="2d",
    **kwargs,
):
    import datashader as ds
    import datashader.transfer_functions as tf
    import matplotlib.pyplot as plt

    dpi = plt.rcParams["figure.dpi"]
    width, height = width * dpi, height * dpi

    """Use datashader to plot points"""
    extent = _get_extent(points)
    canvas = ds.Canvas(
        plot_width=int(width),
        plot_height=int(height),
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )
    data = pd.DataFrame(points, columns=("x", "y"))

    legend_elements = None

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(labels.shape[0], points.shape[0])
            )

        labels = np.array(labels, dtype="str")
        data["label"] = pd.Categorical(labels)
        if color_key is None and color_key_cmap is None:
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
            result = tf.shade(aggregation, how="eq_hist")
        elif color_key is None:
            cmap = matplotlib.cm.get_cmap(color_key_cmap)
            cmap.set_bad("lightgray")
            # add plotnonfinite=True to canvas.points

            if highlights is None:
                aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
                unique_labels = np.unique(labels)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
            else:
                highlights.append("other")
                unique_labels = np.array(highlights)
                num_labels = unique_labels.shape[0]
                color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
                color_key[-1] = "#bdbdbd"  # lightgray hex code https://www.color-hex.com/color/d3d3d3

                labels[[i not in highlights for i in labels]] = "other"
                data["label"] = pd.Categorical(labels)

                # reorder data so that highlighting points will be on top of background points
                highlight_ids, background_ids = (
                    data["label"] != "other",
                    data["label"] == "other",
                )
                reorder_data = data.copy(deep=True)
                (reorder_data.iloc[: sum(background_ids), :], reorder_data.iloc[sum(background_ids) :, :],) = (
                    data.iloc[background_ids, :],
                    data.iloc[highlight_ids, :],
                )
                aggregation = canvas.points(reorder_data, "x", "y", agg=ds.count_cat("label"))

            legend_elements = [Patch(facecolor=color_key[i], label=k) for i, k in enumerate(unique_labels)]
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")
        else:
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))

            legend_elements = [Patch(facecolor=color_key[k], label=k) for k in color_key.keys()]
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")

    # Color by values
    elif values is not None:
        cmap_ = matplotlib.cm.get_cmap(cmap)
        cmap_.set_bad("lightgray")

        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(values.shape[0], points.shape[0])
            )
        # reorder data so that high values data will be on top of background data
        sorted_id = np.argsort(abs(values)) if sort == "abs" else np.argsort(values)
        values, data = values[sorted_id], data.iloc[sorted_id, :]

        values[np.isnan(values)] = 0
        _vmin = np.min(values) if vmin is None else np.percentile(values, vmin)
        _vmax = np.min(values) if vmin is None else np.percentile(values, vmax)

        values = np.clip(values, _vmin, _vmax)

        unique_values = np.unique(values)
        if unique_values.shape[0] >= 256:
            min_val, max_val = np.min(values), np.max(values)
            bin_size = (max_val - min_val) / 255.0
            data["val_cat"] = pd.Categorical(np.round((values - min_val) / bin_size).astype(np.int16))
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")
        else:
            data["val_cat"] = pd.Categorical(values)
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key_cols = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, unique_values.shape[0])))
            color_key = dict(zip(unique_values, color_key_cols))
            result = tf.shade(aggregation, color_key=color_key, how="eq_hist")

    # Color by density (default datashader option)
    else:
        aggregation = canvas.points(data, "x", "y", agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap))

    if background is not None:
        result = tf.set_background(result, background)

    if ax is not None:
        _embed_datashader_in_an_axis(result, ax)
        if show_legend and legend_elements is not None:
            if len(unique_labels) > 1 and show_legend == "on data":
                font_color = "white" if background == "black" else "black"
                for i in unique_labels:
                    color_cnt = np.nanmedian(points.iloc[np.where(labels == i)[0], :2], 0)
                    txt = plt.text(
                        color_cnt[0],
                        color_cnt[1],
                        str(i),
                        color=_select_font_color(font_color),
                        zorder=1000,
                        verticalalignment="center",
                        horizontalalignment="center",
                        weight="bold",
                    )  #
                    txt.set_path_effects(
                        [
                            PathEffects.Stroke(linewidth=1.5, foreground=font_color, alpha=0.8),
                            PathEffects.Normal(),
                        ]
                    )
            else:
                if type(show_legend) == "str":
                    ax.legend(
                        handles=legend_elements,
                        loc=show_legend,
                        ncol=len(unique_labels) // 15 + 1,
                    )
                else:
                    ax.legend(
                        handles=legend_elements,
                        loc="best",
                        ncol=len(unique_labels) // 15 + 1,
                    )
        return ax
    else:
        return result


def interactive(
    umap_object,
    labels=None,
    values=None,
    hover_data=None,
    theme=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=7,
    height=5,
    point_size=None,
):
    """Create an interactive bokeh plot of a UMAP embedding.
    While static plots are useful, sometimes a plot that
    supports interactive zooming, and hover tooltips for
    individual points is much more desireable. This function
    provides a simple interface for creating such plots. The
    result is a bokeh plot that will be displayed in a notebook.
    Note that more complex tooltips etc. will require custom
    code -- this is merely meant to provide fast and easy
    access to interactive plotting.
    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.
    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.
    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.
    hover_data: DataFrame, shape (n_samples, n_tooltip_features)
    (optional, default None)
        A dataframe of tooltip data. Each column of the dataframe
        should be a Series of length ``n_samples`` providing a value
        for each data point. Column names will be used for
        identifying information within the tooltip.
    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'
    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    width: int (optional, default 800)
        The desired width of the plot in pixels.
    height: int (optional, default 800)
        The desired height of the plot in pixels
    Returns
    -------
    """
    import bokeh.plotting as bpl
    import bokeh.transform as btr

    # from bokeh.plotting import output_notebook, output_file, show
    import datashader as ds
    import holoviews as hv
    import holoviews.operation.datashader as hd
    import matplotlib.pyplot as plt

    dpi = plt.rcParams["figure.dpi"]
    width, height = width * dpi, height * dpi

    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError("Conflicting options; only one of labels or values should be set")

    points = umap_object.embedding_

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(umap_object.embedding_, columns=("x", "y"))

    if labels is not None:
        data["label"] = labels

        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))

        if isinstance(color_key, dict):
            data["color"] = pd.Series(labels).map(color_key)
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError("Color key must have enough colors for the number of labels")

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            data["color"] = pd.Series(labels).map(new_color_key)

        colors = "color"

    elif values is not None:
        data["value"] = values
        palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        colors = btr.linear_cmap("value", palette, low=np.min(values), high=np.max(values))

    else:
        colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

    if points.shape[0] <= width * height // 10:

        if hover_data is not None:
            tooltip_dict = {}
            for col_name in hover_data:
                data[col_name] = hover_data[col_name]
                tooltip_dict[col_name] = "@" + col_name
            tooltips = list(tooltip_dict.items())
        else:
            tooltips = None

        # bpl.output_notebook(hide_banner=True) # this doesn't work for non-notebook use
        data_source = bpl.ColumnDataSource(data)

        plot = bpl.figure(
            width=width,
            height=height,
            tooltips=tooltips,
            background_fill_color=background,
        )
        plot.circle(x="x", y="y", source=data_source, color=colors, size=point_size)

        plot.grid.visible = False
        plot.axis.visible = False

        # bpl.show(plot)
    else:
        if hover_data is not None:
            warnings.warn(
                "Too many points for hover data -- tooltips will not" "be displayed. Sorry; try subssampling your data."
            )
        hv.extension("bokeh")
        hv.output(size=300)
        hv.opts('RGB [bgcolor="{}", xaxis=None, yaxis=None]'.format(background))
        if labels is not None:
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["color"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("color"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        elif values is not None:
            min_val = data.values.min()
            val_range = data.values.max() - min_val
            data["val_cat"] = pd.Categorical((data.values - min_val) // (val_range // 256))
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["val_cat"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("val_cat"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        else:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count(),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )

    return plot


# ---------------------------------------------------------------------------------------------------
# plotting utilities borrow from velocyto
# link - https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb


def despline(ax=None):
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def despline_all(ax=None, sides=None):
    # removing the default axis on all sides:
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax

    if sides is None:
        sides = ["bottom", "right", "top", "left"]
    for side in sides:
        ax.spines[side].set_visible(False)


def deaxis_all(ax=None):
    # removing the axis ticks
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def minimal_xticks(start, end):
    import matplotlib.pyplot as plt

    end_ = np.around(end, -int(np.log10(end)) + 1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""] * len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)


def minimal_yticks(start, end):
    import matplotlib.pyplot as plt

    end_ = np.around(end, -int(np.log10(end)) + 1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""] * len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)


def set_spine_linewidth(ax, lw):
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(lw)

    return ax


# ---------------------------------------------------------------------------------------------------
# scatter plot utilities


def scatter_with_colorbar(fig, ax, x, y, c, cmap, **scatter_kwargs):
    # https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    g = ax.scatter(x, y, c=c, cmap=cmap, **scatter_kwargs)
    fig.colorbar(g, cax=cax, orientation="vertical")

    return fig, ax


def scatter_with_legend(fig, ax, df, font_color, x, y, c, cmap, legend, **scatter_kwargs):
    import matplotlib.patheffects as PathEffects
    import seaborn as sns

    unique_labels = np.unique(c)

    if legend == "on data":
        _ = sns.scatterplot(x, y, hue=c, palette=cmap, ax=ax, legend=False, **scatter_kwargs)

        for i in unique_labels:
            color_cnt = np.nanmedian(df.iloc[np.where(c == i)[0], :2], 0)
            txt = ax.text(
                color_cnt[0],
                color_cnt[1],
                str(i),
                color=font_color,
                zorder=1000,
                verticalalignment="center",
                horizontalalignment="center",
                weight="bold",
            )  # c
            txt.set_path_effects(
                [
                    PathEffects.Stroke(linewidth=1.5, foreground=font_color, alpha=0.8),
                    PathEffects.Normal(),
                ]  # 'w'
            )
    else:
        _ = sns.scatterplot(x, y, hue=c, palette=cmap, ax=ax, legend="full", **scatter_kwargs)
        ax.legend(loc=legend, ncol=unique_labels // 15)

    return fig, ax


def set_colorbar(ax, inset_dict={}):
    """https://matplotlib.org/3.1.0/gallery/axes_grid1/demo_colorbar_with_inset_locator.html"""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if len(inset_dict) == 0:
        # see more at https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html
        axins = inset_axes(
            ax,
            width="12%",  # width = 5% of parent_bbox width
            height="100%",  # height : 50%
            loc="upper right",
            bbox_to_anchor=(0.85, 0.97, 0.145, 0.17),
            bbox_transform=ax.transAxes,
            borderpad=1.85,
        )
    else:
        axins = inset_axes(ax, bbox_transform=ax.transAxes, **inset_dict)

    return axins


def arrowed_spines(ax, columns, background="white"):
    """https://stackoverflow.com/questions/33737736/matplotlib-axis-arrow-tip
    modified based on Answer 6
    """
    if type(columns) == str:
        columns = [columns.upper() + " 0", columns.upper() + " 1"]
    import matplotlib.pyplot as plt

    fig = plt.gcf()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    despline_all(ax)

    # removing the axis ticks
    deaxis_all(ax)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length (x-axis)
    hw = 1.0 / 20.0 * (ymax - ymin)
    hl = 1.0 / 20.0 * (xmax - xmin)
    lw = 1.0  # axis line width
    ohg = 0.2  # arrow overhang

    # compute matching arrowhead length and width (y-axis)
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    fc, ec = ("w", "w") if background in ["black", "#ffffff"] else ("k", "k")
    ax.arrow(
        xmin,
        ymin,
        hl * 5 / 2,
        0,
        fc=fc,
        ec=ec,
        lw=lw,
        head_width=hw / 2,
        head_length=hl / 2,
        overhang=ohg / 2,
        length_includes_head=True,
        clip_on=False,
    )
    ax.arrow(
        xmin,
        ymin,
        0,
        hw * 5 / 2,
        fc=fc,
        ec=ec,
        lw=lw,
        head_width=yhw / 2,
        head_length=yhl / 2,
        overhang=ohg / 2,
        length_includes_head=True,
        clip_on=False,
    )

    ax.text(
        xmin + hl * 2.5 / 2,
        ymin - 1.5 * hw / 2,
        columns[0],
        ha="center",
        va="center",
        rotation=0,
        # size=hl * 5 / (2 * len(str(columns[0]))) * 20,
        # size=matplotlib.rcParams['axes.titlesize'],
        size=np.clip((hl + yhw) * 8 / 2, 6, 18),
    )
    ax.text(
        xmin - 1.5 * yhw / 2,
        ymin + hw * 2.5 / 2,
        columns[1],
        ha="center",
        va="center",
        rotation=90,
        # size=hw * 5 / (2 * len(str(columns[1]))) * 20,
        # size=matplotlib.rcParams['axes.titlesize'],
        size=np.clip((hl + yhw) * 8 / 2, 6, 18),
    )

    return ax


# ---------------------------------------------------------------------------------------------------
# vector field plot related utilities


def quiver_autoscaler(X_emb, V_emb):
    """Function to automatically calculate the value for the scale parameter of quiver plot, adapted from scVelo
    Parameters
    ----------
        X_emb: `np.ndarray`
            X, Y-axis coordinates
        V_emb:  `np.ndarray`
            Velocity (U, V) values on the X, Y-axis
    Returns
    -------
        The scale for quiver plot
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    scale_factor = np.ptp(X_emb, 0).mean()
    X_emb = X_emb - X_emb.min(0)

    if len(V_emb.shape) == 3:
        Q = ax.quiver(
            X_emb[0] / scale_factor,
            X_emb[1] / scale_factor,
            V_emb[0],
            V_emb[1],
            angles="xy",
            scale_units="xy",
            scale=None,
        )
    else:
        Q = ax.quiver(
            X_emb[:, 0] / scale_factor,
            X_emb[:, 1] / scale_factor,
            V_emb[:, 0],
            V_emb[:, 1],
            angles="xy",
            scale_units="xy",
            scale=None,
        )

    Q._init()
    fig.clf()
    plt.close(fig)

    return Q.scale / scale_factor * 2


def default_quiver_args(arrow_size, arrow_len=None):
    if isinstance(arrow_size, (list, tuple)) and len(arrow_size) == 3:
        head_w, head_l, ax_l = arrow_size
    elif type(arrow_size) in [int, float]:
        head_w, head_l, ax_l = 10 * arrow_size, 12 * arrow_size, 8 * arrow_size
    else:
        head_w, head_l, ax_l = 10, 12, 8

    scale = 1 / arrow_len if arrow_len is not None else 1 / arrow_size

    return head_w, head_l, ax_l, scale


# ---------------------------------------------------------------------------------------------------
def _plot_traj(y0, t, args, integration_direction, ax, color, lw, f):
    from dynamo.tools.utils import integrate_vf

    _, y = integrate_vf(y0, t, args, integration_direction, f)  # integrate_vf_ivp

    ax.plot(*y.transpose(), color=color, lw=lw, linestyle="dashed", alpha=0.5)

    ax.scatter(*y0.transpose(), color=color, marker="*")

    return ax


# ---------------------------------------------------------------------------------------------------
# streamline related aesthetics
# ---------------------------------------------------------------------------------------------------


def set_arrow_alpha(ax=None, alpha=1):
    from matplotlib import patches

    ax = plt.gca() if ax is None else ax

    # iterate through the children of ax
    for art in ax.get_children():
        # we are only interested in FancyArrowPatches
        if not isinstance(art, patches.FancyArrowPatch):
            continue
        art.set_alpha(alpha)


def set_stream_line_alpha(s=None, alpha=1):
    """s has to be a StreamplotSet"""
    s.lines.set_alpha(alpha)


# ---------------------------------------------------------------------------------------------------
# save_fig figure related
# ---------------------------------------------------------------------------------------------------


def save_fig(
    path=None,
    prefix=None,
    dpi=None,
    ext="pdf",
    transparent=True,
    close=True,
    verbose=True,
):
    """Save a figure from pyplot.
    code adapated from http://www.jesshamrick.com/2012/09/03/saving-figures-from-pyplot/
    Parameters
    ----------
         path: `string`
            The path (and filename, without the extension) to save_fig the
            figure to.
        prefix: `str` or `None`
            The prefix added to the figure name. This will be automatically set
            accordingly to the plotting function used.
        dpi: [ None | scalar > 0 | 'figure' ]
            The resolution in dots per inch. If None, defaults to rcParams["savefig.dpi"].
            If 'figure', uses the figure's dpi value.
        ext: `string` (default='pdf')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
        close: `boolean` (default=True)
            Whether to close the figure after saving.  If you want to save_fig
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.
        verbose: boolean (default=True)
            Whether to print information about when and where the image
            has been saved.
    """
    import matplotlib.pyplot as plt

    if path is None:
        path = os.getcwd() + "/"

    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == "":
        directory = "."
    if filename == "":
        filename = "spateo_savefig"

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save_fig to
    savepath = (
        os.path.join(directory, filename + "." + ext)
        if prefix is None
        else os.path.join(directory, prefix + "_" + filename + "." + ext)
    )

    if verbose:
        print(f"Saving figure to {savepath}...")

    # Actually save the figure
    plt.savefig(
        savepath,
        dpi=300 if dpi is None else dpi,
        transparent=transparent,
        format=ext,
        bbox_inches="tight",
    )

    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")


# ---------------------------------------------------------------------------------------------------
def alpha_shape(x, y, alpha):
    # Start Using SHAPELY
    try:
        import shapely.geometry as geometry
        from shapely.geometry import MultiPoint
        from shapely.ops import cascaded_union, polygonize
    except ImportError:
        raise ImportError(
            "If you want to use the tricontourf in plotting function, you need to install `shapely` "
            "package via `pip install shapely` see more details at https://pypi.org/project/Shapely/,"
        )

    from scipy.spatial import Delaunay

    crds = np.array([x.flatten(), y.flatten()]).transpose()
    points = MultiPoint(crds)

    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points


# View the polygon and adjust alpha if needed
def plot_polygon(polygon, margin=1, fc="#999999", ec="#000000", fill=True, ax=None, **kwargs):
    try:
        from descartes.patch import PolygonPatch
    except ImportError:
        raise ImportError(
            "If you want to use the tricontourf in plotting function, you need to install `descartes` "
            "package via `pip install descartes` see more details at https://pypi.org/project/descartes/,"
        )

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    margin = margin
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc=fc, ec=ec, fill=fill, zorder=-1, lw=3, alpha=0.4, **kwargs)
    ax.add_patch(patch)

    return ax


# ---------------------------------------------------------------------------------------------------
# the following Loess class is taken from:
# link: https://github.com/joaofig/pyloess/blob/master/pyloess/Loess.py


def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):
    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n - 1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n - 1:
                min_range.insert(0, i0 - 1)
            elif distances[i0 - 1] < distances[i1 + 1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)


def _convert_to_geo_dataframe(adata, basis):
    # convert to AnnData with GeoDataFrame as obs
    adata.obs[basis] = pd.Series(adata.obsm[basis]).apply(loads, hex=True).values
    adata.obs = gpd.GeoDataFrame(adata.obs, geometry=basis)
    return adata


def save_return_show_fig_utils(
    save_show_or_return: Literal["save", "show", "return", "both", "all"],
    show_legend: bool,
    background: str,
    prefix: str,
    save_kwargs: Dict,
    total_panels: int,
    fig: matplotlib.figure.Figure,
    axes: matplotlib.axes.Axes,
    return_all: bool,
    return_all_list: Union[List, Tuple, None],
) -> Optional[Tuple]:

    from ...configuration import reset_rcParams
    from ...tools.utils import update_dict

    if show_legend:
        plt.subplots_adjust(right=0.85)

    if save_show_or_return in ["save", "both", "all"]:
        s_kwargs = {
            "path": None,
            "prefix": prefix,
            "dpi": None,
            "ext": "pdf",
            "transparent": True,
            "close": True if save_show_or_return == "save" else False,
            "verbose": True,
        }
        s_kwargs = update_dict(s_kwargs, save_kwargs)

        save_fig(**s_kwargs)
        if background is not None:
            reset_rcParams()
    if save_show_or_return in ["show", "both", "all"]:
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     plt.tight_layout()

        plt.show()
        if background is not None:
            reset_rcParams()
    if save_show_or_return in ["return", "all"]:
        if background is not None:
            reset_rcParams()

        if return_all:
            return (fig, *return_all_list) if total_panels > 1 else (fig, *return_all_list)
        else:
            return (fig, axes) if total_panels > 1 else (fig, axes)


# ---------------------------------------------------------------------------------------------------
# for plotting: subset and reorder data array
# ---------------------------------------------------------------------------------------------------
def _get_array_values(
    X: Union[np.ndarray, scipy.sparse.base.spmatrix],
    dim_names: pd.Index,
    keys: List[str],
    axis: Literal[0, 1],
    backed: bool,
):
    """
    Subset and reorder data array, given array and corresponding array index.

    Args:
        X : np.ndarray or scipy sparse matrix
        dim_names : pd.Index
            Names of
        keys : list of str
            Index names to subset
        axis : int, 0 or 1
            Subset rows or columns of 'X' (0 for rows, 1 for columns)
        backed : bool
            Interfaces w/ AnnData objects; is True if AnnData is backed to disk

    Returns:
         matrix : np.ndarray
    """
    mutable_idxer = [slice(None), slice(None)]
    idx = dim_names.get_indexer(keys)

    if backed:
        idx_order = np.argsort(idx)
        rev_idxer = mutable_idxer.copy()
        mutable_idxer[axis] = idx[idx_order]
        rev_idxer[axis] = np.argsort(idx_order)
        matrix = X[tuple(mutable_idxer)][tuple(rev_idxer)]
    else:
        mutable_idxer[axis] = idx
        matrix = X[tuple(mutable_idxer)]

    from scipy.sparse import issparse

    if issparse(matrix):
        matrix = matrix.toarray()

    return matrix


# ---------------------------------------------------------------------------------------------------
# for plotting: generating object to map from feature magnitudes to color intensities
# ---------------------------------------------------------------------------------------------------
def check_colornorm(
    vmin: Union[None, float] = None,
    vmax: Union[None, float] = None,
    vcenter: Union[None, float] = None,
    norm: Union[None, matplotlib.colors.Normalize] = None,
):
    """
    When plotting continuous variables, configure a normalizer object for the purposes of mapping the data to varying
    color intensities.

    Args:
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

    Returns:
         normalize : `matplotlib.colors.Normalize` object
            The normalizing object that scales data, typically into the interval [0, 1], for the purposes of
            mapping to color intensities for plotting.
    """
    from matplotlib.colors import Normalize

    try:
        from matplotlib.colors import TwoSlopeNorm as DivNorm
    except ImportError:
        from matplotlib.colors import DivergingNorm as DivNorm

    if norm is not None:
        if (vmin is not None) or (vmax is not None) or (vcenter is not None):
            raise ValueError("Passing both norm and vmin/vmax/vcenter is not allowed.")
    else:
        if vcenter is not None:
            norm = DivNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


# ---------------------------------------------------------------------------------------------------
# for plotting: ensure no duplicate keyword arguments
# ---------------------------------------------------------------------------------------------------
def deduplicate_kwargs(kwargs_dict, **kwargs):
    """
    Given a dictionary of plot parameters (kwargs_dict) and any number of additional keyword arguments,
    merge the parameters into a single consolidated dictionary to avoid argument duplication errors.
    If kwargs_dict contains a key that matches any of the additional keyword arguments, only the value in kwargs_dict is
    kept.

    Args:
        kwargs_dict : dict
            Each key is a variable name and each value is the value of that variable
        kwargs :
            Any additional keyword arguments, the keywords of which may or may not already be in 'kwargs_dict'
    """
    kwargs.update(kwargs_dict)

    return kwargs


# ---------------------------------------------------------------------------------------------------
# Dendrogram and utilities for dendrogram generation
# ---------------------------------------------------------------------------------------------------
def _dendrogram_sig(data: np.ndarray, method: str, **kwargs) -> Tuple[List[int], List[int], List[int], List[int]]:
    sch_linkage_params = {k for k in signature(sch.linkage).parameters.keys()}
    sch_dendro_params = {k for k in signature(sch.dendrogram).parameters.keys()}
    # Extract the kwargs that correspond to each function:
    link_kwargs = {k: v for k, v in kwargs.items() if k in sch_linkage_params}
    dendro_kwargs = {k: v for k, v in kwargs.items() if k in sch_dendro_params}

    # Row cluster:
    row_link = sch.linkage(data, method=method, **link_kwargs)
    row_dendro = sch.dendrogram(row_link, no_plot=True, **dendro_kwargs)
    row_order = row_dendro["leaves"]

    # Column cluster:
    col_link = sch.linkage(np.transpose(data), method=method, **link_kwargs)
    col_dendro = sch.dendrogram(col_link, no_plot=True, **dendro_kwargs)
    col_order = col_dendro["leaves"]

    return row_order, col_order, row_link, col_link


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def dendrogram(
    adata: AnnData,
    cat_key: str,
    n_pcs: int = 30,
    use_rep: Union[None, str] = None,
    var_names: Union[None, List[str]] = None,
    cor_method: str = "pearson",
    linkage_method: str = "complete",
    optimal_ordering: bool = False,
    key_added: Union[None, str] = None,
    inplace: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Computes a hierarchical clustering for the categories given by 'cat_key'.

    By default, the PCA representation is used unless `.X` has less than 50 variables.

    Alternatively, a list of `var_names` (e.g. genes) can be given. If this is the case, will subset to these
    features and use them for the dendrogram.

    Args:
        adata: object of class `anndata.AnnData`
        cat_key: Name of key in .obs specifying group labels for each sample
        n_pcs: Number of principal components to use in computing hierarchical clustering
        use_rep: Entry in .obsm to use for computing hierarchical clustering
        var_names: List of genes to define a subset of 'adata' to compute hierarchical clustering directly on
            expression values.
        cor_method: Correlation method to use. Options are 'pearson', 'kendall', and 'spearman'
        linkage_method: Linkage method to use. See :func:`scipy.cluster.hierarchy.linkage` for more information.
        optimal_ordering: Same as the optimal_ordering argument of :func:`scipy.cluster.hierarchy.linkage`
            which reorders the linkage matrix so that the distance between successive leaves is minimal.
        key_added: Sets key in .uns in which dendrogram information is saved.
            By default, the dendrogram information is added to `.uns[f'dendrogram_{cat_key}']`.
        inplace: If `True`, adds dendrogram information to `adata.uns[key_added]`, else this function returns the
            information.

    Returns:
        If `inplace=False`, returns dendrogram information, else adata object is updated in place with information
        stored in `adata.uns[key_added]`.
    """
    logger = lm.get_main_logger()

    if not isinstance(cat_key, list):
        cat_key = [cat_key]
    # For each category label given in 'cat_key':
    for cat in cat_key:
        if cat not in adata.obs_keys():
            logger.error(
                "'cat_key' has to be a valid observation. "
                f"Given value: {cat}, valid observations: {adata.obs_keys()}"
            )
        if not is_categorical_dtype(adata.obs[cat_key]):
            logger.error(
                "'cat_key' has to be a categorical observation. "
                f"Given value: {cat}, Column type: {adata.obs[cat].dtype}"
            )

    if var_names is None:
        # Choose representation to use for hierarchical clustering:
        if use_rep is None and n_pcs == 0:
            use_rep = "X"

        if use_rep is None:
            if adata.n_vars > n_pcs:
                if "X_pca" in adata.obsm.keys():
                    if n_pcs is not None and n_pcs > adata.obsm["X_pca"].shape[1]:
                        logger.error("Existing 'X_pca' does not have enough PCs.")
                    X = adata.obsm["X_pca"][:, :n_pcs]
                    logger.info(f"Using 'X_pca' with n_pcs = {X.shape[1]} to compute dendrogram...")
                else:
                    logger.warning(
                        "'n_pcs' was provided, but 'X_pca' does not already exist. If you meant to use "
                        "gene expression, set 'use_rep' = 'X' or 'n_pcs' = 0. For now, will proceed with "
                        "computing PCA representation and using rep 'X_pca'."
                    )
                    pca = PCA(
                        n_components=min(n_pcs, adata.X.shape[1] - 1),
                        svd_solver="arpack",
                        random_state=0,
                    )
                    fit = pca.fit(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else pca.fit(adata.X)
                    X_pca = (
                        fit.transform(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else fit.transform(adata.X)
                    )
                    adata.obsm["X_pca"] = X_pca

            else:
                logger.info("Using data matrix X directly")
                X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

        else:
            if use_rep in adata.obsm.keys() and n_pcs is not None:
                if n_pcs > adata.obsm[use_rep].shape[1]:
                    logger.error(
                        f"{use_rep} does not have enough dimensions. Provide a representation with equal or more "
                        f"dimensions than 'n_pcs' or lower 'n_pcs'."
                    )
                X = adata.obsm[use_rep][:, :n_pcs]
            elif use_rep in adata.obsm.keys() and n_pcs is None:
                X = adata.obsm[use_rep]
            elif use_rep == "X":
                X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
            else:
                logger.error("Did not find {} in `.obsm.keys()`. Needs to be compute first.".format(use_rep))

        rep_df = pd.DataFrame(X)

        categorical = adata.obs[cat_key[0]]
        # If multiple category keys are given, create new categories by merging their combinations:
        if len(cat_key) > 1:
            for cat in cat_key[1:]:
                categorical = (categorical.astype(str) + "_" + adata.obs[cat].astype(str)).astype("category")
        categorical.name = "_".join(cat_key)

        rep_df.set_index(categorical, inplace=True)
        categories = rep_df.index.categories
    else:
        gene_names = adata.var_names
        from .dotplot import adata_to_frame

        categories, rep_df = adata_to_frame(adata, gene_names, cat_key)

    # Aggregate values within categories using "mean":
    mean_df = rep_df.groupby(level=0).mean()

    corr_matrix = mean_df.T.corr(method=cor_method)
    corr_condensed = distance.squareform(1 - corr_matrix)
    z_var = sch.linkage(corr_condensed, method=linkage_method, optimal_ordering=optimal_ordering)
    dendro_info = sch.dendrogram(z_var, labels=list(categories), no_plot=True)

    dat = dict(
        linkage=z_var,
        cat_key=cat_key,
        use_rep=use_rep,
        cor_method=cor_method,
        linkage_method=linkage_method,
        categories_ordered=dendro_info["ivl"],
        categories_idx_ordered=dendro_info["leaves"],
        dendrogram_info=dendro_info,
        correlation_matrix=corr_matrix.values,
    )

    if inplace:
        if key_added is None:
            key_added = f'dendrogram_{"_".join(cat_key)}'
        logger.info_insert_adata(key_added, adata_attr="uns")
        adata.uns[key_added] = dat
    else:
        return dat


def plot_dendrogram(
    dendro_ax: matplotlib.axes.Axes,
    adata: AnnData,
    cat_key: str,
    dendrogram_key: Union[None, str] = None,
    orientation: Literal["top", "bottom", "left", "right"] = "right",
    remove_labels: bool = True,
    ticks: Union[None, Collection[float]] = None,
):
    """
    Plots dendrogram on the provided Axes, using the dendrogram information stored in `.uns[dendrogram_key]`

    Args:
        dendro_ax: object of class `matplotlib.axes.Axes`
        adata: object of class `anndata.AnnData`
            Contains dendrogram information as well as the data that will be plotted (and was used to hierarchically
            cluster)
        cat_key: Key in .obs containing category labels for all samples
        dendrogram_key:
        orientation: Specifies dendrogram placement relative to the plotting window.
            Options: 'top', 'bottom', 'left', 'right'
        remove_labels: Removes labels along the side that dendrogram is on, if any
        ticks: Assumes original ticks come from `scipy.cluster.hierarchy.dendrogram`, but if not can also pass a list
            of custom tick values.
    """

    logger = lm.get_main_logger()

    # Get dendrogram key:
    if not isinstance(dendrogram_key, str):
        if isinstance(cat_key, str):
            dendrogram_key = f"dendrogram_{cat_key}"
        elif isinstance(cat_key, list):
            dendrogram_key = f'dendrogram_{"_".join(cat_key)}'

    if dendrogram_key not in adata.uns:
        logger.warning(
            f"Dendrogram data not found (using key={dendrogram_key}). Running :func `st.pl.dendrogram` with "
            f"default parameters. For fine tuning it is recommended to run `st.pl.dendrogram` independently."
        )
        dendrogram(adata, cat_key, key_added=dendrogram_key)

    if "dendrogram_info" not in adata.uns[dendrogram_key]:
        raise ValueError(
            f"The given dendrogram key ({dendrogram_key!r}) does not contain valid dendrogram information."
        )

    def translate_pos(pos_list: List[float], new_ticks: List[int], old_ticks: Union[np.ndarray, List[int]]):
        """
        Transforms the dendrogram coordinates to a given new position.
        The xlabel_pos ('pos_list') and orig_ticks ('old_ticks') should be of the same length.

        This is mostly done for the heatmap case, where the position of the dendrogram leaves needs to be adjusted
        depending on the category size.

        Args:
            pos_list: List of dendrogram positions that should be translated
            new_ticks: Sorted list of desired tick positions (e.g. [0, 1, 2, 3])
            old_ticks: sorted list of original tick positions (e.g. [5, 15, 25, 35])
            This list is usually the default position used by `scipy.cluster.hierarchy.dendrogram`.

        Returns:
             new_xs: Translated list of positions
        """
        if not isinstance(old_ticks, list):
            old_ticks = old_ticks.tolist()
        new_xs = []
        for x_val in pos_list:
            if x_val in old_ticks:
                new_x_val = new_ticks[old_ticks.index(x_val)]
            else:
                # find smaller and bigger indices
                idx_next = np.searchsorted(old_ticks, x_val, side="left")
                idx_prev = idx_next - 1
                old_min = old_ticks[idx_prev]
                old_max = old_ticks[idx_next]
                new_min = new_ticks[idx_prev]
                new_max = new_ticks[idx_next]
                new_x_val = ((x_val - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            new_xs.append(new_x_val)

        return new_xs

    dendro_info = adata.uns[dendrogram_key]["dendrogram_info"]
    leaves = dendro_info["ivl"]
    icoord = np.array(dendro_info["icoord"])
    dcoord = np.array(dendro_info["dcoord"])

    orig_ticks = np.arange(5, len(leaves) * 10 + 5, 10).astype(float)
    # check that ticks has the same length as orig_ticks
    if ticks is not None and len(orig_ticks) != len(ticks):
        logger.warning("'ticks' argument does not have the same size as orig_ticks. The argument will be ignored.")
        ticks = None

    for xs, ys in zip(icoord, dcoord):
        if ticks is not None:
            xs = translate_pos(xs, ticks, orig_ticks)
        if orientation in ["right", "left"]:
            xs, ys = ys, xs
        dendro_ax.plot(xs, ys, color="#555555")

    dendro_ax.tick_params(bottom=False, top=False, left=False, right=False)
    ticks = ticks if ticks is not None else orig_ticks
    if orientation in ["right", "left"]:
        dendro_ax.set_yticks(ticks)
        dendro_ax.set_yticklabels(leaves, fontsize="small", rotation=0)
        dendro_ax.tick_params(labelbottom=False, labeltop=False)
        if orientation == "left":
            xmin, xmax = dendro_ax.get_xlim()
            dendro_ax.set_xlim(xmax, xmin)
            dendro_ax.tick_params(labelleft=False, labelright=True)
    else:
        dendro_ax.set_xticks(ticks)
        dendro_ax.set_xticklabels(leaves, fontsize="small", rotation=90)
        dendro_ax.tick_params(labelleft=False, labelright=False)
        if orientation == "bottom":
            ymin, ymax = dendro_ax.get_ylim()
            dendro_ax.set_ylim(ymax, ymin)
            dendro_ax.tick_params(labeltop=True, labelbottom=False)

    if remove_labels:
        dendro_ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    dendro_ax.grid(False)

    dendro_ax.spines["right"].set_visible(False)
    dendro_ax.spines["top"].set_visible(False)
    dendro_ax.spines["left"].set_visible(False)
    dendro_ax.spines["bottom"].set_visible(False)
