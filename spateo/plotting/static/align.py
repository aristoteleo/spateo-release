import math
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from ...tools.cluster.utils import integrate, to_dense_matrix
from ...tools.utils import compute_smallest_distance
from .utils import save_return_show_fig_utils


def slices_2d(
    slices: Union[AnnData, List[AnnData]],
    slices_key: Optional[Union[bool, str]] = None,
    label_key: Optional[str] = None,
    label_type: Optional[str] = None,
    spatial_key: str = "spatial",
    point_size: Optional[float] = None,
    n_sampling: int = -1,
    palette: Optional[dict] = None,
    ncols: int = 4,
    title: str = "",
    title_kwargs: Optional[dict] = None,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    axis_off: bool = True,
    axis_kwargs: Optional[dict] = None,
    ticks_off: bool = True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    height: float = 2,
    alpha: float = 1.0,  # TODO: alpha to be a key in adata
    cmap="tab20",
    center_coordinate: bool = False,
    gridspec_kws: Optional[dict] = None,
    return_palette: bool = False,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):
    # Check slices object.
    if isinstance(slices, AnnData):
        slices = [slices]

    # get spatial coords and labels
    spatial_coords = []
    labels = []
    slice_ids = []
    for i, s in enumerate(slices):
        if spatial_key in s.obsm.keys():
            spatial_coords.append(s.obsm[spatial_key].copy())
        else:
            raise ValueError(f"adata.obsm['{spatial_key}'] does not exist.")

        if label_key in s.obs.keys():
            labels.append(s.obs[label_key].copy())
            # label_type = "cluster"
        elif label_key in s.var_names:
            labels.append(s[:, label_key].X.A.copy().squeeze())
            # label_type = "scalar"
        else:
            raise ValueError(f"adata.obs['{label_key}'] or adata.var['{label_key}'] does not exist.")

        if (slices_key is not None) and (slices_key in s.obs.keys()):
            unique_id = np.unique(s.obs[slices_key])
            if len(unique_id) == 1:
                slice_ids.append(unique_id[0])
            else:
                raise ValueError(f"adata.obs['{slices_key}'] must have only one unique value.")
        else:
            slice_ids.append(str(i))

        assert (
            spatial_coords[-1].shape[0] == labels[-1].shape[0]
        ), "The number of spatial coordinates and labels must be the same. Please check the data."

    # infer the label_type if not specified
    if label_type is None:
        if labels[0].values.dtype in ["float16", "float32", "float64", "int16", "int32", "int64"]:
            label_type = "scalar"
        else:
            label_type = "cluster"

    # downsampling if n_sampling is set
    for i in range(len(slices)):
        sampling_idx = (
            np.random.choice(spatial_coords[i].shape[0], n_sampling, replace=False)
            if n_sampling > 0 and n_sampling < spatial_coords[i].shape[0]
            else np.arange(spatial_coords[i].shape[0])
        )
        spatial_coords[i] = spatial_coords[i][sampling_idx]
        labels[i] = labels[i][sampling_idx]

    # center the coordinates
    if center_coordinate:
        for i in range(len(slices)):
            spatial_coords[i] = spatial_coords[i] - np.mean(spatial_coords[i], axis=0)

    # Set the arrangement of subgraphs
    nrows = math.ceil(len(slices) / ncols)
    # create dataframe for ploting
    slices_spatial_data = pd.DataFrame(columns=["x", "y", "label", "slice_id", "col", "row"])
    for i in range(len(slices)):
        slices_spatial_data = pd.concat(
            [
                slices_spatial_data,
                pd.DataFrame(
                    {
                        "x": spatial_coords[i][:, 0],
                        "y": spatial_coords[i][:, 1],
                        "label": labels[i],
                        "slice_id": slice_ids[i],
                        "col": i % ncols,
                        "row": i // ncols,
                    }
                ),
            ],
            axis=0,
        )

    # set the aspect ratio of each subplot
    ptp_vec = slices_spatial_data[["x", "y"]].values.ptp(0)
    aspect_ratio = ptp_vec[0] / ptp_vec[1]

    # Set multi-plot grid for plotting.
    sns.set_theme(
        context="paper",
        style="white",
        font="Arial",
        font_scale=1,
        rc={
            # "font.size": font_size,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["Arial", "sans-serif", "Helvetica", "DejaVu Sans", "Bitstream Vera Sans"],
        },
    )

    # generate palette
    if (palette is None) and (label_type == "cluster"):
        palette = _agenerate_palette(*labels, cmap=cmap)
    else:
        palette = cmap

    # adjust the gridspec
    _gridspec_kws = {"wspace": 0.1, "hspace": 0.2}
    if gridspec_kws is not None:
        _gridspec_kws.update(gridspec_kws)

    if slices_key is False:
        _gridspec_kws["hspace"] = _gridspec_kws["wspace"] * aspect_ratio

    # determine the pointsize if not specified
    if point_size is None:
        point_size = 500 * height**2 * aspect_ratio / (slices_spatial_data.shape[0] / len(slices))

    # plotting
    g = sns.FacetGrid(
        slices_spatial_data,
        col="col",
        row="row",
        hue="label",
        palette=palette,
        sharex=True,
        sharey=True,
        height=height,
        aspect=aspect_ratio,
        despine=False,
        gridspec_kws=_gridspec_kws,
    )

    scatterplot_kwargs = {"x": "x", "y": "y", "alpha": alpha, "s": point_size, "legend": False, "edgecolor": None}
    scatterplot_kwargs.update(kwargs)

    g.map_dataframe(sns.scatterplot, **scatterplot_kwargs)

    for i, (col_val, ax) in enumerate(g.axes_dict.items()):
        if i < len(slices):
            if slices_key is False:
                ax.set_title("")
            else:
                ax.set_title(f"Slice {slice_ids[i]}", title_kwargs)
        else:
            ax.set_title("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        ax.set_aspect("equal")
        if axis_off:
            ax.axis("off")
        if ticks_off:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")

    # create legend
    if show_legend:
        if label_type == "cluster":
            _legend_kwargs = {
                "loc": "center left",
                "bbox_to_anchor": (1, 0.5),
                "prop": {"family": "Arial", "size": 10},
                "fancybox": False,
                "edgecolor": "black",
                "framealpha": 1,
                "columnspacing": 0.8,
                "handletextpad": 0.5,
                "frameon": True,
            }
            if legend_kwargs:
                _legend_kwargs.update(legend_kwargs)
                # if legend_kwargs.get('loc', None) == 'upper center':
                #     _legend_kwargs['bbox_to_anchor'] = (0.5, 0)
            legend_elements = [
                mpl.lines.Line2D(
                    [0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=6, markeredgecolor="k"
                )
                for k, v in palette.items()
            ]
            g.figure.legend(handles=legend_elements, **_legend_kwargs)
        else:
            _legend_kwargs = {
                "loc": "center left",
                # 'bbox_to_anchor': (1, 0.5, 0.5, 1.0),
                # 'prop': {'family': 'Arial', 'size': 10},
                # 'fancybox': False,
                # 'edgecolor': 'black',
                # 'framealpha': 1,
                # 'columnspacing': 0.5,
                # 'handletextpad': 0.1,
                # 'frameon': True,
            }
            if legend_kwargs:
                _legend_kwargs.update(legend_kwargs)
                # if legend_kwargs.get('loc', None) == 'upper center':
                #     _legend_kwargs['bbox_to_anchor'] = (0.5, 0, 0.5, 1.0)
            # TODO: add colorbar for scalar value input
            label_values = slices_spatial_data["label"].values
            norm = mpl.colors.Normalize(vmin=None, vmax=None)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
            mappable.set_array(label_values)
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            g.figure.colorbar(
                mappable,
                use_gridspec=False,
                shrink=0.5,
                cax=inset_axes(
                    ax,
                    width="15%",
                    height="75%",
                    loc="center left",
                    # **_legend_kwargs,
                    bbox_to_anchor=(1.02, 0.0, 0.5, 1.0),
                    bbox_transform=ax.transAxes,
                ),
            )

    # TODO: add save_return_show_fig_utils
    # plt.tight_layout()
    if return_palette:
        return (
            save_return_show_fig_utils(
                save_show_or_return=save_show_or_return,
                show_legend=show_legend,
                background="white",
                prefix="multi_slices",
                save_kwargs=save_kwargs,
                total_panels=len(slice_ids),
                fig=g,
                axes=g,
                return_all=False,
                return_all_list=None,
            ),
            palette,
        )
    else:
        return save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=show_legend,
            background="white",
            prefix="multi_slices",
            save_kwargs=save_kwargs,
            total_panels=len(slice_ids),
            fig=g,
            axes=g,
            return_all=False,
            return_all_list=None,
        )
    # return g, palette


def overlay_slices_2d(
    slices: Union[AnnData, List[AnnData]],
    slices_key: Optional[Union[bool, str]] = None,
    label_key: Optional[str] = None,
    overlay_type: Literal["forward", "backward", "both"] = "both",
    spatial_key: str = "spatial",
    point_size: Optional[float] = None,
    n_sampling: int = -1,
    palette: Optional[dict] = None,
    ncols: int = 4,
    title: str = "",
    title_kwargs: Optional[dict] = None,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    axis_off: bool = True,
    axis_kwargs: Optional[dict] = None,
    ticks_off: bool = True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    height: float = 2,
    alpha: float = 1.0,  # TODO: alpha to be a key in adata
    cmap="tab20",
    center_coordinate: bool = True,  # different from slices_2d
    gridspec_kws: Optional[dict] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):
    # Check slices object.
    if isinstance(slices, AnnData):
        slices = [slices]

    # get spatial coords and labels
    spatial_coords = []
    labels = []
    slice_ids = []
    for i, s in enumerate(slices):
        if spatial_key in s.obsm.keys():
            spatial_coords.append(s.obsm[spatial_key].copy())
        else:
            raise ValueError(f"adata.obsm['{spatial_key}'] does not exist.")

        if label_key is not None:
            if label_key in s.obs.keys():
                labels.append(s.obs[label_key].copy())
                label_type = "cluster"
            elif label_key in s.var_names:
                labels.append(s[:, label_key].X.A.copy().squeeze())
                label_type = "scalar"
            else:
                raise ValueError(f"adata.obs['{label_key}'] or adata.var['{label_key}'] does not exist.")

            assert (
                spatial_coords[-1].shape[0] == labels[-1].shape[0]
            ), "The number of spatial coordinates and labels must be the same. Please check the data."
        else:
            label_type = "cluster"

        if (slices_key is not None) and (slices_key in s.obs.keys()):
            unique_id = np.unique(s.obs[slices_key])
            if len(unique_id) == 1:
                slice_ids.append(unique_id[0])
            else:
                raise ValueError(f"adata.obs['{slices_key}'] must have only one unique value.")
        else:
            slice_ids.append(str(i))

    # downsampling if n_sampling is set
    for i in range(len(slices)):
        sampling_idx = (
            np.random.choice(spatial_coords[i].shape[0], n_sampling, replace=False)
            if n_sampling > 0 and n_sampling < spatial_coords[i].shape[0]
            else np.arange(spatial_coords[i].shape[0])
        )
        spatial_coords[i] = spatial_coords[i][sampling_idx]
        if label_key is not None:
            labels[i] = labels[i][sampling_idx]

    # center the coordinates
    if center_coordinate:
        for i in range(len(slices)):
            spatial_coords[i] = spatial_coords[i] - np.mean(spatial_coords[i], axis=0)

    # Set the arrangement of subgraphs
    nrows = math.ceil(len(slices) / ncols)
    # create dataframe for ploting
    slices_spatial_data = pd.DataFrame(columns=["x", "y", "label", "overlay_id", "slice_id", "col", "row"])
    for i in range(len(slices)):
        slices_spatial_data = pd.concat(
            [
                slices_spatial_data,
                pd.DataFrame(
                    {
                        "x": spatial_coords[i][:, 0],
                        "y": spatial_coords[i][:, 1],
                        "label": labels[i] if label_key is not None else "unknow",
                        "overlay_id": "current",
                        "slice_id": slice_ids[i],
                        "col": i % ncols,
                        "row": i // ncols,
                    }
                ),
            ],
            axis=0,
        )
        if (i > 0) and ((overlay_type == "forward") or (overlay_type == "both")):
            slices_spatial_data = pd.concat(
                [
                    slices_spatial_data,
                    pd.DataFrame(
                        {
                            "x": spatial_coords[i - 1][:, 0],
                            "y": spatial_coords[i - 1][:, 1],
                            "label": labels[i - 1] if label_key is not None else "unknow",
                            "overlay_id": "forward",
                            "slice_id": slice_ids[i - 1],
                            "col": i % ncols,
                            "row": i // ncols,
                        }
                    ),
                ],
                axis=0,
            )
        if (i < len(slices) - 1) and ((overlay_type == "backward") or (overlay_type == "both")):
            slices_spatial_data = pd.concat(
                [
                    slices_spatial_data,
                    pd.DataFrame(
                        {
                            "x": spatial_coords[i + 1][:, 0],
                            "y": spatial_coords[i + 1][:, 1],
                            "label": labels[i + 1] if label_key is not None else "unknow",
                            "overlay_id": "backward",
                            "slice_id": slice_ids[i + 1],
                            "col": i % ncols,
                            "row": i // ncols,
                        }
                    ),
                ],
                axis=0,
            )

    # set the aspect ratio of each subplot
    ptp_vec = slices_spatial_data[["x", "y"]].values.ptp(0)
    aspect_ratio = ptp_vec[0] / ptp_vec[1]

    # Set multi-plot grid for plotting.
    sns.set_theme(
        context="paper",
        style="white",
        font="Arial",
        font_scale=1,
        rc={
            # "font.size": font_size,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["Arial", "sans-serif", "Helvetica", "DejaVu Sans", "Bitstream Vera Sans"],
        },
    )

    # generate palette
    if label_key is not None:
        if (palette is None) and (label_type == "cluster"):
            palette = _agenerate_palette(*labels, cmap=cmap)
        else:
            palette = cmap
    else:
        palette = {
            "current": "red",
            "forward": "green",
            "backward": "blue",
        }

    # adjust the gridspec
    _gridspec_kws = {"wspace": 0.1, "hspace": 0.2}
    if gridspec_kws is not None:
        _gridspec_kws.update(gridspec_kws)

    if slices_key is False:
        _gridspec_kws["hspace"] = _gridspec_kws["wspace"] * aspect_ratio

    # determine the pointsize if not specified
    if point_size is None:
        point_size = 500 * height**2 * aspect_ratio / (slices_spatial_data.shape[0] / len(slices))

    # plotting
    g = sns.FacetGrid(
        slices_spatial_data,
        col="col",
        row="row",
        hue="label" if label_key is not None else "overlay_id",
        palette=palette,
        sharex=True,
        sharey=True,
        height=height,
        aspect=aspect_ratio,
        despine=False,
        gridspec_kws=_gridspec_kws,
    )

    scatterplot_kwargs = {"x": "x", "y": "y", "alpha": alpha, "s": point_size, "legend": False, "edgecolor": None}
    scatterplot_kwargs.update(kwargs)

    g.map_dataframe(sns.scatterplot, **scatterplot_kwargs)

    for i, (col_val, ax) in enumerate(g.axes_dict.items()):
        if i < len(slices):
            if slices_key is False:
                ax.set_title("")
            else:
                ax.set_title(f"Slice {slice_ids[i]}", title_kwargs)
        else:
            ax.set_title("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
        ax.set_aspect("equal")
        if axis_off:
            ax.axis("off")
        if ticks_off:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")

    # create legend
    if show_legend:
        if label_type == "cluster":
            _legend_kwargs = {
                "loc": "center left",
                "bbox_to_anchor": (1, 0.5),
                "prop": {"family": "Arial", "size": 10},
                "fancybox": False,
                "edgecolor": "black",
                "framealpha": 1,
                "columnspacing": 0.8,
                "handletextpad": 0.5,
                "frameon": True,
            }
            if legend_kwargs:
                _legend_kwargs.update(legend_kwargs)
                # if legend_kwargs.get('loc', None) == 'upper center':
                #     _legend_kwargs['bbox_to_anchor'] = (0.5, 0)
            legend_elements = [
                mpl.lines.Line2D(
                    [0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=6, markeredgecolor="k"
                )
                for k, v in palette.items()
            ]
            g.figure.legend(handles=legend_elements, **_legend_kwargs)
        else:
            _legend_kwargs = {
                "loc": "center left",
                # 'bbox_to_anchor': (1, 0.5, 0.5, 1.0),
                # 'prop': {'family': 'Arial', 'size': 10},
                # 'fancybox': False,
                # 'edgecolor': 'black',
                # 'framealpha': 1,
                # 'columnspacing': 0.5,
                # 'handletextpad': 0.1,
                # 'frameon': True,
            }
            if legend_kwargs:
                _legend_kwargs.update(legend_kwargs)
                # if legend_kwargs.get('loc', None) == 'upper center':
                #     _legend_kwargs['bbox_to_anchor'] = (0.5, 0, 0.5, 1.0)
            # TODO: add colorbar for scalar value input
            label_values = slices_spatial_data["label"].values
            norm = mpl.colors.Normalize(vmin=None, vmax=None)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
            mappable.set_array(label_values)
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            g.figure.colorbar(
                mappable,
                use_gridspec=False,
                shrink=0.5,
                cax=inset_axes(
                    ax,
                    width="15%",
                    height="75%",
                    loc="center left",
                    # **_legend_kwargs,
                    bbox_to_anchor=(1.02, 0.0, 0.5, 1.0),
                    bbox_transform=ax.transAxes,
                ),
            )

    # TODO: add save_return_show_fig_utils
    # plt.tight_layout()
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=show_legend,
        background="white",
        prefix="multi_slices",
        save_kwargs=save_kwargs,
        total_panels=len(slice_ids),
        fig=g,
        axes=g,
        return_all=False,
        return_all_list=None,
    )
    # return g, palette


def multi_slices(
    slices: Union[AnnData, List[AnnData]],
    slices_key: Optional[str] = None,
    label: Optional[str] = None,
    spatial_key: str = "align_spatial",
    layer: str = "X",
    point_size: Optional[float] = None,
    font_size: Optional[float] = 20,
    color: Optional[str] = "skyblue",
    palette: Optional[str] = None,
    alpha: float = 1.0,
    ncols: int = 4,
    ax_height: float = 1,
    dpi: int = 100,
    show_legend: bool = True,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):
    # Check slices object.
    if isinstance(slices, list):
        adatas = [s.copy() for s in slices]
        for i, s in enumerate(adatas):
            s.X = s.layers[layer].copy() if layer != "X" else s.X.copy()
            s.uns = {"__type": "UMI"}
            if slices_key is None:
                slices_key = "slices"
                s.obs[slices_key] = f"slice_{i}"
        adata = integrate(adatas=adatas, batch_key=slices_key)
    else:
        assert slices_key != None, "When `slices` is an anndata object, `slices_key` cannot be None."
        adata = slices.copy()
        adata.X = adata.layers[layer].copy() if layer != "X" else adata.X.copy()

    # Check label data and generate plotting data.
    slices_data = pd.DataFrame(adata.obsm[spatial_key][:, :2], columns=["x", "y"], dtype=float)
    slices_data[slices_key] = adata.obs[slices_key].values

    if label is None:
        label = "spatial coordinates"
        slices_data[label] = label
    elif label in adata.obs_keys():
        slices_data[label] = adata.obs[label].values
    elif label in adata.var_names:
        adata.X = to_dense_matrix(adata.X)
        slices_data[label] = adata[:, label].X
    else:
        raise ValueError("`label` is not a valid column names or gene name.")

    # Set the arrangement of subgraphs
    slices_id = slices_data[slices_key].unique().tolist()
    nrows = math.ceil(len(slices_id) / ncols)

    # Set the aspect ratio of each subplot
    spatial_coords = slices_data[["x", "y"]].values.copy()
    ptp_vec = spatial_coords.ptp(0)
    aspect_ratio = ptp_vec[0] / ptp_vec[1]
    ax_height = 2 if nrows == 1 and ax_height == 1 else ax_height
    axsize = (ax_height * aspect_ratio, ax_height * 2)

    # Set multi-plot grid for plotting.
    sns.set_theme(
        context="paper",
        style="white",
        font="Arial",
        font_scale=1,
        rc={
            "font.size": font_size,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["Arial", "sans-serif", "Helvetica", "DejaVu Sans", "Bitstream Vera Sans"],
        },
    )

    g = sns.FacetGrid(
        slices_data.copy(),
        col=slices_key,
        hue=label,
        palette=palette,
        sharex=True,
        sharey=True,
        height=axsize[1] * nrows,
        aspect=aspect_ratio,
        col_wrap=ncols,
        despine=False,
    )

    # Calculate the most suitable size of the point.
    if point_size is None:
        group_slices_data = slices_data.groupby(by=slices_key)
        min_dist_list = []
        for key, data in group_slices_data:
            sample_num = 1000 if len(data) > 1000 else len(data)
            min_dist_list.append(compute_smallest_distance(coords=data[["x", "y"]].values, sample_num=sample_num))

        point_size = min(min_dist_list) * axsize[0] / ptp_vec[0] * dpi
        point_size = point_size**2 * ncols * nrows

    # Draw scatter plots.
    g.map_dataframe(sns.scatterplot, x="x", y="y", alpha=alpha, color=color, s=point_size, legend="brief", **kwargs)

    # Set legend.
    label_values = slices_data[label].values
    if label_values.dtype in ["float16", "float32", "float64", "int16", "int32", "int64"]:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax = g.facet_axis(row_i=0, col_j=ncols - 1)
        norm = mpl.colors.Normalize(vmin=None, vmax=None)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
        mappable.set_array(label_values)
        plt.colorbar(
            mappable,
            cax=inset_axes(
                ax,
                width="12%",
                height="100%",
                loc="center left",
                bbox_to_anchor=(1.02, 0.0, 0.5, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=1.85,
            ),
            ax=ax,
            orientation="vertical",
            alpha=alpha,
            label=label,
        )
    else:
        g.add_legend()

    plt.tight_layout()
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=show_legend,
        background="white",
        prefix="multi_slices",
        save_kwargs=save_kwargs,
        total_panels=len(slices_id),
        fig=g,
        axes=g,
        return_all=False,
        return_all_list=None,
    )


# TODO: Add docstring, add multi slices plot, legend scatter plot should keep the same size of the text
# def plot_clusters(
#     adata: Union[AnnData, np.ndarray],
#     spatial_key: str = 'spatial',
#     label_key: Union[str, List[str], pd.DataFrame] = 'clusters',
#     ax: Optional[mpl.axes.Axes] = None,
#     point_size: float = 10,
#     n_sampling: int = -1,
#     palette: Optional[dict] = None,
#     title: str = '',
#     title_kwargs: Optional[dict] = None,
#     # title_fontsize = 16,
#     show_legend: bool = True,
#     legend_kwargs: Optional[dict] = None,
#     axis_off: bool = True,
#     axis_kwargs: Optional[dict] = None,
#     ticks_off: bool = True,
#     color=None,
#     x_min = None,
#     x_max = None,
#     y_min = None,
#     y_max = None,
#     **kwargs,
# ):
#     """
#     Plots spatial data with cluster labels stored in adata.obs[col]
#     """

#     # get spatial coords and labels
#     if isinstance(adata, AnnData):
#         if spatial_key in adata.obsm.keys():
#             spatial_coords = adata.obsm[spatial_key].copy()
#         else:
#             raise ValueError(f"adata.obsm['{spatial_key}'] does not exist.")
#     elif isinstance(adata, np.ndarray):
#         if adata.shape[1] > 1:
#             spatial_coords = adata.copy()[:,:2]
#         else:
#             raise ValueError("the input spatial coordinates must have at least 2 columns.")

#     if isinstance(adata, AnnData):
#         if label_key in adata.obs.keys():
#             label = adata.obs[label_key].copy()
#         else:
#             raise ValueError(f"adata.obs['{label_key}'] does not exist.")
#     elif isinstance(label_key, list):
#         label = pd.Series(label_key)
#     elif isinstance(label_key, pd.DataFrame):
#         label = label_key
#     else:
#         raise ValueError(f"label_key must be a string, list, or DataFrame.")

#     assert spatial_coords.shape[0] == label.shape[0], "The number of spatial coordinates and labels must be the same."

#     # downsampling if n_sampling is set
#     sampling_idx = np.random.choice(spatial_coords.shape[0], n_sampling, replace=False) if n_sampling > 0 and n_sampling < spatial_coords.shape[0] else np.arange(spatial_coords.shape[0])
#     x = spatial_coords[sampling_idx, 0]
#     y = spatial_coords[sampling_idx, 1]
#     label = label[sampling_idx]

#     # get unique labels
#     unique_labels = np.unique(label)

#     # get color palette if not provided
#     if palette is None:
#         n_colors = len(unique_labels)
#         palette = sns.color_palette("tab20", n_colors)

#     # plot
#     scatterplot_kwargs = {
#         'x': x,
#         'y': y,
#         'hue': label,
#         'palette': palette,
#         'ax': ax,
#         's': point_size,
#         'legend': show_legend,
#     }
#     scatterplot_kwargs.update(kwargs)
#     sns.scatterplot(**scatterplot_kwargs)

#     # adjust the legend
#     if show_legend:
#         default_legend_kwargs = {
#             'loc': 'center left',
#             'bbox_to_anchor': (1, 0.5),
#             'prop': {'family': 'Arial', 'size': 10},
#             'fancybox': False,
#             'edgecolor': 'black',
#             'framealpha': 1,
#             'columnspacing': 0.5,
#             'handletextpad': 0.1,
#         }
#         if legend_kwargs:
#             default_legend_kwargs.update(legend_kwargs)
#         ax.legend(**default_legend_kwargs)

#     # set axis limits
#     if x_min is not None and x_max is not None:
#         ax.set_xlim(x_min, x_max)
#     if y_min is not None and y_max is not None:
#         ax.set_ylim(y_min, y_max)

#     # set other axis properties
#     if axis_off:
#         ax.axis('off')

#     if ticks_off:
#         ax.set_xticks([])
#         ax.set_yticks([])

#     if title_kwargs:
#         default_title_kwargs = {
#             'label': title,
#             'fontsize': 16,
#         }
#         default_title_kwargs.update(title_kwargs)
#     ax.set_title(title_kwargs)
#     ax.set_aspect('equal')


# def _spatial_scatter(
#     spatial_x: np.ndarray,
#     spatial_y: np.ndarray,
#     color: Union[str, np.ndarray],
#     point_size: Union[float, np.ndarray],
#     alpha: Union[float, np.ndarray],
#     edgecolors: Optional[Union[str, np.ndarray]] = None,
#     linewidths: Optional[Union[float, np.ndarray]] = None,
#     marker: Optional[str] = "o",
#     palette: Optional[Union[str, dict]] = None,
#     ax: Optional[mpl.axes.Axes] = None,
# ):
#     if ax is None:
#         _, ax = plt.subplots()

#     if isinstance(color, str):
#         color = np.array([color] * len(spatial_x))

#     if isinstance(point_size, float):
#         point_size = np.array([point_size] * len(spatial_x))

#     if isinstance(alpha, float):
#         alpha = np.array([alpha] * len(spatial_x))

#     if edgecolors is None:
#         edgecolors = "none"

#     if linewidths is None:
#         linewidths = 0

#     if palette is not None:
#         if isinstance(palette, str):
#             palette = sns.color_palette(palette, n_colors=len(np.unique(color)))
#         elif isinstance(palette, dict):
#             palette = [palette.get(c, "gray") for c in np.unique(color)]
#         else:
#             raise ValueError("`palette` must be a string or a dictionary.")

#         color = [palette[np.where(np.unique(color) == c)[0][0]] for c in color]

#     ax.scatter(
#         spatial_x,
#         spatial_y,
#         c=color,
#         s=point_size,
#         alpha=alpha,
#         edgecolors=edgecolors,
#         linewidths=linewidths,
#         marker=marker,
#     )
#     ax.set_aspect('equal')

#     return ax


def _agenerate_palette(*labels, cmap="tab20"):
    if len(labels) == 1:
        labels = labels[0]
    elif len(labels) > 1:
        labels = np.concatenate(labels)
    else:
        raise ValueError("No labels provided.")
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    palette = {l: sns.color_palette(cmap, n_labels)[i] for i, l in enumerate(unique_labels)}
    return palette
