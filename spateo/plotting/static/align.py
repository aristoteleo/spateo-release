import math
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.animation as animation
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
    axis_off: bool = False,
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
        if labels[0].dtype in ["float16", "float32", "float64", "int16", "int32", "int64"]:
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
    elif label_type == "scalar":
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
    axis_off: bool = False,
    axis_kwargs: Optional[dict] = None,
    ticks_off: bool = True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    height: float = 2,
    alpha: float = 1.0,  # TODO: alpha to be a key in adata
    cmap="tab20",
    center_coordinate: bool = False,  # different from slices_2d
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
        if (
            (overlay_type == "both")
            or ((overlay_type == "backward") and (i < len(slices) - 1))
            or ((overlay_type == "forward") and (i > 0))
        ):
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
        if overlay_type == "both":
            palette = {
                "current": "red",
                "forward": "green",
                "backward": "blue",
            }
        elif overlay_type == "forward":
            palette = {
                "current": "red",
                "forward": "green",
            }
        elif overlay_type == "backward":
            palette = {
                "current": "red",
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
        xlim=(x_min, x_max) if x_min is not None and x_max is not None else None,
        ylim=(y_min, y_max) if y_min is not None and y_max is not None else None,
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

        # if x_max is not None and x_min is not None:
        #     ax.set_xlim(x_min, x_max)
        # if y_max is not None and y_min is not None:
        #     ax.set_ylim(y_min, y_max)

        ax.set_xlabel("")
        ax.set_ylabel("")

    # create legend
    if show_legend:
        if label_type == "cluster":
            _legend_kwargs = {
                "loc": "upper center",
                "bbox_to_anchor": (0.5, 0),
                "prop": {"family": "Arial", "size": 10},
                "fancybox": False,
                "edgecolor": "black",
                "framealpha": 1,
                "columnspacing": 0.8,
                "handletextpad": 0.5,
                "frameon": True,
                "ncol": 8,
                "borderaxespad": -4,
                "frameon": False,
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


def optimization_animation(
    aligned_slices: List[AnnData],
    label_key: Optional[str] = None,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    iter_key_added: Optional[str] = "iter_spatial",
    filename: Optional[str] = "Visualization2D",
    fps: int = 10,
    stepsize: int = 10,
    cmap="Set1",
    palette: Optional[dict] = None,
    point_size: Optional[float] = None,
    n_sampling: int = -1,
):
    assert len(aligned_slices) == 2, "Input aligned_slices must be 2 slices!"

    if label_key is not None:
        if [label_key in s.obs.keys() for s in aligned_slices]:
            labels = [s.obs[label_key] for s in aligned_slices]
            label1 = aligned_slices[0].obs[label_key]
            label2 = aligned_slices[1].obs[label_key]
    else:
        label1 = np.zeros((aligned_slices[0].shape[0],), dtype=np.int32)
        label2 = np.ones((aligned_slices[1].shape[0],), dtype=np.int32)

    if n_sampling > 0:
        sampling_idx1 = (
            np.random.choice(aligned_slices[0].shape[0], n_sampling, replace=False)
            if n_sampling < aligned_slices[0].shape[0]
            else np.arange(aligned_slices[0].shape[0])
        )
        sampling_idx2 = (
            np.random.choice(aligned_slices[1].shape[0], n_sampling, replace=False)
            if n_sampling < aligned_slices[1].shape[0]
            else np.arange(aligned_slices[1].shape[0])
        )
    else:
        sampling_idx1 = np.arange(aligned_slices[0].shape[0])
        sampling_idx2 = np.arange(aligned_slices[1].shape[0])

    # generate palette
    if (palette is None) and (label_key is not None):
        palette = _agenerate_palette(*labels, cmap=cmap)

    if label_key is not None:
        label1_colors = [palette[cat] for cat in label1[sampling_idx1]]
        label2_colors = [palette[cat] for cat in label2[sampling_idx2]]
    else:
        label1_colors = ["#e41a1c" for cat in label1[sampling_idx1]]
        label2_colors = ["#377eb8" for cat in label2[sampling_idx2]]

    if point_size is None:
        point_size = 500 * 10 / (len(sampling_idx1) + len(sampling_idx2))

    coordsB = aligned_slices[0].obsm[spatial_key]
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    artists = []
    iter_dict = aligned_slices[1].uns[iter_key_added]
    iter = len(iter_dict[key_added])
    iteration = range(0, iter, stepsize)
    ax.scatter(
        coordsB[sampling_idx1, 0], coordsB[sampling_idx1, 1], marker="o", s=point_size, c=label1_colors, edgecolors=None
    )
    for i in iteration:
        frame = ax.scatter(
            iter_dict[key_added][i][sampling_idx2, 0],
            iter_dict[key_added][i][sampling_idx2, 1],
            marker="o",
            s=point_size,
            c=label2_colors,
            edgecolors=None,
        )
        title_text = "Iter: {}, sigma2: {:.3f}.".format(i, iter_dict["sigma2"][i])
        tit = ax.text(0.5, 1.02, title_text, ha="center", va="bottom", size=16, weight="bold", transform=ax.transAxes)
        artists.append([frame, tit])
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=4, blit=False)
    ani.save(filename + ".gif", fps=fps, dpi=100)
    plt.close()


def plot_deformation_grid(
    adata,
    spatial_key,
    origin_spatial_key,
    label_key,
    predict_func,
    ax,
    point_size,
    grid_num=10,
    line_width=0.5,
    grid_color="black",
    expand_scale=0.1,
    palette=None,
    title="",
    legend=True,
    fontsize=8,
    fill=False,
):
    x = adata.obsm[spatial_key][:, 0]
    y = adata.obsm[spatial_key][:, 1]
    origin_x = adata.obsm[origin_spatial_key][:, 0]
    origin_y = adata.obsm[origin_spatial_key][:, 1]
    label = adata.obs[label_key]
    if palette is None:
        n_colors = len(np.unique(label))
        palette = sns.color_palette("Paired", n_colors)

    # plot deformation grid

    # Generate the grid points
    horizontal_lines_data = pd.DataFrame(columns=["x", "y", "type"])
    vertical_lines_data = pd.DataFrame(columns=["x", "y", "type"])
    x_min, x_max = origin_x.min(), origin_x.max()
    y_min, y_max = origin_y.min(), origin_y.max()
    # expand the min max
    x_length = x_max - x_min
    y_length = y_max - y_min
    x_min -= x_length * expand_scale
    x_max += x_length * expand_scale
    y_min -= y_length * expand_scale
    y_max += y_length * expand_scale
    horizontal_values = np.linspace(y_min, y_max, grid_num)
    vertical_values = np.linspace(x_min, x_max, grid_num)
    horizontal_lines, vertical_lines = [], []

    if fill:
        for i, vertical_value in enumerate(vertical_values):
            vertical_line = np.linspace(y_min, y_max, 1000)[:, np.newaxis]
            vertical_line = np.concatenate([np.ones_like(vertical_line) * vertical_value, vertical_line], axis=1)
            deformed_vertical_line = predict_func(vertical_line)
            if (i == 0) or (i == len(vertical_values) - 1):
                if i == 0:
                    edge_points_vert_up = deformed_vertical_line
                if i == len(vertical_values) - 1:
                    edge_points_vert_down = deformed_vertical_line
            else:
                continue

        for i, horizontal_value in enumerate(horizontal_values):
            horizontal_line = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
            horizontal_line = np.concatenate(
                [horizontal_line, np.ones_like(horizontal_line) * horizontal_value], axis=1
            )
            deformed_horizontal_line = predict_func(horizontal_line)
            if (i == 0) or (i == len(horizontal_values) - 1):
                if i == 0:
                    edge_points_hor_right = deformed_horizontal_line
                if i == len(vertical_values) - 1:
                    edge_points_hor_left = deformed_horizontal_line
            else:
                continue
        edge_x = [
            edge_points_vert_up[:, 0],
            edge_points_hor_right[:, 0],
            np.flip(edge_points_vert_down[:, 0]),
            np.flip(edge_points_hor_left[:, 0]),
        ]
        edge_y = [
            edge_points_vert_up[:, 1],
            edge_points_hor_right[:, 1],
            np.flip(edge_points_vert_down[:, 1]),
            np.flip(edge_points_hor_left[:, 1]),
        ]
        ax.fill(edge_x, edge_y, color=np.array([249, 249, 249]) / 255, alpha=0.5)
    sns.scatterplot(x=x, y=y, hue=label, palette=palette, ax=ax, s=point_size, legend=legend)

    # Plot horizontal lines
    for i, vertical_value in enumerate(vertical_values):
        vertical_line = np.linspace(y_min, y_max, 1000)[:, np.newaxis]
        vertical_line = np.concatenate([np.ones_like(vertical_line) * vertical_value, vertical_line], axis=1)
        deformed_vertical_line = predict_func(vertical_line)
        if (i == 0) or (i == len(vertical_values) - 1):
            # ax.plot(deformed_vertical_line[:,0], deformed_vertical_line[:,1], color=np.array([233,72,23])/255, linewidth=line_width, alpha=1)
            continue
        else:
            ax.plot(
                deformed_vertical_line[:, 0],
                deformed_vertical_line[:, 1],
                color=grid_color,
                linewidth=line_width,
                alpha=0.8,
            )

    # Plot vertical line
    for i, horizontal_value in enumerate(horizontal_values):
        horizontal_line = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
        horizontal_line = np.concatenate([horizontal_line, np.ones_like(horizontal_line) * horizontal_value], axis=1)
        deformed_horizontal_line = predict_func(horizontal_line)
        if (i == 0) or (i == len(horizontal_values) - 1):
            # ax.plot(deformed_horizontal_line[:,0], deformed_horizontal_line[:,1], color=np.array([233,72,23])/255, linewidth=line_width, alpha=1)
            continue
        else:
            ax.plot(
                deformed_horizontal_line[:, 0],
                deformed_horizontal_line[:, 1],
                color=grid_color,
                linewidth=line_width,
                alpha=0.8,
            )

    # edge_points_vert_up = []
    # edge_points_vert_down = []

    for i, vertical_value in enumerate(vertical_values):
        vertical_line = np.linspace(y_min, y_max, 1000)[:, np.newaxis]
        vertical_line = np.concatenate([np.ones_like(vertical_line) * vertical_value, vertical_line], axis=1)
        deformed_vertical_line = predict_func(vertical_line)
        if (i == 0) or (i == len(vertical_values) - 1):
            ax.plot(
                deformed_vertical_line[:, 0],
                deformed_vertical_line[:, 1],
                color=np.array([91, 139, 200]) / 255,
                linewidth=1.5 * line_width,
                alpha=1,
            )
        else:
            continue
            # ax.plot(deformed_vertical_line[:,0], deformed_vertical_line[:,1], color=grid_color, linewidth=line_width, alpha=1)

    for i, horizontal_value in enumerate(horizontal_values):
        horizontal_line = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
        horizontal_line = np.concatenate([horizontal_line, np.ones_like(horizontal_line) * horizontal_value], axis=1)
        deformed_horizontal_line = predict_func(horizontal_line)
        if (i == 0) or (i == len(horizontal_values) - 1):
            ax.plot(
                deformed_horizontal_line[:, 0],
                deformed_horizontal_line[:, 1],
                color=np.array([91, 139, 200]) / 255,
                linewidth=1.5 * line_width,
                alpha=1,
            )
        else:
            # ax.plot(deformed_horizontal_line[:,0], deformed_horizontal_line[:,1], color=grid_color, linewidth=line_width, alpha=1)
            continue

    if legend:
        ax.legend().remove()
    ax.set_facecolor("white")
    ax.axis("off")
    if title != "":
        ax.set_title(title + " mapping", fontsize=fontsize)
    ax.set_aspect("equal")


# def plot_align_correspondence_2d(
#     slices: List[AnnData],
#     mapping: List[np.ndarray],
#     label_key: Optional[str] = None,
#     spatial_key: str = "spatial",
#     point_size: Optional[float] = None,
#     linewidth: Optional[float] = None,
#     n_sampling: int = -1,
#     mapping_sampling: int = -1,
#     robust_threshold: Optional[float] = None,
#     palette: Optional[dict] = None,
#     show_legend: bool = True,
#     legend_kwargs: Optional[dict] = None,
#     axis_off: bool = False,
#     axis_kwargs: Optional[dict] = None,
#     ticks_off: bool = True,
#     x_min=None,
#     x_max=None,
#     y_min=None,
#     y_max=None,
#     height: float = 2,
#     alpha: float = 1.0,  # TODO: alpha to be a key in adata
#     cmap="tab20",
#     center_coordinate: bool = False,  # different from slices_2d
# ):

#     assert len(mapping) == len(slices) - 1, "The length of mapping should be len(slices) - 1."
#     # get spatial coords and labels
#     spatial_coords = []
#     labels = []

#     for i, s in enumerate(slices):
#         if spatial_key in s.obsm.keys():
#             spatial_coords.append(s.obsm[spatial_key].copy())
#         else:
#             raise ValueError(f"adata.obsm['{spatial_key}'] does not exist.")

#         if label_key in s.obs.keys():
#             labels.append(s.obs[label_key].copy())
#             # label_type = "cluster"
#         elif label_key in s.var_names:
#             labels.append(s[:, label_key].X.A.copy().squeeze())
#             # label_type = "scalar"
#         else:
#             raise ValueError(f"adata.obs['{label_key}'] or adata.var['{label_key}'] does not exist.")

#         assert (
#             spatial_coords[-1].shape[0] == labels[-1].shape[0]
#         ), "The number of spatial coordinates and labels must be the same. Please check the data."

#     # add mapping
#     correspondences = []
#     for i in range(len(mapping)):
#         if mapping[i].shape[1] == 2:
#             correspondences.append(mapping[i])
#         elif (mapping[i].shape[1] == slices[i+1].shape[0]) and (mapping[i].shape[0] == slices[i].shape[0]):
#             sampling_idx = (
#                 np.random.choice(mapping[i].shape[0], mapping_sampling, replace=False)
#                 if mapping_sampling > 0 and mapping_sampling < mapping[i].shape[0]
#                 else np.arange(mapping[i].shape[0])
#             )
#             mapping_argmax = np.argmax(mapping[i][sampling_idx], axis=1)
#             mapping_valmax = mapping[i][sampling_idx][np.arange(mapping_sampling), mapping_argmax]
#             mask = np.arange(sampling_idx.shape[0]) if robust_threshold is None else mapping_valmax > robust_threshold
#             correspondence = np.array([sampling_idx[mask], mapping_argmax[mask]]).T
#             correspondences.append(correspondence)
#         else:
#             raise ValueError("The shape of mapping is not correct.")

#     # infer the label_type if not specified
#     if label_type is None:
#         if labels[0].values.dtype in ["float16", "float32", "float64", "int16", "int32", "int64"]:
#             label_type = "scalar"
#         else:
#             label_type = "cluster"

#     # downsampling if n_sampling is set
#     # TODO: implement downsampling

#     # center the coordinates
#     if center_coordinate:
#         for i in range(len(slices)):
#             spatial_coords[i] = spatial_coords[i] - np.mean(spatial_coords[i], axis=0)

#     # determine the interval of slices
#     slices_interval = []
#     for i in range(len(slices) - 1):
#         slices_interval.append(
#             _compute_smallest_distance(spatial_coords[i], spatial_coords[i+1])
#         )

#     # Update the spatial coordinates
#     cur_pos = 0
#     for i in range(len(slices) - 1):
#         cur_pos += slices_interval[i]
#         spatial_coords[i+1] += cur_pos

#     # determine the mapping line position and label
#     mapping_lines = []
#     mapping_labels = []
#     for i in range(len(correspondences)):
#         mapping_lines.append(
#             np.concatenate([spatial_coords[i][correspondences[i][:, 0]], spatial_coords[i+1][correspondences[i][:, 1]]], axis=1)
#         )
#         mapping_labels.append(
#             labels[i][correspondences[i][:, 0]]
#         )

#     # Set plot theme
#     sns.set_theme(
#         context="paper",
#         style="white",
#         font="Arial",
#         font_scale=1,
#         rc={
#             # "font.size": font_size,
#             "font.family": ["sans-serif"],
#             "font.sans-serif": ["Arial", "sans-serif", "Helvetica", "DejaVu Sans", "Bitstream Vera Sans"],
#         },
#     )

#     # generate palette
#     if (palette is None) and (label_type == "cluster"):
#         palette = _agenerate_palette(*labels, cmap=cmap)
#     else:
#         palette = cmap

#     # generate figure
#     fig, ax = plt.subplots(1, 1, figsize=(height * aspect_ratio, height))

#     # determine the pointsize if not specified
#     if point_size is None:
#         point_size = 500 * height**2 * aspect_ratio / (slices_spatial_data.shape[0] / len(slices))

#     # plotting
#     sns.scatterplot(x=x, y=y, hue=label, legend=legend, palette = palette, ax=ax, s=point_size, edgecolor=edgecolor)


# TODO: finish this
# def plot_align_correspondence_3d(
#     slices: List[AnnData],
#     label_key: Optional[str] = None,
#     spatial_key: str = "spatial",
#     point_size: Optional[float] = None,
#     n_sampling: int = -1,
#     palette: Optional[dict] = None,
#     show_legend: bool = True,
#     legend_kwargs: Optional[dict] = None,
#     axis_off: bool = False,
#     axis_kwargs: Optional[dict] = None,
#     ticks_off: bool = True,
#     x_min=None,
#     x_max=None,
#     y_min=None,
#     y_max=None,
#     height: float = 2,
#     alpha: float = 1.0,  # TODO: alpha to be a key in adata
#     cmap="tab20",
#     center_coordinate: bool = False,  # different from slices_2d
# ):
#     # get spatial coords and labels
#     spatial_coords = []
#     labels = []
#     slice_ids = []
#     for i, s in enumerate(slices):
#         if spatial_key in s.obsm.keys():
#             spatial_coords.append(s.obsm[spatial_key].copy())
#         else:
#             raise ValueError(f"adata.obsm['{spatial_key}'] does not exist.")

#         if label_key is not None:
#             if label_key in s.obs.keys():
#                 labels.append(s.obs[label_key].copy())
#                 label_type = "cluster"
#             elif label_key in s.var_names:
#                 labels.append(s[:, label_key].X.A.copy().squeeze())
#                 label_type = "scalar"
#             else:
#                 raise ValueError(f"adata.obs['{label_key}'] or adata.var['{label_key}'] does not exist.")

#             assert (
#                 spatial_coords[-1].shape[0] == labels[-1].shape[0]
#             ), "The number of spatial coordinates and labels must be the same. Please check the data."
#         else:
#             label_type = "cluster"

#         if (slices_key is not None) and (slices_key in s.obs.keys()):
#             unique_id = np.unique(s.obs[slices_key])
#             if len(unique_id) == 1:
#                 slice_ids.append(unique_id[0])
#             else:
#                 raise ValueError(f"adata.obs['{slices_key}'] must have only one unique value.")
#         else:
#             slice_ids.append(str(i))


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


def _compute_smallest_distance(spatial_coord1, spatial_coord2, direction="x", scale_factor=1.1):
    if direction == "x":
        spatial_coord1_max = np.max(spatial_coord1[:, 0])
        spatial_coord2_min = -np.min(spatial_coord2[:, 0])
        interval = (spatial_coord1_max + spatial_coord2_min) * scale_factor
    elif direction == "y":
        spatial_coord1_max = np.max(spatial_coord1[:, 1])
        spatial_coord2_min = -np.min(spatial_coord2[:, 1])
        interval = (spatial_coord1_max + spatial_coord2_min) * scale_factor
    else:
        raise ValueError("`direction` must be 'x' or 'y'.")

    return interval


def transform_by_min_max(x, _min, _max, interval=0.1):
    x = x - _min
    x = x / _max
    x = x * (1 - 2 * interval)
    x = x + interval
    return x


def get_min_max(x):
    _min = x.min(0)
    x = x - _min
    _max = x.max(0)
    return _min, _max


def transform_H(x, H, z_shift=0):
    x_H = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    transformed_x = (H @ x_H.T).T
    transformed_x = transformed_x / transformed_x[:, 2:]
    transformed_x[:, 1] = transformed_x[:, 1] + z_shift
    return transformed_x[:, :2]


def get_H(h=0.5, w=0.2):
    corner_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # transformed_corner_points = np.array([[0,0], [w,h], [1,0], [1-w, h]])
    transformed_corner_points = np.array([[w, h], [1 - w, h], [0, 0], [1, 0]])
    import cv2

    H, _ = cv2.findHomography(srcPoints=corner_points, dstPoints=transformed_corner_points)
    return H
