try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import List, Optional, Union

import numpy as np
from anndata import AnnData
from pyvista import PolyData

from spateo.tdr import (
    add_model_labels,
    center_to_zero,
    collect_models,
    construct_pc,
    merge_models,
    translate_model,
)

from .three_dims_plots import three_d_multi_plot


def _check_cpos_in_multi_plot(
    models: List,
    window_size: Optional[tuple] = None,
    cpo: Union[str, list] = "xy",
):
    cpos, cpos_size = [], []
    for model in models:
        _window_size = (512, 512) if window_size is None else window_size
        _cpo = model.plot(
            cpos=cpo,
            jupyter_backend="none",
            return_cpos=True,
            off_screen=True,
            window_size=_window_size,
        )
        cpos.append(_cpo)
        cpos_size.append(_cpo[0][2])
    cpo_index = np.argmax(np.asarray(cpos_size))
    cpo = cpos[cpo_index]
    return cpo


def multi_models(
    *adata: AnnData,
    layer: str = "X",
    group_key: Union[str, list] = None,
    spatial_key: str = "align_spatial",
    id_key: str = "slices",
    mode: Literal["single", "overlap", "both"] = "single",
    center_zero: bool = False,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    cpo: Union[str, list] = "xy",
    shape: Union[str, list, tuple] = None,
    window_size: Optional[tuple] = None,
    background: str = "white",
    colormap: Union[str, list, dict] = "red",
    overlap_cmap: Union[str, list, dict] = "dodgerblue",
    alphamap: Union[float, list, dict] = 1.0,
    overlap_amap: Union[float, list, dict] = 0.5,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Visualize multiple models separately in one figure.

    Args:
        *adata: A list of models[Anndata object].
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        group_key: The key that stores clustering or annotation information in ``.obs``, a gene name or a list of gene names in ``.var``.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        id_key: The key in ``.obs`` that corresponds to the model id of each bucket.
        mode: Three modes of visualization.  Available ``mode`` are:

                * ``'single'`` - Visualize each model individually.
                * ``'overlap'`` - Simultaneously visualize two models aligned front to back in one subplot.
                * ``'both'`` - Simultaneously visualize both types above.
        center_zero: Whether to move the center point of the model to the (0, 0, 0).
        filename:  Filename of output file. Writer type is inferred from the extension of the filename.

                * Output an image file,please enter a filename ending with
                  ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
        jupyter: Whether to plot in jupyter notebook. Available ``jupyter`` are:

                * ``'none'`` - Do not display in the notebook.
                * ``'pythreejs'`` - Show a pythreejs widget
                * ``'static'`` - Display a static figure.
                * ``'ipygany'`` - Show an ipygany widget
                * ``'panel'`` - Show a panel widget.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        cpo: Camera position of the active render window. Available ``cpo`` are:

                * Iterable containing position, focal_point, and view up.
                    ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
                * Iterable containing a view vector.
                    ``E.g.: [-1.0, 2.0, -5.0].``
                * A string containing the plane orthogonal to the view direction.
                    ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
        shape: Number of sub-render windows inside the main window. By default, there is only one render window.

               * Specify two across with ``shape``=(2, 1) and a two by two grid with ``shape``=(2, 2).
               * ``shape`` Can also accept a string descriptor as shape.

                    ``E.g.: shape="3|1" means 3 plots on the left and 1 on the right,``
                    ``E.g.: shape="4/2" means 4 plots on top and 2 at the bottom.``
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        colormap: Colors to use for plotting pc. The default colormap is ``'dodgerblue'``.
        overlap_cmap: Colors to use for plotting overlapped pc. The default colormap is ``'red'``.
        alphamap: The opacity of the colors to use for plotting pc. The default alphamap is ``1.0``.
        overlap_amap: The opacity of the colors to use for plotting overlapped pc. The default alphamap is ``.5``.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.

                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.

                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_size: The point size of any nodes in the dataset plotted.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.

                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None,  "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
        **kwargs: Additional parameters that will be passed to ``three_d_multi_plot`` function.
    """
    adata_list = adata[0]
    adata_list = adata_list if isinstance(adata_list, list) else [adata_list]

    # Construct a point cloud model
    pcs, ids, keys, cmaps = [], [], [], []
    for i, adata in enumerate(adata_list):
        adata = adata.copy()
        adata_id = str(adata.obs[id_key].unique().tolist()[0])
        group_key = id_key if group_key is None else group_key

        if adata.obsm[spatial_key].shape[1] == 2:
            z = np.zeros(shape=(adata.obsm[spatial_key].shape[0], 1))
            adata.obsm[spatial_key] = np.c_[adata.obsm[spatial_key], z]

        pc, plot_cmap = construct_pc(
            adata=adata.copy(),
            layer=layer,
            spatial_key=spatial_key,
            groupby=group_key,
            key_added=f"{adata_id}-{group_key}",
            colormap=colormap,
            alphamap=alphamap,
        )
        if center_zero is True:
            center_to_zero(model=pc, inplace=True)
        ids.append(adata_id)
        pcs.append(pc)
        keys.append(f"{adata_id}-{group_key}")
        cmaps.append(plot_cmap)

    # Check the shared cpo again
    cpo = _check_cpos_in_multi_plot(models=pcs, window_size=window_size, cpo=cpo)

    # Visualization.
    if mode == "single":
        three_d_multi_plot(
            model=collect_models(pcs),
            key=keys,
            filename=filename,
            jupyter=jupyter,
            off_screen=off_screen,
            shape=shape,
            window_size=window_size,
            background=background,
            ambient=ambient,
            opacity=opacity,
            colormap=cmaps,
            cpo=[cpo],
            model_style=["points"],
            model_size=[model_size],
            show_legend=show_legend,
            legend_kwargs=legend_kwargs,
            text=[f"\nModel id: {id}" for id in ids] if text is True else text,
            text_kwargs=text_kwargs,
            **kwargs,
        )
    elif mode in ["overlap", "both"]:
        overlap_pcs, overlap_ids, overlap_keys, overlap_cmaps = [], [], [], []
        for i in range(len(pcs) - 1):
            pc1, pc2 = pcs[i].copy(), pcs[i + 1].copy()
            npc1, opc1_1 = add_model_labels(
                model=pc1.copy(),
                labels=np.asarray(pc1.point_data[keys[i]]),
                key_added=f"overlap-{keys[i]}-{keys[i + 1]}",
                where="point_data",
                colormap=overlap_cmap,
                alphamap=overlap_amap,
                inplace=False,
            )
            _, opc1_2 = add_model_labels(
                model=npc1,
                labels=np.asarray(pc1.point_data[keys[i]]),
                key_added=f"overlap-{keys[i]}",
                where="point_data",
                colormap=overlap_cmap,
                alphamap=overlap_amap,
                inplace=True,
            )

            npc2, opc2_1 = add_model_labels(
                model=pc2.copy(),
                labels=np.asarray(pc2.point_data[keys[i + 1]]),
                key_added=f"overlap-{keys[i]}-{keys[i + 1]}",
                where="point_data",
                colormap=colormap,
                alphamap=alphamap,
                inplace=False,
            )
            _, opc2_2 = add_model_labels(
                model=npc2,
                labels=np.asarray(npc2.point_data[keys[i + 1]]),
                key_added=f"overlap-{keys[i + 1]}",
                where="point_data",
                colormap=colormap,
                alphamap=alphamap,
                inplace=True,
            )
            overlap_pc = merge_models([npc1, npc2])
            if not (cmaps[i] is None):
                overlap_pc = merge_models([npc1, npc2])
                _, plot_cmap = add_model_labels(
                    model=overlap_pc,
                    labels=np.asarray(overlap_pc.point_data[f"overlap-{keys[i]}-{keys[i + 1]}"]),
                    key_added=f"overlap-{keys[i]}-{keys[i + 1]}",
                    where="point_data",
                    colormap=colormap,
                    inplace=True,
                )
            else:
                plot_cmap = None

            if mode == "overlap":
                overlap_pcs.append(overlap_pc)
                overlap_ids.append(f"{ids[i]} & {ids[i + 1]}")
                overlap_keys.append(f"overlap-{keys[i]}-{keys[i + 1]}")
                overlap_cmaps.append(plot_cmap)
            elif mode == "both":
                overlap_pcs.extend([npc1, npc2, overlap_pc])
                overlap_ids.extend([ids[i], ids[i + 1], f"{ids[i]} & {ids[i + 1]}"])
                overlap_keys.extend(
                    [
                        f"overlap-{keys[i]}",
                        f"overlap-{keys[i + 1]}",
                        f"overlap-{keys[i]}-{keys[i + 1]}",
                    ]
                )
                overlap_cmaps.extend([opc1_2, opc2_2, plot_cmap])

        three_d_multi_plot(
            model=collect_models(overlap_pcs),
            key=overlap_keys,
            filename=filename,
            jupyter=jupyter,
            off_screen=off_screen,
            shape=shape,
            window_size=window_size,
            background=background,
            ambient=ambient,
            opacity=opacity,
            colormap=overlap_cmaps,
            cpo=[cpo],
            model_style=["points"],
            model_size=[model_size],
            show_legend=show_legend,
            legend_kwargs=legend_kwargs,
            text=[f"\nModel id: {id}" for id in overlap_ids] if text is True else text,
            text_kwargs=text_kwargs,
            **kwargs,
        )


def deformation(
    *adata: AnnData,
    deformed_grid: Union[PolyData, List[PolyData]],
    layer: str = "X",
    group_key: Union[str, list] = None,
    spatial_key: str = "align_spatial",
    id_key: str = "slices",
    deformation_key: Optional[str] = "deformation",
    center_zero: bool = False,
    show_model: bool = True,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    cpo: Union[str, list] = "xy",
    shape: Union[str, list, tuple] = None,
    window_size: Optional[tuple] = (1024, 756),
    background: str = "white",
    model_color: Union[str, list] = "red",
    model_alpha: Union[float, list, dict] = 1,
    colormap: Union[str, list, dict] = "black",
    alphamap: Union[float, list, dict] = 1.0,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    grid_size: Union[float, list] = 2.0,
    model_size: Union[float, list] = 3.0,
    show_legend: bool = False,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    adata_list = adata[0]
    adata_list = adata_list if isinstance(adata_list, list) else [adata_list]

    grid_list = deformed_grid if isinstance(deformed_grid, list) else [deformed_grid]
    assert len(adata_list) == len(
        grid_list
    ), "The number of Anndata objects is not equal to the number of deformed grids."

    # Construct a point cloud model
    plot_models, ids, keys, cmaps = [], [], [], []
    for adata, grid in zip(adata_list, grid_list):
        adata, raw_grid = adata.copy(), grid.copy()
        adata_id = str(adata.obs[id_key].unique().tolist()[0])
        group_key = id_key if group_key is None else group_key

        if deformation_key is None:
            labels = np.asarray(grid.n_points * [f"{adata_id}-deformed grid"])
        else:
            labels = np.asarray(grid.point_data[deformation_key])
        _, plot_grid_cmap = add_model_labels(
            model=grid,
            labels=labels,
            key_added=f"{adata_id}-{group_key}",
            where="point_data",
            colormap=colormap,
            alphamap=alphamap,
            inplace=True,
        )
        if center_zero is True:
            center_to_zero(model=grid, inplace=True)

        if show_model:
            if adata.obsm[spatial_key].shape[1] == 2:
                z = np.zeros(shape=(adata.obsm[spatial_key].shape[0], 1))
                adata.obsm[spatial_key] = np.c_[adata.obsm[spatial_key], z]
            pc, plot_pc_cmap = construct_pc(
                adata=adata.copy(),
                layer=layer,
                spatial_key=spatial_key,
                groupby=group_key,
                key_added=f"{adata_id}-{group_key}",
                colormap=model_color,
                alphamap=model_alpha,
            )
            if center_zero is True:
                translate_distance = (
                    -raw_grid.center[0],
                    -raw_grid.center[1],
                    -raw_grid.center[2],
                )
                translate_model(model=pc, distance=translate_distance, inplace=True)
            plot_model = collect_models([pc, grid])
        else:
            plot_model = grid.copy()

        ids.append(adata_id)
        plot_models.append(plot_model)
        keys.append(f"{adata_id}-{group_key}")
        cmaps.append([plot_pc_cmap, plot_grid_cmap] if show_model else plot_grid_cmap)

    # Check the shared cpo again
    cpo = _check_cpos_in_multi_plot(models=grid_list, window_size=window_size, cpo=cpo)

    # Visualization.
    three_d_multi_plot(
        model=collect_models(plot_models),
        key=keys,
        filename=filename,
        jupyter=jupyter,
        off_screen=off_screen,
        shape=shape,
        window_size=window_size,
        background=background,
        ambient=ambient,
        opacity=opacity,
        colormap=cmaps,
        cpo=[cpo],
        model_style=[["points", "wireframe"]] if show_model else ["wireframe"],
        model_size=[[model_size, grid_size]] if show_model else [grid_size],
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=[f"\nModel id: {id}" for id in ids] if text is True else text,
        text_kwargs=text_kwargs,
        **kwargs,
    )
