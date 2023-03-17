try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from ....alignment import get_optimal_mapping_relationship
from ....tdr import collect_models, construct_align_lines, construct_pc
from .three_dims_plots import three_d_animate, three_d_plot


def pi_heatmap(
    pi: np.ndarray,
    model1_name: str = "model1",
    model2_name: str = "model2",
    colormap: str = "hot_r",
    fig_height: Union[int, float] = 5,
    robust: bool = False,
    vmin: Optional[Union[int, float]] = None,
    vmax: Optional[Union[int, float]] = None,
    fontsize: Union[int, float] = 12,
    filename: Optional[str] = None,
    **kwargs,
):
    """
    Visualize a heatmap of the pi matrix.

    Args:
        pi: The pi matrix obtained by alignment.
        model1_name: The name/id of model1.
        model2_name: The name/id of model2.
        colormap: Colors to use for plotting heatmap. The default colormap is ``'hot_r'``.
        fig_height: Figure height.
        robust: If True and vmin or vmax are absent, the colormap range is computed with robust quantiles instead of the extreme values.
        vmin: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
        vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
        fontsize: The font size of x label and y label.
        filename:  Filename of output file.
        **kwargs: Additional parameters that will be passed to ``sns.heatmap`` function.
    """
    pi_shape = pi.shape
    aspect_ratio = pi_shape[1] / pi_shape[0]
    figsize = (fig_height * aspect_ratio, fig_height)
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(data=pi, cmap=colormap, vmin=vmin, vmax=vmax, robust=robust, ax=ax, **kwargs)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(
        xlabel=model2_name,
        labelpad=5,
        loc="center",
        fontsize=fontsize,
        fontweight="regular",
    )
    ax.set_ylabel(
        ylabel=model1_name,
        labelpad=5,
        loc="center",
        fontsize=fontsize,
        fontweight="regular",
    )
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)
    if not (filename is None):
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    else:
        return fig


def pairwise_connections(
    adataA: AnnData,
    adataB: AnnData,
    pi: np.ndarray,
    layer: str = "X",
    group_key: Union[str, list] = None,
    spatial_key: str = "align_spatial",
    id_key: str = "slices",
    keep_all: bool = False,
    distance: Optional[Union[int, float]] = None,
    direction: Optional[Literal["x", "y", "z"]] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    cpo: Optional[Union[str, list]] = None,
    window_size: Optional[tuple] = None,
    background: str = "black",
    modelA_cmap: str = "dodgerblue",
    modelB_cmap: str = "red",
    alphamap: Union[float, list] = 1.0,
    line_color: str = "gainsboro",
    ambient: float = 0.3,
    model_opacity: float = 1,
    line_opacity: float = 0.03,
    model_size: Union[float, list] = 6.0,
    line_size: Union[float, list] = 2.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Visualize the pairing of cells between two models.

    Args:
        adataA: Anndata object of modelA.
        adataB: Anndata object of modelB.
        pi: The pi matrix obtained by alignment.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        group_key: The key that stores clustering or annotation information in ``.obs``, a gene name or a list of gene names in ``.var``.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        id_key: The key in ``.obs`` that corresponds to the model id of each bucket.
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.
        distance: Distance between modelA and modelB when visualizing.
        direction: The direction between modelA and modelB when visualizing.
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
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        modelA_cmap: Colors to use for plotting modelA. The default colormap is ``'dodgerblue'``.
        modelB_cmap: Colors to use for plotting modelB. The default colormap is ``'red'``.
        alphamap: The opacity of the colors to use for plotting modelA&modelB. The default alphamap is ``1.0``.
        line_color: Colors to use for plotting lines. The default colormap is ``'gainsboro'``.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        model_opacity: Opacity of the modelA and modelB.
        line_opacity: Opacity of the lines.
        model_size: The point size of any nodes in the dataset plotted.
        line_size: The line size of lines in the dataset plotted.
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
        **kwargs: Additional parameters that will be passed to ``three_d_plot`` function.

    Returns:
        pcA: The point cloud models of adataA.
        pcB: The point cloud models of adataB.
        connection_lines: Cell connection lines between modelA and modelB
    """
    adataA, adataB = adataA.copy(), adataB.copy()

    # Check the spatial coordinates
    if adataA.obsm[spatial_key].shape[1] == 2:
        cpo = "xy" if cpo is None else cpo
        direction = "y" if direction is None else direction
        distance = 500 if distance is None else distance

        z = np.zeros(shape=(adataA.obsm[spatial_key].shape[0], 1))
        adataA.obsm[spatial_key] = np.c_[adataA.obsm[spatial_key], z]
    else:
        cpo = "iso" if cpo is None else cpo
        direction = "z" if direction is None else direction
        distance = 300 if distance is None else distance
    if adataB.obsm[spatial_key].shape[1] == 2:
        z = np.zeros(shape=(adataB.obsm[spatial_key].shape[0], 1))
        adataB.obsm[spatial_key] = np.c_[adataB.obsm[spatial_key], z]

    # Obtain the optimal mapping connections between two samples
    max_index, pi_value, _, _ = get_optimal_mapping_relationship(
        X=adataA.obsm[spatial_key].copy(),
        Y=adataB.obsm[spatial_key].copy(),
        pi=pi,
        keep_all=keep_all,
    )

    mapping_data = pd.DataFrame(
        np.concatenate([max_index, pi_value], axis=1),
        columns=["index_x", "index_y", "pi_value"],
    ).astype(
        dtype={
            "index_x": np.int64,
            "index_y": np.int64,
            "pi_value": np.float64,
        }
    )
    mapping_data.sort_values(by=["index_x", "pi_value"], ascending=[True, False], inplace=True)
    mapping_data.drop_duplicates(subset=["index_x"], keep="first", inplace=True)

    coordsA = adataA.obsm[spatial_key].copy()
    coordsB = adataB.obsm[spatial_key][mapping_data["index_y"].values]

    # Construct connection lines
    if direction is "x":
        models_distance = np.asarray([-distance, 0, 0])
    elif direction is "y":
        models_distance = np.asarray([0, -distance, 0])
    else:
        models_distance = np.asarray([0, 0, -distance])
    connection_lines, plot_cmapL = construct_align_lines(
        model1_points=coordsA,
        model2_points=coordsB + models_distance,
        key_added="connections",
        label="lines",
        color=line_color,
    )

    # Construct point cloud models
    group_key = id_key if group_key is None else group_key
    idA = str(adataA.obs[id_key].unique().tolist()[0])
    idB = str(adataB.obs[id_key].unique().tolist()[0])

    pcA, plot_cmapA = construct_pc(
        adata=adataA.copy(),
        layer=layer,
        spatial_key=spatial_key,
        groupby=group_key,
        key_added="connections",
        colormap=modelA_cmap,
        alphamap=alphamap,
    )
    pcB, plot_cmapB = construct_pc(
        adata=adataB.copy(),
        layer=layer,
        spatial_key=spatial_key,
        groupby=group_key,
        key_added="connections",
        colormap=modelB_cmap,
        alphamap=alphamap,
    )
    pcB.points = pcB.points + models_distance

    # Visualization
    three_d_plot(
        model=collect_models([connection_lines, pcA, pcB]),
        key="connections",
        filename=filename,
        jupyter=jupyter,
        off_screen=off_screen,
        cpo=cpo,
        background=background,
        window_size=window_size,
        ambient=ambient,
        opacity=[line_opacity, model_opacity, model_opacity],
        colormap=None,
        model_style=["wireframe", "points", "points"],
        model_size=[line_size, model_size, model_size],
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=f"\nModels id: {idA} & {idB}" if text is True else text,
        text_kwargs=text_kwargs,
        **kwargs,
    )

    return pcA, pcB, connection_lines


"""
def pairwise_exp_similarity(
    adataA: AnnData,
    adataB: AnnData,
    cells: Union[int, str, list],
    layer: str = "X",
    spatial_key: str = "spatial",
    id_key: str = "slices",
    dissimilarity: Literal["euc", "kl", "both"] = "both",
    beta2: Union[int, float] = 0.5,
    normalize_c: bool = True,
    normalize_g: bool = False,
    select_high_exp_genes: Union[bool, float, int] = False,
    center_zero: bool = True,
    filename: Optional[str] = None,
    jupyter: Union[
        bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]
    ] = False,
    off_screen: bool = False,
    cpo: Union[str, list] = "xy",
    shape: Union[str, list, tuple] = None,
    window_size: Optional[tuple] = None,
    background: str = "black",
    cell_color: str = "dodgerblue",
    star_cell_color: str = "red",
    colormap: Union[str, list, dict] = "viridis",
    alphamap: Union[float, list, dict] = 1.0,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_size: float = 5.0,
    star_cell_size: float = 12.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    # Preprocessing
    adataA, adataB = adataA.copy(), adataB.copy()
    (
        nx,
        type_as,
        new_samples,
        exp_matrices,
        spatial_coords,
        normalize_scale,
        normalize_mean_list,
    ) = align_preprocess(
        samples=[adataA.copy(), adataB.copy()],
        spatial_key=spatial_key,
        normalize_c=normalize_c,
        normalize_g=normalize_g,
        select_high_exp_genes=select_high_exp_genes,
        dtype="float64",
        device="cpu",
        verbose=False,
    )

    X_A, X_B = exp_matrices[1], exp_matrices[0]

    # Calculate expression dissimilarity
    if dissimilarity in ["euc", "both"]:
        EucGeneDistMat = calc_exp_dissimilarity(
            X_A=X_A.copy(), X_B=X_B.copy(), dissimilarity="euc"
        )
        # EucGeneDistMinusMat = EucGeneDistMat - nx.min(EucGeneDistMat, axis=1, keepdims=True)
        EucGeneAssignmentMat = np.exp(-EucGeneDistMat / (2 * beta2))
    if dissimilarity in ["kl", "both"]:
        KLGeneDistMat = calc_exp_dissimilarity(
            X_A=X_A.copy(), X_B=X_B.copy(), dissimilarity="kl"
        )
        # KLGeneDistMinusMat = KLGeneDistMat - nx.min(KLGeneDistMat, axis=1, keepdims=True)
        KLGeneAssignmentMat = np.exp(-KLGeneDistMat / (2 * beta2))

    # Select Cells
    cells = cells if isinstance(cells, list) else [cells]
    adataA_cells = np.asarray(adataA.obs.index.tolist())
    if isinstance(cells[0], str):
        cell_indices = np.argwhere(np.isin(adataA_cells, cells)).flatten()
        cell_names = cells.copy()
    else:
        cell_indices = np.asarray(cells)
        cell_names = adataA_cells[cell_indices]

    # Construct a point cloud model
    if adataA.obsm[spatial_key].shape[1] == 2:
        z = np.zeros(shape=(adataA.obsm[spatial_key].shape[0], 1))
        adataA.obsm[spatial_key] = np.c_[adataA.obsm[spatial_key], z]
    if adataB.obsm[spatial_key].shape[1] == 2:
        z = np.zeros(shape=(adataB.obsm[spatial_key].shape[0], 1))
        adataB.obsm[spatial_key] = np.c_[adataB.obsm[spatial_key], z]

    adataA_id = str(adataA.obs[id_key].unique().tolist()[0])
    adataB_id = str(adataB.obs[id_key].unique().tolist()[0])
    pcA, plot_campA = construct_pc(
        adata=adataA,
        layer=layer,
        spatial_key=spatial_key,
        groupby=id_key,
        key_added="cell",
        colormap=cell_color,
    )
    pcB, plot_cmapB = construct_pc(
        adata=adataB,
        layer=layer,
        spatial_key=spatial_key,
        groupby=id_key,
        key_added="cell",
        colormap=cell_color,
    )
    if center_zero is True:
        center_to_zero(model=pcA, inplace=True)
        center_to_zero(model=pcB, inplace=True)

    pcs, ids, mss, cmaps, keys = [], [], [], [], []
    for cell_ind, cell_name in zip(cell_indices, cell_names):
        star_cell = pv.PolyData(pcA.points[[cell_ind], :])
        _, star_plot_cmap = add_model_labels(
            model=star_cell,
            labels=np.asarray(["star cell"]),
            key_added="cell",
            where="point_data",
            colormap=star_cell_color,
            alphamap=alphamap,
            inplace=True,
        )
        pcs.append(collect_models([pcA.copy(), star_cell]))
        ids.append(adataA_id)
        mss.append([model_size, star_cell_size])
        cmaps.append(star_plot_cmap)
        keys.append("cell")

        if dissimilarity in ["euc", "both"]:
            npcB, B_plot_cmap = add_model_labels(
                model=pcB.copy(),
                labels=np.asarray(EucGeneAssignmentMat[:, cell_ind].flatten()),
                key_added=f"{cell_ind}-euc",
                where="point_data",
                colormap=colormap,
                alphamap=alphamap,
                inplace=False,
            )
            pcs.append(npcB)
            ids.append(f"{adataB_id}, EUC")
            mss.append(model_size)
            cmaps.append(B_plot_cmap)
            keys.append(f"{cell_ind}-euc")

        if dissimilarity in ["kl", "both"]:
            npcB, B_plot_cmap = add_model_labels(
                model=pcB.copy(),
                labels=np.asarray(KLGeneAssignmentMat[:, cell_ind].flatten()),
                key_added=f"{cell_ind}-kl",
                where="point_data",
                colormap=colormap,
                alphamap=alphamap,
                inplace=False,
            )
            pcs.append(npcB.copy())
            ids.append(f"{adataB_id}, KL")
            mss.append(model_size)
            cmaps.append(B_plot_cmap)
            keys.append(f"{cell_ind}-kl")

    # Check the shared cpo again
    cpo = _check_cpos_in_multi_plot(models=[pcA, pcB], window_size=window_size, cpo=cpo)

    # Visualization.
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
        model_size=mss,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=[f"\nModel id: {id}" for id in ids] if text is True else text,
        text_kwargs=text_kwargs,
        **kwargs,
    )
"""


def pairwise_iteration(
    adataA: AnnData,
    adataB: AnnData,
    layer: str = "X",
    group_key: Union[str, list] = None,
    spatial_key: str = "align_spatial",
    iter_key: str = "iter_spatial",
    id_key: str = "slices",
    filename: str = "animate.mp4",
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    cpo: Optional[Union[str, list]] = None,
    window_size: Optional[tuple] = None,
    background: str = "black",
    modelA_cmap: str = "dodgerblue",
    modelB_cmap: str = "red",
    ambient: Union[int, float] = 0.3,
    modelA_opacity: Union[int, float] = 0.8,
    modelB_opacity: Union[int, float] = 1.0,
    model_size: Union[float, list] = 6.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    framerate: int = 6,
    **kwargs,
):
    """
    Visualize the results of each iteration in the alignment process.

    Args:
        adataA: Anndata object of modelA.
        adataB: Anndata object of modelB.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        group_key: The key that stores clustering or annotation information in ``.obs``, a gene name or a list of gene names in ``.var``.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        iter_key: The key in ``.uns`` that corresponds to the result of each iteration of the iterative process.
        id_key: The key in ``.obs`` that corresponds to the model id of each bucket.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.

                * Output a gif file, please enter a filename ending with ``.gif``.
                * Output a mp4 file, please enter a filename ending with ``.mp4``.
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
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        modelA_cmap: Colors to use for plotting modelA. The default colormap is ``'dodgerblue'``.
        modelB_cmap: Colors to use for plotting modelB. The default colormap is ``'red'``.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        modelA_opacity: Opacity of the modelA.
        modelB_opacity: Opacity of the modelB.
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
        framerate: Frames per second.
        **kwargs: Additional parameters that will be passed to ``three_d_animate`` function.
    """
    adataA, adataB = adataA.copy(), adataB.copy()

    group_key = id_key if group_key is None else group_key
    idA = str(adataA.obs[id_key].unique().tolist()[0])
    idB = str(adataB.obs[id_key].unique().tolist()[0])

    spatialA_dims = adataA.obsm[spatial_key].shape[1]
    spatialB_dims = adataB.obsm[spatial_key].shape[1]
    if cpo is None:
        cpo = "xy" if spatialA_dims == 2 else "iso"

    # Check the spatial coordinates
    if spatialA_dims == 2:
        z = np.zeros(shape=(adataA.obsm[spatial_key].shape[0], 1))
        adataA.obsm[spatial_key] = np.c_[adataA.obsm[spatial_key], z]
    if spatialB_dims == 2:
        z = np.zeros(shape=(adataB.obsm[spatial_key].shape[0], 1))
        adataB.obsm[spatial_key] = np.c_[adataB.obsm[spatial_key], z]

    # Construct point cloud models
    stable_pc, _ = construct_pc(
        adata=adataA.copy(),
        layer=layer,
        spatial_key=spatial_key,
        groupby=group_key,
        key_added="iter",
        colormap=modelA_cmap,
    )

    iter_pcs = []
    iteration_coords = adataB.uns[iter_key][spatial_key]
    for iter_key in iteration_coords.keys():
        coords = iteration_coords[iter_key]
        if spatialB_dims == 2:
            coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]

        adataB.obsm["iter_spatial"] = coords
        iter_pc, _ = construct_pc(
            adata=adataB.copy(),
            layer=layer,
            spatial_key="iter_spatial",
            groupby=group_key,
            key_added="iter",
            colormap=modelB_cmap,
        )
        iter_pcs.append(iter_pc)

    # Animation
    three_d_animate(
        models=collect_models(iter_pcs),
        stable_model=stable_pc,
        stable_kwargs=dict(
            key="iter",
            ambient=ambient,
            opacity=modelA_opacity,
            model_style="points",
            model_size=model_size,
            background=background,
            show_legend=show_legend,
            legend_kwargs=legend_kwargs,
        ),
        key="iter",
        filename=filename,
        jupyter=jupyter,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
        cpo=cpo,
        ambient=ambient,
        opacity=modelB_opacity,
        model_style="points",
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=f"\nModels id: {idA} & {idB}" if text is True else text,
        text_kwargs=text_kwargs,
        framerate=framerate,
        **kwargs,
    )
