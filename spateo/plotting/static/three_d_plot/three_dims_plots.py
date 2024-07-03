import math
import os
import re
from typing import List, Optional, Union

import anndata
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from spateo.tdr import collect_models

from ..colorlabel import vega_10
from .three_dims_plotter import (
    _set_jupyter,
    add_legend,
    add_model,
    add_outline,
    add_text,
    create_plotter,
    output_plotter,
    save_plotter,
)


def wrap_to_plotter(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Union[str, list] = None,
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
):
    """
    What needs to be added to the visualization window.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
        background: The background color of the window.
        cpo: Camera position of the active render window. Available ``cpo`` are:

                * Iterable containing position, focal_point, and view up.
                    ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
                * Iterable containing a view vector.
                    ``E.g.: [-1.0, 2.0, -5.0].``
                * A string containing the plane orthogonal to the view direction.
                    ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.

                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.

                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.

                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_style: Visualization style of the model. One of the following:

                * ``model_style = 'surface'``,
                * ``model_style = 'wireframe'``,
                * ``model_style = 'points'``.
        model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.

                    If ``model_style = 'wireframe'``, thickness of lines.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None, "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        show_outline:  whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the ``add_outline`` function.

                        By default, it is an empty dictionary and the `add_legend` function will use the
                        ``{"outline_width": 5.0, "outline_color": "black", "show_labels": True, "font_size": 16,
                        "font_color": "white", "font_family": "arial"}`` as its parameters. Otherwise,
                        you can provide a dictionary that properly modify those keys according to your needs.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
    """

    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    # Add model(s) to the plotter.
    add_model(
        plotter=plotter,
        model=model,
        key=key,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_size=model_size,
        model_style=model_style,
    )

    # Set the camera position of plotter.
    plotter.camera_position = cpo

    # Add a legend to the plotter.
    if show_legend:
        lg_kwargs = dict(
            title=key if isinstance(key, str) else key[-1],
            legend_size=None,
            legend_loc=None,
            font_color=cbg_rgb,
            title_font_size=-1,
            label_font_size=12,
            font_family="arial",
            fmt="%.2e",
            n_labels=5,
            vertical=True,
        )
        if not (legend_kwargs is None):
            lg_kwargs.update((k, legend_kwargs[k]) for k in lg_kwargs.keys() & legend_kwargs.keys())

        add_legend(
            plotter=plotter,
            model=model,
            key=key,
            colormap=colormap,
            **lg_kwargs,
        )

    # Add an outline to the plotter.
    if show_outline:
        ol_kwargs = dict(
            outline_width=5.0,
            outline_color=cbg_rgb,
            show_labels=True,
            font_size=16,
            font_color=bg_rgb,
            font_family="arial",
        )
        if not (outline_kwargs is None):
            ol_kwargs.update((k, outline_kwargs[k]) for k in ol_kwargs.keys() & outline_kwargs.keys())
        add_outline(plotter=plotter, model=model, **ol_kwargs)

    # Add text to the plotter.
    if not (text is None):
        t_kwargs = dict(
            font_family="arial",
            font_size=12,
            font_color=cbg_rgb,
            text_loc="upper_left",
        )
        if not (text_kwargs is None):
            t_kwargs.update((k, text_kwargs[k]) for k in t_kwargs.keys() & text_kwargs.keys())
        add_text(plotter=plotter, text=text, **t_kwargs)


def three_d_plot(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Union[str, list] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    window_size: tuple = (512, 512),
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 24,
    plotter_filename: Optional[str] = None,
):
    """
    Visualize reconstructed 3D model.

    Args:
        model: A reconstructed model.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.

                * Output an image file,please enter a filename ending with
                  ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
                * Output a gif file, please enter a filename ending with ``.gif``.
                * Output a mp4 file, please enter a filename ending with ``.mp4``.
        jupyter: Whether to plot in jupyter notebook. Available ``jupyter`` are:

                * ``'none'`` - Do not display in the notebook.
                * ``'pythreejs'`` - Show a pythreejs widget
                * ``'static'`` - Display a static figure.
                * ``'ipygany'`` - Show an ipygany widget
                * ``'panel'`` - Show a panel widget.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        cpo: Camera position of the active render window. Available ``cpo`` are:

                * Iterable containing position, focal_point, and view up.
                    ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
                * Iterable containing a view vector.
                    ``E.g.: [-1.0, 2.0, -5.0].``
                * A string containing the plane orthogonal to the view direction.
                    ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.

                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.

                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.

                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_style: Visualization style of the model. One of the following:

                * ``model_style = 'surface'``,
                * ``model_style = 'wireframe'``,
                * ``model_style = 'points'``.
        model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.

                    If ``model_style = 'wireframe'``, thickness of lines.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None,  "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        show_outline:  whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the ``add_outline`` function.

                        By default, it is an empty dictionary and the `add_legend` function will use the
                        ``{"outline_width": 5.0, "outline_color": "black", "show_labels": True, "font_size": 16,
                        "font_color": "white", "font_family": "arial"}`` as its parameters. Otherwise,
                        you can provide a dictionary that properly modify those keys according to your needs.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
        view_up: The normal to the orbital plane. Only available when filename ending with ``.mp4`` or ``.gif``.
        framerate: Frames per second. Only available when filename ending with ``.mp4`` or ``.gif``.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.

                * Output a gltf file, please enter a filename ending with ``.gltf``.
                * Output a html file, please enter a filename ending with ``.html``.
                * Output an obj file, please enter a filename ending with ``.obj``.
                * Output a vtkjs file, please enter a filename without format.

    Returns:

        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
    """
    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
    )
    model_kwargs = dict(
        background=background,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        show_outline=show_outline,
        outline_kwargs=outline_kwargs,
        text=text,
        text_kwargs=text_kwargs,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    wrap_to_plotter(plotter=p, model=model, key=key, cpo=cpo, **model_kwargs)
    cpo = p.show(return_cpos=True, jupyter_backend="none", cpos=cpo)

    # Create another plotting object to save pyvista/vtk model.
    p = create_plotter(off_screen=off_screen2, **plotter_kws)
    wrap_to_plotter(plotter=p, model=model, key=key, cpo=cpo, **model_kwargs)

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)

    # Output the plotting object.
    return output_plotter(
        plotter=p,
        filename=filename,
        view_up=view_up,
        framerate=framerate,
        jupyter=jupyter_backend,
    )


def three_d_multi_plot(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Union[str, list] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    shape: Union[str, list, tuple] = None,
    window_size: Optional[tuple] = None,
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Union[str, list] = None,
    text_kwargs: Optional[dict] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 24,
    plotter_filename: Optional[str] = None,
):
    """
    Multi-view visualization of reconstructed 3D model.
    If you want to draw a legend in each sub-window, please ensure that the key names used in each legend are different.

    Args:
        model: A MultiBlock of reconstructed models or a reconstructed model.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.

                * Output an image file,please enter a filename ending with
                  ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
                * Output a gif file, please enter a filename ending with ``.gif``.
                * Output a mp4 file, please enter a filename ending with ``.mp4``.
        jupyter: Whether to plot in jupyter notebook. Available ``jupyter`` are:

                * ``'none'`` - Do not display in the notebook.
                * ``'pythreejs'`` - Show a pythreejs widget
                * ``'static'`` - Display a static figure.
                * ``'ipygany'`` - Show an ipygany widget
                * ``'panel'`` - Show a panel widget.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        shape: Number of sub-render windows inside the main window. By default, there is only one render window.

               * Specify two across with ``shape``=(2, 1) and a two by two grid with ``shape``=(2, 2).
               * ``shape`` Can also accept a string descriptor as shape.

                    ``E.g.: shape="3|1" means 3 plots on the left and 1 on the right,``
                    ``E.g.: shape="4/2" means 4 plots on top and 2 at the bottom.``
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        cpo: Camera position of the active render window. Available ``cpo`` are:

                * Iterable containing position, focal_point, and view up.
                    ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
                * Iterable containing a view vector.
                    ``E.g.: [-1.0, 2.0, -5.0].``
                * A string containing the plane orthogonal to the view direction.
                    ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.

                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.

                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.

                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_style: Visualization style of the model. One of the following:

                * ``model_style = 'surface'``,
                * ``model_style = 'wireframe'``,
                * ``model_style = 'points'``.
        model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.

                    If ``model_style = 'wireframe'``, thickness of lines.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None,  "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        show_outline:  whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the ``add_outline`` function.

                        By default, it is an empty dictionary and the `add_legend` function will use the
                        ``{"outline_width": 5.0, "outline_color": "black", "show_labels": True, "font_size": 16,
                        "font_color": "white", "font_family": "arial"}`` as its parameters. Otherwise,
                        you can provide a dictionary that properly modify those keys according to your needs.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
        view_up: The normal to the orbital plane. Only available when filename ending with ``.mp4`` or ``.gif``.
        framerate: Frames per second. Only available when filename ending with ``.mp4`` or ``.gif``.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.

                * Output a gltf file, please enter a filename ending with ``.gltf``.
                * Output a html file, please enter a filename ending with ``.html``.
                * Output an obj file, please enter a filename ending with ``.obj``.
                * Output a vtkjs file, please enter a filename without format.
    """
    models = model if isinstance(model, (MultiBlock, list)) else [model]
    keys = key if isinstance(key, list) else [key]
    cpos = cpo if isinstance(cpo, list) else [cpo]
    mts = model_style if isinstance(model_style, list) else [model_style]
    mss = model_size if isinstance(model_size, list) else [model_size]
    cmaps = colormap if isinstance(colormap, list) else [colormap]
    ams = ambient if isinstance(ambient, list) else [ambient]
    ops = opacity if isinstance(opacity, list) else [opacity]
    texts = text if isinstance(text, list) else [text]

    n_window = max(
        len(models),
        len(keys),
        len(cpos),
        len(mts),
        len(mss),
        len(cmaps),
        len(ams),
        len(ops),
        len(texts),
    )

    models = collect_models([models[0].copy() for i in range(n_window)]) if len(models) == 1 else models
    keys = keys * n_window if len(keys) == 1 else keys
    cpos = cpos * n_window if len(cpos) == 1 else cpos
    mts = mts * n_window if len(mts) == 1 else mts
    mss = mss * n_window if len(mss) == 1 else mss
    cmaps = cmaps * n_window if len(cmaps) == 1 else cmaps
    ams = ams * n_window if len(ams) == 1 else ams
    ops = ops * n_window if len(ops) == 1 else ops
    texts = texts * n_window if len(texts) == 1 else texts

    shape = (math.ceil(n_window / 3), n_window if n_window < 3 else 3) if shape is None else shape
    if isinstance(shape, (tuple, list)):
        n_subplots = shape[0] * shape[1]
        subplots = []
        for i in range(n_subplots):
            col = math.floor(i / shape[1])
            ind = i - col * shape[1]
            subplots.append([col, ind])
    else:
        shape_x, shape_y = re.split("[/|]", shape)
        n_subplots = int(shape_x) * int(shape_y)
        subplots = [i for i in range(n_subplots)]

    win_x, win_y = shape[1], shape[0]
    window_size = (
        (512 * win_x, 512 * win_y) if window_size is None else (window_size[0] * win_x, window_size[1] * win_y)
    )

    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
        shape=shape,
    )

    model_kwargs = dict(
        background=background,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        show_outline=show_outline,
        outline_kwargs=outline_kwargs,
        text_kwargs=text_kwargs,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    for (
        sub_model,
        sub_key,
        sub_cpo,
        sub_mt,
        sub_ms,
        sub_cmap,
        sub_am,
        sub_op,
        sub_text,
        subplot_index,
    ) in zip(models, keys, cpos, mts, mss, cmaps, ams, ops, texts, subplots):
        p.subplot(subplot_index[0], subplot_index[1])
        wrap_to_plotter(
            plotter=p,
            model=sub_model,
            key=sub_key,
            cpo=sub_cpo,
            text=sub_text,
            model_style=sub_mt,
            model_size=sub_ms,
            colormap=sub_cmap,
            ambient=sub_am,
            opacity=sub_op,
            **model_kwargs,
        )
        p.add_axes()

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)

    # Output the plotting object.
    return output_plotter(
        plotter=p,
        filename=filename,
        view_up=view_up,
        framerate=framerate,
        jupyter=jupyter_backend,
    )


def three_d_animate(
    models: Union[List[PolyData or UnstructuredGrid], MultiBlock],
    stable_model: Optional[Union[PolyData, UnstructuredGrid, MultiBlock]] = None,
    stable_kwargs: Optional[dict] = None,
    key: Optional[str] = None,
    filename: str = "animate.mp4",
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    window_size: tuple = (512, 512),
    background: str = "white",
    cpo: Union[str, list] = "iso",
    colormap: Optional[Union[str, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    show_outline: bool = False,
    outline_kwargs: Optional[dict] = None,
    text: Optional[str] = None,
    text_kwargs: Optional[dict] = None,
    framerate: int = 24,
    plotter_filename: Optional[str] = None,
):
    """
    Animated visualization of 3D reconstruction model.

    Args:
        models: A List of reconstructed models or a MultiBlock.
        stable_model: The model that do not change with time in animation.
        stable_kwargs: Parameters for plotting stable model. Available ``stable_kwargs`` are:

                * ``'key'``
                * ``'ambient'``
                * ``'opacity'``
                * ``'model_style'``
                * ``'model_size'``
                * ``'background'``
                * ``'show_legend'``
                * ``'legend_kwargs'``
                * ``'show_outline'``
                * ``'outline_kwargs'``
                * ``'text'``
                * ``'text_kwargs'``
        key: The key under which are the labels.
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
        window_size: Window size in pixels. The default window_size is ``[512, 512]``.
        background: The background color of the window.
        cpo: Camera position of the active render window. Available ``cpo`` are:

                * Iterable containing position, focal_point, and view up.
                    ``E.g.: [(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)].``
                * Iterable containing a view vector.
                    ``E.g.: [-1.0, 2.0, -5.0].``
                * A string containing the plane orthogonal to the view direction.
                    ``E.g.: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'.``
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.

                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.

                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.

                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_style: Visualization style of the model. One of the following:

                * ``model_style = 'surface'``,
                * ``model_style = 'wireframe'``,
                * ``model_style = 'points'``.
        model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.

                    If ``model_style = 'wireframe'``, thickness of lines.
        show_legend: whether to add a legend to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None,  "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        show_outline:  whether to produce an outline of the full extent for the model.
        outline_kwargs: A dictionary that will be pass to the ``add_outline`` function.

                        By default, it is an empty dictionary and the `add_legend` function will use the
                        ``{"outline_width": 5.0, "outline_color": "black", "show_labels": True, "font_size": 16,
                        "font_color": "white", "font_family": "arial"}`` as its parameters. Otherwise,
                        you can provide a dictionary that properly modify those keys according to your needs.
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
        framerate: Frames per second. Only available when filename ending with ``.mp4`` or ``.gif``.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.

                * Output a gltf file, please enter a filename ending with ``.gltf``.
                * Output a html file, please enter a filename ending with ``.html``.
                * Output an obj file, please enter a filename ending with ``.obj``.
                * Output a vtkjs file, please enter a filename without format.
    """

    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
    )
    model_kwargs = dict(
        background=background,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        show_outline=show_outline,
        outline_kwargs=outline_kwargs,
        text=text,
        text_kwargs=text_kwargs,
    )

    if not (stable_model is None):
        stable_kwargs = model_kwargs if stable_kwargs is None else stable_kwargs
        if "key" not in stable_kwargs.keys():
            stable_kwargs["key"] = key

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Check models.
    blocks = collect_models(models) if isinstance(models, list) else models
    blocks_name = blocks.keys()

    # Create a plotting object to display the end model of blocks.
    end_block = blocks[blocks_name[-1]].copy()
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    if not (stable_model is None):
        wrap_to_plotter(plotter=p, model=stable_model, cpo=cpo, **stable_kwargs)
    wrap_to_plotter(plotter=p, model=end_block, key=key, cpo=cpo, **model_kwargs)
    cpo = p.show(return_cpos=True, jupyter_backend="none", cpos=cpo)

    # Create another plotting object to save pyvista/vtk model.
    start_block = blocks[blocks_name[0]].copy()
    p = create_plotter(off_screen=off_screen2, **plotter_kws)
    if not (stable_model is None):
        wrap_to_plotter(plotter=p, model=stable_model, cpo=cpo, **stable_kwargs)
    wrap_to_plotter(plotter=p, model=start_block, key=key, cpo=cpo, **model_kwargs)

    filename_format = filename.split(".")[-1]
    if filename_format == "gif":
        p.open_gif(filename)
    elif filename_format == "mp4":
        p.open_movie(filename, framerate=framerate, quality=5)

    for block_name in blocks_name[1:]:
        block = blocks[block_name]
        start_block.overwrite(block)
        p.write_frame()

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)

    # Close the plotting object.
    p.close()


def merge_animations(
    mp4_files: Optional[list] = None,
    mp4_folder: Optional[list] = None,
    filename: str = "merged_animation.mp4",
):
    """
    Use MoviePy to compose a new animation and play multiple animations together in the new animation.

    Args:
        mp4_files: A list containing absolute paths to mp4 files that need to be played together.
        mp4_folder: Absolute path to the folder containing all mp4 files that need to be played together. If ``mp4_files`` is provided, ``mp4_folder`` cannot also be provided.
        filename: Absolute path to save the newly composed animation.

    Examples:
        st.pl.merge_animations(mp4_files=["animation1.mp4", "animation2.mp4"], filename=f"merged_animation.mp4")
    """
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        raise ImportError(
            "You need to install the package `moviepy`." "\nInstall moviepy via `pip install --upgrade moviepy`"
        )

    try:
        from natsort import natsorted
    except ImportError:
        raise ImportError(
            "You need to install the package `natsort`." "\nInstall natsort via `pip install --upgrade natsort`"
        )

    clips = []
    if mp4_files is None and mp4_folder is not None:
        for root, dirs, files in os.walk(mp4_folder):
            for file in files:
                mp4_file = os.path.join(root, file)
                assert str(mp4_file).endswith(".mp4"), f"``{mp4_file}`` is not the mp4 file."
                clips.append(VideoFileClip(mp4_file))
    elif mp4_files is not None and mp4_folder is None:
        for mp4_file in mp4_files:
            assert str(mp4_file).endswith(".mp4"), f"``{mp4_file}`` is not the mp4 file."
            clips.append(VideoFileClip(mp4_file))
    else:
        raise ValueError("One of ``mp4_files`` and ``mp4_folder`` must be None.")

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(filename)


def quick_plot_3D_celltypes(
    adata: anndata.AnnData,
    save_path: str,
    colors: Optional[list] = None,
    coords_key: str = "spatial",
    group_key: str = "celltype",
    opacity: float = 1.0,
    title: Optional[str] = None,
    ct_subset: Optional[list] = None,
    size: float = 2.0,
):
    """Using plotly, save a 3D plot where cells are drawn as points and colored by their cell type.

    Args:
        adata: AnnData object containing spatial coordinates and cell type labels
        save_path: Path to save the plot
        colors: Optional, used to specify colors for each cell type, given as a list that is at least as long as the
            set of cell types (in the AnnData object if all cell types are used, and in "ct_subset" if "ct_subset"
            is given). If None, a default color palette will be used.
        coords_key: Key in adata.obsm where spatial coordinates are stored
        group_key: Key in adata.obs where cell type labels are stored
        opacity: Sets only the transparency of the "Other" labeled points. Default is 1.0 (fully opaque).
        title: Optional, can be used to provide a title for the plot
        ct_subset: Optional, used to specify cell types of interest. If given, only cells with these types will be
            plotted, and other cells will be labeled "Other". If None, all cell types will be plotted.
        size: Size of the points in the plot. Defaults to 2.
    """
    from ..colorlabel import godsnot_102

    if colors is None:
        colors = godsnot_102

    if coords_key not in adata.obsm.keys():
        raise ValueError(f"adata.obsm does not contain {coords_key}- spatial coordinates could not be found.")
    if group_key not in adata.obs.keys():
        raise ValueError(f"adata.obs does not contain {group_key}- cell type labels could not be found.")

    if adata.obsm[coords_key].shape[1] != 3:
        raise ValueError(f"{coords_key} must be 3-dimensional.")

    spatial_coords = adata.obsm[coords_key]
    x, y, z = spatial_coords[:, 0], spatial_coords[:, 1], spatial_coords[:, 2]

    all_cts = adata.obs[group_key].unique()
    if ct_subset is not None:
        if len(ct_subset) < len(all_cts):
            adata.obs["temp"] = adata.obs[group_key].apply(lambda x: x if x in ct_subset else "Other")
            group_key = "temp"

    # Dictionary mapping cell types to colors:
    ct_color_mapping = dict(zip(adata.obs[group_key].value_counts().index, colors))
    if group_key == "temp":
        ct_color_mapping["Other"] = "#D3D3D3"

    traces = []
    for ct, color in ct_color_mapping.items():
        ct_mask = adata.obs[group_key] == ct
        if ct == "Other":
            opacity = opacity
        scatter = go.Scatter3d(
            x=x[ct_mask],
            y=y[ct_mask],
            z=z[ct_mask],
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity),
            showlegend=False,
        )
        traces.append(scatter)

        # Invisible trace for the legend (so the colored point is larger than the plot points):
        legend_target = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=30, color=color),
            name=ct,
            showlegend=True,
        )
        traces.append(legend_target)

    fig = go.Figure(data=traces)
    if title is None:
        title = "Cell Types of Interest" if ct_subset is not None else "Cells, Colored by Type"
    title_dict = dict(
        text=title,
        y=0.9,
        yanchor="top",
        x=0.5,
        xanchor="center",
        font=dict(size=28),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.65, y=0.85, orientation="v", font=dict(size=18)),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            zaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=50),  # Adjust margins to minimize spacing
        title=title_dict,
    )
    fig.write_html(save_path)


def plot_expression_3D(
    adata: anndata.AnnData,
    save_path: str,
    gene: str,
    coords_key: str = "spatial",
    group_key: Optional[str] = None,
    ct_subset: Optional[list] = None,
    pcutoff: Optional[float] = 99.7,
    zero_opacity: float = 1.0,
    size: int = 2,
):
    """Visualize gene expression in a 3D space.

    Args:
        adata: AnnData object containing spatial coordinates and cell type labels
        save_path: Path to save the plot
        gene: Will plot expression pattern of this gene
        coords_key: Key in adata.obsm where spatial coordinates are stored
        group_key: Optional key for grouping in adata.obs, but needed if "ct_subset" is provided
        ct_subset: Optional list of cell types to include in the plot. If None, all cell types will be included.
        pcutoff: Percentile cutoff for gene expression. Default is 99.7, which will set the max value plotted to the
            99.7th percentile of gene expression values.
        zero_opacity: Opacity of points with zero expression. Between 0.0 and 1.0. Default is 1.0.
        size: Size of the points in the plot. Defaults to 2.
    """
    if group_key is not None:
        if group_key not in adata.obs.keys():
            raise ValueError(f"adata.obs does not contain {group_key}- cell type labels could not be found.")
        adata = adata[adata.obs[group_key].isin(ct_subset), :].copy()

    coords = adata.obsm[coords_key]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    gene_expr = adata[:, gene].X.toarray().flatten()
    # Lenient w/ the max value cutoff so that the colored dots are more distinct from black background
    cutoff = np.percentile(gene_expr, pcutoff)
    gene_expr[gene_expr > cutoff] = cutoff

    # Separately plot zeros and nonzeros:
    zero_indices = gene_expr == 0
    non_zero_indices = gene_expr > 0

    x_zeros, y_zeros, z_zeros = x[zero_indices], y[zero_indices], z[zero_indices]
    x_non_zeros, y_non_zeros, z_non_zeros = x[non_zero_indices], y[non_zero_indices], z[non_zero_indices]
    gene_expr_non_zeros = gene_expr[non_zero_indices]

    # Plot non-zero expression values including one zero for color consistency
    gene_expr_nz = np.append(gene_expr_non_zeros, 0)  # Include one zero value
    x_nz = np.append(x_non_zeros, x[zero_indices][0]) if len(x_zeros) > 0 else x_non_zeros
    y_nz = np.append(y_non_zeros, y[zero_indices][0]) if len(y_zeros) > 0 else y_non_zeros
    z_nz = np.append(z_non_zeros, z[zero_indices][0]) if len(z_zeros) > 0 else z_non_zeros

    scatter_expr_nz = go.Scatter3d(
        x=x_nz,
        y=y_nz,
        z=z_nz,
        mode="markers",
        marker=dict(
            color=gene_expr_nz,
            colorscale="Hot",
            size=size,
            colorbar=dict(title=f"{gene}", x=0.75, titlefont=dict(size=24), tickfont=dict(size=24)),
        ),
        showlegend=False,
    )

    # Add separate trace for zero expression values, if any, with specified opacity
    if len(x_zeros) > 0:
        scatter_expr_zeros = go.Scatter3d(
            x=x_zeros,
            y=y_zeros,
            z=z_zeros,
            mode="markers",
            marker=dict(
                color="#000000",  # Use zero for color to match color scale
                size=size,
                opacity=zero_opacity,  # Apply custom opacity for zeros
            ),
            showlegend=False,
        )
    else:
        scatter_expr_zeros = None

    fig = go.Figure(data=[scatter_expr_nz])
    if scatter_expr_zeros is not None:
        fig.add_trace(scatter_expr_zeros)

    title_dict = dict(
        text=f"{gene}",
        y=0.9,
        yanchor="top",
        x=0.5,
        xanchor="center",
        font=dict(size=36),
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            zaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=50),  # Adjust margins to minimize spacing
        title=title_dict,
    )
    fig.write_html(save_path)


def plot_multiple_genes_3D(
    adata: anndata.AnnData,
    genes: list,
    save_path: str,
    colors: Optional[list] = None,
    coords_key: str = "spatial",
    group_key: Optional[str] = None,
    ct_subset: Optional[list] = None,
    size: int = 2,
):
    """Visualize the exclusivity or overlap of multiple gene expression patterns in 3D space.

    Args:
        adata: An AnnData object containing gene expression data.
        genes: List of genes to visualize (e.g., ["gene1", "gene2", "gene3"]).
        save_path: Path to save the figure to (will save as HTML file).
        colors: Optional, list of colors to use for each gene. If None, will use a default Spateo color palette. Must be
            at least the same length as "genes", plus one.
        coords_key: Key for spatial coordinates in adata.obsm.
        group_key: Optional key for grouping in adata.obs, but needed if "ct_subset" is provided.
        ct_subset: Optional list of cell types to include in the plot. If None, all cell types will be included.
        size: Size of the points in the plot. Defaults to 2.
    """
    if colors is None:
        colors = vega_10

    if group_key is not None:
        if group_key not in adata.obs.keys():
            raise ValueError(f"adata.obs does not contain {group_key} - cell type labels could not be found.")
        adata = adata[adata.obs[group_key].isin(ct_subset), :].copy()

    coords = adata.obsm[coords_key]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    for gene in genes:
        adata.obs.loc[adata[:, gene].X.toarray().flatten() > 0, gene] = True
    adata.obs["gene_expressed"] = adata.obs[genes].sum(axis=1)

    adata.obs["gene_expr_category"] = "None"
    # Assign multiple genes expressed category:
    adata.obs.loc[adata.obs["gene_expressed"] > 1, "gene_expr_category"] = "Multiple genes"

    # Assign individual gene labels where only one gene is expressed:
    for gene in genes:
        adata.obs.loc[(adata.obs[gene] == True) & (adata.obs["gene_expr_category"] == "None"), "gene_expr_category"] = (
            gene
        )

    traces = []
    for gene, color in zip(genes + ["Multiple genes"], colors):
        mask = adata.obs["gene_expr_category"] == gene
        if gene == "Multiple genes":
            color = "#D3D3D3"
        scatter = go.Scatter3d(
            x=x[mask],
            y=y[mask],
            z=z[mask],
            mode="markers",
            marker=dict(color=color, size=size),
            name=gene,
            showlegend=False,
        )
        traces.append(scatter)

        # Invisible trace for the legend (so the colored point is larger than the plot points):
        legend_target = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=30, color=color),
            name=gene,
            showlegend=True,
        )
        traces.append(legend_target)

    fig = go.Figure(data=traces)
    title_dict = dict(
        text="Expression Patterns",
        y=0.9,
        yanchor="top",
        x=0.5,
        xanchor="center",
        font=dict(size=36),
    )

    fig.update_layout(
        legend=dict(x=0.65, y=0.85, orientation="v", font=dict(size=24)),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            zaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        title=title_dict,
    )
    fig.write_html(save_path)


def visualize_3D_increasing_direction_gradient(
    adata: anndata.AnnData,
    save_path: str,
    color_key: str = "spatial",
    coord_key: str = "spatial",
    coord_column: int = 0,
    cmap: str = "viridis",
    center: float = 0.5,
    opacity: float = 1.0,
    title: Optional[str] = None,
):
    """Given a key in adata.obsm or adata.obs and optionally a column index, plot a 3D scatterplot where points
    are colored according to increasing value in the specified column (typically, a coordinate axis,
    e.g. the y-axis).

    Args:
        adata: AnnData object containing all required information
        save_path: Save the figure as an HTML file to this path
        color_key: Key in adata.obs or adata.obsm containing numerical information that will be used to color the
            cells with gradient. Defaults to "spatial" (the default assumption also for the key in .obsm
            containing spatial coordinates).
        coord_key: Key in adata.obsm specifying the coordinates of each point in 3D space. Defaults to "spatial".
        coord_column: Column index to use for plotting
        cmap: Colormap to use for plotting
        center: Coordinates will be normalized to [0, 1] and centered around this value. Defaults to 0.5. Larger
            values will result in more points being colored in the upper half of the colormap, and vice versa.
        opacity: Transparency of the points
        title: Optional, can be used to provide a title for the plot
    """
    if color_key not in adata.obsm.keys() and color_key not in adata.obs.keys():
        raise ValueError(f"Key {color_key} not found in adata.obsm or adata.obs.")
    if coord_key not in adata.obsm.keys():
        raise ValueError(f"Key {coord_key} pointing to array containing 3D coordinates not found in adata.obsm.")

    if color_key in adata.obsm.keys():
        coords = adata.obsm[color_key]
        if isinstance(coords, pd.DataFrame):
            coords = coords.values[:, coord_column]
        else:
            coords = coords[:, coord_column]
    else:
        coords = adata.obs[color_key].values.reshape(-1, 1)

    coords_norm = (coords - np.min(coords)) / (np.max(coords) - np.min(coords))
    # Shift center if necessary
    if center != 0.5:
        new_center = center
        coords_norm = np.where(
            coords_norm <= 0.5,
            coords_norm * new_center / 0.5,  # Expand the lower half
            1 - (1 - coords_norm) * (1 - new_center) / 0.5,  # Compress the upper half
        )

    colors = mpl.colormaps[cmap](coords_norm)
    # Convert colors to hex format:
    colors = ["#" + "".join([f"{int(c * 255):02x}" for c in color[:3]]) for color in colors]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=adata.obsm[coord_key][:, 0],
                y=adata.obsm[coord_key][:, 1],
                z=adata.obsm[coord_key][:, 2],
                mode="markers",
                marker=dict(size=2, color=colors, opacity=opacity),
                showlegend=False,
            )
        ]
    )

    if title is not None:
        title_dict = dict(
            text=title,
            y=0.9,
            yanchor="top",
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        )
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.65, y=0.85, orientation="v", font=dict(size=18)),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
            zaxis=dict(
                showgrid=False,
                showline=False,
                linewidth=2,
                linecolor="black",
                backgroundcolor="white",
                title="",
                showticklabels=False,
                ticks="",
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=50),  # Adjust margins to minimize spacing
        title=title_dict,
    )
    fig.write_html(save_path)
