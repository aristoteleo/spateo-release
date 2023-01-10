import math
import re
from typing import List, Optional, Union

import matplotlib as mpl
import numpy as np
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ....tdr import collect_models
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
            title_font_size=15,
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

    n_window = max(len(models), len(keys), len(cpos), len(mts), len(mss), len(cmaps), len(ams), len(ops), len(texts))
    models = collect_models([models[0].copy() for i in range(n_window)]) if len(models) == 1 else models
    keys = keys * n_window if len(keys) == 1 else keys
    cpos = cpos * n_window if len(cpos) == 1 else cpos
    mts = mts * n_window if len(mts) == 1 else mts
    mss = mss * n_window if len(mss) == 1 else mss
    cmaps = cmaps * n_window if len(cmaps) == 1 else cmaps
    ams = ams * n_window if len(ams) == 1 else ams
    ops = ops * n_window if len(ops) == 1 else ops
    texts = texts * n_window if len(texts) == 1 else texts

    shape = (math.ceil(n_window / 4), n_window if n_window < 4 else 4) if shape is None else shape
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

    if window_size is None:
        win_x, win_y = (shape[1], shape[0]) if isinstance(shape, (tuple, list)) else (1, 1)
        window_size = (512 * win_x, 512 * win_y)

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
    for sub_model, sub_key, sub_cpo, sub_mt, sub_ms, sub_cmap, sub_am, sub_op, sub_text, subplot_index in zip(
        models, keys, cpos, mts, mss, cmaps, ams, ops, texts, subplots
    ):
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
