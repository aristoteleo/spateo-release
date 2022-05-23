import math
import re
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from spateo.tdr import collect_model

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


def _add2plotter(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    legend_size: Optional[Tuple] = None,
    legend_loc: Literal[
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ] = "lower right",
    outline: bool = False,
    outline_width: float = 5.0,
    outline_labels: bool = True,
    text: Optional[str] = None,
    text_font: Literal["times", "courier", "arial"] = "times",
    text_size: Union[int, float] = 18,
    text_color: Union[str, tuple, list, None] = None,
    text_loc: Literal[
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
        "lower_edge",
        "upper_edge",
        "right_edge",
        "left_edge",
    ] = "upper_left",
):
    """What needs to be added to the visualization window."""
    add_model(
        plotter=plotter,
        model=model,
        key=key,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
    )

    add_legend(
        plotter=plotter,
        model=model,
        key=key,
        legend_size=legend_size,
        legend_loc=legend_loc,
    )

    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    if outline is True:
        add_outline(
            plotter=plotter,
            model=model,
            outline_width=outline_width,
            outline_color=cbg_rgb,
            labels=outline_labels,
            labels_color=bg_rgb,
        )

    if not (text is None):
        add_text(
            plotter=plotter,
            text=text,
            text_font=text_font,
            text_size=text_size,
            text_color=cbg_rgb if text_color is None else text_color,
            text_loc=text_loc,
        )


def three_d_plot(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    initial_cpo: Union[str, list] = "iso",
    legend_size: Optional[Tuple] = None,
    legend_loc: Literal[
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ] = "lower right",
    outline: bool = False,
    outline_width: float = 5.0,
    outline_labels: bool = True,
    text: Optional[str] = None,
    text_font: Literal["times", "courier", "arial"] = "times",
    text_size: Union[int, float] = 18,
    text_color: Union[str, tuple, list, None] = None,
    text_loc: Literal[
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
        "lower_edge",
        "upper_edge",
        "right_edge",
        "left_edge",
    ] = "upper_left",
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    plotter_filename: Optional[str] = None,
):
    """
    Visualize reconstructed 3D model.

    Args:
        model: A reconstructed model.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output an image file,
                  please enter a filename ending with
                  `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`, `.svg`, `.eps`, `.ps`, `.pdf`, `.tex`.
                * Output a gif file, please enter a filename ending with `.gif`.
                * Output a mp4 file, please enter a filename ending with `.mp4`.
        jupyter: Whether to plot in jupyter notebook.
                * `'none'` : Do not display in the notebook.
                * `'pythreejs'` : Show a pythreejs widget
                * `'static'` : Display a static figure.
                * `'ipygany'` : Show an ipygany widget
                * `'panel'` : Show a panel widget.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model. If a single float value is given, it will be the global opacity of the model and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        point_size: Point size of any nodes in the dataset plotted.
        model_style: Visualization style of the model. One of the following: style='surface', style='wireframe', style='points'.
        initial_cpo: Camera position of the window. Available `initial_cpo` are:
                * Iterable containing position, focal_point, and view up. E.g.:
                    `[(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)]`
                * Iterable containing a view vector. E.g.:
                   ` [-1.0, 2.0, -5.0]`
                * A string containing the plane orthogonal to the view direction. E.g.:
                    `'xy'`, `'xz'`, `'yz'`, `'yx'`, `'zx'`, `'zy'`, `'iso'`
        legend_size: Two float tuple, each float between 0 and 1.
                     For example (0.1, 0.1) would make the legend 10% the size of the entire figure window.
                     If legend_size is None, legend_size will be adjusted adaptively.
        legend_loc: The location of the legend in the window. Available `legend_loc` are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'center right'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
        outline: Produce an outline of the full extent for the model.
        outline_width: The width of outline.
        outline_labels: Whether to add the length, width and height information of the model to the outline.
        text: The text to add the rendering.
        text_font: The font of the text. Available `text_font` are:
                * `'times'`
                * `'courier'`
                * `'arial'`
        text_size: The size of the text.
        text_color: The color of the text.
        text_loc: The location of the text in the window. Available `text_loc` are:
                * `'lower_left'`
                * `'lower_right'`
                * `'upper_left'`
                * `'upper_right'`
                * `'lower_edge'`
                * `'upper_edge'`
                * `'right_edge'`
                * `'left_edge'`
        view_up: The normal to the orbital plane. Only available when filename ending with `.mp4` or `.gif`.
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.
                * Output a gltf file, please enter a filename ending with `.gltf`.
                * Output a html file, please enter a filename ending with `.html`.
                * Output an obj file, please enter a filename ending with `.obj`.
                * Output a vtkjs file, please enter a filename without format.

    Returns:
        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`, `.svg`, `.eps`, `.ps`, `.pdf`, `.tex`.
        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`, `.svg`, `.eps`, `.ps`, `.pdf`, `.tex`.
    """
    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
    )

    model_kws = dict(
        model=model,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
        legend_size=legend_size,
        legend_loc=legend_loc,
        outline=outline,
        outline_width=outline_width,
        outline_labels=outline_labels,
        text=text,
        text_font=text_font,
        text_size=text_size,
        text_color=text_color,
        text_loc=text_loc,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    _add2plotter(plotter=p, **model_kws)
    cpo = p.show(return_cpos=True, cpos=initial_cpo, jupyter_backend="none")

    # Create another plotting object to save pyvista/vtk model.
    p = create_plotter(off_screen=off_screen2, **plotter_kws)
    _add2plotter(plotter=p, **model_kws)
    p.camera_position = cpo

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)

    # Output the plotting object.
    return output_plotter(plotter=p, filename=filename, view_up=view_up, framerate=framerate, jupyter=jupyter)


def three_d_plot_multi_cpos(
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    cpos: Optional[list] = None,
    shape: Union[str, list, tuple] = None,
    window_size: Optional[tuple] = None,
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    legend_size: Optional[Tuple] = None,
    legend_loc: Literal[
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ] = "lower right",
    outline: bool = False,
    outline_width: float = 5.0,
    outline_labels: bool = True,
    text: Optional[str] = None,
    text_font: Literal["times", "courier", "arial"] = "times",
    text_size: Union[int, float] = 18,
    text_color: Union[str, tuple, list, None] = None,
    text_loc: Literal[
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
        "lower_edge",
        "upper_edge",
        "right_edge",
        "left_edge",
    ] = "upper_left",
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    plotter_filename: Optional[str] = None,
):
    """
    Multi-view visualization of reconstructed 3D model.

    Args:
        model: A reconstructed model.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output an image file,
                  please enter a filename ending with
                  `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`, `.svg`, `.eps`, `.ps`, `.pdf`, `.tex`.
                * Output a gif file, please enter a filename ending with `.gif`.
                * Output a mp4 file, please enter a filename ending with `.mp4`.
        jupyter: Whether to plot in jupyter notebook.
                * `'none'` : Do not display in the notebook.
                * `'pythreejs'` : Show a pythreejs widget
                * `'static'` : Display a static figure.
                * `'ipygany'` : Show an ipygany widget
                * `'panel'` : Show a panel widget.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        cpos: The list of camera positions of the window.
        shape: Number of sub-render windows inside the main window. Specify two across with shape=(2, 1) and a two by
               two grid with shape=(2, 2). By default, there is only one render window. Can also accept a string descriptor
               as shape. E.g.:
               shape="3|1" means 3 plots on the left and 1 on the right,
               shape="4/2" means 4 plots on top and 2 at the bottom.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model. If a single float value is given, it will be the global opacity of the model and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        point_size: Point size of any nodes in the dataset plotted.
        model_style: Visualization style of the model. One of the following: style='surface', style='wireframe', style='points'.
        legend_size: Two float tuple, each float between 0 and 1.
                     For example (0.1, 0.1) would make the legend 10% the size of the entire figure window.
                     If legend_size is None, legend_size will be adjusted adaptively.
        legend_loc: The location of the legend in the window. Available `legend_loc` are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'center right'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
        outline: Produce an outline of the full extent for the model.
        outline_width: The width of outline.
        outline_labels: Whether to add the length, width and height information of the model to the outline.
        text: The text to add the rendering.
        text_font: The font of the text. Available `text_font` are:
                * `'times'`
                * `'courier'`
                * `'arial'`
        text_size: The size of the text.
        text_color: The color of the text.
        text_loc: The location of the text in the window. Available `text_loc` are:
                * `'lower_left'`
                * `'lower_right'`
                * `'upper_left'`
                * `'upper_right'`
                * `'lower_edge'`
                * `'upper_edge'`
                * `'right_edge'`
                * `'left_edge'`
        view_up: The normal to the orbital plane. Only available when filename ending with `.mp4` or `.gif`.
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.
                * Output a gltf file, please enter a filename ending with `.gltf`.
                * Output a html file, please enter a filename ending with `.html`.
                * Output an obj file, please enter a filename ending with `.obj`.
                * Output a vtkjs file, please enter a filename without format.

    """
    shape = (2, 3) if shape is None else shape
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
        window_size = (1024 * win_x, 768 * win_y)

    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
        shape=shape,
    )

    model_kws = dict(
        model=model,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
        legend_size=legend_size,
        legend_loc=legend_loc,
        outline=outline,
        outline_width=outline_width,
        outline_labels=outline_labels,
        text=text,
        text_font=text_font,
        text_size=text_size,
        text_color=text_color,
        text_loc=text_loc,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    cpos = ["xy", "xz", "yz", "yx", "zx", "zy"] if cpos is None else cpos
    for cpo, subplot_index in zip(cpos, subplots):
        p.subplot(subplot_index[0], subplot_index[1])
        _add2plotter(plotter=p, **model_kws)
        p.camera_position = cpo
        p.add_axes()

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)

    # Output the plotting object.
    return output_plotter(plotter=p, filename=filename, view_up=view_up, framerate=framerate, jupyter=jupyter)


def three_d_animate(
    models: Union[List[PolyData or UnstructuredGrid], MultiBlock],
    key: Optional[str] = None,
    filename: str = "animate.mp4",
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    initial_cpo: Union[str, list] = "iso",
    legend_size: Optional[Tuple] = None,
    legend_loc: Literal[
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ] = "lower right",
    text: Optional[str] = None,
    text_font: Literal["times", "courier", "arial"] = "times",
    text_size: Union[int, float] = 18,
    text_color: Union[str, tuple, list, None] = None,
    text_loc: Literal[
        "lower_left",
        "lower_right",
        "upper_left",
        "upper_right",
        "lower_edge",
        "upper_edge",
        "right_edge",
        "left_edge",
    ] = "upper_left",
    framerate: int = 15,
    plotter_filename: Optional[str] = None,
):
    """
    Visualize reconstructed 3D models.

    Args:
        models: A List of reconstructed models or a MultiBlock.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output a gif file, please enter a filename ending with `.gif`.
                * Output a mp4 file, please enter a filename ending with `.mp4`.
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model. If a single float value is given, it will be the global opacity of the model and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        point_size: Point size of any nodes in the dataset plotted.
        model_style: Visualization style of the model. One of the following: style='surface', style='wireframe', style='points'.
        initial_cpo: Camera position of the window. Available `initial_cpo` are:
                * Iterable containing position, focal_point, and view up. E.g.:
                    `[(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)]`
                * Iterable containing a view vector. E.g.:
                   ` [-1.0, 2.0, -5.0]`
                * A string containing the plane orthogonal to the view direction. E.g.:
                    `'xy'`, `'xz'`, `'yz'`, `'yx'`, `'zx'`, `'zy'`, `'iso'`
        legend_size: Two float tuple, each float between 0 and 1.
                     For example (0.1, 0.1) would make the legend 10% the size of the entire figure window.
                     If legend_size==None, legend_size will be adjusted adaptively.
        legend_loc: The location of the legend in the window. Available `legend_loc` are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'center right'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
        text: The text to add the rendering.
        text_font: The font of the text. Available `text_font` are:
                * `'times'`
                * `'courier'`
                * `'arial'`
        text_size: The size of the text.
        text_color: The color of the text.
        text_loc: The location of the text in the window. Available `text_loc` are:
                * `'lower_left'`
                * `'lower_right'`
                * `'upper_left'`
                * `'upper_right'`
                * `'lower_edge'`
                * `'upper_edge'`
                * `'right_edge'`
                * `'left_edge'`
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.
                * Output a gltf file, please enter a filename ending with `.gltf`.
                * Output a html file, please enter a filename ending with `.html`.
                * Output an obj file, please enter a filename ending with `.obj`.
                * Output a vtkjs file, please enter a filename without format.
    """

    plotter_kws = dict(
        jupyter=False if jupyter is False else True,
        window_size=window_size,
        background=background,
    )

    model_kws = dict(
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
        legend_size=legend_size,
        legend_loc=legend_loc,
        text=text,
        text_font=text_font,
        text_size=text_size,
        text_color=text_color,
        text_loc=text_loc,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)

    # Check models.
    blocks = collect_model(models) if isinstance(models, list) else models
    blocks_name = blocks.keys()

    # Create a plotting object to display the end model of blocks.
    end_block = blocks[blocks_name[-1]]
    p = create_plotter(off_screen=off_screen1, **plotter_kws)
    _add2plotter(plotter=p, model=end_block, **model_kws)
    cpo = p.show(return_cpos=True, cpos=initial_cpo, jupyter_backend="none")

    # Create another plotting object to save pyvista/vtk model.
    start_block = blocks[blocks_name[0]]
    p = create_plotter(off_screen=off_screen2, **plotter_kws)
    _add2plotter(plotter=p, model=start_block, **model_kws)
    p.camera_position = cpo

    filename_format = filename.split(".")[-1]
    if filename_format == "gif":
        p.open_gif(filename)
    elif filename_format == "mp4":
        p.open_movie(filename, framerate=framerate, quality=5)

    for block_name in blocks_name[1:]:
        block = blocks[block_name]
        start_block.overwrite(block)
        _add2plotter(plotter=p, model=start_block, **model_kws)
        p.write_frame()

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(plotter=p, filename=plotter_filename)
