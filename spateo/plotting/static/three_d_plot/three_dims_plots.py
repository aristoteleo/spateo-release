from typing import List, Optional, Tuple, Union

import matplotlib as mpl
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ....tools.TDR.models import collect_model
from .three_dims_plotter import (
    _set_jupyter,
    add_legend,
    add_model,
    add_outline,
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
    model_style: Literal["points", "surface", "wireframe"] = "surface",
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
    if outline is True:
        bg_rgb = mpl.colors.to_rgb(background)
        cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])
        add_outline(
            plotter=plotter,
            model=model,
            outline_width=outline_width,
            outline_color=cbg_rgb,
            labels=outline_labels,
            labels_color=bg_rgb,
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
    model_style: Literal["points", "surface", "wireframe"] = "surface",
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
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    plotter_filename: Optional[str] = None,
):
    """
    Visualize reconstructed 3D models.
    Args:
        model: A reconstructed model.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output an image file,
                  please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
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
        view_up: The normal to the orbital plane. Only available when filename ending with `.mp4` or `.gif`.
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.
                * Output a gltf file, please enter a filename ending with `.gltf`.
                * Output a html file, please enter a filename ending with `.html`.
                * Output an obj file, please enter a filename ending with `.obj`.
                * Output a vtkjs file, please enter a filename without format.
    Returns:
        img: Numpy array of the last image.
             Returned only if filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
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
        save_plotter(p, filename=plotter_filename)

    # Output the plotting object.
    return output_plotter(p=p, filename=filename, view_up=view_up, framerate=framerate, jupyter=jupyter)


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
    model_style: Literal["points", "surface", "wireframe"] = "surface",
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
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        plotter_filename: The filename of the file where the plotter is saved.
                          Writer type is inferred from the extension of the filename.
                * Output a gltf file, please enter a filename ending with `.gltf`.
                * Output a html file, please enter a filename ending with `.html`.
                * Output an obj file, please enter a filename ending with `.obj`.
                * Output a vtkjs file, please enter a filename without format.
    Returns:
        img: Numpy array of the last image.
             Returned only if filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
    """

    blocks = collect_model(models) if isinstance(models, list) else models
    blocks_name = blocks.keys()

    # Create a plotting object to display the end model of blocks.
    end_block = blocks[blocks_name[-1]]
    p1 = create_plotter(
        jupyter=jupyter,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
    )
    _add2plotter(
        plotter=p1,
        model=end_block,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
        legend_size=legend_size,
        legend_loc=legend_loc,
    )
    jupyter_backend = "panel" if jupyter is True else None
    cpo = p1.show(return_cpos=True, jupyter_backend=jupyter_backend, cpos=initial_cpo)

    # Create another plotting object to save.
    start_block = blocks[blocks_name[0]]
    p2 = create_plotter(
        jupyter=jupyter,
        off_screen=True,
        window_size=window_size,
        background=background,
    )
    _add2plotter(
        plotter=p2,
        model=start_block,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        model_style=model_style,
        legend_size=legend_size,
        legend_loc=legend_loc,
    )

    p2.camera_position = cpo

    filename_format = filename.split(".")[-1]
    if filename_format == "gif":
        p2.open_gif(filename)
    elif filename_format == "mp4":
        p2.open_movie(filename, framerate=framerate, quality=5)

    for block_name in blocks_name[1:]:
        block = blocks[block_name]
        start_block.overwrite(block)
        _add2plotter(
            plotter=p2,
            model=start_block,
            key=key,
            background=background,
            ambient=ambient,
            opacity=opacity,
            point_size=point_size,
            legend_size=legend_size,
            legend_loc=legend_loc,
        )
        p2.write_frame()

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(p2, filename=plotter_filename)
