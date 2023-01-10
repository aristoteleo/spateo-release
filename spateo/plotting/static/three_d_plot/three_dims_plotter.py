import math
from typing import Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ....logging import logger_manager as lm


def create_plotter(
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (512, 512),
    background: str = "white",
    shape: Union[str, list, tuple] = (1, 1),
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk model.

    Args:
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is ``[1024, 768]``.
        background: The background color of the window.
        shape: Number of sub-render windows inside the main window. By default, there is only one render window.

               * Specify two across with ``shape``=(2, 1) and a two by two grid with ``shape``=(2, 2).
               * ``shape`` Can also accept a string descriptor as shape.

                    ``E.g.: shape="3|1" means 3 plots on the left and 1 on the right,``
                    ``E.g.: shape="4/2" means 4 plots on top and 2 at the bottom.``
    Returns:
        plotter: The plotting object to display pyvista/vtk model.
    """

    # Create an initial plotting object.
    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=window_size,
        notebook=jupyter,
        lighting="light_kit",
        shape=shape,
    )

    # Set the background color of the active render window.
    plotter.background_color = background

    # Add a camera orientation widget to the active renderer (This Widget cannot be used in jupyter notebook).
    if shape == (1, 1):
        plotter.add_camera_orientation_widget()
    else:
        plotter.add_axes()

    return plotter


def _set_jupyter(
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
):
    if jupyter is False:
        off_screen1, off_screen2 = off_screen, True
        jupyter_backend = "none"
    elif jupyter is True:
        off_screen1, off_screen2 = True, off_screen
        jupyter_backend = "static"
    elif jupyter in ["panel", "none", "pythreejs", "static", "ipygany"]:
        off_screen1, off_screen2 = True, off_screen
        jupyter_backend = jupyter
    else:
        raise ValueError(
            "`jupyter` value is wrong."
            "\nAvailable `jupyter` value are: `True`, `False`, `'panel'`, `'none'`, `'pythreejs'`, `'static'`, `'ipygany'`."
        )

    return off_screen1, off_screen2, jupyter_backend


def add_model(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    key: Union[str, list] = None,
    colormap: Optional[Union[str, list]] = None,
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, list, np.ndarray] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "surface",
    model_size: Union[float, list] = 3.0,
):
    """
    Add model(s) to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
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
    """

    def _add_model(_p, _model, _key, _colormap, _style, _ambient, _opacity, _model_size):
        """Add any PyVista/VTK model to the scene."""
        if _style == "points":
            render_spheres, render_tubes, smooth_shading = True, False, True
        elif _style == "wireframe":
            render_spheres, render_tubes, smooth_shading = False, True, False
        else:
            render_spheres, render_tubes, smooth_shading = False, False, True
        mesh_kwargs = dict(
            style=_style,
            render_points_as_spheres=render_spheres,
            render_lines_as_tubes=render_tubes,
            point_size=_model_size,
            line_width=_model_size,
            ambient=_ambient,
            opacity=_opacity,
            smooth_shading=smooth_shading,
            show_scalar_bar=False,
        )

        if _colormap is None:
            added_kwargs = dict(
                scalars=f"{_key}_rgba" if _key in _model.array_names else _model.active_scalars_name,
                rgba=True,
            )
        else:
            added_kwargs = (
                dict(scalars=_key if _key in _model.array_names else _model.active_scalars_name, cmap=_colormap)
                if _colormap in list(mpl.colormaps())
                else dict(color=_colormap)
            )

        mesh_kwargs.update(added_kwargs)
        _p.add_mesh(_model, **mesh_kwargs)

    # Add model(s) to the plotter.
    if isinstance(model, (MultiBlock, list)):
        n_model = len(model)

        keys = key if isinstance(key, list) else [key]
        keys = keys * n_model if len(keys) == 1 else keys

        cmaps = colormap if isinstance(colormap, list) else [colormap]
        cmaps = cmaps * n_model if len(cmaps) == 1 else cmaps

        mts = model_style if isinstance(model_style, list) else [model_style]
        mts = mts * n_model if len(mts) == 1 else mts

        mss = model_size if isinstance(model_size, list) else [model_size]
        mss = mss * n_model if len(mss) == 1 else mss

        ams = ambient if isinstance(ambient, list) else [ambient]
        ams = ams * n_model if len(ams) == 1 else ams

        ops = opacity if isinstance(opacity, list) else [opacity]
        ops = ops * n_model if len(ops) == 1 else ops

        for sub_model, sub_key, sub_cmap, sub_mt, sub_ms, sub_am, sub_op in zip(model, keys, cmaps, mts, mss, ams, ops):
            _add_model(
                _p=plotter,
                _model=sub_model,
                _key=sub_key,
                _colormap=sub_cmap,
                _style=sub_mt,
                _model_size=sub_ms,
                _ambient=sub_am,
                _opacity=sub_op,
            )
    else:
        _add_model(
            _p=plotter,
            _model=model,
            _key=key,
            _colormap=colormap,
            _style=model_style,
            _model_size=model_size,
            _ambient=ambient,
            _opacity=opacity,
        )


def add_str_legend(
    plotter: Plotter,
    labels: np.ndarray,
    colors: np.ndarray,
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
):
    """
    Add a string legend to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        labels: Legend labels.
        colors: The colors corresponding to the legend labels.
        legend_size: The percentage (0 to 1) width and height of the window for the legend.

                     For example (0.1, 0.1) would make the legend 10% the size of the entire figure window.

                     If legend_size is None, legend_size will be adjusted adaptively.
        legend_loc: The location of the legend in the window. Available ``legend_loc`` are:

                * ``legend_loc = 'upper right'``
                * ``legend_loc = 'upper left'``
                * ``legend_loc = 'lower left'``
                * ``legend_loc = 'lower right'``
                * ``legend_loc = 'center left'``
                * ``legend_loc = 'center right'``
                * ``legend_loc = 'lower center'``
                * ``legend_loc = 'upper center'``
                * ``legend_loc = 'center'``
    """

    legend_data = np.concatenate([labels.reshape(-1, 1).astype(object), colors.reshape(-1, 1).astype(object)], axis=1)
    legend_data = legend_data[legend_data[:, 0] != "mask", :]
    assert len(legend_data) != 0, "No legend can be added, please set `show_legend=False`."

    legend_entries = legend_data[np.lexsort(legend_data[:, ::-1].T)]
    if legend_size is None:
        legend_num = 10 if len(legend_entries) >= 10 else len(legend_entries)
        legend_size = (0.1 + 0.01 * legend_num, 0.1 + 0.012 * legend_num)

    plotter.add_legend(
        legend_entries.tolist(),
        face="circle",
        bcolor=None,
        loc=legend_loc,
        size=legend_size,
    )


def add_num_legend(
    plotter: Plotter,
    title: Optional[str] = "",
    title_font_size: Optional[Union[int, float]] = None,
    label_font_size: Optional[Union[int, float]] = None,
    font_color: str = "black",
    font_family: Literal["times", "courier", "arial"] = "arial",
    fmt="%.2e",
    n_labels: int = 5,
    legend_size: Union[tuple, list] = (0.1, 0.4),
    legend_loc: Union[tuple, list] = (0.85, 0.3),
    vertical: bool = True,
):
    """
    Add a numeric legend (colorbar) to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        title: Title of the legend. Default '' which is rendered as an empty title.
        title_font_size: Sets the size of the title font.
        label_font_size: Sets the size of the label font.
        font_color: The color of the font.
        font_family: The font of the text. Available `font_family` are:

                * ``font_family = times``
                * ``font_family = courier``
                * ``font_family = arial``
        fmt: printf format for labels.
        n_labels: Number of labels to use for the legend.
        legend_size: The percentage (0 to 1) width and height of the window for the legend.
        legend_loc: The location of the legend in the window.

                      legend_loc[0]: The percentage (0 to 1) along the windows’s horizontal direction to place the bottom left corner of the legend.
                      legend_loc[1]: The percentage (0 to 1) along the windows’s vertical direction to place the bottom left corner of the legend.
        vertical: Use vertical or horizontal legend.
    """

    plotter.add_scalar_bar(
        title=title,
        n_labels=n_labels,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
        color=font_color,
        font_family=font_family,
        use_opacity=True,
        width=legend_size[0],
        height=legend_size[1],
        position_x=legend_loc[0],
        position_y=legend_loc[1],
        vertical=vertical,
        fmt=fmt,
    )


def add_legend(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[Union[str, list]] = None,
    colormap: Optional[Union[str, list]] = None,
    legend_size: Optional[Tuple] = None,
    legend_loc: Union[str, tuple, list] = None,
    title: Optional[str] = "",
    title_font_size: Optional[Union[int, float]] = None,
    label_font_size: Optional[Union[int, float]] = None,
    font_color: str = "black",
    font_family: Literal["times", "courier", "arial"] = "arial",
    fmt="%.2e",
    n_labels: int = 5,
    vertical: bool = True,
):
    """
    Add a legend to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the legend labels.
        colormap: Name of the Matplotlib colormap to use when mapping the scalars.

                  When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        legend_size: The percentage (0 to 1) width and height of the window for the legend.
        legend_loc: The location of the legend in the window.
        title: Title of the legend. Default '' which is rendered as an empty title.
        title_font_size: Sets the size of the title font. Only available when colormap is not None.
        label_font_size: Sets the size of the label font. Only available when colormap is not None.
        font_color: The color of the font. Only available when colormap is not None.
        font_family: The font of the text. Only available when colormap is not None.

                     Available `font_family` are:

                * ``font_family = times``
                * ``font_family = courier``
                * ``font_family = arial``
        fmt: printf format for labels. Only available when colormap is not None.
        n_labels: Number of labels to use for the legend. Only available when colormap is not None.
        vertical: Use vertical or horizontal legend. Only available when colormap is not None.
    """
    if colormap is None:
        assert key is not None, "When colormap is None, key cannot be None at the same time."

        if isinstance(model, MultiBlock):
            keys = key if isinstance(key, list) else [key] * len(model)

            legend_label_data, legend_color_data = [], []
            for m, k in zip(model, keys):
                legend_label_data.append(np.asarray(m[k]).flatten())
                legend_color_data.append(np.asarray([mpl.colors.to_hex(i) for i in m[f"{k}_rgba"]]).flatten())
            legend_label_data = np.concatenate(legend_label_data, axis=0)
            legend_color_data = np.concatenate(legend_color_data, axis=0)
        else:
            legend_label_data = np.asarray(model[key]).flatten()
            legend_color_data = np.asarray([mpl.colors.to_hex(i) for i in model[f"{key}_rgba"]]).flatten()
        legend_data = np.concatenate([legend_label_data.reshape(-1, 1), legend_color_data.reshape(-1, 1)], axis=1)
        unique_legend_data = np.unique(legend_data, axis=0)

        add_str_legend(
            plotter=plotter,
            labels=unique_legend_data[:, 0],
            colors=unique_legend_data[:, 1],
            legend_size=legend_size,
            legend_loc="lower right" if legend_loc is None else legend_loc,
        )
    else:
        add_num_legend(
            plotter=plotter,
            legend_size=(0.1, 0.4) if legend_size is None else legend_size,
            legend_loc=(0.85, 0.3) if legend_loc is None else legend_loc,
            title=title,
            n_labels=n_labels,
            title_font_size=title_font_size,
            label_font_size=label_font_size,
            font_color=font_color,
            font_family=font_family,
            fmt=fmt,
            vertical=vertical,
        )


def add_outline(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    outline_width: float = 5.0,
    outline_color: Union[str, tuple] = "black",
    show_labels: bool = True,
    font_size: int = 16,
    font_color: Union[str, tuple] = "white",
    font_family: Literal["times", "courier", "arial"] = "arial",
):
    """
    Produce an outline of the full extent for the model.
    If labels is True, add the length, width and height information of the model to the outline.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        outline_width: The width of the outline.
        outline_color: The color of the outline.
        show_labels: Whether to add the length, width and height information of the model to the outline.
        font_family: The font family of the text. Available `font_family` are:

                * ``font_family = times``
                * ``font_family = courier``
                * ``font_family = arial``
        font_size: The size of the font.
        font_color: The color of the font.
    """

    model_outline = model.outline()
    plotter.add_mesh(model_outline, color=outline_color, line_width=outline_width)

    if show_labels is True:
        mo_points = np.asarray(model_outline.points)
        model_x = mo_points[:, 0].max() - mo_points[:, 0].min()
        model_y = mo_points[:, 1].max() - mo_points[:, 1].min()
        model_z = mo_points[:, 2].max() - mo_points[:, 2].min()
        model_x, model_y, model_z = (
            round(model_x.astype(float), 5),
            round(model_y.astype(float), 5),
            round(model_z.astype(float), 5),
        )

        momid_points = [
            mo_points[1, :] - [model_x / 2, 0, 0],
            mo_points[1, :] + [0, model_y / 2, 0],
            mo_points[1, :] + [0, 0, model_z / 2],
        ]
        momid_labels = [model_x, model_y, model_z]
        plotter.add_point_labels(
            points=momid_points,
            labels=momid_labels,
            bold=True,
            font_size=font_size,
            font_family=font_family,
            shape="rounded_rect",
            shape_color=outline_color,
            show_points=False,
            text_color=font_color,
        )


def add_text(
    plotter: Plotter,
    text: str,
    font_family: Literal["times", "courier", "arial"] = "arial",
    font_size: Union[int, float] = 12,
    font_color: Union[str, tuple, list] = "black",
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
    """
    Add text to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        text: The text to add the rendering.
        font_family: The font family of the text. Available `font_family` are:

                * ``font_family = times``
                * ``font_family = courier``
                * ``font_family = arial``
        font_size: The size of the font.
        font_color: The color of the font.
        text_loc: The location of the text in the window. Available `text_loc` are:

                * ``text_loc = 'lower_left'``
                * ``text_loc = 'lower_right'``
                * ``text_loc = 'upper_left'``
                * ``text_loc = 'upper_right'``
                * ``text_loc = 'lower_edge'``
                * ``text_loc = 'upper_edge'``
                * ``text_loc = 'right_edge'``
                * ``text_loc = 'left_edge'``
    """
    plotter.add_text(
        text=text,
        font=font_family,
        color=font_color,
        font_size=font_size,
        position=text_loc,
    )


def output_plotter(
    plotter: Plotter,
    filename: Optional[str] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
):
    """
    Output plotter as image, gif file or mp4 file.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.

                * Output an image file,please enter a filename ending with
                  ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
                * Output a gif file, please enter a filename ending with ``.gif``.
                * Output a mp4 file, please enter a filename ending with ``.mp4``.
        view_up: The normal to the orbital plane. Only available when filename ending with ``.mp4`` or ``.gif``.
        framerate: Frames per second. Only available when filename ending with ``.mp4`` or ``.gif``.
        jupyter: Whether to plot in jupyter notebook. Available ``jupyter`` are:

                * ``'none'`` - Do not display in the notebook.
                * ``'pythreejs'`` - Show a pythreejs widget
                * ``'static'`` - Display a static figure.
                * ``'ipygany'`` - Show an ipygany widget
                * ``'panel'`` - Show a panel widget.

    Returns:
        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
    """

    def _to_graph(_screenshot, _jupyter_backend):
        if jupyter is False or jupyter is "none":
            cpo, img = plotter.show(
                screenshot=_screenshot,
                return_img=True,
                return_cpos=True,
                jupyter_backend=_jupyter_backend,
            )
            return cpo, img
        else:
            plotter.show(screenshot=_screenshot, jupyter_backend=_jupyter_backend)

    def _to_gif(_filename, _view_up):
        """Output plotter to gif file."""
        path = plotter.generate_orbital_path(factor=2.0, shift=0, viewup=_view_up, n_points=20)
        plotter.open_gif(_filename)
        plotter.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        plotter.close()

    def _to_mp4(_filename, _view_up, _framerate):
        """Output plotter to mp4 file."""
        path = plotter.generate_orbital_path(factor=2.0, shift=0, viewup=_view_up, n_points=20)
        plotter.open_movie(_filename, framerate=_framerate, quality=5)
        plotter.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        plotter.close()

    _, _, jupyter_backend = _set_jupyter(jupyter=jupyter)

    # The format of the output file.
    if filename is None:
        # p.show(jupyter_backend=jupyter_backend)
        if jupyter is False or jupyter is "none":
            cpo, img = plotter.show(return_img=True, return_cpos=True, jupyter_backend=jupyter_backend)
            return cpo, img
        else:
            plotter.show(jupyter_backend=jupyter_backend)
    else:
        filename_format = filename.split(".")[-1]

        # Output the plotter in the format of the output file.
        if filename_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
            _to_graph(_screenshot=filename, _jupyter_backend=jupyter_backend)
        elif filename_format in ["svg", "eps", "ps", "pdf", "tex"]:
            plotter.save_graphic(filename, title="PyVista Export", raster=True, painter=True)
            _to_graph(_screenshot=None, _jupyter_backend=jupyter_backend)
        elif filename_format == "gif":
            _to_gif(_filename=filename, _view_up=view_up)
            return None
        elif filename_format == "mp4":
            _to_mp4(_filename=filename, _view_up=view_up, _framerate=framerate)
            return None
        else:
            raise ValueError(
                "\nFilename is wrong."
                "\nIf outputting an image file, "
                "please enter a filename ending with "
                "`.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`, `.svg`, `.eps`, `.ps`, `.pdf`, `.tex`."
                "\nIf outputting a gif file, please enter a filename ending with `.gif`."
                "\nIf outputting a mp4 file, please enter a filename ending with `.mp4`."
            )


def save_plotter(
    plotter: Plotter,
    filename: str,
):
    """Save plotter as gltf file, html file, obj file or vtkjs file.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        filename: The filename of the file where the plotter is saved.
                  Writer type is inferred from the extension of the filename.

                * Output a gltf file, please enter a filename ending with ``.gltf``.
                * Output a html file, please enter a filename ending with ``.html``.
                * Output an obj file, please enter a filename ending with ``.obj``.
                * Output a vtkjs file, please enter a filename without format.
    """

    # The format of the save file.
    filename_format = filename.split(".")[-1]

    # Save the plotter in the format of the output file.
    if filename_format == "gltf":
        plotter.export_gltf(filename)
    elif filename_format == "html":
        plotter.export_html(filename)
    elif filename_format == "obj":
        plotter.export_obj(filename)
    else:
        plotter.export_vtkjs(filename)
