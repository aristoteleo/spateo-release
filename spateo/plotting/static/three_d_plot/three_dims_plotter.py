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


def create_plotter(
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    shape: Union[str, list, tuple] = (1, 1),
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk model.

    Args:
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        shape: Number of sub-render windows inside the main window. Specify two across with shape=(2, 1) and a two by
               two grid with shape=(2, 2). By default, there is only one render window. Can also accept a string descriptor
               as shape. E.g.:
               shape="3|1" means 3 plots on the left and 1 on the right,
               shape="4/2" means 4 plots on top and 2 at the bottom.
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
        jupyter_backend = None
    elif jupyter is True:
        off_screen1, off_screen2 = True, off_screen
        jupyter_backend = "static"
    elif jupyter in ["panel", "none", "pythreejs", "static", "ipygany"]:
        off_screen1, off_screen2 = True, off_screen
        jupyter_backend = jupyter
    else:
        raise ValueError(
            "\n`jupyter` value is wrong."
            "\nAvailable `jupyter` value are: `True`, `False`, `'panel'`, `'none'`, `'pythreejs'`, `'static'`, `'ipygany'`."
        )

    return off_screen1, off_screen2, jupyter_backend


def add_model(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    model_style: Literal["points", "surface", "wireframe"] = "surface",
):
    """
    Add model(s) to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model. If a single float value is given, it will be the global opacity of the model and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        point_size: Point size of any nodes in the dataset plotted.
        model_style: Visualization style of the model. One of the following: style='surface', style='wireframe', style='points'.
    """

    def _add_model(_p, _model):
        """Add any PyVista/VTK model to the scene."""

        scalars = f"{key}_rgba" if key in _model.array_names else _model.active_scalars_name

        _p.add_mesh(
            _model,
            scalars=scalars,
            rgba=True,
            render_points_as_spheres=True,
            style=model_style,
            point_size=point_size,
            ambient=ambient,
            opacity=opacity,
            smooth_shading=True,
        )

    # Add model(s) to the plotter.
    if isinstance(model, MultiBlock):
        for sub_model in model:
            _add_model(_p=plotter, _model=sub_model)
    else:
        _add_model(_p=plotter, _model=model)


def add_outline(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    outline_width: float = 5.0,
    outline_color: Union[str, tuple] = "black",
    labels: bool = True,
    labels_size: int = 16,
    labels_color: Union[str, tuple] = "white",
):
    """
    Produce an outline of the full extent for the model.
    If labels is True, add the length, width and height information of the model to the outline.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        outline_width: The width of the outline.
        outline_color: The color of the outline.
        labels: Whether to add the length, width and height information of the model to the outline.
        labels_size: The size of the label font.
        labels_color: The color of the label.
    """

    model_outline = model.outline()
    plotter.add_model(model_outline, color=outline_color, line_width=outline_width)

    if labels is True:
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
            font_size=labels_size,
            font_family="arial",
            shape="rounded_rect",
            shape_color=outline_color,
            show_points=False,
            text_color=labels_color,
        )


def add_legend(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
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
    Add a legend to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
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
    """

    if isinstance(model, MultiBlock):
        legends = pd.DataFrame()
        for sub_model in model:
            if key in sub_model.array_names:
                sub_labels = pd.Series(sub_model[key])
                sub_labels_hex = pd.Series([mpl.colors.to_hex(i) for i in sub_model[f"{key}_rgba"]])
                sub_legends = pd.concat([sub_labels, sub_labels_hex], axis=1)
                legends = pd.concat([legends, sub_legends])
    else:
        labels = pd.Series(model[key])
        labels_hex = pd.Series([mpl.colors.to_hex(i) for i in model[f"{key}_rgba"]])
        legends = pd.concat([labels, labels_hex], axis=1)

    legends.columns = ["label", "hex"]
    legends.drop_duplicates(inplace=True)

    legends = legends[legends["label"] != "mask"]
    if len(legends.index) != 0:
        legends.sort_values(by=["label", "hex"], inplace=True)
        legends.index = range(len(legends.index))

        gap = 1
        gene_dtypes = ["float32", "float64", "int16", "int32", "int64"]
        if legends["label"].dtype in gene_dtypes:
            legends["label"] = legends["label"].round(2).astype(np.str)
            gap = math.ceil(len(legends.index) / 10) - 1

        legend_entries = [[legends["label"].iloc[i], legends["hex"].iloc[i]] for i in range(0, len(legends.index), gap)]

        if legend_size is None:
            legend_num = len(legend_entries)
            legend_num = 10 if legend_num >= 10 else legend_num
            legend_size = (0.1 + 0.01 * legend_num, 0.1 + 0.012 * legend_num)

        plotter.add_legend(
            legend_entries,
            face="circle",
            bcolor=None,
            loc=legend_loc,
            size=legend_size,
        )


def output_plotter(
    p: Plotter,
    filename: Optional[str] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
):
    """
    Output plotter as image, gif file or mp4 file.

    Args:
        p: The plotting object to display pyvista/vtk model.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output an image file,
                  please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
                * Output a gif file, please enter a filename ending with `.gif`.
                * Output a mp4 file, please enter a filename ending with `.mp4`.
        view_up: The normal to the orbital plane. Only available when filename ending with `.mp4` or `.gif`.
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        jupyter: Whether to plot in jupyter notebook.
                * `'none'` : Do not display in the notebook.
                * `'pythreejs'` : Show a pythreejs widget
                * `'static'` : Display a static figure.
                * `'ipygany'` : Show an ipygany widget
                * `'panel'` : Show a panel widget.

    Returns:
        img: Numpy array of the last image.
             Returned only if filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
    """

    def _to_gif(_filename, _view_up):
        """Output plotter to gif file."""
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=_view_up, n_points=20)
        p.open_gif(_filename)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()

    def _to_mp4(_filename, _view_up, _framerate):
        """Output plotter to mp4 file."""
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=_view_up, n_points=20)
        p.open_movie(_filename, framerate=_framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()

    _, _, jupyter_backend = _set_jupyter(jupyter=jupyter)

    # The format of the output file.
    if filename is None:
        # p.show(jupyter_backend=jupyter_backend)
        if jupyter is False:
            cpo, img = p.show(return_img=True, return_cpos=True, jupyter_backend=jupyter_backend)
            return cpo, img
        else:
            p.show(jupyter_backend=jupyter_backend)
    else:
        filename_format = filename.split(".")[-1]

        # Output the plotter in the format of the output file.
        if filename_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
            if jupyter is False:
                cpo, img = p.show(
                    screenshot=filename,
                    return_img=True,
                    return_cpos=True,
                    jupyter_backend=jupyter_backend,
                )
                return cpo, img
            else:
                p.show(screenshot=filename, jupyter_backend=jupyter_backend)
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
                "please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`."
                "\nIf outputting a gif file, please enter a filename ending with `.gif`."
                "\nIf outputting a mp4 file, please enter a filename ending with `.mp4`."
            )


def save_plotter(
    p: Plotter,
    filename: str,
):
    """Save plotter as gltf file, html file, obj file or vtkjs file.

    Args:
       p: The plotting object to display pyvista/vtk model.
       filename: The filename of the file where the plotter is saved.
                 Writer type is inferred from the extension of the filename.
           * Output a gltf file, please enter a filename ending with `.gltf`.
           * Output a html file, please enter a filename ending with `.html`.
           * Output an obj file, please enter a filename ending with `.obj`.
           * Output a vtkjs file, please enter a filename without format.
    """

    # The format of the save file.
    filename_format = filename.split(".")[-1]

    # Save the plotter in the format of the output file.
    if filename_format == "gltf":
        p.export_gltf(filename)
    elif filename_format == "html":
        p.export_html(filename)
    elif filename_format == "obj":
        p.export_obj(filename)
    else:
        p.export_vtkjs(filename)
