import math
import numpy as np

import matplotlib as mpl
import pandas as pd
import pyvista as pv

from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def create_plotter(
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    initial_cpo: Union[str, tuple, list] = "iso",
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk mesh.

    Args:
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        initial_cpo: Camera position of the window. Available `initial_cpo` are:
                * `'xy'`, `'xz'`, `'yz'`, `'yx'`, `'zx'`, `'zy'`, `'iso'`
                * Customize a tuple. E.g.: (7, 0, 20.).
    Returns:
        plotter: The plotting object to display pyvista/vtk mesh.
    """

    # Create an initial plotting object.
    notebook = True if jupyter is True else False
    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=window_size,
        notebook=notebook,
        lighting="light_kit",
    )

    # Set camera position of the active render window.
    plotter.camera_position = initial_cpo

    # Set the background color of the active render window.
    plotter.background_color = background

    # Contrasting color of the background color.
    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    if jupyter is True:
        # Description of control 3D images in jupyter notebook.
        plotter.add_text(
            "The method to control 3D images in jupyter notebook is as follows:"
            "CTRL Left Mouse spins the camera around its view plane normal;"
            "SHIFT Left Mouse pans the camera; "
            "CTRL SHIFT Left Mouse dollies (a positional zoom) the camera;"
            "Left mouse button dollies the camera.",
            font_size=12,
            color=cbg_rgb,
            font="arial",
        )
    else:
        # Add a camera orientation widget to the active renderer (This Widget cannot be used in jupyter notebook).
        plotter.add_camera_orientation_widget()

    return plotter


def add_mesh(
    plotter: Plotter,
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    ambient: float = 0.2,
    opacity: float = 1.0,
    style: Literal["points", "surface", "volume"] = "surface",
    point_size: float = 5.0,
):
    """
    Add mesh(es) to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk mesh.
        mesh: A reconstructed mesh.
        key: The key under which are the labels.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        style: Visualization style of the mesh. One of the following: style='surface', style='volume', style='points'.
        point_size: Point size of any nodes in the dataset plotted.
    """

    def _add_mesh(_p, _mesh):
        """Add any PyVista/VTK mesh to the scene."""

        mesh_style = "points" if style == "points" else None
        _p.add_mesh(
            _mesh,
            scalars=f"{key}_rgba",
            rgba=True,
            render_points_as_spheres=True,
            style=mesh_style,
            point_size=point_size,
            ambient=ambient,
            opacity=opacity,
        )

    # Add mesh(es) to the plotter.
    if isinstance(mesh, MultiBlock):
        for sub_mesh in mesh:
            _add_mesh(_p=plotter, _mesh=sub_mesh)
    else:
        _add_mesh(_p=plotter, _mesh=mesh)


def add_legend(
    plotter: Plotter,
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    legend_loc: Literal[
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center left",
        "lower center",
        "upper center",
        "center",
    ] = "lower right",
):
    """
    Add a legend to the plotter.

    Args:
        plotter: The plotting object to display pyvista/vtk mesh.
        mesh: A reconstructed mesh.
        key: The key under which are the labels.
        legend_loc: The location of the legend in the window. Available `legend_loc` are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
    """

    if isinstance(mesh, MultiBlock):
        legends = pd.DataFrame()
        for sub_mesh in mesh:
            sub_labels = pd.Series(sub_mesh[key])
            sub_labels_hex = pd.Series([mpl.colors.to_hex(i) for i in sub_mesh[f"{key}_rgba"]])
            sub_legends = pd.concat([sub_labels, sub_labels_hex], axis=1)
            legends = pd.concat([legends, sub_legends])
    else:
        labels = pd.Series(mesh[key])
        labels_hex = pd.Series([mpl.colors.to_hex(i) for i in mesh[f"{key}_rgba"]])
        legends = pd.concat([labels, labels_hex], axis=1)

    legends.columns = ["label", "hex"]
    legends.drop_duplicates(inplace=True)

    legends = legends[legends["label"] != "mask"]
    if len(legends.index) != 0:
        legends.sort_values(by=["label", "hex"], inplace=True)
        legends.index = range(len(legends.index))

        gene_dtypes = ["float16", "float32", "float64", "int16", "int32", "int64"]
        gap = math.ceil(len(legends.index) / 10) - 1 if legends["label"].dtype in gene_dtypes else 1
        legend_entries = [[legends["label"].iloc[i], legends["hex"].iloc[i]] for i in range(0, len(legends.index), gap)]

        legend_num = len(legend_entries)
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
    filename: str,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
):
    """
    Output plotter as image, gif file or mp4 file.

    Args:
        p: The plotting object to display pyvista/vtk mesh.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
            * Output an image file,
              please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
            * Output a gif file, please enter a filename ending with `.gif`.
            * Output a mp4 file, please enter a filename ending with `.mp4`.
        view_up: The normal to the orbital plane. Only available when filename ending with `.mp4` or `.gif`.
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
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

    # The format of the output file.
    filename_format = filename.split(".")[-1]

    # Output the plotter in the format of the output file.
    if filename_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
        _, img = p.show(
            screenshot=filename,
            return_img=True,
            return_cpos=True,
        )
        return img
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
       p: The plotting object to display pyvista/vtk mesh.
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


def three_d_plot(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    filename: Optional[str] = None,
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    style: Literal["points", "surface", "volume"] = "surface",
    point_size: float = 5.0,
    initial_cpo: Union[str, tuple] = "iso",
    legend_loc: Literal[
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center left",
        "lower center",
        "upper center",
        "center",
    ] = "lower right",
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    plotter_filename: Optional[str] = None,
):
    """
    Visualize reconstructed 3D meshes.

    Args:
        mesh: A reconstructed mesh.
        key: The key under which are the labels.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output an image file,
                  please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
                * Output a gif file, please enter a filename ending with `.gif`.
                * Output a mp4 file, please enter a filename ending with `.mp4`.
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        style: Visualization style of the mesh. One of the following: style='surface', style='volume', style='points'.
        point_size: Point size of any nodes in the dataset plotted.
        initial_cpo: Camera position of the window. Available `initial_cpo` are:
                * `'xy'`, `'xz'`, `'yz'`, `'yx'`, `'zx'`, `'zy'`, `'iso'`
                * Customize a tuple. E.g.: (7, 0, 20.).
        legend_loc: The location of the legend in the window. Available `legend_loc` are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
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

    # Create a plotting object to display pyvista/vtk mesh.
    p1 = create_plotter(
        jupyter=jupyter,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
        initial_cpo=initial_cpo,
    )
    add_mesh(
        plotter=p1,
        mesh=mesh,
        key=key,
        ambient=ambient,
        opacity=opacity,
        style=style,
        point_size=point_size,
    )
    add_legend(
        plotter=p1,
        mesh=mesh,
        key=key,
        legend_loc=legend_loc,
    )
    jupyter_backend = "panel" if jupyter is True else None
    cpo = p1.show(return_cpos=True, jupyter_backend=jupyter_backend)

    # Create another plotting object to save pyvista/vtk mesh.
    p2 = create_plotter(
        jupyter=jupyter,
        off_screen=True,
        window_size=window_size,
        background=background,
        initial_cpo=cpo,
    )
    add_mesh(
        plotter=p2,
        mesh=mesh,
        key=key,
        ambient=ambient,
        opacity=opacity,
        style=style,
        point_size=point_size,
    )
    add_legend(
        plotter=p2,
        mesh=mesh,
        key=key,
        legend_loc=legend_loc,
    )
    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(p2, filename=plotter_filename)

    # Output the plotting object.
    if filename is not None:
        return output_plotter(p=p2, filename=filename, view_up=view_up, framerate=framerate)
    else:
        return None
