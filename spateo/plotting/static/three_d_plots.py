import math

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
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "black",
    background_r: str = "white",
    ambient: float = 0.3,
    opacity: float = 1.0,
    initial_cpo: Union[str, tuple, list] = "iso",
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
    legend_size: tuple = (0.1, 0.1),
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk mesh.

    Args:
        mesh: A reconstructed mesh.
        key: The key under which are the labels.
        off_screen: Renders off screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        background_r: A color that is clearly different from the background color.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
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
        legend_size: The size of the legend in the window. Two float sequence, each float between 0 and 1.
                     E.g.: (0.1, 0.1) would make the legend 10% the size of the entire figure window.
    Returns:
        plotter: The plotting object to display pyvista/vtk mesh.
    """
    # Create an initial plotting object.
    plotter = pv.Plotter(
        off_screen=off_screen,
        lighting="light_kit",
        window_size=window_size,
        border=True,
        border_color=background_r,
    )
    plotter.camera_position = initial_cpo
    plotter.background_color = background

    # Add a mesh to the plotter.
    plotter.add_mesh(
        mesh,
        scalars=f"{key}_rgba",
        rgba=True,
        render_points_as_spheres=True,
        ambient=ambient,
        opacity=opacity,
    )

    # Add a camera orientation widget to the plotter.
    plotter.add_camera_orientation_widget()

    # Add a legend to the plotter.

    _hex = pd.Series([mpl.colors.to_hex(i) for i in mesh[f"{key}_rgba"]])
    _label = pd.Series(mesh[key])

    _legend_data = pd.concat([_label, _hex], axis=1)
    _legend_data.columns = ["label", "hex"]
    _legend_data = _legend_data[_legend_data["label"] != "mask"]
    _legend_data.drop_duplicates(inplace=True)
    _legend_data.sort_values(by=["label", "hex"], inplace=True)
    _legend_data = _legend_data.astype(str)

    try:
        _label_new = _legend_data["label"]
        _label_new = _label_new.astype(float)
        _label_type = "float"
    except:
        _label_type = "str"

    gap = math.ceil(len(_legend_data.index) / 5) if _label_type == "float" else 1
    legend_entries = [
        [_legend_data["label"].iloc[i], _legend_data["hex"].iloc[i]] for i in range(0, len(_legend_data.index), gap)
    ]
    if _label_type == "float":
        legend_entries.append([_legend_data["label"].iloc[-1], _legend_data["hex"].iloc[-1]])

    plotter.add_legend(
        legend_entries,
        face="circle",
        bcolor=None,
        loc=legend_loc,
        size=legend_size,
    )

    return plotter


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
            * Output an image file, please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
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

    def _to_mp4(_filename, _view_up, _framerate):
        """Output plotter to mp4 file."""
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
        p.open_movie(filename, framerate=framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)

    # The format of the output file.
    filename_format = filename.split(".")[-1]

    # Output the plotter in the format of the output file.
    if filename_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
        _, img = p.show(
            screenshot=filename_format,
            interactive=False,
            return_img=True,
            return_cpos=True,
        )
        return img
    elif filename_format == "gif":
        _to_gif(_filename=filename, _view_up=view_up)
    elif filename_format == "mp4":
        _to_mp4(_filename=filename, _view_up=view_up, _framerate=framerate)
    else:
        raise ValueError(
            "\nFilename is wrong."
            "\nIf outputting an image file, please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`."
            "\nIf outputting a gif file, please enter a filename ending with `.gif`."
            "\nIf outputting a mp4 file, please enter a filename ending with `.mp4`."
        )

    # Close the plotter when finished.
    p.close()


def save_plotter(
    p: Plotter,
    filename: str,
):
    """Save plotter as gltf file, html file, obj file or vtkjs file.

    Args:
       p: The plotting object to display pyvista/vtk mesh.
       filename: The filename of the file where the plotter is saved. Writer type is inferred from the extension of the filename.
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
    key: str,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "black",
    background_r: str = "white",
    ambient: float = 0.3,
    opacity: float = 1.0,
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
    legend_size: tuple = (0.1, 0.1),
    filename: Optional[str] = None,
    view_up: tuple = (0.5, 0.5, 1),
    framerate: int = 15,
    plotter_filename: Optional[str] = None,
):
    """

    Args:
        mesh: A reconstructed mesh.
        key: The key under which are the labels.
        off_screen: Renders off-screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        background_r: A color that is clearly different from the background color.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
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
        legend_size: The size of the legend in the window. Two float sequence, each float between 0 and 1.
                     E.g.: (0.1, 0.1) would make the legend 10% the size of the entire figure window.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                * Output an image file, please enter a filename ending with `.png`, `.tif`, `.tiff`, `.bmp`, `.jpeg`, `.jpg`.
                * Output a gif file, please enter a filename ending with `.gif`.
                * Output a mp4 file, please enter a filename ending with `.mp4`.
        view_up: The normal to the orbital plane. Only available when filename ending with `.mp4` or `.gif`.
        framerate: Frames per second. Only available when filename ending with `.mp4` or `.gif`.
        plotter_filename: The filename of the file where the plotter is saved. Writer type is inferred from the extension of the filename.
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
        mesh=mesh,
        key=key,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
        background_r=background_r,
        ambient=ambient,
        opacity=opacity,
        initial_cpo=initial_cpo,
        legend_loc=legend_loc,
        legend_size=legend_size,
    )
    p2 = create_plotter(
        mesh=mesh,
        key=key,
        off_screen=True,
        window_size=window_size,
        background=background,
        background_r=background_r,
        ambient=ambient,
        opacity=opacity,
        initial_cpo=initial_cpo,
        legend_loc=legend_loc,
        legend_size=legend_size,
    )
    p2.camera_position = p1.show(return_cpos=True)

    # Save the plotting object.
    if plotter_filename is not None:
        save_plotter(p2, filename=plotter_filename)

    # Output the plotting object.
    if filename is not None:
        try:
            return output_plotter(p=p2, filename=filename, view_up=view_up, framerate=framerate)
        except:
            output_plotter(p=p2, filename=filename, view_up=view_up, framerate=framerate)
