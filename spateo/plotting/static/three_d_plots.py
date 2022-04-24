import math
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import MultiBlock, Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ...tools.TDR.mesh.utils import collect_mesh


def create_plotter(
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    shape: Union[str, list, tuple] = (1, 1),
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk mesh.

    Args:
        jupyter: Whether to plot in jupyter notebook.
        off_screen: Renders off screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        background: The background color of the window.
        shape: Number of sub-render windows inside of the main window. Specify two across with shape=(2, 1) and a two by
               two grid with shape=(2, 2). By default there is only one render window. Can also accept a string descriptor
               as shape. E.g.:
               shape="3|1" means 3 plots on the left and 1 on the right,
               shape="4/2" means 4 plots on top and 2 at the bottom.
    Returns:
        plotter: The plotting object to display pyvista/vtk mesh.
    """

    # Create an initial plotting object.
    notebook = True if jupyter else False
    plotter = pv.Plotter(
        off_screen=off_screen, window_size=window_size, notebook=notebook, lighting="light_kit", shape=shape
    )

    # Set the background color of the active render window.
    plotter.background_color = background

    # Add a camera orientation widget to the active renderer (This Widget cannot be used in jupyter notebook).
    if shape == (1, 1):
        plotter.add_camera_orientation_widget()
    else:
        plotter.add_axes()

    return plotter


def add_mesh(
    plotter: Plotter,
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    mesh_style: Literal["points", "surface", "wireframe"] = "surface",
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
        point_size: Point size of any nodes in the dataset plotted.
        mesh_style: Visualization style of the mesh. One of the following: style='surface', style='wireframe', style='points'.
    """

    def _add_mesh(_p, _mesh):
        """Add any PyVista/VTK mesh to the scene."""

        scalars = f"{key}_rgba" if key in _mesh.array_names else _mesh.active_scalars_name

        _p.add_mesh(
            _mesh,
            scalars=scalars,
            rgba=True,
            render_points_as_spheres=True,
            style=mesh_style,
            point_size=point_size,
            ambient=ambient,
            opacity=opacity,
            smooth_shading=True,
        )

    # Add mesh(es) to the plotter.
    if isinstance(mesh, MultiBlock):
        for sub_mesh in mesh:
            _add_mesh(_p=plotter, _mesh=sub_mesh)
    else:
        _add_mesh(_p=plotter, _mesh=mesh)


def add_outline(
    plotter: Plotter,
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    outline_width: float = 5.0,
    outline_color: Union[str, tuple] = "black",
    labels: bool = True,
    labels_size: float = 16,
    labels_color: Union[str, tuple] = "white",
):
    """
    Produce an outline of the full extent for the mesh.
    If labels is True, add the length, width and height information of the mesh to the outline.

    Args:
        plotter: The plotting object to display pyvista/vtk mesh.
        mesh: A reconstructed mesh.
        outline_width: The width of the outline.
        outline_color: The color of the outline.
        labels: Whether to add the length, width and height information of the mesh to the outline.
        labels_size: The size of the label font.
        labels_color: The color of the label.
    """

    mesh_outline = mesh.outline()
    plotter.add_mesh(mesh_outline, color=outline_color, line_width=outline_width)

    if labels is True:
        mo_points = np.asarray(mesh_outline.points)
        mesh_x = mo_points[:, 0].max() - mo_points[:, 0].min()
        mesh_y = mo_points[:, 1].max() - mo_points[:, 1].min()
        mesh_z = mo_points[:, 2].max() - mo_points[:, 2].min()
        mesh_x, mesh_y, mesh_z = (
            round(mesh_x.astype(float), 5),
            round(mesh_y.astype(float), 5),
            round(mesh_z.astype(float), 5),
        )

        momid_points = [
            mo_points[1, :] - [mesh_x / 2, 0, 0],
            mo_points[1, :] + [0, mesh_y / 2, 0],
            mo_points[1, :] + [0, 0, mesh_z / 2],
        ]
        momid_labels = [mesh_x, mesh_y, mesh_z]
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
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
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
        plotter: The plotting object to display pyvista/vtk mesh.
        mesh: A reconstructed mesh.
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

    if isinstance(mesh, MultiBlock):
        legends = pd.DataFrame()
        for sub_mesh in mesh:
            if key in sub_mesh.array_names:
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
        cpo, img = p.show(
            screenshot=filename,
            return_img=True,
            return_cpos=True,
        )
        return cpo, img
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


def _add2plotter(
    plotter: Plotter,
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: Optional[str] = None,
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    mesh_style: Literal["points", "surface", "wireframe"] = "surface",
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
    add_mesh(
        plotter=plotter,
        mesh=mesh,
        key=key,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        mesh_style=mesh_style,
    )
    add_legend(
        plotter=plotter,
        mesh=mesh,
        key=key,
        legend_size=legend_size,
        legend_loc=legend_loc,
    )
    if outline is True:
        bg_rgb = mpl.colors.to_rgb(background)
        cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])
        add_outline(
            plotter=plotter,
            mesh=mesh,
            outline_width=outline_width,
            outline_color=cbg_rgb,
            labels=outline_labels,
            labels_color=bg_rgb,
        )


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
    point_size: float = 5.0,
    mesh_style: Literal["points", "surface", "wireframe"] = "surface",
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
        point_size: Point size of any nodes in the dataset plotted.
        mesh_style: Visualization style of the mesh. One of the following: style='surface', style='wireframe', style='points'.
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
        outline: Produce an outline of the full extent for the mesh.
        outline_width: The width of outline.
        outline_labels: Whether to add the length, width and height information of the mesh to the outline.
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
    p = create_plotter(
        jupyter=jupyter,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
    )
    _add2plotter(
        plotter=p,
        mesh=mesh,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        mesh_style=mesh_style,
        legend_size=legend_size,
        legend_loc=legend_loc,
        outline=outline,
        outline_width=outline_width,
        outline_labels=outline_labels,
    )

    if jupyter is True:
        p.show(jupyter_backend="panel", cpos=initial_cpo)
    else:
        cpo = p.show(return_cpos=True, cpos=initial_cpo)
        # Create another plotting object to save pyvista/vtk mesh.
        p = create_plotter(
            jupyter=jupyter,
            off_screen=True,
            window_size=window_size,
            background=background,
        )
        _add2plotter(
            plotter=p,
            mesh=mesh,
            key=key,
            background=background,
            ambient=ambient,
            opacity=opacity,
            point_size=point_size,
            mesh_style=mesh_style,
            legend_size=legend_size,
            legend_loc=legend_loc,
            outline=outline,
            outline_width=outline_width,
            outline_labels=outline_labels,
        )

        p.camera_position = cpo

        # Save the plotting object.
        if plotter_filename is not None:
            save_plotter(p, filename=plotter_filename)

        # Output the plotting object.
        if filename is not None:
            return output_plotter(p=p, filename=filename, view_up=view_up, framerate=framerate)
        else:
            return None


def three_d_animate(
    meshes: Union[List[PolyData or UnstructuredGrid], MultiBlock],
    key: Optional[str] = None,
    filename: str = "animate.mp4",
    jupyter: bool = False,
    off_screen: bool = False,
    window_size: tuple = (1024, 768),
    background: str = "white",
    ambient: float = 0.2,
    opacity: float = 1.0,
    point_size: float = 5.0,
    mesh_style: Literal["points", "surface", "wireframe"] = "surface",
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
    Visualize reconstructed 3D meshes.

    Args:
        meshes: A List of reconstructed meshes or a MultiBlock.
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
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        point_size: Point size of any nodes in the dataset plotted.
        mesh_style: Visualization style of the mesh. One of the following: style='surface', style='wireframe', style='points'.
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

    blocks = collect_mesh(meshes) if isinstance(meshes, list) else meshes
    blocks_name = blocks.keys()

    # Create a plotting object to display the end mesh of blocks.
    end_block = blocks[blocks_name[-1]]
    p1 = create_plotter(
        jupyter=jupyter,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
    )
    _add2plotter(
        plotter=p1,
        mesh=end_block,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        mesh_style=mesh_style,
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
        mesh=start_block,
        key=key,
        background=background,
        ambient=ambient,
        opacity=opacity,
        point_size=point_size,
        mesh_style=mesh_style,
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
            mesh=start_block,
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
