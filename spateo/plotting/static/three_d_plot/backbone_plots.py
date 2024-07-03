from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib as mpl
import numpy as np
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from .three_dims_plots import wrap_to_plotter
from .three_dims_plotter import (_set_jupyter, add_model, create_plotter,
                                 output_plotter)


def backbone(
    backbone_model: PolyData,
    backbone_key: str = "backbone",
    backbone_model_size: Union[float, int] = 8,
    backbone_colormap: Optional[str] = None,
    backbone_ambient: Union[float, list] = 0.2,
    backbone_opacity: Union[float, np.ndarray, list] = 1.0,
    nodes_key: Optional[str] = "nodes",
    nodes_label_size: Union[float, int] = 18,
    bg_model: Optional[Union[PolyData, UnstructuredGrid, MultiBlock]] = None,
    bg_key: Optional[Union[str, list]] = None,
    bg_model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    bg_model_size: Union[float, list] = 10,
    bg_colormap: Optional[Union[str, list]] = None,
    bg_ambient: Union[float, list] = 0.2,
    bg_opacity: Union[float, np.ndarray, list] = 0.6,
    show_axes: bool = True,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["none", "static", "trame"]] = False,
    off_screen: bool = False,
    window_size: tuple = (2048, 2048),
    background: str = "white",
    cpo: Union[str, list] = "iso",
    **kwargs,
):
    """
    Visualize constructed 3D backbone model.

    Args:
        backbone_model: The constructed backbone model.
        backbone_key: Any point_data names or cell_data names to be used for coloring ``backbone_model``.
        backbone_model_size: The thickness of backbone.
        backbone_colormap: Name of the Matplotlib colormap to use when mapping the scalars of ``backbone_model``.

                           When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        backbone_ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                          the actor when not directed at the light source emitted from the viewer.
        backbone_opacity: Opacity of the model.

                          If a single float value is given, it will be the global opacity of the model and uniformly applied
                          everywhere, elif a numpy.ndarray with single float values is given, it
                          will be the opacity of each point. - should be between 0 and 1.

                          A string can also be specified to map the scalars range to a predefined opacity transfer function
                          (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        nodes_key: The key that corresponds to the coordinates of the nodes in the backbone.
        nodes_label_size: Sets the size of the title font.
        bg_model: The background model used to construct backbone model.
        bg_key: Any point_data names or cell_data names to be used for coloring ``bg_model``.
        bg_model_style: Visualization style of ``bg_model``. One of the following:

                * ``bg_model_style = 'surface'``,
                * ``bg_model_style = 'wireframe'``,
                * ``bg_model_style = 'points'``.
        bg_model_size: If ``bg_model_style = 'points'``, point size of any nodes in the dataset plotted.

                       If ``bg_model_style = 'wireframe'``, thickness of lines.
        bg_colormap: Name of the Matplotlib colormap to use when mapping the scalars of ``bg_model``.

                     When the colormap is None, use {key}_rgba to map the scalars, otherwise use the colormap to map scalars.
        bg_ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                    the actor when not directed at the light source emitted from the viewer.
        bg_opacity: Opacity of the model.

                    If a single float value is given, it will be the global opacity of the model and uniformly applied
                    everywhere, elif a numpy.ndarray with single float values is given, it
                    will be the opacity of each point. - should be between 0 and 1.

                    A string can also be specified to map the scalars range to a predefined opacity transfer function
                    (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        show_axes: Whether to add a camera orientation widget to the active renderer.
        show_legend: whether to add a legend of ``bg_model`` to the plotter.
        legend_kwargs: A dictionary that will be pass to the ``add_legend`` function.
                       By default, it is an empty dictionary and the ``add_legend`` function will use the
                       ``{"legend_size": None, "legend_loc": None,  "legend_size": None, "legend_loc": None,
                       "title_font_size": None, "label_font_size": None, "font_family": "arial", "fmt": "%.2e",
                       "n_labels": 5, "vertical": True}`` as its parameters. Otherwise, you can provide a dictionary
                       that properly modify those keys according to your needs.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.

                * Output an image file,please enter a filename ending with
                  ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
                  When ``jupyter=False``, if you want to save '.png' file, please ensure ``off_screen=True``.
        jupyter: Whether to plot in jupyter notebook. Available ``jupyter`` are:

                * ``'none'`` - Do not display in the notebook.
                * ``'trame'`` - Show a trame widget
                * ``'static'`` - Display a static figure.
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
        **kwargs: Additional parameters that will be passed to ``.add_point_labels`` function.

    Returns:
        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
    """

    plotter_kws = dict(window_size=window_size, background=background, show_axes=show_axes)
    backbone_model_kwargs = dict(
        key=backbone_key,
        colormap=backbone_colormap,
        ambient=backbone_ambient,
        opacity=backbone_opacity,
        model_style="wireframe",
        model_size=backbone_model_size,
    )
    bg_model_kwargs = dict(
        background=background,
        key=bg_key,
        colormap=bg_colormap,
        ambient=bg_ambient,
        opacity=bg_opacity,
        model_style=bg_model_style,
        model_size=bg_model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
    )

    # Set jupyter.
    off_screen1, off_screen2, jupyter_backend = _set_jupyter(jupyter=jupyter, off_screen=off_screen)
    bg_rgb = mpl.colors.to_rgb(background)
    cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])

    # Create a plotting object to display pyvista/vtk model.
    p = create_plotter(off_screen=off_screen1, jupyter=jupyter, **plotter_kws)
    if not (bg_model is None):
        wrap_to_plotter(plotter=p, model=bg_model, cpo=cpo, **bg_model_kwargs)
    add_model(plotter=p, model=backbone_model, **backbone_model_kwargs)
    if not (nodes_key is None):
        p.add_point_labels(
            backbone_model,
            labels=nodes_key,
            font_size=nodes_label_size,
            font_family="arial",
            text_color=bg_rgb,
            shape_color=cbg_rgb,
            always_visible=True,
            **kwargs,
        )
    cpo = p.show(return_cpos=True, jupyter_backend="none", cpos=cpo)

    # Create another plotting object to save pyvista/vtk model.
    p = create_plotter(off_screen=off_screen2, jupyter=jupyter, **plotter_kws)
    if not (bg_model is None):
        wrap_to_plotter(plotter=p, model=bg_model, cpo=cpo, **bg_model_kwargs)
    add_model(plotter=p, model=backbone_model, **backbone_model_kwargs)
    p.camera_position = cpo
    if not (nodes_key is None):
        p.add_point_labels(
            backbone_model,
            labels=nodes_key,
            font_size=nodes_label_size,
            font_family="arial",
            text_color=bg_rgb,
            shape_color=cbg_rgb,
            always_visible=True,
            **kwargs,
        )

    # Output the plotting object.
    return output_plotter(
        plotter=p,
        filename=filename,
        jupyter=jupyter_backend,
    )
