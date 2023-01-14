from typing import Optional, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from pyvista import MultiBlock, PolyData, UnstructuredGrid

from ....tdr import add_model_labels, collect_models
from .three_dims_plots import three_d_multi_plot, three_d_plot

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _get_default_cmap():
    if "default_cmap" not in mpl.colormaps():
        colors = ["#4B0082", "#800080", "#F97306", "#FFA500", "#FFD700", "#FFFFCB"]
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        mpl.colormaps.register(LinearSegmentedColormap.from_list("default_cmap", list(zip(nodes, colors))))
    return "default_cmap"


def _check_index_in_adata(adata, model):
    adata_obs_index = pd.DataFrame(range(len(adata.obs.index)), index=adata.obs.index, columns=["ind"])
    obs_index = (
        np.asarray(model.point_data["obs_index"]) if "obs_index" in model.point_data else np.asarray(adata.obs.index)
    )
    obs_index_ind = adata_obs_index.loc[obs_index, "ind"].values
    return obs_index_ind


def _check_key_in_adata(adata: AnnData, key: str, where: str):
    if where == "obs":
        assert key in adata.obs_keys(), f"``{key}`` does not exist in adata.obs."
        return adata.obs[key]
    elif where == "obsm":
        assert key in adata.obsm_keys(), f"``{key}`` does not exist in adata.obsm."
        return adata.obsm[key]
    elif where == "obsp":
        assert key in adata.obsp, f"``{key}`` does not exist in adata.layers."
        return adata.obsp[key]
    elif where == "var":
        assert key in adata.var_keys(), f"``{key}`` does not exist in adata.var."
        return adata.var[key]
    elif where == "varm":
        assert key in adata.varm_keys(), f"``{key}`` does not exist in adata.varm."
        return adata.varm[key]
    elif where == "varp":
        assert key in adata.varp, f"``{key}`` does not exist in adata.varp."
        return adata.varp[key]
    elif where == "uns":
        assert key in adata.uns_keys(), f"``{key}`` does not exist in adata.uns."
        return adata.uns[key]
    elif where == "layers":
        assert key in adata.layers, f"``{key}`` does not exist in adata.layers."
        return adata.layers[key]
    else:
        raise ValueError("``where`` value is error.")


def jacobian(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    jacobian_key: str = "jacobian",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    shape: Union[str, list, tuple] = (3, 3),
    window_size: Optional[tuple] = (512 * 3, 512 * 3),
    background: str = "black",
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = None,
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Visualize the jacobian result.

    Args:
        adata: An anndata object contain jacobian matrix in ``.uns[jacobian_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        jacobian_key: The key in ``.uns`` that corresponds to the jacobian matrix in the anndata object.
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
        shape: Number of sub-render windows inside the main window. By default, there are nine render window.

               * Specify two across with ``shape``=(2, 1) and a two by two grid with ``shape``=(2, 2).
               * ``shape`` Can also accept a string descriptor as shape.

                    ``E.g.: shape="3|1" means 3 plots on the left and 1 on the right,``
                    ``E.g.: shape="4/2" means 4 plots on top and 2 at the bottom.``
        window_size: Window size in pixels. The default window_size is ``[512*3, 512*3]``.
        background: The background color of the window.
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
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
        **kwargs: Additional parameters that will be passed into the ``st.pl.three_d_multi_plot`` function.

    Returns:
        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

    Examples:

        Visualize only in one model:

        st.pl.jacobian(
            adata=stage_adata,
            model=stage_pc,
            jacobian_key="jacobian",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.jacobian(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            jacobian_key="jacobian",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """
    adata, model = adata.copy(), model.copy()
    jacobian_martix = _check_key_in_adata(adata=adata, key=jacobian_key, where="uns")

    # Add values in the jacobian matrix to the model separately.
    models = model if isinstance(model, (MultiBlock, list)) else collect_models([model])
    for m in models:
        obs_index_ind = _check_index_in_adata(adata=adata, model=m)
        m_jacobian_martix = jacobian_martix[:, :, obs_index_ind].copy()
        for f_i, f in enumerate(["fx", "fy", "fz"]):
            for i_i, i in enumerate(["x", "y", "z"]):
                add_model_labels(
                    model=m,
                    labels=m_jacobian_martix[f_i, i_i, :],
                    key_added=f"∂{f}/∂{i}",
                    where="point_data",
                    inplace=True,
                )

    # Visualization.
    j_keys = [f"∂{f}/∂{i}" for f in ["fx", "fy", "fz"] for i in ["x", "y", "z"]]
    colormap = _get_default_cmap() if colormap is None or colormap == "default_cmap" else colormap
    return three_d_multi_plot(
        model=collect_models([models]),
        key=j_keys,
        filename=filename,
        jupyter=jupyter,
        off_screen=off_screen,
        shape=shape,
        window_size=window_size,
        background=background,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=[model_style],
        model_size=[model_size],
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=[f"\njacobian: {i}" for i in j_keys] if text is True else text,
        text_kwargs=text_kwargs,
        **kwargs,
    )


def feature(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    feature_key: str,
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    off_screen: bool = False,
    window_size: Optional[tuple] = (512, 512),
    background: str = "black",
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    show_legend: bool = True,
    legend_kwargs: Optional[dict] = dict(title=""),
    text: Union[bool, str] = True,
    text_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Visualize the feature values.

    Args:
        adata: An anndata object contain feature values in ``.obs[feature_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        feature_key: The key in ``.obs`` that corresponds to the feature values in the anndata object.
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
        text: The text to add the rendering.
        text_kwargs: A dictionary that will be pass to the ``add_text`` function.

                     By default, it is an empty dictionary and the ``add_legend`` function will use the
                     ``{ "font_family": "arial", "font_size": 12, "font_color": "black", "text_loc": "upper_left"}``
                     as its parameters. Otherwise, you can provide a dictionary that properly modify those keys
                     according to your needs.
        **kwargs: Additional parameters that will be passed into the ``st.pl.three_d_plot`` function.

    Examples:

        Visualize only in one model:

        st.pl.feature(
            adata=stage_adata,
            model=stage_pc,
            feature_key="torsion",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.feature(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            feature_key="torsion",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """

    adata, model = adata.copy(), model.copy()
    feature_values = _check_key_in_adata(adata=adata, key=feature_key, where="obs")

    models = model if isinstance(model, (MultiBlock, list)) else [model]
    for m in models:
        obs_index_ind = _check_index_in_adata(adata=adata, model=m)
        m_feature_values = feature_values[obs_index_ind].copy()
        add_model_labels(model=m, labels=m_feature_values, key_added=feature_key, where="point_data", inplace=True)

    # Visualization.
    colormap = _get_default_cmap() if colormap is None or colormap == "default_cmap" else colormap
    return three_d_plot(
        model=models,
        key=feature_key,
        filename=filename,
        jupyter=jupyter,
        off_screen=off_screen,
        window_size=window_size,
        background=background,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        show_legend=show_legend,
        legend_kwargs=legend_kwargs,
        text=f"\nFeature: {feature_key}" if text is True else text,
        text_kwargs=text_kwargs,
        **kwargs,
    )


def torsion(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    torsion_key: str = "torsion",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    **kwargs,
):
    """
    Visualize the torsion result.

    Args:
        adata: An anndata object contain torsion values in ``.obs[torsion_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        torsion_key: The key in ``.obs`` that corresponds to the torsion values in the anndata object.
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
        **kwargs: Additional parameters that will be passed into the ``st.pl.feature`` function.

    Returns:

        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.
    Examples:

        Visualize only in one model:

        st.pl.torsion(
            adata=stage_adata,
            model=stage_pc,
            torsion_key="torsion",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.torsion(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            torsion_key="torsion",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """

    return feature(
        adata=adata,
        model=model,
        feature_key=torsion_key,
        filename=filename,
        jupyter=jupyter,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        **kwargs,
    )


def acceleration(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    acceleration_key: str = "acceleration",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    **kwargs,
):
    """
    Visualize the torsion result.

    Args:
        adata: An anndata object contain acceleration values in ``.obs[acceleration_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        acceleration_key: The key in ``.obs`` that corresponds to the acceleration values in the anndata object.
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
        **kwargs: Additional parameters that will be passed into the ``st.pl.feature`` function.

    Returns:

        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

    Examples:

        Visualize only in one model:

        st.pl.acceleration(
            adata=stage_adata,
            model=stage_pc,
            acceleration_key="acceleration",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.acceleration(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            acceleration_key="acceleration",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """

    return feature(
        adata=adata,
        model=model,
        feature_key=acceleration_key,
        filename=filename,
        jupyter=jupyter,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        **kwargs,
    )


def curvature(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    curvature_key: str = "curvature",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    **kwargs,
):
    """
    Visualize the curvature result.

    Args:
        adata: An anndata object contain curvature values in ``.obs[curvature_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        curvature_key: The key in ``.obs`` that corresponds to the curvature values in the anndata object.
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
        **kwargs: Additional parameters that will be passed into the ``st.pl.feature`` function.

    Returns:

        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

    Examples:

        Visualize only in one model:

        st.pl.curvature(
            adata=stage_adata,
            model=stage_pc,
            curvature_key="curvature",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.curvature(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            curvature_key="curvature",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """

    return feature(
        adata=adata,
        model=model,
        feature_key=curvature_key,
        filename=filename,
        jupyter=jupyter,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        **kwargs,
    )


def curl(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    curl_key: str = "curl",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    **kwargs,
):
    """
    Visualize the curl result.

    Args:
        adata: An anndata object contain curl values in ``.obs[curl_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        curl_key: The key in ``.obs`` that corresponds to the curl values in the anndata object.
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
        **kwargs: Additional parameters that will be passed into the ``st.pl.feature`` function.

    Returns:

        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

    Examples:

        Visualize only in one model:

        st.pl.curl(
            adata=stage_adata,
            model=stage_pc,
            curl_key="curl",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.curl(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            curl_key="curl",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """

    return feature(
        adata=adata,
        model=model,
        feature_key=curl_key,
        filename=filename,
        jupyter=jupyter,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        **kwargs,
    )


def divergence(
    adata: AnnData,
    model: Union[PolyData, UnstructuredGrid, MultiBlock, list],
    divergence_key: str = "divergence",
    filename: Optional[str] = None,
    jupyter: Union[bool, Literal["panel", "none", "pythreejs", "static", "ipygany"]] = False,
    colormap: Optional[Union[str, list]] = "default_cmap",
    ambient: Union[float, list] = 0.2,
    opacity: Union[float, np.ndarray, list] = 1.0,
    model_style: Union[Literal["points", "surface", "wireframe"], list] = "points",
    model_size: Union[float, list] = 3.0,
    **kwargs,
):
    """
    Visualize the divergence result.

    Args:
        adata: An anndata object contain curl values in ``.obs[divergence_key]``.
        model: A reconstructed model contains ``obs_index`` values.
        divergence_key: The key in ``.obs`` that corresponds to the divergence values in the anndata object.
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
        **kwargs: Additional parameters that will be passed into the ``st.pl.feature`` function.

    Returns:

        cpo: List of camera position, focal point, and view up.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

        img: Numpy array of the last image.
             Returned only if filename is None or filename ending with
             ``'.png', '.tif', '.tiff', '.bmp', '.jpeg', '.jpg', '.svg', '.eps', '.ps', '.pdf', '.tex'``.

    Examples:

        Visualize only in one model:

        st.pl.divergence(
            adata=stage_adata,
            model=stage_pc,
            divergence_key="divergence",
            jupyter="static",
            model_style="points",
            model_size=3
        )

        Visualize in multiple model:

        st.pl.divergence(
            adata=stage_adata,
            model=[stage_pc, trajectory_model],
            divergence_key="divergence",
            jupyter="static",
            model_style=["points", "wireframe"],
            model_size=[3, 1]
        )
    """

    return feature(
        adata=adata,
        model=model,
        feature_key=divergence_key,
        filename=filename,
        jupyter=jupyter,
        colormap=colormap,
        ambient=ambient,
        opacity=opacity,
        model_style=model_style,
        model_size=model_size,
        **kwargs,
    )
