import math
from typing import Union, Optional, List

import pandas as pd
import pyvista as pv
import matplotlib as mpl
from anndata import AnnData


def set_mesh(
    adata: AnnData,
    cluster: str = "cluster",
    cluster_show: Union[str, list] = "all",
    gene_show: Union[str, list] = "all",
) -> Union[Tuple[pv.PolyData, pv.PolyData, pv.PolyData], Tuple[pv.PolyData, pv.PolyData]]:
    """Create mesh.

    Args:
        adata: an Annodata object.
        cluster: Column name in .obs DataFrame that stores clustering results.
        cluster_show: Clustering categories that need to be displayed.
        gene_show: Genes that need to be displayed.

    Returns
        mask_grid: Dataset consisting of undisplayed clustering vertices.
            (if cluster_show != "all", return the mask_grid.)
        other_grid: Dataset consisting of displayed clustering vertices.
        surf: Clipped surface.
    """

    points = adata.obs[["x", "y", "z"]].values
    grid = pv.PolyData(points)

    if cluster_show == "all":
        grid["cluster"] = adata.obs[cluster]
    elif isinstance(cluster_show, list) or isinstance(cluster_show, tuple):
        grid["cluster"] = adata.obs[cluster].map(lambda x: str(x) if x in cluster_show else "mask")
    else:
        grid["cluster"] = adata.obs[cluster].map(lambda x: x if x == cluster_show else "mask")

    if gene_show == "all":
        grid["gene"] = adata.X.sum(axis=1, keepdims=True)
    else:
        grid["gene"] = adata[:, gene_show].X.sum(axis=1, keepdims=True)

    surf = grid.delaunay_3d().extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clipped_grid = grid.clip_surface(surf)
    clipped_grid_data = pd.DataFrame(clipped_grid.points)
    clipped_grid_data["cluster"], clipped_grid_data["gene"] = (
        clipped_grid["cluster"],
        clipped_grid["gene"],
    )

    other_data = clipped_grid_data[clipped_grid_data["cluster"] != "mask"]
    other_grid = pv.PolyData(other_data[[0, 1, 2]].values)
    other_grid["cluster"] = other_data["cluster"]
    other_grid["gene"] = other_data["gene"]

    if cluster_show != "all":
        mask_data = clipped_grid_data[clipped_grid_data["cluster"] == "mask"]
        mask_grid = pv.PolyData(mask_data[[0, 1, 2]].values)
        return mask_grid, other_grid, surf
    else:
        return other_grid, surf


def recon_3d(
    adata: AnnData,
    cluster: str = "cluster",
    save: str = "3d.png",
    cluster_show: Union[str, list] = "all",
    gene_show: Union[str, list] = "all",
    show: str = "cluster",
    colormap: str = "RdYlBu_r",
    background_color: str = "black",
    other_color: str = "white",
    off_screen: bool = True,
    window_size: Optional[List[int]] = None,
    cpos: Optional[list] = None,
    bar_position: Optional[list] = None,
    bar_height: float = 0.3,
    viewup: Optional[list] = None,
    framerate: int = 15,
):
    """Draw a 3D image that integrates all the slices through pyvista,
    and you can output a png image file, or a gif image file, or an MP4 video file.

    Args:
        adata: an Annodata object.
        cluster: Column name in .obs DataFrame that stores clustering results.
        save: If a str, save the figure. Infer the file type if ending on
            {'.png','.jpeg', '.jpg', '.bmp', '.tif', '.tiff', '.gif', '.mp4'}.
        cluster_show: Clustering categories that need to be displayed.
        gene_show: Genes that need to be displayed.
        show: Display gene expression (`gene`) or clustering results (`cluster`).
        colormap: The name of a matplotlib colormap to use for categorical coloring.
        background_color: The background color of the active render window.
        other_color: The color of the font and border.
        off_screen: Renders off screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. Defaults to [1024, 768].
        cpos: List of Camera position. Available camera positions are: "xy", "xz", "yz", "yx", "zx", "zy", "iso".
            Defaults to ["xy", "xz", "yz", "iso"].
        bar_position: The percentage (0 to 1) along the windows’s horizontal direction and vertical direction to place the bottom
            left corner of the colorbar. Defaults to [0.9, 0.1].
        bar_height: The percentage (0 to 1) height of the window for the colorbar.
        viewup: The normal to the orbital plane. Defaults to [0.5, 0.5, 1].
        framerate: Frames per second.

    Examples:
        >>> adata
        AnnData object with n_obs × n_vars = 35145 × 16131
        obs: 'slice_ID', 'x', 'y', 'z', 'cluster'
        obsm: 'spatial'
        >>> recon_3d(adata=adata, cluster="cluster",cluster_show=["muscle", "testis"],gene_show=["128up", "14-3-3epsilon"],
        >>>          show='cluster', save="3d.png", viewup=[0, 0, 0], colormap="RdYlBu_r", bar_height=0.2)
    """

    if window_size is None:
        window_size = [1024, 768]
    if cpos is None:
        cpos = ["xy", "xz", "yz", "iso"]
    if viewup is None:
        viewup = [0.5, 0.5, 1]
    if bar_position is None:
        bar_position = [0.9, 0.1]
    if cluster_show != "all":
        mask_grid, other_grid, surf = set_mesh(
            adata=adata,
            cluster=cluster,
            cluster_show=cluster_show,
            gene_show=gene_show,
        )
    else:
        mask_grid = None
        other_grid, surf = set_mesh(
            adata=adata,
            cluster=cluster,
            cluster_show=cluster_show,
            gene_show=gene_show,
        )

    # Plotting object to display vtk meshes
    p = pv.Plotter(
        shape="3|1",
        off_screen=off_screen,
        border=True,
        border_color=other_color,
        lighting="light_kit",
        window_size=window_size,
    )
    p.background_color = background_color
    for i, _cpos in enumerate(cpos):
        p.subplot(i)

        # Add clipped surface
        p.add_mesh(
            surf,
            show_scalar_bar=False,
            show_edges=False,
            opacity=0.2,
            color="whitesmoke",
        )
        if mask_grid != None:
            # Add undisplayed clustering vertices
            p.add_mesh(
                mask_grid,
                opacity=0.02,
                color="whitesmoke",
                render_points_as_spheres=True,
                ambient=0.5,
            )
        # Add displayed clustering vertices
        p.add_mesh(
            other_grid,
            opacity=0.7,
            scalars=show,
            colormap=colormap,
            render_points_as_spheres=True,
            ambient=0.5,
        )

        p.show_axes()
        p.remove_scalar_bar()
        p.camera_position = _cpos

        if i == 3 and show == "cluster":
            p.add_scalar_bar(
                title=show,
                fmt="%.2f",
                n_labels=0,
                font_family="arial",
                color=other_color,
                vertical=True,
                use_opacity=True,
                position_x=bar_position[0],
                position_y=bar_position[1],
                height=bar_height,
            )
        elif i == 3 and show == "gene":
            p.add_scalar_bar(
                title=show,
                fmt="%.2f",
                font_family="arial",
                color=other_color,
                vertical=True,
                use_opacity=True,
                position_x=bar_position[0],
                position_y=bar_position[1],
                height=bar_height,
            )
        fontsize = math.ceil(window_size[0] / 100)
        p.add_text(
            f"\n " f" Camera position = '{_cpos}' \n " f" Cluster(s): {cluster_show} \n " f" Gene(s): {gene_show} ",
            position="upper_left",
            font="arial",
            font_size=fontsize,
            color=mpl.colors.to_hex(other_color),
        )

    # Save 3D reconstructed image or GIF or video
    if (
        save.endswith(".png")
        or save.endswith(".tif")
        or save.endswith(".tiff")
        or save.endswith(".bmp")
        or save.endswith(".jpeg")
        or save.endswith(".jpg")
    ):
        p.show(screenshot=save)
    else:
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=viewup, n_points=20)
        if save.endswith(".gif"):
            p.open_gif(save)
        elif save.endswith(".mp4"):
            p.open_movie(save, framerate=framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()
