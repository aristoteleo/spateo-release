import pyvista as pv

def plot_3D(adata=None, cluster=None, colormap=None, window_size=(1024, 768), off_screen=True,
            background_color="black", font_color="white", font_size=12, cpos=("xy", "xz", "yz", "iso"),
            save=None, framerate=15, viewup=(0.5, 0.5, 1)):
    '''

    Draw a 3D image that integrates all the slices through pyvista, and you can output a png image file, or a gif image
    file, or an MP4 video file.

    Parameters
    ----------
        adata: 'anndata.AnnData'
            An Integrate all sliced AnnData object. adata.obsm['spatial'] includes x, y, z axis information, and adata.obs includes various
            clusters of information
        cluster: 'str'
            Cluster column name in adata.obs.
        colormap: 'list'
            A list of colors to override an existing colormap with a custom one.
            For example, to create a three color colormap you might specify ['green', 'red', 'blue'].
        window_size: 'tuple' (default: (1024, 768))
            Window size in pixels.
        off_screen: 'bool' (default: True)
            Whether to close the pop-up interactive window.
        background_color: 'str' (default: "black")
            The background color of the window.
        font_color: 'str' (default: "white")
            Set the font color.
        font_size: 'int' (default: 12)
            Set the font size.
        cpos: 'tuple' (default: ("xy", "xz", "yz", "iso"))
            Tuple of camera position. You can choose 4 perspectives from the following seven perspectives for drawing,
            and the last one is the main perspective. ("xy", "xz", "yz", "yx", "zx", "zy", "iso")
        save: 'str'
            Output file name. Filename should end in png, gif or mp4.
        framerate: 'int' (default: 15)
            Frames per second.The larger the framerate, the faster the rotation speed. (Used when the output file is MP4)
        viewup: 'tuple' (default: (0.5, 0.5, 1))
            In the process of generating the track path around the data scene, viewup is the normal of the track plane.

    '''

    if colormap is None:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "autocmap",["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6",
                        "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C", "#F9BD3F", "#DAB370",
                        "#877F6C", "#268785"]
        )
        mpl.cm.register_cmap(cmap=cmap)
        colormap = sns.color_palette(palette="autocmap", n_colors=len(adata.obs[cluster].unique()), as_cmap=False)
        colormap = [mpl.colors.to_hex(i, keep_alpha=False) for i in colormap]

    points = adata.obsm["spatial"].values
    grid = pv.PolyData(points)
    grid["cluster"] = adata.obs[cluster]
    volume = grid.delaunay_3d()
    surf = volume.extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clipped = grid.clip_surface(surf)
    p = pv.Plotter(shape="3|1", off_screen=off_screen, border=True,border_color=font_color,
                   lighting="light_kit", window_size=list(window_size))
    p.background_color = background_color
    for i, _cpos in enumerate(cpos):
        p.subplot(i)
        p.add_mesh(surf, show_scalar_bar=False, show_edges=False, opacity=0.8, color="gray")
        p.add_mesh(clipped, opacity=0.8, scalars="cluster", colormap=plot_color)
        p.remove_scalar_bar()
        p.camera_position = _cpos
        p.add_text(f" camera_position = '{_cpos}' ", position="upper_left",
                   font_size=font_size, color=font_color, font="arial")
        if i == 3:
            p.add_scalar_bar(title="cluster", title_font_size=font_size+10, label_font_size=font_size, color=font_color,
                             font_family="arial", vertical=True, fmt="%Id", n_labels=0, use_opacity=True)
    if save.endswith("png"):
        p.show(screenshot=save)
    else:
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=viewup, n_points=20)
        if save.endswith("gif"):
            p.open_gif(save)
        elif save.endswith("mp4"):
            p.open_movie(save, framerate=framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()
