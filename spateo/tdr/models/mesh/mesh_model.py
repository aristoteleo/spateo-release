from typing import Optional, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista import PolyData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..utilities import add_model_labels, merge_models, scale_model
from .mesh_utils import (
    clean_mesh,
    fix_mesh,
    smooth_mesh,
    uniform_larger_pc,
    uniform_mesh,
)

###################################
# Construct cell-level mesh model #
###################################


def construct_cells(
    pc: PolyData,
    cell_size: np.ndarray,
    geometry: Literal["cube", "sphere", "ellipsoid"] = "cube",
    xyz_scale: tuple = (1, 1, 1),
    n_scale: tuple = (1, 1),
    factor: float = 0.5,
):
    """
    Reconstructing cells from point clouds.

    Args:
        pc: A point cloud object, including ``pc.point_data["obs_index"]``.
        geometry: The geometry of generating cells. Available ``geometry`` are:

                * geometry = ``'cube'``
                * geometry = ``'sphere'``
                * geometry = ``'ellipsoid'``
        cell_size: A numpy.ndarray object including the relative radius/length size of each cell.
        xyz_scale: The scale factor for the x-axis, y-axis and z-axis.
        n_scale: The ``squareness`` parameter in the x-y plane adn z axis. Only works if ``geometry = 'ellipsoid'``.
        factor: Scale factor applied to scaling array.

    Returns:
        ds_glyph: A cells mesh including `ds_glyph.point_data["cell_size"]`, `ds_glyph.point_data["cell_centroid"]` and
        the data contained in the pc.
    """
    if not (cell_size is None):
        pc.point_data["cell_size"] = cell_size.flatten()
    else:
        raise ValueError("`cell_size` value is wrong. \nPlease enter a value for `cell_size`")

    if geometry == "cube":
        geom = pv.Box(
            bounds=(
                -xyz_scale[0],
                xyz_scale[0],
                -xyz_scale[1],
                xyz_scale[1],
                -xyz_scale[2],
                xyz_scale[2],
            )
        )
        # geom = pv.Cube(x_length=xyz_scale[0], y_length=xyz_scale[1],  z_length=xyz_scale[2], clean=True)
    elif geometry == "sphere":
        geom = pv.Sphere(radius=xyz_scale[0])
    elif geometry == "ellipsoid":
        geom = pv.ParametricSuperEllipsoid(
            xradius=xyz_scale[0],
            yradius=xyz_scale[1],
            zradius=xyz_scale[2],
            n1=n_scale[0],
            n2=n_scale[1],
        )
    else:
        raise ValueError("`geometry` value is wrong. \nAvailable `geometry` are: `'cube'`, `'sphere'`, `'ellipsoid'`.")

    ds_glyph = pc.glyph(geom=geom, scale="cell_size", factor=factor)
    centroid_coords = {index: coords for index, coords in zip(pc.point_data["obs_index"], pc.points)}
    ds_glyph.point_data["cell_centroid"] = np.asarray([centroid_coords[i] for i in ds_glyph.point_data["obs_index"]])

    return ds_glyph


#####################################
# Construct tissue-level mesh model #
#####################################


def construct_surface(
    pc: PolyData,
    key_added: str = "groups",
    label: str = "surface",
    color: Optional[str] = "gainsboro",
    alpha: Union[float, int] = 1.0,
    uniform_pc: bool = False,
    uniform_pc_alpha: Union[float, int] = 0,
    cs_method: Literal["pyvista", "alpha_shape", "ball_pivoting", "poisson", "marching_cube"] = "marching_cube",
    cs_args: Optional[dict] = None,
    nsub: Optional[int] = 3,
    nclus: int = 20000,
    smooth: Optional[int] = 1000,
    scale_distance: Union[float, int, list, tuple] = None,
    scale_factor: Union[float, int, list, tuple] = None,
) -> Tuple[PolyData, PolyData]:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pc: A point cloud model.
        key_added: The key under which to add the labels.
        label: The label of reconstructed surface mesh model.
        color: Color to use for plotting mesh. The default ``color`` is ``'gainsboro'``.
        alpha: The opacity of the color to use for plotting mesh. The default ``alpha`` is ``0.8``.
        uniform_pc: Generates a uniform point cloud with a larger number of points.
        uniform_pc_alpha: Specify alpha (or distance) value to control output of this filter.
        cs_method: The methods of generating a surface mesh. Available ``cs_method`` are:

                * ``'pyvista'``: Generate a 3D tetrahedral mesh based on pyvista.
                * ``'alpha_shape'``: Computes a triangle mesh on the alpha shape algorithm.
                * ``'ball_pivoting'``: Computes a triangle mesh based on the Ball Pivoting algorithm.
                * ``'poisson'``: Computes a triangle mesh based on thee Screened Poisson Reconstruction.
                * ``'marching_cube'``: Computes a triangle mesh based on the marching cube algorithm.
        cs_args: Parameters for various surface reconstruction methods. Available ``cs_args`` are:
                * ``'pyvista'``: {'alpha': 0}
                * ``'alpha_shape'``: {'alpha': 2.0}
                * ``'ball_pivoting'``: {'radii': [1]}
                * ``'poisson'``: {'depth': 8, 'width'=0, 'scale'=1.1, 'linear_fit': False, 'density_threshold': 0.01}
                * ``'marching_cube'``: {'levelset': 0, 'mc_scale_factor': 1}
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.
        smooth: Number of iterations for Laplacian smoothing.
        scale_distance: The distance by which the model is scaled. If ``scale_distance`` is float, the model is scaled same
                        distance along the xyz axis; when the ``scale factor`` is list, the model is scaled along the xyz
                        axis at different distance. If ``scale_distance`` is None, there will be no scaling based on distance.
        scale_factor: The scale by which the model is scaled. If ``scale factor`` is float, the model is scaled along the
                      xyz axis at the same scale; when the ``scale factor`` is list, the model is scaled along the xyz
                      axis at different scales. If ``scale_factor`` is None, there will be no scaling based on scale factor.

    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            ``uniform_surf.cell_data[key_added]``, the ``label`` array;
            ``uniform_surf.cell_data[f'{key_added}_rgba']``, the rgba colors of the ``label`` array.
        inside_pc: A point cloud, which contains the following properties:
            ``inside_pc.point_data['obs_index']``, the obs_index of each coordinate in the original adata.
            ``inside_pc.point_data[key_added]``, the ``groupby`` information.
            ``inside_pc.point_data[f'{key_added}_rgba']``, the rgba colors of the ``groupby`` information.
    """

    # Generates a uniform point cloud with a larger number of points or not.
    cloud = uniform_larger_pc(pc=pc, alpha=uniform_pc_alpha, nsub=3, nclus=20000) if uniform_pc else pc.copy()

    # Reconstruct surface mesh.
    if cs_method == "pyvista":
        _cs_args = {"alpha": 0}
        if not (cs_args is None):
            _cs_args.update(cs_args)

        from .reconstruction_methods import pv_mesh

        surf = pv_mesh(pc=cloud, alpha=_cs_args["alpha"])

    elif cs_method == "alpha_shape":
        _cs_args = {"alpha": 2.0}
        if not (cs_args is None):
            _cs_args.update(cs_args)

        from .reconstruction_methods import alpha_shape_mesh

        surf = alpha_shape_mesh(pc=cloud, alpha=_cs_args["alpha"])

    elif cs_method == "ball_pivoting":
        _cs_args = {"radii": [1]}
        if not (cs_args is None):
            _cs_args.update(cs_args)

        from .reconstruction_methods import ball_pivoting_mesh

        surf = ball_pivoting_mesh(pc=cloud, radii=_cs_args["radii"])

    elif cs_method == "poisson":
        _cs_args = {
            "depth": 8,
            "width": 0,
            "scale": 1.1,
            "linear_fit": False,
            "density_threshold": None,
        }
        if not (cs_args is None):
            _cs_args.update(cs_args)

        from .reconstruction_methods import poisson_mesh

        surf = poisson_mesh(
            pc=cloud,
            depth=_cs_args["depth"],
            width=_cs_args["width"],
            scale=_cs_args["scale"],
            linear_fit=_cs_args["linear_fit"],
            density_threshold=_cs_args["density_threshold"],
        )
    elif cs_method == "marching_cube":
        _cs_args = {"levelset": 0, "mc_scale_factor": 1}
        if not (cs_args is None):
            _cs_args.update(cs_args)

        from .reconstruction_methods import marching_cube_mesh

        surf = marching_cube_mesh(pc=cloud, levelset=_cs_args["levelset"], mc_scale_factor=_cs_args["mc_scale_factor"])

    else:
        raise ValueError(
            "`cs_method` value is wrong."
            "\nAvailable `cs_method` are: `'pyvista'`, `'alpha_shape'`, `'ball_pivoting'`, `'poisson'`, `'marching_cube'`."
        )

    # Removes unused points and degenerate cells.
    csurf = clean_mesh(mesh=surf)

    uniform_surfs = []
    for sub_surf in csurf.split_bodies():
        # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
        sub_fix_surf = fix_mesh(mesh=sub_surf.extract_surface())

        # Get a uniformly meshed surface using voronoi clustering.
        sub_uniform_surf = uniform_mesh(mesh=sub_fix_surf, nsub=nsub, nclus=nclus)
        uniform_surfs.append(sub_uniform_surf)
    uniform_surf = merge_models(models=uniform_surfs)
    uniform_surf = uniform_surf.extract_surface().triangulate().clean()

    # Adjust point coordinates using Laplacian smoothing.
    if not (smooth is None):
        uniform_surf = smooth_mesh(mesh=uniform_surf, n_iter=smooth)

    # Scale the surface mesh.
    uniform_surf = scale_model(model=uniform_surf, distance=scale_distance, scale_factor=scale_factor)

    # Add labels and the colormap of the surface mesh.
    labels = np.asarray([label] * uniform_surf.n_cells, dtype=str)
    add_model_labels(
        model=uniform_surf,
        labels=labels,
        key_added=key_added,
        where="cell_data",
        colormap=color,
        alphamap=alpha,
        inplace=True,
    )

    # Clip the original pc using the reconstructed surface and reconstruct new point cloud.
    select_pc = pc.select_enclosed_points(surface=uniform_surf, check_surface=False)
    select_pc1 = select_pc.threshold(0.5, scalars="SelectedPoints").extract_surface()
    select_pc2 = select_pc.threshold(0.5, scalars="SelectedPoints", invert=True).extract_surface()
    inside_pc = select_pc1 if select_pc1.n_points > select_pc2.n_points else select_pc2

    return uniform_surf, inside_pc
