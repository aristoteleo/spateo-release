from pyvista import PolyData, UnstructuredGrid, MultiBlock
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import _interactive_plotter
from ..mesh.utils import collect_mesh, multiblock2mesh


def three_d_slice(
    mesh: Union[PolyData, UnstructuredGrid],
    method: Literal["axis", "orthogonal"] = "axis",
    n_slices: int = 10,
    axis: Literal["x", "y", "z"] = "x",
    center: Optional[tuple] = None,
):
    """
    Create many slices of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.
    Args:
        mesh: Reconstructed 3D mesh.
        method: The methods of slicing a mesh. Available `method` are:
                * `'axis'`: Create many slices of the input dataset along a specified axis.
                * `'orthogonal'`: Create three orthogonal slices through the dataset on the three cartesian planes.
                                  This method is usually used interactively without entering a position which slices are taken.
        n_slices: The number of slices to create along a specified axis. Only works when `method` is 'axis'.
        axis: The axis to generate the slices along. Only works when `method` is 'axis'.
        center: A 3-length sequence specifying the position which slices are taken. Defaults to the center of the mesh.
    Returns:
        A MultiBlock that contains all meshes you sliced.
    """
    # Check input mesh.
    mesh = multiblock2mesh(mesh=mesh, message="slicing") if isinstance(mesh, MultiBlock) else mesh.copy()

    if method == "axis":
        # Create many slices of the input dataset along a specified axis.
        return mesh.slice_along_axis(n=n_slices, axis=axis, center=center)
    else:
        # Create three orthogonal slices through the dataset on the three cartesian planes.
        center = (None, None, None) if center is None else center
        return mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])


def interactive_slice(
    mesh: Union[PolyData, UnstructuredGrid, MultiBlock],
    key: str = "groups",
    method: Literal["axis", "orthogonal"] = "axis",
    axis: Literal["x", "y", "z"] = "x",
) -> MultiBlock:
    """
    Create a slice of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.

    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
        method: The methods of slicing a mesh. Available `method` are:
                * `'axis'`: Create a slice of the input dataset along a specified axis.
                * `'orthogonal'`: Create three orthogonal slices through the dataset on the three cartesian planes.
        axis: The axis to generate the slices along. Only works when `method` is 'axis'.
    Returns:
        sliced_mesh: A MultiBlock that contains all meshes you sliced.
    """
    # Check input mesh.
    mesh = multiblock2mesh(mesh=mesh, message="slicing") if isinstance(mesh, MultiBlock) else mesh.copy()

    # Create an interactive window for using widgets.
    p = _interactive_plotter(message=True)

    # Slice a mesh using a slicing widget.
    if method == "axis":
        p.add_mesh_slice(
            mesh,
            normal=axis,
            scalars=f"{key}_rgba",
            rgba=True,
            tubing=True,
            widget_color="black",
        )
    else:
        p.add_mesh_slice_orthogonal(mesh, scalars=f"{key}_rgba", rgba=True, tubing=True, widget_color="black")
    p.show()

    return collect_mesh(p.plane_sliced_meshes)
