from typing import Optional, Union

from pyvista import MultiBlock, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import _MultiBlock


def three_d_slice(
    mesh: Union[PolyData, UnstructuredGrid],
    key: str = "groups",
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
        key: The key under which are the labels.
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
    mesh = _MultiBlock(mesh=mesh, message="slicing") if isinstance(mesh, MultiBlock) else mesh

    if method == "axis":
        # Create many slices of the input dataset along a specified axis.
        return mesh.slice_along_axis(n=n_slices, axis=axis, center=center)
    else:
        # Create three orthogonal slices through the dataset on the three cartesian planes.
        center = (None, None, None) if center is None else center
        return mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])
