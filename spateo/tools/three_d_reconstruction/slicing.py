import warnings

from pyvista import PolyData, UnstructuredGrid
from typing import List, Optional, Sequence, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def three_d_slice(
    mesh: Union[PolyData, UnstructuredGrid],
    key: str = "groups",
    slice_method: Literal["axis", "orthogonal"] = "axis",
    slice_method_args: dict = None,
    interactive: bool = True,
) -> List[PolyData]:
    """
    Create many slices of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.
    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
        slice_method: The methods of slicing a mesh. Available `slice_method` are:
                * `'axis'`: Create many slices of the input dataset along a specified axis.
                * `'orthogonal'`: Create three orthogonal slices through the dataset on the three cartesian planes. This method is usually used interactively without entering a position which slices are taken.
        slice_method_args: Parameters for various slicing methods. Available `slice_method_args` are:
                * `'axis'` method: {"n_slices": 10, "axis": "x", center: None}
                        n_slices: The number of slices to create along a specified axis. Only works when `interactive` is False.
                        axis: The axis to generate the slices along.
                        center: A 3-length sequence specifying the position which slices are taken. Defaults to the center of the mesh.
                * `'orthogonal'`method: {center: None}
                        center: A 3-length sequence specifying the position which slices are taken. Defaults to the center of the mesh.
        interactive: Whether to slice interactively.
    Returns:
        A list of slices
    """

    if isinstance(mesh, UnstructuredGrid) is False:
        warnings.warn("The mesh should be a pyvista.UnstructuredGrid object.")
        mesh = mesh.cast_to_unstructured_grid()

    _slice_method_args = {"n_slices": 10, "axis": "x", "center": None}
    if slice_method_args is not None:
        _slice_method_args.update(slice_method_args)

    if interactive:
        p = pv.Plotter()
        if slice_method == "axis":
            p.add_mesh_slice(
                mesh,
                normal=_slice_method_args["axis"],
                scalars=f"{key}_rgba",
                rgba=True,
                tubing=True,
            )
        elif slice_method == "orthogonal":
            p.add_mesh_slice_orthogonal(mesh, scalars=f"{key}_rgba", rgba=True, tubing=True)
        else:
            raise ValueError(
                "\n`slice_method` value is wrong." "\nAvailable `slice_method` are: `'axis'`, `'orthogonal'`."
            )
        p.show()
        slices = p.plane_sliced_meshes
    else:
        mesh.set_active_scalars(f"{key}_rgba")
        if slice_method == "axis":
            slices_blocks = mesh.slice_along_axis(n=n_slices, axis=axis, center=_slice_method_args["center"])
        elif slice_method == "orthogonal":
            # Create three orthogonal slices through the dataset on the three cartesian planes.
            center = (None, None, None) if _slice_method_args["center"] is None else _slice_method_args["center"]
            slices_blocks = mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])
        else:
            raise ValueError(
                "\n`slice_method` value is wrong." "\nAvailable `slice_method` are: `'axis'`, `'orthogonal'`."
            )

        slices = [_slice for _slice in slices_blocks]

    return slices
