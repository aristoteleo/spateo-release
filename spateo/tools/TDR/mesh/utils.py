import warnings

import matplotlib as mpl
import numpy as np
import pyvista as pv

from pyvista import PolyData, UnstructuredGrid, MultiBlock
from typing import Optional, List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def mesh_type(
    mesh: Union[PolyData, UnstructuredGrid],
    mtype: Literal["polydata", "unstructuredgrid"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """Get a new representation of this mesh as a new type."""
    if mtype == "polydata":
        return mesh if isinstance(mesh, PolyData) else pv.PolyData(mesh.points, mesh.cells)
    elif mtype == "unstructured":
        return mesh.cast_to_unstructured_grid() if isinstance(mesh, PolyData) else mesh
    else:
        raise ValueError("\n`mtype` value is wrong." "\nAvailable `mtype` are: `'polydata'` and `'unstructuredgrid'`.")


def merge_mesh(
    meshes: List[PolyData or UnstructuredGrid],
) -> PolyData or UnstructuredGrid:
    """Merge all meshes in the `meshes` list. The format of all meshes must be the same."""

    merged_mesh = meshes[0]
    for mesh in meshes[1:]:
        merged_mesh.merge(mesh, inplace=True)

    return merged_mesh


def collect_mesh(
    meshes: List[PolyData or UnstructuredGrid],
    meshes_name: Optional[List[str]] = None,
) -> MultiBlock:
    """
    A composite class to hold many data sets which can be iterated over.
    You can think of MultiBlock like lists or dictionaries as we can iterate over this data structure by index
    and we can also access blocks by their string name.

    If the input is a dictionary, it can be iterated in the following ways:
        >>> blocks = collect_mesh(meshes, meshes_name)
        >>> for name in blocks.keys():
        ...     print(blocks[name])

    If the input is a list, it can be iterated in the following ways:
        >>> blocks = collect_mesh(meshes)
        >>> for block in blocks:
        ...    print(block)
    """

    if meshes_name is not None:
        meshes = {name: mesh for name, mesh in zip(meshes_name, meshes)}

    return pv.MultiBlock(meshes)


def multiblock2mesh(mesh, message=None):
    """Merge all meshes in MultiBlock into one mesh"""
    if message is not None:
        warnings.warn(
            f"\nMultiBlock does not support {message}. "
            f"\nHere, all meshes contained in MultiBlock will be automatically merged into one mesh before {message}."
        )
    meshes = [mesh[name] for name in mesh.keys()]
    return merge_mesh(meshes=meshes)


def add_mesh_labels(
    mesh: Union[PolyData, UnstructuredGrid],
    labels: np.ndarray,
    key_added: str = "groups",
    where: Literal["point_data", "cell_data"] = "cell_data",
    colormap: Union[str, list, dict, np.ndarray] = "rainbow",
    alphamap: Union[float, list, dict, np.ndarray] = 1.0,
    mask_color: Optional[str] = "gainsboro",
    mask_alpha: Optional[float] = 0.0,
    copy: bool = False,
) -> PolyData or UnstructuredGrid:
    """
    Add rgba color to each point of mesh based on labels.

    Args:
        mesh: A reconstructed mesh.
        labels: An array of labels of interest.
        key_added: The key under which to add the labels.
        where: The location where the label information is recorded in the mesh.
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Color to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
        copy: Whether to copy `pcd` or modify it inplace.
    Returns:
         A mesh, which contains the following properties:
            `mesh.cell_data[key_added]` or `mesh.point_data[key_added]`, the labels array;
            `mesh.cell_data[f'{key_added}_rgba']` or `mesh.point_data[f'{key_added}_rgba']`, the rgba colors of the labels.
    """

    mesh = mesh.copy() if copy else mesh

    new_labels = labels.copy().astype(object)
    raw_labels_hex = new_labels.copy()
    raw_labels_alpha = new_labels.copy()
    cu_arr = np.unique(new_labels)
    cu_arr = np.sort(cu_arr, axis=0)

    raw_labels_hex[raw_labels_hex == "mask"] = mpl.colors.to_hex(mask_color)
    raw_labels_alpha[raw_labels_alpha == "mask"] = mask_alpha

    # Set raw hex.
    if isinstance(colormap, str):
        if colormap in list(mpl.colormaps):
            lscmap = mpl.cm.get_cmap(colormap)
            raw_hex_list = [mpl.colors.to_hex(lscmap(i)) for i in np.linspace(0, 1, len(cu_arr))]
            for label, color in zip(cu_arr, raw_hex_list):
                raw_labels_hex[raw_labels_hex == label] = color
        else:
            raw_labels_hex[raw_labels_hex != "mask"] = mpl.colors.to_hex(colormap)
    elif isinstance(colormap, dict):
        for label, color in colormap.items():
            raw_labels_hex[raw_labels_hex == label] = mpl.colors.to_hex(color)
    elif isinstance(colormap, list) or isinstance(colormap, np.ndarray):
        raw_labels_hex = np.array([mpl.colors.to_hex(color) for color in colormap]).astype(object)
    else:
        raise ValueError("\n`colormap` value is wrong." "\nAvailable `colormap` types are: `str`, `list` and `dict`.")

    # Set raw alpha.
    if isinstance(alphamap, float):
        raw_labels_alpha[raw_labels_alpha != "mask"] = alphamap
    elif isinstance(alphamap, dict):
        for label, alpha in alphamap.items():
            raw_labels_alpha[raw_labels_alpha == label] = alpha
    elif isinstance(alphamap, list) or isinstance(alphamap, np.ndarray):
        raw_labels_alpha = np.asarray(alphamap).astype(object)
    else:
        raise ValueError("\n`alphamap` value is wrong." "\nAvailable `alphamap` types are: `float`, `list` and `dict`.")

    # Set rgba.
    labels_rgba = [mpl.colors.to_rgba(c, alpha=a) for c, a in zip(raw_labels_hex, raw_labels_alpha)]
    labels_rgba = np.array(labels_rgba).astype(np.float64)

    # Added labels and rgba of the labels
    if where == "point_data":
        mesh.point_data[key_added] = labels
        mesh.point_data[f"{key_added}_rgba"] = labels_rgba
    else:
        mesh.cell_data[key_added] = labels
        mesh.cell_data[f"{key_added}_rgba"] = labels_rgba

    return mesh if copy else None
