import matplotlib as mpl
import numpy as np
import open3d as o3d
import pandas as pd
import pyacvd
import pymeshfix as mf
import pyvista as pv
import PVGeo
import warnings

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista import PolyData, UnstructuredGrid, MultiBlock, DataSet
from typing import Optional, Tuple, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def read_vtk(filename: str):
    """
    Read vtk file.
    Args:
        filename: The string path to the file to read.
    Returns:
        Wrapped PyVista dataset.
    """
    mesh = pv.read(filename)
    del mesh.cell_data["orig_extract_id"]

    return mesh


def save_mesh(
    mesh: DataSet,
    filename: str,
    binary: bool = True,
    texture: Union[str, np.ndarray] = None,
):
    """
    Save the pvvista/vtk mesh to vtk files.
    Args:
        mesh: A reconstructed mesh.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
        binary: If True, write as binary. Otherwise, write as ASCII.
                Binary files write much faster than ASCII and have a smaller file size.
        texture: Write a single texture array to file when using a PLY file.
                 Texture array must be a 3 or 4 component array with the datatype np.uint8.
                 Array may be a cell array or a point array,
                 and may also be a string if the array already exists in the PolyData.
                 If a string is provided, the texture array will be saved to disk as that name.
                 If an array is provided, the texture array will be saved as 'RGBA'
    """

    if filename.endswith(".vtk"):
        mesh.save(filename=filename, binary=binary, texture=texture)
    else:
        raise ValueError(
            "\nFilename is wrong. This function is only available when saving vtk files."
            "\nPlease enter a filename ending with `.vtk`."
        )


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


def _MultiBlock(mesh, message=None):
    if message is not None:
        warnings.warn(
            f"\nMultiBlock does not support {message}. "
            f"\nHere, all meshes contained in MultiBlock will be automatically merged into one mesh before {message}."
        )
    meshes = [mesh[name] for name in mesh.keys()]
    return merge_mesh(meshes=meshes)
