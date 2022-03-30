from typing import List, Optional, Union

import numpy as np
import pyvista as pv
from pyvista import DataSet, MultiBlock, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def read_mesh(filename: str):
    """
    Read any file type supported by vtk or meshio.
    Args:
        filename: The string path to the file to read.
    Returns:
        Wrapped PyVista dataset.
    """
    mesh = pv.read(filename)

    if "vtkOriginalPointIds" in mesh.point_data.keys():
        del mesh.point_data["vtkOriginalPointIds"]
    if "vtkOriginalCellIds" in mesh.cell_data.keys():
        del mesh.cell_data["vtkOriginalCellIds"]

    return mesh


def save_mesh(
    mesh: Union[DataSet, MultiBlock],
    filename: str,
    binary: bool = True,
    texture: Union[str, np.ndarray] = None,
):
    """
    Save the pvvista/vtk mesh to vtk/vtm file.
    Args:
        mesh: A reconstructed mesh.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
                  If mesh is a pyvista.MultiBlock object, please enter a filename ending with `.vtm`;
                  else please enter a filename ending with `.vtk`.
        binary: If True, write as binary. Otherwise, write as ASCII.
                Binary files write much faster than ASCII and have a smaller file size.
        texture: Write a single texture array to file when using a PLY file.
                 Texture array must be a 3 or 4 component array with the datatype np.uint8.
                 Array may be a cell array or a point array,
                 and may also be a string if the array already exists in the PolyData.
                 If a string is provided, the texture array will be saved to disk as that name.
                 If an array is provided, the texture array will be saved as 'RGBA'
    """

    if filename.endswith(".vtk") or filename.endswith(".vtm"):
        mesh.save(filename=filename, binary=binary, texture=texture)
    else:
        raise ValueError(
            "\nFilename is wrong. This function is only available when saving vtk or vtm files."
            "\nIf mesh is a pyvista.MultiBlock object, please enter a filename ending with `.vtm`;"
            "else please enter a filename ending with `.vtk`."
        )
