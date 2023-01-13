from typing import Union

import numpy as np
import pyvista as pv
from pyvista import DataSet, MultiBlock

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def read_model(filename: str):
    """
    Read any file type supported by vtk or meshio.
    Args:
        filename: The string path to the file to read.
    Returns:
        Wrapped PyVista dataset.
    """
    model = pv.read(filename)

    return model


def save_model(
    model: Union[DataSet, MultiBlock],
    filename: str,
    binary: bool = True,
    texture: Union[str, np.ndarray] = None,
):
    """
    Save the pvvista/vtk model to vtk/vtm file.
    Args:
        model: A reconstructed model.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.

                  If model is a pyvista.MultiBlock object, please enter a filename ending with ``.vtm``;
                  else please enter a filename ending with ``.vtk``.
        binary: If True, write as binary. Otherwise, write as ASCII.
                Binary files write much faster than ASCII and have a smaller file size.
        texture: Write a single texture array to file when using a PLY file.

                 Texture array must be a 3 or 4 component array with the datatype np.uint8.
                 Array may be a cell array or a point array, and may also be a string if the array already exists in the PolyData.

                 If a string is provided, the texture array will be saved to disk as that name.
                 If an array is provided, the texture array will be saved as 'RGBA'
    """

    if filename.endswith(".vtk") or filename.endswith(".vtm"):
        model.save(filename=filename, binary=binary, texture=texture)
    else:
        raise ValueError(
            "\n`filename` is wrong."
            "\nFor pyvista.MultiBlock object please save as `.vtm` file; For other objects please save as `.vtk` file."
        )
