from typing import Optional, Tuple, Union, List
from scipy.sparse import csr_matrix, spmatrix
import numpy as np


def rescaling(
    mat: Union[np.ndarray, spmatrix], new_shape: Union[List, Tuple]
) -> Union[np.ndarray, spmatrix]:
    """This function rescale the resolution of the input matrix that represents a spatial domain. For example, if you
    want to decrease the resolution of a matrix by a factor of 2, the new_shape will be `mat.shape / 2`.

    Args:
        mat: The input matrix of the spatial domain (or an image).
        new_shape: The rescaled shape of the spatial domain, each dimension must be an factorial of the original
                    dimension.

    Returns:
        res: the spatial resolution rescaled matrix.
    """
    shape = (new_shape[0], mat.shape[0] // mat[0], new_shape[1], mat.shape[1] // mat[1])

    res = mat.reshape(shape).sum(-1).sum(1)
    return res
