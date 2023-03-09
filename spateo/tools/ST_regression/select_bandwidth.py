"""
Guided selection of spatial search radii, e.g. bandwidth for spatial kernels
"""
from typing import List, Tuple, Union

import numpy as np


class BW_Selector:
    """
    Select optimal kernel bandwidth.

    Math and concepts from: Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). Geographically weighted
    regression: the analysis of spatially varying relationships.

    Args:

    """

    def __init__(
        self,
        coords: List[Tuple[float, float]],
        y: np.ndarray,
        X_loc: np.ndarray,
        X_global: np.ndarray,
    ):
        "filler"
