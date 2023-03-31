"""
Regression function that is considerate of the spatial heterogeneity of (and thus the context-dependency of the
relationships of) the response variable.
"""
import copy
from typing import Optional

import numpy as np
import numpy.linalg

# ---------------------------------------------------------------------------------------------------
# GWR
# ---------------------------------------------------------------------------------------------------


# MGWR:
