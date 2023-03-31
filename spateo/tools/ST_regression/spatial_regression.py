"""
Regression function that is considerate of the spatial heterogeneity of (and thus the context-dependency of the
relationships of) the response variable.
"""
import copy
from typing import Optional

import numpy as np
import numpy.linalg
from mpi4py import MPI


# ---------------------------------------------------------------------------------------------------
# GWR
# ---------------------------------------------------------------------------------------------------
class STGWR:
    """Geographically weighted regression on spatial omics data with parallel processing.

    Args:
        communicators: MPI communicators for parallel processing, initialized using mpi4py
        coords: Array-like of shape [n_samples, 2]; list of coordinates (x, y) for each sample
        y: Array-like of shape [n_samples]; response variable
        X: Array-like of shape [n_samples, n_loc_features]; independent variables
        bw: Bandwidth for the spatial kernel. Consists of either a distance value or N for the number of nearest
            neighbors. Can be obtained using BW_Selector or some other user-defined method.
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".
        fixed_bw: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        constant: Set True to include intercept in the model and False to exclude intercept
        n_jobs: Number of processes to use
    """


# MGWR:
