"""
Regression function that is considerate of the spatial heterogeneity of (and thus the context-dependency of the
relationships of) the response variable.
"""
import copy
from multiprocessing import Pool
from typing import Optional, Union

import numpy as np
import numpy.linalg

from ...logging import logger_manager as lm
from ..find_neighbors import Kernel
from .regression_utils import compute_betas_local, iwls


# ---------------------------------------------------------------------------------------------------
# GWR
# ---------------------------------------------------------------------------------------------------
class STGWR:
    """Geographically weighted regression on spatial omics data with parallel processing.

    Args:
        n_processes: Number of processes to use for parallel processing. If None, use all available processes.
        coords: Array-like of shape [n_samples, 2]; list of coordinates (x, y) for each sample
        y: Array-like of shape [n_samples]; response variable
        X: Array-like of shape [n_samples, n_features]; independent variables
        bw: Used to provide previously obtained bandwidth for the spatial kernel. Consists of either a distance
            value or N for the number of nearest neighbors. Can be obtained using BW_Selector or some other
            user-defined method. Pass "np.inf" if all other points should have the same spatial weight.
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".
        fixed_bw: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        fit_intercept: Set True to include intercept in the model and False to exclude intercept
    """

    def __init__(
        self,
        n_processes: Optional[int],
        coords: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
        bw: Union[float, int],
        distr: str = "gaussian",
        kernel: str = "bisquare",
        fixed_bw: bool = False,
        fit_intercept: bool = True,
    ):
        self.logger = lm.get_main_logger()

        self.n_processes = n_processes
        self.coords = coords
        self.y = y
        self.X = X
        self.bw = bw
        self.distr = distr
        self.kernel = kernel
        self.fixed_bw = fixed_bw
        self.fit_intercept = fit_intercept
        # Model fitting parameters:
        self.init_params = {}

        self.n_samples = self.X.shape[0]

    def _get_wi(self, i: int, bw: Union[float, int]) -> np.ndarray:
        """Get spatial weights for each sample.

        Args:
            i: Index of sample for which weights are to be calculated to all other samples in the dataset
            bw: Bandwidth for the spatial kernel

        Returns:
            wi: Array of weights for sample of interest
        """

        if bw == np.inf:
            wi = np.ones(self.n_samples)
            return wi

        try:
            wi = Kernel(i, self.coords, bw, fixed=self.fixed_bw, function=self.kernel, subset_idxs=None).kernel
        except:
            self.logger.error(f"Error in getting weights for sample {i}")
        return wi

    def _local_fit(self, i: int):
        """Fit a local regression model for each sample.

        Args:
            i: Index of sample for which local regression model is to be fitted
        """
        wi = self._get_wi(i, self.bw).reshape(-1, 1)

        if self.distr == "gaussian":
            betas, influence_matrix = compute_betas_local(self.y, self.X, wi)
            y_hat = np.dot(self.X[i], betas)[0]
            residual = self.y[i] - y_hat

            # Effect of deleting sample i from the dataset on the estimated coefficients
            influence_i = np.dot(self.X[i], influence_matrix[:, i])
            w = 1

        elif self.distr == "poisson" or self.distr == "nb":
            # init_betas (initial coefficients) to be incorporated at runtime:
            betas, y_hat, n_iter, spatial_weights, linear_predictor, adjusted_predictor, influence_matrix = iwls(
                self.y, self.X, distr=self.distr, init_betas="filler"
            )


# MGWR:
