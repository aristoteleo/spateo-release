"""
Additional functionalities to characterize signaling patterns from spatial transcriptomics

These include:
    - prediction of the effects of spatial perturbation on gene expression- this can include the effect of perturbing
    known regulators of ligand/receptor expression or the effect of perturbing the ligand/receptor itself.
    - following spatially-aware regression (or a sequence of spatially-aware regressions), combine regression results
    with data such that each cell can be associated with region-specific coefficient(s).
    - following spatially-aware regression (or a sequence of spatially-aware regressions), overlay the directionality
    of the predicted influence of the ligand on downstream expression.
"""
import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from MuSIC import MuSIC

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm


# ---------------------------------------------------------------------------------------------------
# Statistical testing, correlated differential expression analysis
# ---------------------------------------------------------------------------------------------------
class MuSIC_Interpreter(MuSIC):
    """
    Interpretation and downstream analysis of spatially weighted regression models.

    Args:
        comm: MPI communicator object initialized with mpi4py, to control parallel processing operations
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.
    """

    def __init__(self, comm: MPI.Comm, parser: argparse.ArgumentParser):
        super().__init__(comm, parser)

        # Coefficients:
        if not self.set_up:
            self.logger.info("Model has not yet been set up to run, running :func `SWR._set_up_model()` now...")
            self._set_up_model()

        self.coeffs = self.return_outputs()
        self.coeffs = self.comm.bcast(self.coeffs, root=0)

    def compute_coeff_significance(
        self,
        significance_threshold: float = 0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Computes local statistical significance for fitted coefficients.

        Args:
            significance_threshold: p-value (or q-value) needed to call a parameter significant.

        Returns:
            is_significant: Dataframe of identical shape to coeffs, where each element is True or False if it meets the
            threshold for significance
            pvalues: Dataframe of identical shape to coeffs, where each element is a p-value for that instance of that
                feature
            qvalues: Dataframe of identical shape to coeffs, where each element is a q-value for that instance of that
                feature
        """
        y_arr = self.targets_expr if hasattr(self, "targets_expr") else self.target
        y_arr = self.comm.bcast(y_arr, root=0)

        try:
            parent_dir = os.path.dirname(self.output_path)
            pred_path = os.path.join(parent_dir, "predictions.csv")
            y_pred = pd.read_csv(pred_path, index_col=0)
        except:
            y_pred = self.predict(self.X, self.coeffs)

        for target in y_arr.columns:
            # Check if predictions exist, and if not perform prediction:
            y = y_arr[target].values.reshape(-1, 1)
            resids = y - y_pred[target].values.reshape(-1, 1)
            sigma_sq = np.sum(resids**2) / (self.n_samples - self.n_features - 1)

    def chunk_compute_coeff_significance(
        self,
    ):
        """Perform statistical inference by chunks to reduce memory footprint.

        Returns:

        """
