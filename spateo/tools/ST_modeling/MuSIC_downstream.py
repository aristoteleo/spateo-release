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
import math
import os
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse
from mpi4py import MPI
from MuSIC import MuSIC
from sklearn.preprocessing import normalize

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm
from spateo.tools.ST_modeling.regression_utils import multitesting_correction, wald_test


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

        # Dictionary containing coefficients:
        self.coeffs, self.standard_errors = self.return_outputs()
        self.coeffs = self.comm.bcast(self.coeffs, root=0)
        self.standard_errors = self.comm.bcast(self.standard_errors, root=0)

        chunk_size = int(math.ceil(float(len(range(self.n_samples))) / self.comm.size))
        self.x_chunk = np.arange(self.n_samples)[self.comm.rank * chunk_size : (self.comm.rank + 1) * chunk_size]
        self.x_chunk = self.comm.bcast(self.x_chunk, root=0)

        # Save directory:
        parent_dir = os.path.dirname(self.output_path)
        if not os.path.exists(os.path.join(parent_dir, "significance")):
            os.makedirs(os.path.join(parent_dir, "significance"))

    def compute_coeff_significance(
        self,
        method: str = "fdr_bh",
        significance_threshold: float = 0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Computes local statistical significance for fitted coefficients.

        Args:
             method: Method to use for correction. Available methods can be found in the documentation for
                statsmodels.stats.multitest.multipletests(), and are also listed below (in correct case) for
                convenience:
            - Named methods:
                - bonferroni
                - sidak
                - holm-sidak
                - holm
                - simes-hochberg
                - hommel
            - Abbreviated methods:
                - fdr_bh: Benjamini-Hochberg correction
                - fdr_by: Benjamini-Yekutieli correction
                - fdr_tsbh: Two-stage Benjamini-Hochberg
                - fdr_tsbky: Two-stage Benjamini-Krieger-Yekutieli method
            significance_threshold: p-value (or q-value) needed to call a parameter significant.

        Returns:
            is_significant: Dataframe of identical shape to coeffs, where each element is True or False if it meets the
            threshold for significance
            pvalues: Dataframe of identical shape to coeffs, where each element is a p-value for that instance of that
                feature
            qvalues: Dataframe of identical shape to coeffs, where each element is a q-value for that instance of that
                feature
        """

        for target in self.coeffs.keys():
            # Get coefficients and standard errors for this key
            coef = self.coeffs[target]
            coef = self.comm.bcast(coef, root=0)
            se = self.standard_errors[target]
            se = self.comm.bcast(se, root=0)

            # Parallelize computations over observations and features:
            local_p_values_all = np.zeros((len(self.x_chunk), self.n_features))

            # Compute p-values for local observations and features
            for i, obs_index in enumerate(self.x_chunk):
                for j in range(self.n_features):
                    local_p_values_all[i, j] = wald_test(coef[obs_index, j], se[obs_index, j])

            # Collate p-values from all processes:
            p_values_all = self.comm.gather(local_p_values_all, root=0)

            if self.comm.rank == 0:
                p_values_all = np.concatenate(p_values_all, axis=0)
                p_values_df = pd.DataFrame(p_values_all, index=self.sample_names, columns=self.feature_names)
                # Multiple testing correction for each observation:
                qvals = np.zeros_like(p_values_all)
                for i in range(p_values_all.shape[0]):
                    qvals[i, :] = multitesting_correction(
                        p_values_all[i, :], method=method, alpha=significance_threshold
                    )
                q_values_df = pd.DataFrame(qvals, index=self.sample_names, columns=self.feature_names)

                # Significance:
                is_significant_df = q_values_df < significance_threshold

                # Save dataframes:
                p_values_df.to_csv(os.path.join(self.output_path, "significance", f"{target}_p_values.csv"))
                q_values_df.to_csv(os.path.join(self.output_path, "significance", f"{target}_q_values.csv"))
                is_significant_df.to_csv(os.path.join(self.output_path, "significance", f"{target}_is_significant.csv"))

    def inferred_effect_direction(self, bw: float, k: int = 10, targets: Optional[Union[str, List[str]]] = None):
        """For visualization purposes, used for models that consider ligand expression (:attr `mod_type` is 'ligand' or
        'lr'. Construct spatial vector fields to infer the directionality of observed effects (the "sources" of the
        downstream expression).

        Parts of this function are inspired by 'communication_direction' from COMMOT: https://github.com/zcang/COMMOT

        Args:
            bw: Bandwidth used for model fitting. Defines the search area around each observation to consider when
                computing effect directions.
            k: Top k receivers to consider when computing effect directions
            target: Optional string or list of strings to select targets from among the genes used to fit the model.
                If not given, will use all targets.
        """
        if not self.mod_type == "ligand" or self.mod_type == "lr":
            raise ValueError(
                "Direction of effect can only be inferred if ligand expression is used as part of the " "model."
            )

        # Get spatial weights given bandwidth value:
        spatial_weights = self._compute_all_wi(bw, exclude_self=True)
        # Columns consist of the spatial weights of each observation- convolve with expression of each ligand to
        # get proxy of ligand signal "sent", weight by the local coefficient value to get a proxy of the "signal
        # functionally received" in generating the downstream effect and store in .obsp.
        if targets is None:
            targets = self.coeffs.keys()
        elif isinstance(targets, str):
            targets = [targets]

        for target in targets:
            coeffs = self.coeffs[target]

            for j, col in enumerate(self.ligands_expr.columns):
                ligand_expr = self.ligands_expr[col].values.reshape(-1, 1)
                # Referred to as "sent potential"
                sent_potential = spatial_weights * ligand_expr
                coeff = coeffs.iloc[:, j].reshape(1, -1)
                # Weight each column by the coefficient magnitude and store as sparse array:
                sig_potential = scipy.sparse.csc_matrix(sent_potential * coeff)
                self.adata.obsp[f"spatial_effect_{col}_{target}"] = sig_potential

                sum_sent_potential = np.array(sig_potential.sum(axis=1)).reshape(-1)
                sum_received_potential = np.array(sig_potential.sum(axis=0)).reshape(-1)

                # Arrays to store vector field components:
                sending_vf = np.zeros_like(self.coords)
                receiving_vf = np.zeros_like(self.coords)

                # Vector field for sent signal:
                sig_potential_lil = sig_potential.tolil()
                # Find the most notable senders for each observation and create the vector field from these senders
                # to the observation in question:
                for i in range(self.n_samples):
                    # Check if the number of nonzero indices in the given row is less than or equal to k- if so,
                    # take only the nonzero rows:
                    if len(sig_potential_lil.rows[i]) <= k:
                        top_idx = np.array(sig_potential_lil.rows[i], dtype="int")
                        top_data = np.array(sig_potential_lil.data[i], dtype="float")
                    else:
                        all_indices = np.array(sig_potential_lil.rows[i], dtype="int")
                        all_data = np.array(sig_potential_lil.data[i], dtype="float")
                        # Sort by descending order of data:
                        sort_idx = np.argsort(-all_data)[:k]
                        top_idx = all_indices[sort_idx]
                        top_data = all_data[sort_idx]

                    # Compute the vector field components:
                    # Check if there are no nonzero senders, or only one nonzero sender:
                    if len(top_idx) == 0:
                        continue
                    elif len(top_idx) == 1:
                        v_i = -self.coords[top_idx[0], :] + self.coords[i, :]
                    else:
                        v_i = -self.coords[top_idx, :] + self.coords[i, :]
                        v_i = normalize(v_i, norm="l2")
                        v_i = np.sum(v_i * top_data.reshape(-1, 1), axis=0)
                    v_i = normalize(v_i.reshape(1, -1))
                    sending_vf[i, :] = v_i[0, :] * sum_sent_potential[i]

                # Vector field for received signal:
                sig_potential_lil = sig_potential.transpose().tolil()
                # Find the most notable receivers for each observation and create the vector field from the
                # observation to these receivers:
                for i in range(self.n_samples):
                    # Check if the number of nonzero indices in the given row is less than or equal to k- if so,
                    # take only the nonzero rows:
                    if len(sig_potential_lil.rows[i]) <= k:
                        top_idx = np.array(sig_potential_lil.rows[i], dtype="int")
                        top_data = np.array(sig_potential_lil.data[i], dtype="float")
                    else:
                        all_indices = np.array(sig_potential_lil.rows[i], dtype="int")
                        all_data = np.array(sig_potential_lil.data[i], dtype="float")
                        # Sort by descending order of data:
                        sort_idx = np.argsort(-all_data)[:k]
                        top_idx = all_indices[sort_idx]
                        top_data = all_data[sort_idx]

                    # Compute the vector field components:
                    # Check if there are no nonzero senders, or only one nonzero sender:
                    if len(top_idx) == 0:
                        continue
                    elif len(top_idx) == 1:
                        v_i = -self.coords[top_idx, :] + self.coords[i, :]
                    else:
                        v_i = -self.coords[top_idx, :] + self.coords[i, :]
                        v_i = normalize(v_i, norm="l2")
                        v_i = np.sum(v_i * top_data.reshape(-1, 1), axis=0)
                    v_i = normalize(v_i.reshape(1, -1))
                    receiving_vf[i, :] = v_i[0, :] * sum_received_potential[i]

                self.adata.obsm[f"spatial_effect_sender_vf_{col}_{target}"] = sending_vf
                self.adata.obsm[f"spatial_effect_receiver_vf_{col}_{target}"] = receiving_vf
