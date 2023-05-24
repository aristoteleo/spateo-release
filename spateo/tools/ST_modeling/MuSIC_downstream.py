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

        self.search_bw = self.arg_retrieve.search_bw
        self.k = self.arg_retrieve.top_k_receivers

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
                    local_p_values_all[i, j] = wald_test(coef.iloc[obs_index, j], se.iloc[obs_index, j])

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
                parent_dir = os.path.dirname(self.output_path)
                p_values_df.to_csv(os.path.join(parent_dir, "significance", f"{target}_p_values.csv"))
                q_values_df.to_csv(os.path.join(parent_dir, "significance", f"{target}_q_values.csv"))
                is_significant_df.to_csv(os.path.join(parent_dir, "significance", f"{target}_is_significant.csv"))

    def inferred_effect_direction(self, targets: Optional[Union[str, List[str]]] = None):
        """For visualization purposes, used for models that consider ligand expression (:attr `mod_type` is 'ligand' or
        'lr'. Construct spatial vector fields to infer the directionality of observed effects (the "sources" of the
        downstream expression).

        Parts of this function are inspired by 'communication_direction' from COMMOT: https://github.com/zcang/COMMOT

        Args:
            bw: Bandwidth used for model fitting. Defines the search area around each observation to consider when
                computing effect directions. For models that include both secreted and membrane-bound signaling,
                it is recommended to set this higher than the optimal bandwidth found in fitting to be able to cover
                the larger search area in which secreted signals can have effects.
            k: Top k receivers to consider when computing effect directions
            target: Optional string or list of strings to select targets from among the genes used to fit the model.
                If not given, will use all targets.
        """
        if not self.mod_type == "ligand" or self.mod_type == "lr":
            raise ValueError(
                "Direction of effect can only be inferred if ligand expression is used as part of the " "model."
            )

        # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver:
        spatial_weights = self._compute_all_wi(self.search_bw, bw_fixed=self.bw_fixed, exclude_self=True).toarray()
        # Columns consist of the spatial weights of each observation- convolve with expression of each ligand to
        # get proxy of ligand signal "sent", weight by the local coefficient value to get a proxy of the "signal
        # functionally received" in generating the downstream effect and store in .obsp.
        if targets is None:
            targets = self.coeffs.keys()
        elif isinstance(targets, str):
            targets = [targets]

        for target in targets:
            coeffs = self.coeffs[target]
            target_expr = self.targets_expr[target].values.reshape(1, -1)
            target_indicator = np.where(target_expr != 0, 1, 0)

            for j, col in enumerate(self.ligands_expr.columns):
                # Use the non-lagged ligand expression array:
                ligand_expr = self.ligands_expr_nonlag[col].values.reshape(-1, 1)
                # Referred to as "sent potential"
                sent_potential = spatial_weights * ligand_expr
                coeff = coeffs.iloc[:, j].values.reshape(1, -1)
                # Weight each column by the coefficient magnitude and finally by the indicator for expression/no
                # expression of the target and store as sparse array:
                sig_potential = sent_potential * coeff * target_indicator
                self.adata.obsp[f"spatial_effect_{col}_{target}"] = sig_potential

                # Vector field for sent signal:
                top_senders_each_receiver = np.argsort(-sig_potential, axis=1)[:, : self.k]
                avg_v = np.zeros_like(self.coords)
                for ik in range(self.k):
                    tmp_v = (
                        self.coords[top_senders_each_receiver[:, ik]]
                        - self.coords[np.arange(self.n_samples, dtype=int)]
                    )
                    tmp_v = normalize(tmp_v, norm="l2")
                    avg_v = avg_v + tmp_v * sig_potential[
                        np.arange(self.n_samples, dtype=int), top_senders_each_receiver[:, ik]
                    ].reshape(-1, 1)
                avg_v = normalize(avg_v)
                # factor = normalize(np.sum(sig_potential, axis=1).reshape(-1, 1))
                sum_values = np.sum(sig_potential, axis=1).reshape(-1, 1)
                # Normalize to range [0, 1]:
                normalized_sum_values = (sum_values - np.min(sum_values)) / (np.max(sum_values) - np.min(sum_values))
                # factor = np.where(sum_values > 1, np.log10(sum_values), sum_values / 4)
                sending_vf = avg_v * normalized_sum_values
                sending_vf = np.clip(sending_vf, -0.05, 0.05)

                # Vector field for received signal:
                received_potential = sig_potential.T
                top_receivers_each_sender = np.argsort(-received_potential, axis=1)[:, : self.k]
                avg_v = np.zeros_like(self.coords)
                for ik in range(self.k):
                    tmp_v = (
                        -self.coords[top_receivers_each_sender[:, ik]]
                        + self.coords[np.arange(self.n_samples, dtype=int)]
                    )
                    tmp_v = normalize(tmp_v, norm="l2")
                    avg_v = avg_v + tmp_v * received_potential[
                        np.arange(self.n_samples, dtype=int), top_receivers_each_sender[:, ik]
                    ].reshape(-1, 1)
                avg_v = normalize(avg_v)
                # factor = normalize(np.sum(received_potential, axis=1).reshape(-1, 1))
                sum_values = np.sum(received_potential, axis=1).reshape(-1, 1)
                # Normalize to range [0, 1]:
                normalized_sum_values = (sum_values - np.min(sum_values)) / (np.max(sum_values) - np.min(sum_values))
                # factor = np.where(sum_values > 1, np.log10(sum_values), sum_values / 4)
                receiving_vf = avg_v * normalized_sum_values
                receiving_vf = np.clip(receiving_vf, -0.05, 0.05)

                del sig_potential, received_potential

                self.adata.obsm[f"spatial_effect_sender_vf_{col}_{target}"] = sending_vf
                self.adata.obsm[f"spatial_effect_receiver_vf_{col}_{target}"] = receiving_vf

        # Save AnnData object with effect direction information:
        adata_name = os.path.splitext(self.adata_path)[0]
        self.adata.write(f"{adata_name}_effect_directions.h5ad")
