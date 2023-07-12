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
import itertools
import math
import os
import pickle
import re
import sys
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Literal, Optional, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
import xarray as xr
from mpi4py import MPI
from scipy.stats import pearsonr, spearmanr, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from statsmodels.gam.smooth_basis import BSplines

from ...logging import logger_manager as lm
from ..dimensionality_reduction import find_optimal_pca_components, pca_fit
from .MuSIC import MuSIC
from .regression_utils import multitesting_correction, permutation_testing, wald_test


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
        args_list: If parser is provided by function call, the arguments to parse must be provided as a separate
            list. It is recommended to use the return from :func `define_spateo_argparse()` for this.
    """

    def __init__(self, comm: MPI.Comm, parser: argparse.ArgumentParser, args_list: Optional[List[str]] = None):
        super().__init__(comm, parser, args_list, verbose=False)

        self.search_bw = self.arg_retrieve.search_bw
        if self.search_bw is None:
            self.search_bw = self.n_neighbors
            self.bw_fixed = False
        self.k = self.arg_retrieve.top_k_receivers

        # Coefficients:
        if not self.set_up:
            self.logger.info(
                "Running :func `SWR._set_up_model()` to organize predictors and targets for downstream "
                "analysis now..."
            )
            self._set_up_model()
            self.logger.info("Finished preprocessing, getting fitted coefficients and standard errors.")

        # Dictionary containing coefficients:
        self.coeffs, self.standard_errors = self.return_outputs()
        self.coeffs = self.comm.bcast(self.coeffs, root=0)
        self.standard_errors = self.comm.bcast(self.standard_errors, root=0)

        self.predictions = self.predict(coeffs=self.coeffs)
        self.predictions = self.comm.bcast(self.predictions, root=0)

        chunk_size = int(math.ceil(float(len(range(self.n_samples))) / self.comm.size))
        self.x_chunk = np.arange(self.n_samples)[self.comm.rank * chunk_size : (self.comm.rank + 1) * chunk_size]
        self.x_chunk = self.comm.bcast(self.x_chunk, root=0)

        # Save directory:
        parent_dir = os.path.dirname(self.output_path)
        if not os.path.exists(os.path.join(parent_dir, "significance")):
            os.makedirs(os.path.join(parent_dir, "significance"))

        # Arguments for cell type coupling computation:
        self.filter_targets = self.arg_retrieve.filter_targets
        self.filter_target_threshold = self.arg_retrieve.filter_target_threshold

        # Get targets for the downstream ligand(s), receptor(s), target(s), etc. to use for analysis:
        self.ligand_for_downstream = self.arg_retrieve.ligand_for_downstream
        self.receptor_for_downstream = self.arg_retrieve.receptor_for_downstream
        self.pathway_for_downstream = self.arg_retrieve.pathway_for_downstream
        self.target_for_downstream = self.arg_retrieve.target_for_downstream
        self.sender_ct_for_downstream = self.arg_retrieve.sender_ct_for_downstream
        self.receiver_ct_for_downstream = self.arg_retrieve.receiver_ct_for_downstream

        # Other downstream analysis-pertinent argparse arguments:
        self.cci_degs_model_interactions = self.arg_retrieve.cci_degs_model_interactions
        self.no_cell_type_markers = self.arg_retrieve.no_cell_type_markers
        self.compute_pathway_effect = self.arg_retrieve.compute_pathway_effect
        self.diff_sending_or_receiving = self.arg_retrieve.diff_sending_or_receiving

    def compute_coeff_significance(self, method: str = "fdr_bh", significance_threshold: float = 0.05):
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

    def compute_diagnostics(self):
        """
        For true and predicted gene expression, compute and generate plots of various diagnostics, including the
        Pearson correlation, Spearman correlation and root mean-squared-error (RMSE).
        """
        # Plot title:
        file_name = os.path.splitext(os.path.basename(self.adata_path))[0]

        parent_dir = os.path.dirname(self.output_path)
        pred_path = os.path.join(parent_dir, "predictions.csv")

        predictions = pd.read_csv(pred_path, index_col=0)
        all_genes = predictions.columns
        width = 0.5 * len(all_genes)
        pred_vals = predictions.values

        # Note that the assumption is the same processing arguments (normalize, smooth, etc.) are given for
        # downstream as would be given for model fitting itself, so these are not repeated here since they are
        # handled at initialization.
        def compute_rmse(predictions, targets):
            # Calculate the square of differences between predictions and targets
            squared_diff = (predictions - targets) ** 2

            # Calculate the mean squared difference
            mean_squared_diff = np.mean(squared_diff)

            # Calculate the square root of the mean squared difference (RMSE)
            rmse = np.sqrt(mean_squared_diff)
            return rmse

        pearson_dict = {}
        spearman_dict = {}
        rmse_dict = {}

        for i, gene in enumerate(all_genes):
            y = self.adata[:, gene].X.toarray().reshape(-1)
            music_results_target = pred_vals[:, i]

            # For cases where the values are relatively large (as is often the case in normalized gene expression
            # data), the Pearson correlation can be disproportionately influenced by larger values. Therefore,
            # we take the log transform:
            y = np.log1p(y)
            music_results_target = np.log1p(music_results_target)

            # Remove indices of large predicted values that are observed zero (these are POTENTIALLY zero due to
            # dropout and not biology):
            z_scores = zscore(music_results_target)
            outlier_indices = np.where((y == 0) & (z_scores > 2))[0]
            music_results_target_to_plot = np.delete(music_results_target, outlier_indices)
            y_plot = np.delete(y, outlier_indices)

            rp, _ = pearsonr(y_plot, music_results_target_to_plot)
            r, _ = spearmanr(y_plot, music_results_target_to_plot)
            rmse = compute_rmse(music_results_target_to_plot, y_plot)

            pearson_dict[gene] = rp
            spearman_dict[gene] = r
            rmse_dict[gene] = rmse

        # Mean of diagnostic metrics:
        mean_pearson = sum(pearson_dict.values()) / len(pearson_dict.values())
        mean_spearman = sum(spearman_dict.values()) / len(spearman_dict.values())
        mean_rmse = sum(rmse_dict.values()) / len(rmse_dict.values())

        data = []
        for gene in pearson_dict.keys():
            data.append(
                {
                    "Gene": gene,
                    "Pearson coefficient": pearson_dict[gene],
                    "Spearman coefficient": spearman_dict[gene],
                    "RMSE": rmse_dict[gene],
                }
            )
        # Color palette:
        colors = {"Pearson coefficient": "#FF7F00", "Spearmann coefficient": "#87CEEB", "RMSE": "#0BDA51"}
        df = pd.DataFrame(data)

        # Plot Pearson correlation barplot:
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.figure(figsize=(width, 6))
        plt.xticks(rotation="vertical")
        ax = sns.barplot(
            data=df,
            x="Gene",
            y="Pearson coefficient",
            hue="Model",
            palette=colors["Pearson coefficient"],
            edgecolor="black",
            dodge=True,
        )

        # Mean line:
        line_style = "--"  # Specify the line style (e.g., "--" for dotted)
        line_thickness = 2  # Specify the line thickness
        ax.axhline(mean_pearson, color="black", linestyle=line_style, linewidth=line_thickness)

        # Update legend:
        legend_label = f"Mean: {mean_pearson}"
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="black", linewidth=line_thickness, linestyle=line_style))
        labels.append(legend_label)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.title(f"Pearson correlation {file_name}")
        plt.tight_layout()
        plt.show()

        # Plot Spearman correlation barplot:
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.figure(figsize=(width, 6))
        plt.xticks(rotation="vertical")
        ax = sns.barplot(
            data=df,
            x="Gene",
            y="Spearman coefficient",
            hue="Model",
            palette=colors["Spearman coefficient"],
            edgecolor="black",
            dodge=True,
        )

        # Mean line:
        line_style = "--"  # Specify the line style (e.g., "--" for dotted)
        line_thickness = 2  # Specify the line thickness
        ax.axhline(mean_spearman, color="black", linestyle=line_style, linewidth=line_thickness)

        # Update legend:
        legend_label = f"Mean: {mean_spearman}"
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="black", linewidth=line_thickness, linestyle=line_style))
        labels.append(legend_label)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.title(f"Spearman correlation {file_name}")
        plt.tight_layout()
        plt.show()

        # Plot RMSE barplot:
        sns.set(font_scale=2)
        sns.set_style("white")
        plt.figure(figsize=(width, 6))
        plt.xticks(rotation="vertical")
        ax = sns.barplot(
            data=df, x="Gene", y="RMSE", hue="Model", palette=colors["RMSE"], edgecolor="black", dodge=True
        )

        # Mean line:
        line_style = "--"  # Specify the line style (e.g., "--" for dotted)
        line_thickness = 2  # Specify the line thickness
        ax.axhline(mean_rmse, color="black", linestyle=line_style, linewidth=line_thickness)

        # Update legend:
        legend_label = f"Mean: {mean_rmse}"
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="black", linewidth=line_thickness, linestyle=line_style))
        labels.append(legend_label)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.title(f"RMSE {file_name}")
        plt.tight_layout()
        plt.show()

    def identify_enriched_interactions(
        self,
        target: Optional[str] = None,
    ):
        """Given the target gene of interest, identify interaction features that are differentially expressed where
        the target gene is relatively higher- or lower-expressed."""

    def get_effect_potential(
        self,
        target: Optional[str] = None,
        ligand: Optional[str] = None,
        receptor: Optional[str] = None,
        sender_cell_type: Optional[str] = None,
        receiver_cell_type: Optional[str] = None,
        spatial_weights: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        store_summed_potential: bool = True,
    ) -> Tuple[scipy.sparse.spmatrix, np.ndarray, np.ndarray]:
        """For each cell, computes the 'signaling effect potential', interpreted as a quantification of the strength of
        effect of intercellular communication on downstream expression in a given cell mediated by any given other cell
        with any combination of ligands and/or cognate receptors, as inferred from the model results. Computations are
        similar to those of :func ~`.inferred_effect_direction`, but stops short of computing vector fields.

        Args:
            target: Optional string to select target from among the genes used to fit the model to compute signaling
                effects for. Note that this function takes only one target at a time. If not given, will take the
                first name from among all targets.
            ligand: Needed if :attr `mod_type` is 'ligand'; select ligand from among the ligands used to fit the
                model to compute signaling potential.
            receptor: Needed if :attr `mod_type` is 'lr'; together with 'ligand', used to select ligand-receptor pair
                from among the ligand-receptor pairs used to fit the model to compute signaling potential.
            sender_cell_type: Can optionally be used to select cell type from among the cell types used to fit the model
                to compute sent potential. Must be given if :attr `mod_type` is 'niche'.
            receiver_cell_type: Can optionally be used to condition sent potential on receiver cell type.
            spatial_weights: Optional pairwise spatial weights matrix. If not given, will compute at runtime.
            store_summed_potential: If True, will store both sent and received signaling potential as entries in
                .obs of the AnnData object.

        Returns:
            effect_potential: Sparse array of shape [n_samples, n_samples]; proxy for the "signaling effect potential"
                with respect to a particular target gene between each sender-receiver pair of cells.
            normalized_effect_potential_sum_sender: Array of shape [n_samples,]; for each sending cell, the sum of the
                signaling potential to all receiver cells for a given target gene, normalized between 0 and 1.
            normalized_effect_potential_sum_receiver: Array of shape [n_samples,]; for each receiving cell, the sum of
                the signaling potential from all sender cells for a given target gene, normalized between 0 and 1.
        """

        if self.mod_type == "receptor":
            raise ValueError("Sent potential is not defined for receptor models.")

        if target is None:
            if self.target_for_downstream is not None:
                target = self.target_for_downstream
            else:
                self.logger.info(
                    "Target gene not provided for :func `get_effect_potential`. Using first target " "listed."
                )
                target = list(self.coeffs.keys())[0]

        # Check for valid inputs:
        if ligand is None:
            if self.ligand_for_downstream is not None:
                ligand = self.ligand_for_downstream
            else:
                if self.mod_type == "ligand" or self.mod_type == "lr":
                    raise ValueError("Must provide ligand for ligand models.")

        if receptor is None:
            if self.receptor_for_downstream is not None:
                receptor = self.receptor_for_downstream
            else:
                if self.mod_type == "lr":
                    raise ValueError("Must provide receptor for lr models.")

        if sender_cell_type is None:
            if self.sender_ct_for_downstream is not None:
                sender_cell_type = self.sender_ct_for_downstream
            else:
                if self.mod_type == "niche":
                    raise ValueError("Must provide sender cell type for niche models.")

        if receiver_cell_type is None and self.receiver_ct_for_downstream is not None:
            receiver_cell_type = self.receiver_ct_for_downstream

        if spatial_weights is None:
            # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver.
            # Note: as the default (if bw is not otherwise provided), the n nearest neighbors will be used for the
            # bandwidth:
            spatial_weights = self._compute_all_wi(self.search_bw, bw_fixed=self.bw_fixed, exclude_self=True)

        # Testing: compare both ways:
        coeffs = self.coeffs[target]
        # Set negligible coefficients to zero:
        coeffs[coeffs.abs() < 1e-2] = 0

        # Target indicator array:
        target_expr = self.targets_expr[target].values.reshape(1, -1)
        target_indicator = np.where(target_expr != 0, 1, 0)

        # For ligand models, "signaling potential" can only use the ligand information. For lr models, it can further
        # conditioned on the receptor expression:
        if self.mod_type == "ligand" or self.mod_type == "lr":
            if self.mod_type == "ligand" and ligand is None:
                raise ValueError("Must provide ligand name for ligand model.")
            elif self.mod_type == "lr" and (ligand is None or receptor is None):
                raise ValueError("Must provide both ligand name and receptor name for lr model.")

            if self.mod_type == "lr":
                lr_pair = (ligand, receptor)
                if lr_pair not in self.lr_pairs:
                    raise ValueError(
                        "Invalid ligand-receptor pair given. Check that input to 'lr_pair' is given in "
                        "the form of a tuple."
                    )

            # Use the non-lagged ligand expression to construct ligand indicator array:
            ligand_expr = self.ligands_expr_nonlag[ligand].values.reshape(-1, 1)
            # Referred to as "sent potential"
            sent_potential = spatial_weights.multiply(ligand_expr)
            sent_potential.eliminate_zeros()

            # If "lr", incorporate the receptor expression indicator array:
            if self.mod_type == "lr":
                receptor_expr = self.receptors_expr[receptor].values.reshape(1, -1)
                sent_potential = sent_potential.multiply(receptor_expr)
                sent_potential.eliminate_zeros()

            # Find the location of the correct coefficient:
            if self.mod_type == "ligand":
                ligand_coeff_label = f"b_{ligand}"
                idx = coeffs.columns.get_loc(ligand_coeff_label)
            elif self.mod_type == "lr":
                lr_coeff_label = f"b_{ligand}:{receptor}"
                idx = coeffs.columns.get_loc(lr_coeff_label)

            coeff = coeffs.iloc[:, idx].values.reshape(1, -1)
            effect_sign = np.where(coeff > 0, 1, -1)
            # Weight each column by the coefficient magnitude and finally by the indicator for expression/no
            # expression of the target and store as sparse array:
            sig_interm = sent_potential.multiply(coeff)
            sig_interm.eliminate_zeros()
            effect_potential = sig_interm.multiply(target_indicator)
            effect_potential.eliminate_zeros()

        elif self.mod_type == "niche":
            if sender_cell_type is None:
                raise ValueError("Must provide sending cell type name for niche models.")

            sender_cell_type = self.cell_categories[sender_cell_type].values.reshape(-1, 1)
            # Get sending cells only of the specified type:
            sent_potential = spatial_weights.multiply(sender_cell_type)
            sent_potential.eliminate_zeros()

            # Check whether to condition on receiver cell type:
            if receiver_cell_type is not None:
                receiver_cell_type = self.cell_categories[receiver_cell_type].values.reshape(1, -1)
                sent_potential = sent_potential.multiply(receiver_cell_type)
                sent_potential.eliminate_zeros()

            sending_ct_coeff_label = f"b_Proxim{sender_cell_type}"
            coeff = coeffs[sending_ct_coeff_label].values.reshape(1, -1)
            effect_sign = np.where(coeff > 0, 1, -1)
            # Weight each column by the coefficient magnitude and finally by the indicator for expression/no expression
            # of the target and store as sparse array:
            sig_interm = sent_potential.multiply(coeff)
            sig_interm.eliminate_zeros()
            effect_potential = sig_interm.multiply(target_indicator)
            effect_potential.eliminate_zeros()

        effect_potential_sum_sender = np.array(effect_potential.sum(axis=1)).reshape(-1)
        sign = np.where(effect_potential_sum_sender > 0, 1, -1)
        # Take the absolute value to get the overall measure of the effect- after normalizing, add the sign back in:
        effect_potential_sum_sender = np.abs(effect_potential_sum_sender)
        normalized_effect_potential_sum_sender = (effect_potential_sum_sender - np.min(effect_potential_sum_sender)) / (
            np.max(effect_potential_sum_sender) - np.min(effect_potential_sum_sender)
        )
        normalized_effect_potential_sum_sender = normalized_effect_potential_sum_sender * sign

        effect_potential_sum_receiver = np.array(effect_potential.sum(axis=0)).reshape(-1)
        sign = np.where(effect_potential_sum_receiver > 0, 1, -1)
        # Take the absolute value to get the overall measure of the effect- after normalizing, add the sign back in:
        effect_potential_sum_receiver = np.abs(effect_potential_sum_receiver)
        normalized_effect_potential_sum_receiver = (
            effect_potential_sum_receiver - np.min(effect_potential_sum_receiver)
        ) / (np.max(effect_potential_sum_receiver) - np.min(effect_potential_sum_receiver))
        normalized_effect_potential_sum_receiver = normalized_effect_potential_sum_receiver * sign

        # Store summed sent/received potential:
        if store_summed_potential:
            if self.mod_type == "niche":
                if receiver_cell_type is None:
                    self.adata.obs[
                        f"norm_sum_sent_effect_potential_{sender_cell_type}_for_{target}"
                    ] = normalized_effect_potential_sum_sender

                    self.adata.obs[
                        f"norm_sum_received_effect_potential_from_{sender_cell_type}_for_{target}"
                    ] = normalized_effect_potential_sum_receiver
                else:
                    self.adata.obs[
                        f"norm_sum_sent_{sender_cell_type}_effect_potential_to_{receiver_cell_type}_for_{target}"
                    ] = normalized_effect_potential_sum_sender

                    self.adata.obs[
                        f"norm_sum_{receiver_cell_type}_received_effect_potential_from_{sender_cell_type}_for_{target}"
                    ] = normalized_effect_potential_sum_receiver

            elif self.mod_type == "ligand":
                self.adata.obs[
                    f"norm_sum_sent_effect_potential_{ligand}_for_{target}"
                ] = normalized_effect_potential_sum_sender

                self.adata.obs[
                    f"norm_sum_received_effect_potential_from_{ligand}_for_{target}"
                ] = normalized_effect_potential_sum_receiver

            elif self.mod_type == "lr":
                self.adata.obs[
                    f"norm_sum_sent_effect_potential_{ligand}_for_{target}_via_{receptor}"
                ] = normalized_effect_potential_sum_sender

                self.adata.obs[
                    f"norm_sum_received_effect_potential_from_{ligand}_for_{target}_via_{receptor}"
                ] = normalized_effect_potential_sum_receiver

            self.adata.obs["effect_sign"] = effect_sign.reshape(-1, 1)

        return effect_potential, normalized_effect_potential_sum_sender, normalized_effect_potential_sum_receiver

    def get_pathway_potential(
        self,
        pathway: Optional[str] = None,
        target: Optional[str] = None,
        spatial_weights: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        store_summed_potential: bool = True,
    ):
        """For each cell, computes the 'pathway effect potential', which is an aggregation of the effect potentials
        of all pathway member ligand-receptor pairs (or all pathway member ligands, for ligand-only models).

        Args:
            pathway: Name of pathway to compute pathway effect potential for.
            target: Optional string to select target from among the genes used to fit the model to compute signaling
                effects for. Note that this function takes only one target at a time. If not given, will take the
                first name from among all targets.
            spatial_weights: Optional pairwise spatial weights matrix. If not given, will compute at runtime.
            store_summed_potential: If True, will store both sent and received signaling potential as entries in
                .obs of the AnnData object.

        Returns:
            pathway_sum_potential: Array of shape [n_samples, n_samples]; proxy for the combined "signaling effect
                potential" with respect to a particular target gene for ligand-receptor pairs in a pathway.
            normalized_pathway_effect_potential_sum_sender: Array of shape [n_samples,]; for each sending cell,
                the sum of the pathway sum potential to all receiver cells for a given target gene, normalized between
                0 and 1.
            normalized_pathway_effect_potential_sum_receiver: Array of shape [n_samples,]; for each receiving cell,
                the sum of the pathway sum potential from all sender cells for a given target gene, normalized between
                0 and 1.
        """

        if self.mod_type not in ["lr", "ligand"]:
            raise ValueError("Cannot compute pathway effect potential, since fitted model does not use ligands.")

        # Columns consist of the spatial weights of each observation- convolve with expression of each ligand to
        # get proxy of ligand signal "sent", weight by the local coefficient value to get a proxy of the "signal
        # functionally received" in generating the downstream effect and store in .obsp.
        if target is None and self.target_for_downstream is not None:
            target = self.target_for_downstream
        else:
            self.logger.info("Target gene not provided for :func `get_effect_potential`. Using first target listed.")
            target = list(self.coeffs.keys())[0]

        if pathway is None and self.pathway_for_downstream is not None:
            pathway = self.pathway_for_downstream
        else:
            raise ValueError("Must provide pathway to analyze.")

        lr_db_subset = self.lr_db[self.lr_db["pathway"] == pathway]
        all_senders = list(set(lr_db_subset["from"]))
        all_receivers = list(set(lr_db_subset["to"]))

        if self.mod_type == "lr":
            self.logger.info(
                "Computing pathway effect potential for ligand-receptor pairs in pathway, since :attr "
                "`mod_type` is 'lr'."
            )

            # All possible ligand-receptor combinations:
            possible_lr_combos = list(itertools.product(all_senders, all_receivers))
            valid_lr_combos = list(set(possible_lr_combos).intersection(set(self.lr_pairs)))
            if len(valid_lr_combos) < 3:
                raise ValueError(
                    f"Pathway effect potential computation for pathway {pathway} is unsuitable for this model, "
                    f"since there are fewer than three valid ligand-receptor pairs in the pathway that were "
                    f"incorporated in the initial model."
                )

            all_pathway_member_effects = {}
            for j, col in enumerate(valid_lr_combos):
                ligand = col[0]
                receptor = col[1]
                effect_potential, _, _ = self.get_effect_potential(
                    target=target,
                    ligand=ligand,
                    receptor=receptor,
                    spatial_weights=spatial_weights,
                    store_summed_potential=False,
                )
                all_pathway_member_effects[f"effect_potential_{ligand}_{receptor}_on_{target}"] = effect_potential

        elif self.mod_type == "ligand":
            self.logger.info(
                "Computing pathway effect potential for ligands in pathway, since :attr `mod_type` is " "'ligand'."
            )

            all_pathway_member_effects = {}
            for j, col in enumerate(all_senders):
                ligand = col
                effect_potential, _, _ = self.get_effect_potential(
                    target=target,
                    ligand=ligand,
                    spatial_weights=spatial_weights,
                    store_summed_potential=False,
                )
                all_pathway_member_effects[f"effect_potential_{ligand}_on_{target}"] = effect_potential

        # Combine results for all ligand-receptor pairs in the pathway:
        pathway_sum_potential = None
        for key in all_pathway_member_effects.keys():
            if pathway_sum_potential is None:
                pathway_sum_potential = all_pathway_member_effects[key]
            else:
                pathway_sum_potential += all_pathway_member_effects[key]
        # self.adata.obsp[f"effect_potential_{pathway}_on_{target}"] = pathway_sum_potential

        pathway_effect_potential_sum_sender = np.array(pathway_sum_potential.sum(axis=1)).reshape(-1)
        normalized_pathway_effect_potential_sum_sender = (
            pathway_effect_potential_sum_sender - np.min(pathway_effect_potential_sum_sender)
        ) / (np.max(pathway_effect_potential_sum_sender) - np.min(pathway_effect_potential_sum_sender))

        pathway_effect_potential_sum_receiver = np.array(pathway_sum_potential.sum(axis=0)).reshape(-1)
        normalized_effect_potential_sum_receiver = (
            pathway_effect_potential_sum_receiver - np.min(pathway_effect_potential_sum_receiver)
        ) / (np.max(pathway_effect_potential_sum_receiver) - np.min(pathway_effect_potential_sum_receiver))

        if store_summed_potential:
            if self.mod_type == "lr":
                send_key = f"norm_sum_sent_effect_potential_{pathway}_lr_for_{target}"
                receive_key = f"norm_sum_received_effect_potential_{pathway}_lr_for_{target}"
            elif self.mod_type == "ligand":
                send_key = f"norm_sum_sent_effect_potential_{pathway}_ligands_for_{target}"
                receive_key = f"norm_sum_received_effect_potential_{pathway}_ligands_for_{target}"

            self.adata.obs[send_key] = normalized_pathway_effect_potential_sum_sender
            self.adata.obs[receive_key] = normalized_effect_potential_sum_receiver

        return (
            pathway_sum_potential,
            normalized_pathway_effect_potential_sum_sender,
            normalized_effect_potential_sum_receiver,
        )

    def inferred_effect_direction(
        self,
        targets: Optional[Union[str, List[str]]] = None,
        compute_pathway_effect: bool = False,
    ):
        """For visualization purposes, used for models that consider ligand expression (:attr `mod_type` is 'ligand' or
        'lr' (for receptor models, assigning directionality is impossible and for niche models, it makes much less
        sense to draw/compute a vector field). Construct spatial vector fields to infer the directionality of
        observed effects (the "sources" of the downstream expression).

        Parts of this function are inspired by 'communication_direction' from COMMOT: https://github.com/zcang/COMMOT

        Args:
            targets: Optional string or list of strings to select targets from among the genes used to fit the model
                to compute signaling effects for. If not given, will use all targets.
            compute_pathway_effect: Whether to compute the effect potential for each pathway in the model. If True,
                will collectively take the effect potential of all pathway components. If False, will compute effect
                potential for each for each individual signal.
        """
        if not self.mod_type == "ligand" or self.mod_type == "lr":
            raise ValueError(
                "Direction of effect can only be inferred if ligand expression is used as part of the " "model."
            )

        if self.compute_pathway_effect is not None:
            compute_pathway_effect = self.compute_pathway_effect
        if self.target_for_downstream is not None:
            targets = self.target_for_downstream

        # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver:
        # Note: as the default (if bw is not otherwise provided), the n nearest neighbors will be used for the
        # bandwidth:
        spatial_weights = self._compute_all_wi(self.search_bw, bw_fixed=self.bw_fixed, exclude_self=True)

        # Columns consist of the spatial weights of each observation- convolve with expression of each ligand to
        # get proxy of ligand signal "sent", weight by the local coefficient value to get a proxy of the "signal
        # functionally received" in generating the downstream effect and store in .obsp.
        if targets is None:
            targets = self.coeffs.keys()
        elif isinstance(targets, str):
            targets = [targets]

        if self.filter_targets:
            pearson_dict = {}
            for target in targets:
                observed = self.adata[:, target].X.toarray().reshape(-1, 1)
                predicted = self.predictions[target].reshape(-1, 1)

                # Ignore large predicted values that are actually zero- these have a high likelihood of being
                # dropouts rather than biological zeros:
                z_scores = zscore(predicted)
                outlier_indices = np.where((observed == 0) & (z_scores > 2))[0]
                predicted = np.delete(predicted, outlier_indices)
                observed = np.delete(observed, outlier_indices)

                rp, _ = pearsonr(observed, predicted)
                pearson_dict[target] = rp

            targets = [target for target in targets if pearson_dict[target] > self.filter_target_threshold]

        queries = self.lr_pairs if self.mod_type == "lr" else self.ligands

        if compute_pathway_effect:
            # Find pathways that are represented among the ligands or ligand-receptor pairs:
            pathways = []
            for query in queries:
                if self.mod_type == "lr":
                    ligand = query.split(":")[0]
                    receptor = query.split(":")[1]
                    col_pathways = list(
                        self.lr_db.loc[
                            (self.lr_db["from"] == ligand) & (self.lr_db["to"] == receptor), "pathway"
                        ].values
                    )
                    pathways.extend(col_pathways)
                elif self.mod_type == "ligand":
                    col_pathways = list(self.lr_db.loc[self.lr_db["from"] == query, "pathway"].values)
                    pathways.extend(col_pathways)
            # Before taking the set of pathways, count number of occurrences of each pathway in the list- remove
            # pathways for which there are fewer than three ligands or ligand-receptor pairs- these are not enough to
            # constitute a pathway:
            pathway_counts = Counter(pathways)
            pathways = [pathway for pathway, count in pathway_counts.items() if count >= 3]
            # Take the set of pathways:
            queries = list(set(pathways))

        for target in targets:
            for j, query in enumerate(queries):
                if self.mod_type == "lr":
                    if compute_pathway_effect:
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_pathway_potential(target=target, pathway=query, spatial_weights=spatial_weights)
                    else:
                        ligand = query.split(":")[0]
                        receptor = query.split(":")[1]
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_effect_potential(
                            target=target, ligand=ligand, receptor=receptor, spatial_weights=spatial_weights
                        )
                else:
                    if compute_pathway_effect:
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_pathway_potential(target=target, pathway=query, spatial_weights=spatial_weights)
                    else:
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_effect_potential(target=target, ligand=query, spatial_weights=spatial_weights)

                # Compute vector field:
                self.define_effect_vf(
                    effect_potential,
                    normalized_effect_potential_sum_sender,
                    normalized_effect_potential_sum_receiver,
                    query,
                    target,
                )

        # Save AnnData object with effect direction information:
        adata_name = os.path.splitext(self.adata_path)[0]
        self.adata.write(f"{adata_name}_effect_directions.h5ad")

    def define_effect_vf(
        self,
        effect_potential: scipy.sparse.spmatrix,
        normalized_effect_potential_sum_sender: np.ndarray,
        normalized_effect_potential_sum_receiver: np.ndarray,
        sig: str,
        target: str,
    ):
        """Given the pairwise effect potential array, computes the effect vector field.

        Args:
            effect_potential: Sparse array containing computed effect potentials- output from
                :func:`get_effect_potential`
            normalized_effect_potential_sum_sender: Array containing the sum of the effect potentials sent by each
                cell. Output from :func:`get_effect_potential`.
            normalized_effect_potential_sum_receiver: Array containing the sum of the effect potentials received by
                each cell. Output from :func:`get_effect_potential`.
            sig: Label for the mediating interaction (e.g. name of a ligand, name of a ligand-receptor pair, etc.)
            target: Name of the target that the vector field describes the effect for
        """
        sending_vf = np.zeros_like(self.coords)
        receiving_vf = np.zeros_like(self.coords)

        # Vector field for sent signal:
        effect_potential_lil = effect_potential.tolil()
        for i in range(self.n_samples):
            if len(effect_potential_lil.rows[i]) <= self.k:
                temp_idx = np.array(effect_potential_lil.rows[i], dtype=int)
                temp_val = np.array(effect_potential_lil.data[i], dtype=float)
            else:
                row_np = np.array(effect_potential_lil.rows[i], dtype=int)
                data_np = np.array(effect_potential_lil.data[i], dtype=float)
                temp_idx = row_np[np.argsort(-data_np)[: self.k]]
                temp_val = data_np[np.argsort(-data_np)[: self.k]]
            if len(temp_idx) == 0:
                continue
            elif len(temp_idx) == 1:
                avg_v = self.coords[temp_idx[0], :] - self.coords[i, :]
            else:
                temp_v = self.coords[temp_idx, :] - self.coords[i, :]
                temp_v = normalize(temp_v, norm="l2")
                avg_v = np.sum(temp_v * temp_val.reshape(-1, 1), axis=0)
            avg_v = normalize(avg_v.reshape(1, -1))
            sending_vf[i, :] = avg_v[0, :] * normalized_effect_potential_sum_sender[i]
        sending_vf = np.clip(sending_vf, -0.02, 0.02)

        # Vector field for received signal:
        effect_potential_lil = effect_potential.T.tolil()
        for i in range(self.n_samples):
            if len(effect_potential_lil.rows[i]) <= self.k:
                temp_idx = np.array(effect_potential_lil.rows[i], dtype=int)
                temp_val = np.array(effect_potential_lil.data[i], dtype=float)
            else:
                row_np = np.array(effect_potential_lil.rows[i], dtype=int)
                data_np = np.array(effect_potential_lil.data[i], dtype=float)
                temp_idx = row_np[np.argsort(-data_np)[: self.k]]
                temp_val = data_np[np.argsort(-data_np)[: self.k]]
            if len(temp_idx) == 0:
                continue
            elif len(temp_idx) == 1:
                avg_v = self.coords[temp_idx, :] - self.coords[i, :]
            else:
                temp_v = self.coords[temp_idx, :] - self.coords[i, :]
                temp_v = normalize(temp_v, norm="l2")
                avg_v = np.sum(temp_v * temp_val.reshape(-1, 1), axis=0)
            avg_v = normalize(avg_v.reshape(1, -1))
            receiving_vf[i, :] = avg_v[0, :] * normalized_effect_potential_sum_receiver[i]
        receiving_vf = np.clip(receiving_vf, -0.02, 0.02)

        del effect_potential

        self.adata.obsm[f"spatial_effect_sender_vf_{sig}_{target}"] = sending_vf
        self.adata.obsm[f"spatial_effect_receiver_vf_{sig}_{target}"] = receiving_vf

    # ---------------------------------------------------------------------------------------------------
    # Constructing gene regulatory networks
    # ---------------------------------------------------------------------------------------------------
    def CCI_sender_deg_detection_setup(
        self,
        ligand: Optional[str] = None,
        pathway: Optional[str] = None,
        sender_cell_type: Optional[str] = None,
        perform_dimensionality_reduction: bool = False,
    ):
        """Computes differential expression signatures of cells with various levels of ligand expression.

        Args:
            ligand: Ligand to use for differential expression analysis. Will take precedent over sender/receiver cell
                type if also provided.
            target: Only used if 'diff_sending_or_receiving' is "receiving"- target to use for differential expression
                analysis. If None, will use the first listed target.
            pathway: Optional pathway to use for differential expression analysis. Will use ligands and receptors in
                these pathways to collectively compute signaling potential score. Will take precedent over
                ligand/receptor and sender/receiver cell type if provided.
            sender_cell_type: Sender cell type to use for differential expression analysis. If given,
                this will essentially compute differential expression for the given cell type.
            perform_dimensionality_reduction: Set True to perform dimensionality reduction on the array of
                predictors. If False,

        Returns:

        """
        logger = lm.get_main_logger()

        if pathway is None and ligand is None and sender_cell_type is None:
            raise ValueError("Must provide at least one pathway, ligand, or sender_cell_type.")

        if pathway is not None:
            # For sending analysis, use ligand expression as the dependent variable:
            # First, compute pathway score:
            lr_db_subset = self.lr_db[self.lr_db["pathway"] == pathway]
            all_senders = list(set(lr_db_subset["from"]))
            all_senders = [lig for lig in all_senders if lig in self.adata.var_names]

            if len(all_senders) < 3:
                raise ValueError(
                    f"Pathway effect potential computation for pathway {pathway} is unsuitable for this model, "
                    f"since there are fewer than three valid ligand-receptor pairs in the pathway that were "
                    f"incorporated in the initial model."
                )

            self.adata.obs[f"{pathway}_ligands"] = np.sum(self.ligands_expr[all_senders].values, axis=1)
            send_key = f"{pathway}_ligands"

        elif ligand is not None:
            send_key = ligand

        elif sender_cell_type is not None:
            if self.mod_type != "niche":
                raise ValueError(
                    f"Only sender_cell_type was provided to compute effect-associated DEGs, but the "
                    f"initial {self.mod_type} model does not use cell type identity."
                )

            send_key = sender_cell_type
            # Check that the name of the sender cell type is provided in a recognizable format:
            if send_key not in self.cell_categories.columns:
                send_key = re.sub(
                    r"\b([a-zA-Z0-9])",
                    lambda match: match.group(1).upper(),
                    re.sub(r"[" r"^a-zA-Z0-9]+", "", send_key),
                )

        try:
            # Sent signal from ligand
            sent_signal = self.ligands_expr[send_key]
        except:
            try:
                # Sent signal from pathway
                sent_signal = self.adata.obs[send_key]
            except:
                # Sent signal from cell type- in this case, will just look for differential expression in the cell type
                sent_signal = self.cell_categories[send_key]

        # Check if the array of additional molecules to query has already been created:
        parent_dir = os.path.dirname(self.adata_path)
        file_name = os.path.basename(self.adata_path).split(".")[0]

        # For saving/loading purposes:
        if pathway is not None:
            file_name = f"design_matrix_sent_signal_{pathway}.csv"
        elif ligand is not None:
            file_name = f"design_matrix_sent_signal_{ligand}.csv"
        else:
            file_name = f"design_matrix_sent_signal_{sender_cell_type}.csv"

        if not os.path.exists(os.path.join(parent_dir, "cci_deg_detection")):
            os.makedirs(os.path.join(parent_dir, "cci_deg_detection"))

        if not os.path.exists(os.path.join(parent_dir, "cci_deg_detection", file_name)):
            if self.cci_dir is None:
                raise ValueError("With 'diff_sending_or_receiving' set to 'sending', please provide :attr `cci_dir`.")

            if self.species == "human":
                grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
                rna_bp_db = pd.read_csv(os.path.join(self.cci_dir, "human_RBP_db.csv"), index_col=0)
                cof_db = pd.read_csv(os.path.join(self.cci_dir, "human_cofactors.csv"), index_col=0)
                tf_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_TF_TF_db.csv"), index_col=0)
            elif self.species == "mouse":
                grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
                rna_bp_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_RBP_db.csv"), index_col=0)
                cof_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_cofactors.csv"), index_col=0)
                tf_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_TF_TF_db.csv"), index_col=0)

            # Subset GRN and other databases to only include TFs that are in the adata object:
            grn = grn[np.isin(self.adata.var_names, grn.columns)]
            cof_db = cof_db[np.isin(self.adata.var_names, cof_db.columns)]
            tf_tf_db = tf_tf_db[np.isin(self.adata.var_names, tf_tf_db.columns)]

            self.logger.info(
                "Selecting transcription factors, cofactors and RNA-binding proteins for analysis of differential "
                "expression."
            )

            # Further subset list of additional factors to those that are expressed in at least n% of the cells that are
            # nonzero in sending cells (use the user input 'target_expr_threshold'):
            indices = sent_signal.values.reshape(-1) != 0
            nz_sending = list(self.sample_names[indices])
            adata_subset = self.adata[nz_sending, :]
            n_cells_threshold = int(self.target_expr_threshold * adata_subset.n_obs)

            all_TFs = list(grn.columns)
            all_TFs = [tf for tf in all_TFs if tf in cof_db.columns and tf in tf_tf_db.columns]
            if scipy.sparse.issparse(self.adata.X):
                nnz_counts = np.array(adata_subset[:, all_TFs].X.getnnz(axis=0)).flatten()
            else:
                nnz_counts = np.array(adata_subset[:, all_TFs].X.getnnz(axis=0)).flatten()
            all_TFs = [tf for tf, nnz in zip(all_TFs, nnz_counts) if nnz >= n_cells_threshold]

            # Get the set of transcription cofactors that correspond to these transcription factors, in addition to
            # interacting transcription factors that may not themselves have passed the threshold:
            cof_subset = list(cof_db[(cof_db[all_TFs] == 1).any(axis=1)].index)
            cof_subset = [cof for cof in cof_subset if cof in self.feature_names]
            intersecting_tf_subset = list(tf_tf_db[(tf_tf_db[all_TFs] == 1).any(axis=1)].index)
            intersecting_tf_subset = [tf for tf in intersecting_tf_subset if tf in self.feature_names]

            # Subset to cofactors for which enough signal is present- filter to those expressed in at least n% of the
            # cells that express at least one of the TFs associated with the cofactor:
            all_cofactors = []
            for cofactor in cof_subset:
                cof_row = cof_db.loc[cofactor, :]
                cof_TFs = cof_row[cof_row == 1].index
                tfs_expr_subset_indices = np.where(adata_subset[:, cof_TFs].X.sum(axis=1) > 0)[0]
                tf_subset_cells = adata_subset[tfs_expr_subset_indices, :]
                n_cells_threshold = int(self.target_expr_threshold * tf_subset_cells.n_obs)
                if scipy.sparse.issparse(self.adata.X):
                    nnz_counts = np.array(tf_subset_cells[:, cofactor].X.getnnz(axis=0)).flatten()
                else:
                    nnz_counts = np.array(tf_subset_cells[:, cofactor].X.getnnz(axis=0)).flatten()

                if nnz_counts >= n_cells_threshold:
                    all_cofactors.append(cofactor)

            # And extend the set of transcription factors using interacting pairs that may also be present in the same
            # cells upstream transcription factors are:
            all_interacting_tfs = []
            for tf in intersecting_tf_subset:
                tf_row = tf_tf_db.loc[tf, :]
                tf_TFs = tf_row[tf_row == 1].index
                tfs_expr_subset_indices = np.where(adata_subset[:, tf_TFs].X.sum(axis=1) > 0)[0]
                tf_subset_cells = adata_subset[tfs_expr_subset_indices, :]
                n_cells_threshold = int(self.target_expr_threshold * tf_subset_cells.n_obs)
                if scipy.sparse.issparse(self.adata.X):
                    nnz_counts = np.array(tf_subset_cells[:, tf].X.getnnz(axis=0)).flatten()
                else:
                    nnz_counts = np.array(tf_subset_cells[:, tf].X.getnnz(axis=0)).flatten()

                if nnz_counts >= n_cells_threshold:
                    all_interacting_tfs.append(tf)

            if self.cci_degs_model_interactions:
                # Transcription factor-cofactor combinatorial pairs:
                tf_cof_pairs = {}
                for tf in all_TFs:
                    tf_cofactors = cof_db.loc[all_cofactors, tf]
                    # Find elements that are equal to 1:
                    tf_cofactors = tf_cofactors[tf_cofactors == 1].index.tolist()
                    tf_cof_pairs[tf] = tf_cofactors

                # Transcription factor interactions:
                tf_tf_pairs = {}
                for tf in all_TFs:
                    tf_tf = tf_tf_db.loc[all_interacting_tfs, tf]
                    # Find elements that are equal to 1:
                    tf_tf = tf_tf[tf_tf == 1].index.tolist()
                    tf_tf_pairs[tf] = tf_tf

            # Do the same for RNA-binding proteins:
            all_RBPs = list(rna_bp_db["Gene_Name"].values)
            all_RBPs = [r for r in all_RBPs if r in self.feature_names]
            if len(all_RBPs) > 0:
                if scipy.sparse.issparse(self.adata.X):
                    nnz_counts = np.array(adata_subset[:, all_RBPs].X.getnnz(axis=0)).flatten()
                else:
                    nnz_counts = np.array(adata_subset[:, all_RBPs].X.getnnz(axis=0)).flatten()
                all_RBPs = [tf for tf, nnz in zip(all_RBPs, nnz_counts) if nnz >= n_cells_threshold]
                # Remove RBPs if any happen to be TFs or cofactors:
                all_RBPs = [
                    r for r in all_RBPs if r not in all_TFs and r not in all_interacting_tfs and r not in all_cofactors
                ]

            self.logger.info(f"For this dataset, marked {len(all_TFs)} of interest.")
            self.logger.info(f"For this dataset, marked {len(all_cofactors)} transcriptional cofactors of interest.")
            if len(all_RBPs) > 0:
                self.logger.info(f"For this dataset, marked {len(all_RBPs)} RNA-binding proteins of interest.")

            if not self.cci_degs_model_interactions:
                # Get feature names- for the singleton factors:
                regulator_features = all_TFs + all_interacting_tfs + all_cofactors + all_RBPs
            else:
                # Get feature names- for the interaction factors:
                interacting_pairs = []
                regulator_features = []
                for key, value in tf_cof_pairs.items():
                    if value:
                        interacting_pairs.extend([f"{key}_{val}" for val in value])
                    regulator_features.append(key)
                    regulator_features.extend(value)
                for key, value in tf_tf_pairs.items():
                    if value:
                        interacting_pairs.extend([f"{key}_{val}" for val in value])
                    regulator_features.append(key)
                    regulator_features.extend(value)

            # Take subset of AnnData object corresponding to these regulators:
            counts = self.adata[:, regulator_features].copy()
            self.sender_deg_predictors = pd.DataFrame(
                counts.X.A if scipy.sparse.issparse(counts.X) else counts.X,
                index=counts.obs_names,
                columns=regulator_features,
            )
            # Scale independent variables to reduce the impact of large values:
            self.sender_deg_predictors = self.sender_deg_predictors.applymap(np.log1p)

            if self.cci_degs_model_interactions:
                # Combine columns that are part of interactions- eventually the individual columns should be dropped,
                # but store them in a temporary list to do so later because some may contribute to multiple complexes
                to_drop = []
                for element in interacting_pairs:
                    if "_" in element:
                        parts = element.split("_")
                        # Combine the columns into a new column with the name of the hyphenated element- here we
                        # will compute the geometric mean of the expression values of the interacting pair:
                        self.sender_deg_predictors[element] = self.sender_deg_predictors[parts].apply(
                            lambda x: x.prod() ** (1 / len(parts)), axis=1
                        )
                        to_drop.extend(parts)

                # Drop any possible duplicate ligands alongside any other columns to be dropped:
                to_drop = list(set(to_drop))
                self.sender_deg_predictors.drop(to_drop, axis=1, inplace=True)
                first_occurrences = self.sender_deg_predictors.columns.duplicated(keep="first")
                self.sender_deg_predictors = self.sender_deg_predictors.loc[:, ~first_occurrences]

            # Save independent variable array:
            self.sender_deg_predictors.to_csv(os.path.join(parent_dir, "cci_deg_detection", file_name))
        else:
            logger.info(f"Existing data found at {file_name}, using this for further analysis.")
            self.sender_deg_predictors = pd.read_csv(
                os.path.join(parent_dir, "cci_deg_detection", file_name), index_col=0
            )

        if perform_dimensionality_reduction:
            # Compute latent representation of the AnnData subset ("counts"):
            # Minmax scale interaction features for the purpose of dimensionality reduction:
            sender_deg_predictors_scaled = self.sender_deg_predictors.apply(
                lambda column: (column - column.min()) / (column.max() - column.min())
            )

            # Compute the ideal number of principal components to use- use half the number of features as the max
            # possible number of components:
            logger.info("Computing optimal number of principal components ...")
            n_pca_components = find_optimal_pca_components(
                sender_deg_predictors_scaled.values, pca_func=PCA, max_components=None, drop_ratio=0.25
            )
            logger.info(f"Optimal number of PCs: {n_pca_components}")

            # Perform PCA reduction with the chosen number of components:

            # Using the ideal number of components, compute the ideal number of UMAP components using the silhouette
            # score after calculating Leiden partitioning:
            logger.info("Computing optimal number of UMAP components ...")
            n_umap_components = "filler"

            # Collect information into .csv file:
            self.logger.info(
                "'CCI_sender_deg_detection'- saving regulatory molecules to test as .h5ad file to the "
                "directory of the original AnnData object..."
            )
            counts.write_h5ad(
                os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_{send_key}_queries_singleton.h5ad")
            )

    def CCI_sender_deg_detection(self):
        # Load and process the file for the chosen ligand:
        # counts = anndata.read_h5ad(
        #     os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_{send_key}_queries.h5ad")
        # )
        "filler"

        # Stitch into dataframe, where the first 3 columns are coordinates in UMAP space and the fourth column is the
        # sent signal:

        # Perform logistic regression if using sender cell type as the query

    # ---------------------------------------------------------------------------------------------------
    # Cell type coupling:
    # ---------------------------------------------------------------------------------------------------
    def compute_cell_type_coupling(
        self,
        targets: Optional[Union[str, List[str]]] = None,
        effect_strength_threshold: Optional[float] = None,
    ):
        """Generates heatmap of spatially differentially-expressed features for each pair of sender and receiver
        categories- if :attr `mod_type` is "niche", this directly averages the effects for each neighboring cell type
        for each observation. If :attr `mod_type` is "lr" or "ligand", this correlates cell type prevalence with the
        size of the predicted effect on downstream expression for each L:R pair.

        Args:
            targets: Optional string or list of strings to select targets from among the genes used to fit the model
                to compute signaling effects for. If not given, will use all targets.
            effect_strength_threshold: Optional percentile for filtering the computed signaling effect. If not None,
                will filter to those cells for which a given signaling effect is predicted to have a strong effect
                on target gene expression. Otherwise, will compute cell type coupling over all cells in the sample.

        Returns:
            ct_coupling: 3D array summarizing cell type coupling in terms of effect on downstream expression
            ct_coupling_significance: 3D array summarizing significance of cell type coupling in terms of effect on
                downstream expression
        """

        if self.effect_strength_threshold is not None:
            effect_strength_threshold = self.effect_strength_threshold
            self.logger.info(
                f"Computing cell type coupling for cells in which predicted sent/received effect score "
                f"is higher than {effect_strength_threshold * 100}th percentile score."
            )

        # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver:
        spatial_weights = self._compute_all_wi(self.search_bw, bw_fixed=self.bw_fixed, exclude_self=True, verbose=False)

        if not self.mod_type != "receptor":
            raise ValueError("Knowledge of the source is required to sent effect potential.")

        # Compute signaling potential for each target (mediated by each of the possible signaling patterns-
        # ligand/receptor or cell type/cell type pair):

        # Columns consist of the spatial weights of each observation- convolve with expression of each ligand to
        # get proxy of ligand signal "sent", weight by the local coefficient value to get a proxy of the "signal
        # functionally received" in generating the downstream effect and store in .obsp.
        if targets is None:
            targets = self.coeffs.keys()
        elif isinstance(targets, str):
            targets = [targets]

        if self.filter_targets:
            pearson_dict = {}
            for target in targets:
                observed = self.adata[:, target].X.toarray().reshape(-1, 1)
                predicted = self.predictions[target].reshape(-1, 1)

                # Ignore large predicted values that are actually zero- these have a high likelihood of being
                # dropouts rather than biological zeros:
                z_scores = zscore(predicted)
                outlier_indices = np.where((observed == 0) & (z_scores > 2))[0]
                predicted = np.delete(predicted, outlier_indices)
                observed = np.delete(observed, outlier_indices)

                rp, _ = pearsonr(observed, predicted)
                pearson_dict[target] = rp

            targets = [target for target in targets if pearson_dict[target] > self.filter_target_threshold]

        # Cell type pairings:
        if not hasattr(self, "cell_categories"):
            group_name = self.adata.obs[self.group_key]
            # db = pd.DataFrame({"group": group_name})
            db = pd.DataFrame({"group": group_name})
            categories = np.array(group_name.unique().tolist())
            # db["group"] = pd.Categorical(db["group"], categories=categories)
            db["group"] = pd.Categorical(db["group"], categories=categories)

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)
            # Ensure columns are in order:
            self.cell_categories = X.reindex(sorted(X.columns), axis=1)
            # Ensure each category is one word with no spaces or special characters:
            self.cell_categories.columns = [
                re.sub(r"\b([a-zA-Z0-9])", lambda match: match.group(1).upper(), re.sub(r"[^a-zA-Z0-9]+", "", s))
                for s in self.cell_categories.columns
            ]

        celltype_pairs = list(itertools.product(self.cell_categories.columns, self.cell_categories.columns))
        celltype_pairs = [f"{cat[0]}-{cat[1]}" for cat in celltype_pairs]

        if self.mod_type in ["lr", "ligand"]:
            cols = self.lr_pairs if self.mod_type == "lr" else self.ligands
        else:
            cols = celltype_pairs

        # Storage for cell type-cell type coupling results- primary axis: targets, secondary: L:R pairs/ligands,
        # tertiary: cell type pairs:
        ct_coupling = np.zeros((len(targets), len(cols), len(celltype_pairs)))
        ct_coupling_significance = np.zeros((len(targets), len(cols), len(celltype_pairs)))

        for i, target in enumerate(targets):
            for j, col in enumerate(cols):
                if self.mod_type == "lr":
                    ligand = col[0]
                    receptor = col[1]

                    effect_potential, _, _ = self.get_effect_potential(
                        target=target, ligand=ligand, receptor=receptor, spatial_weights=spatial_weights
                    )

                    # For each cell type pair, compute average effect potential across all cells of the sending and
                    # receiving type for those cells that are sending + receiving signal:
                    for k, pair in enumerate(celltype_pairs):
                        sending_cell_type = pair.split("-")[0]
                        receiving_cell_type = pair.split("-")[1]

                        # Get indices of cells of each type:
                        sending_indices = np.where(self.cell_categories[sending_cell_type] == 1)[0]
                        receiving_indices = np.where(self.cell_categories[receiving_cell_type] == 1)[0]

                        # Get average effect potential across all cells of each type- first filter if threshold is
                        # given:
                        if effect_strength_threshold is not None:
                            effect_potential_data = effect_potential.data
                            # Threshold is taken to be a percentile value:
                            effect_strength_threshold = np.percentile(
                                effect_potential_data, effect_strength_threshold * 100
                            )
                            strong_effect_mask = effect_potential > effect_strength_threshold
                            rem_row_indices, rem_col_indices = strong_effect_mask.nonzero()

                            # Update sending and receiving indices to now include cells of the given sending and
                            # receiving type that send/receive signal:
                            sending_indices = np.intersect1d(sending_indices, rem_row_indices)
                            receiving_indices = np.intersect1d(receiving_indices, rem_col_indices)

                        # Check if there is no signal being transmitted and/or received between cells of the given
                        # two types:
                        if len(sending_indices) == 0 or len(receiving_indices) == 0:
                            ct_coupling[i, j, k] = 0
                            ct_coupling_significance[i, j, k] = 0
                            continue

                        avg_effect_potential = np.mean(effect_potential[sending_indices, receiving_indices])
                        ct_coupling[i, j, k] = avg_effect_potential
                        ct_coupling_significance[i, j, k] = permutation_testing(
                            effect_potential,
                            n_permutations=10000,
                            n_jobs=30,
                            subset_rows=sending_indices,
                            subset_cols=receiving_indices,
                        )

                elif self.mod_type == "ligand":
                    effect_potential, _, _ = self.get_effect_potential(
                        target=target, ligand=col, spatial_weights=spatial_weights
                    )

                    # For each cell type pair, compute average effect potential across all cells of the sending and
                    # receiving type:
                    for k, pair in enumerate(celltype_pairs):
                        sending_cell_type = pair.split("-")[0]
                        receiving_cell_type = pair.split("-")[1]

                        # Get indices of cells of each type:
                        sending_indices = np.where(self.cell_categories[sending_cell_type] == 1)[0]
                        receiving_indices = np.where(self.cell_categories[receiving_cell_type] == 1)[0]

                        # Get average effect potential across all cells of each type:
                        avg_effect_potential = np.mean(effect_potential[sending_indices, receiving_indices])
                        ct_coupling[i, j, k] = avg_effect_potential
                        ct_coupling_significance[i, j, k] = permutation_testing(
                            effect_potential,
                            n_permutations=10000,
                            n_jobs=30,
                            subset_rows=sending_indices,
                            subset_cols=receiving_indices,
                        )

                elif self.mod_type == "niche":
                    sending_cell_type = col.split("-")[0]
                    receiving_cell_type = col.split("-")[1]
                    effect_potential, _, _ = self.get_effect_potential(
                        target=target,
                        sender_cell_type=sending_cell_type,
                        receiver_cell_type=receiving_cell_type,
                        spatial_weights=spatial_weights,
                    )

                    # Directly compute the average- the processing steps when providing sender and receiver cell
                    # types already handle filtering down to the pertinent cells- but for the permutation we still have
                    # to supply indices to keep track of the original indices of the cells:
                    for k, pair in enumerate(celltype_pairs):
                        sending_cell_type = pair.split("-")[0]
                        receiving_cell_type = pair.split("-")[1]

                        # Get indices of cells of each type:
                        sending_indices = np.where(self.cell_categories[sending_cell_type] == 1)[0]
                        receiving_indices = np.where(self.cell_categories[receiving_cell_type] == 1)[0]

                        avg_effect_potential = np.mean(effect_potential)
                        ct_coupling[i, j, k] = avg_effect_potential
                        ct_coupling_significance[i, j, k] = permutation_testing(
                            avg_effect_potential,
                            n_permutations=10000,
                            n_jobs=30,
                            subset_rows=sending_indices,
                            subset_cols=receiving_indices,
                        )

        # Save results:
        parent_dir = os.path.dirname(self.output_path)
        if not os.path.exists(os.path.join(parent_dir, "cell_type_coupling")):
            os.makedirs(os.path.join(parent_dir, "cell_type_coupling"))

        # Convert Numpy array to xarray object for storage and save as .h5 object:
        ct_coupling = xr.DataArray(
            ct_coupling,
            dims=["target", "signal_source", "celltype_pair"],
            coords={"target": targets, "signal_source": cols, "celltype_pair": celltype_pairs},
            name="ct_coupling",
        )

        ct_coupling_significance = xr.DataArray(
            ct_coupling_significance,
            dims=["target", "signal_source", "celltype_pair"],
            coords={"target": targets, "signal_source": cols, "celltype_pair": celltype_pairs},
            name="ct_coupling_significance",
        )
        coupling_results_path = os.path.join(
            parent_dir, "cell_type_coupling", "celltype_effects_coupling_and_significance.nc"
        )

        # Combine coupling and significance into the same dataset:
        ds = xr.merge([ct_coupling, ct_coupling_significance])
        ds.to_netcdf(coupling_results_path)

        return ct_coupling, ct_coupling_significance

    def pathway_coupling(self, pathway: Union[str, List[str]]):
        """From computed cell type coupling results, compute pathway coupling by leveraging the pathway membership of
        constituent ligands/ligand:receptor pairs.

        Args:
            pathway: Name of the pathway(s) to compute coupling for.

        Returns:
            pathway_coupling: Dictionary where pathway names are indices and values are coupling score dataframes for
                the pathway
            pathway_coupling_significance: Dictionary where pathway names are indices and values are coupling score
                significance dataframes for the pathway
        """
        # Check for already existing cell coupling results:
        parent_dir = os.path.dirname(self.output_path)
        coupling_results_path = os.path.join(
            parent_dir, "cell_type_coupling", "celltype_effects_coupling_and_significance.nc"
        )

        try:
            coupling_ds = xr.open_dataset(coupling_results_path)
            coupling_results = coupling_ds["ct_coupling"]
            coupling_significance = coupling_ds["ct_coupling_significance"]
        except FileNotFoundError:
            self.logger.info("No coupling results found. Computing cell type coupling...")
            coupling_results, coupling_significance = self.compute_cell_type_coupling()

        predictors = list(coupling_results["signal_source"].values)

        # For chosen pathway(s), get the ligands/ligand:receptor pairs that are members:
        if isinstance(pathway, str):
            if self.mod_type == "lr":
                pathway_ligands = list(self.lr_db.loc[self.lr_db["pathway"] == pathway, "from"].values)
                pathway_receptors = list(self.lr_db.loc[self.lr_db["pathway"] == pathway, "to"].values)
                all_pathway_lr = [f"{l}:{r}" for l, r in zip(pathway_ligands, pathway_receptors)]
                matched_pathway_lr = list(set(all_pathway_lr).intersection(set(predictors)))
                # Make sure the pathway has at least three ligands or ligand-receptor pairs after processing-
                # otherwise, there is not enough measured signal in the model to constitute a pathway:
                if len(matched_pathway_lr) < 3:
                    raise ValueError(
                        "The chosen pathway has too little representation (<= 3 interactions) in the modeling "
                        "features. Specify a different pathway or fit an additional model."
                    )

                matched_pathway_coupling_scores = [
                    coupling_results.sel(signal_source=lr).values for lr in matched_pathway_lr
                ]
                matched_pathway_coupling_significance = [
                    coupling_significance.sel(signal_source=lr).values for lr in matched_pathway_lr
                ]

            elif self.mod_type == "ligand":
                pathway_ligands = list(self.lr_db.loc[self.lr_db["pathway"] == pathway, "from"].values)
                matched_pathway_ligands = list(set(pathway_ligands).intersection(set(predictors)))
                # Make sure the pathway has at least three ligands or ligand-receptor pairs after processing-
                # otherwise, there is not enough measured signal in the model to constitute a pathway:
                if len(matched_pathway_ligands) < 3:
                    raise ValueError(
                        "The chosen pathway has too little representation (<= 3 interactions) in the modeling "
                        "features. Specify a different pathway or fit an additional model."
                    )

                matched_pathway_coupling_scores = [
                    coupling_results.sel(signal_source=ligand).values for ligand in matched_pathway_ligands
                ]
                matched_pathway_coupling_significance = [
                    coupling_significance.sel(signal_source=ligand).values for ligand in matched_pathway_ligands
                ]

            # Compute mean over pathway:
            stack = np.hstack(matched_pathway_coupling_scores)
            pathway_coupling = np.mean(stack, axis=0)

            # Convert to DataFrame:
            pathway_coupling_df = pd.DataFrame(
                pathway_coupling,
                index=list(coupling_results["target"].values),
                columns=list(coupling_results["celltype_pair"].values),
            )

            # And pathway score significance- if the majority of pathway L:R pairs are significant, then consider
            # the pathway significant for the given cell type pair + target combo:
            stack = np.hstack(matched_pathway_coupling_significance)
            pathway_coupling_significance = np.mean(stack, axis=0)
            pathway_coupling_significance[pathway_coupling_significance >= 0.5] = True

            # Convert to DataFrame:
            pathway_coupling_significance_df = pd.DataFrame(
                pathway_coupling_significance,
                index=list(coupling_results["target"].values),
                columns=list(coupling_results["celltype_pair"].values),
            )

            # Store in dictionary:
            pathway_coupling = {pathway: pathway_coupling_df}
            pathway_coupling_significance = {pathway: pathway_coupling_significance_df}

        elif isinstance(pathway, list):
            pathway_coupling = {}
            pathway_coupling_significance = {}

            for p in pathway:
                if self.mod_type == "lr":
                    pathway_ligands = list(self.lr_db.loc[self.lr_db["pathway"] == p, "from"].values)
                    pathway_receptors = list(self.lr_db.loc[self.lr_db["pathway"] == p, "to"].values)
                    all_pathway_lr = [f"{l}:{r}" for l, r in zip(pathway_ligands, pathway_receptors)]
                    matched_pathway_lr = list(set(all_pathway_lr).intersection(set(predictors)))
                    # Make sure the pathway has at least three ligands or ligand-receptor pairs after processing-
                    # otherwise, there is not enough measured signal in the model to constitute a pathway:
                    if len(matched_pathway_lr) < 3:
                        raise ValueError(
                            "The chosen pathway has too little representation (<= 3 interactions) in the modeling "
                            "features. Specify a different pathway or fit an additional model."
                        )

                    matched_pathway_coupling_scores = [
                        coupling_results.sel(signal_source=lr).values for lr in matched_pathway_lr
                    ]
                    matched_pathway_coupling_significance = [
                        coupling_significance.sel(signal_source=lr).values for lr in matched_pathway_lr
                    ]

                elif self.mod_type == "ligand":
                    pathway_ligands = list(self.lr_db.loc[self.lr_db["pathway"] == p, "from"].values)
                    matched_pathway_ligands = list(set(pathway_ligands).intersection(set(predictors)))
                    # Make sure the pathway has at least three ligands or ligand-receptor pairs after processing-
                    # otherwise, there is not enough measured signal in the model to constitute a pathway:
                    if len(matched_pathway_ligands) < 3:
                        raise ValueError(
                            "The chosen pathway has too little representation (<= 3 interactions) in the modeling "
                            "features. Specify a different pathway or fit an additional model."
                        )

                    matched_pathway_coupling_scores = [
                        coupling_results.sel(signal_source=ligand).values for ligand in matched_pathway_ligands
                    ]
                    matched_pathway_coupling_significance = [
                        coupling_significance.sel(signal_source=ligand).values for ligand in matched_pathway_ligands
                    ]

                # Compute mean over pathway:
                stack = np.hstack(matched_pathway_coupling_scores)
                pathway_coupling = np.mean(stack, axis=0)

                # Convert to DataFrame:
                pathway_coupling_df = pd.DataFrame(
                    pathway_coupling,
                    index=list(coupling_results["target"].values),
                    columns=list(coupling_results["celltype_pair"].values),
                )

                # And pathway score significance- if the majority of pathway L:R pairs are significant, then consider
                # the pathway significant for the given cell type pair + target combo:
                stack = np.hstack(matched_pathway_coupling_significance)
                pathway_coupling_significance = np.mean(stack, axis=0)
                pathway_coupling_significance[pathway_coupling_significance >= 0.5] = True

                # Convert to DataFrame:
                pathway_coupling_significance_df = pd.DataFrame(
                    pathway_coupling_significance,
                    index=list(coupling_results["target"].values),
                    columns=list(coupling_results["celltype_pair"].values),
                )

                # Store in dictionary:
                pathway_coupling[pathway] = pathway_coupling_df
                pathway_coupling_significance[pathway] = pathway_coupling_significance_df

        return pathway_coupling, pathway_coupling_significance

    # ---------------------------------------------------------------------------------------------------
    # Visualization functions
    # ---------------------------------------------------------------------------------------------------
