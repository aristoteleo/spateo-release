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
import re
import sys
from collections import Counter
from multiprocessing import Pool
from typing import List, Literal, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
from mpi4py import MPI
from scipy.stats import pearsonr, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from statsmodels.gam.smooth_basis import BSplines

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.configuration import SKM
from spateo.tools.cluster.leiden import calculate_leiden_partition
from spateo.tools.gene_expression_variance import (
    compute_gene_groups_p_val,
    get_highvar_genes_sparse,
)
from spateo.tools.ST_modeling.MuSIC import MuSIC
from spateo.tools.ST_modeling.regression_utils import (
    DE_GAM_test,
    fit_DE_GAM,
    multitesting_correction,
    permutation_testing,
    wald_test,
)


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
        if self.search_bw is None:
            self.search_bw = self.n_neighbors
            self.bw_fixed = False
        self.k = self.arg_retrieve.top_k_receivers

        # Coefficients:
        if not self.set_up:
            self.logger.info("Model has not yet been set up, running :func `SWR._set_up_model()` now...")
            self._set_up_model()

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
        self.no_cell_type_markers = self.arg_retrieve.no_cell_type_markers
        self.compute_pathway_effect = self.arg_retrieve.compute_pathway_effect
        self.diff_sending_or_receiving = self.arg_retrieve.diff_sending_or_receiving
        self.no_cell_type_markers = self.arg_retrieve.no_cell_type_markers
        self.n_GAM_points = self.arg_retrieve.n_GAM_points
        self.n_leiden_pcs = self.arg_retrieve.n_leiden_pcs
        self.n_leiden_neighbors = self.arg_retrieve.n_leiden_neighbors
        self.leiden_resolution = self.arg_retrieve.leiden_resolution
        self.top_n_DE_genes = self.arg_retrieve.top_n_DE_genes

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
            effect_potential: Array of shape [n_samples, n_samples]; proxy for the "signaling effect potential" with
                respect to a particular target gene between each sender-receiver pair of cells.
            normalized_effect_potential_sum_sender: Array of shape [n_samples,]; for each sending cell, the sum of the
                signaling potential to all receiver cells for a given target gene, normalized between 0 and 1.
            normalized_effect_potential_sum_receiver: Array of shape [n_samples,]; for each receiving cell, the sum of
                the signaling potential from all sender cells for a given target gene, normalized between 0 and 1.
        """

        if self.mod_type == "receptor":
            raise ValueError("Sent potential is not defined for receptor models.")

        if target is None and self.target_for_downstream is not None:
            target = self.target_for_downstream
        else:
            self.logger.info("Target gene not provided for :func `get_effect_potential`. Using first target listed.")
            target = list(self.coeffs.keys())[0]

        # Check for valid inputs:
        if ligand is None and self.ligand_for_downstream is not None:
            ligand = self.ligand_for_downstream
        else:
            if self.mod_type == "ligand" or self.mod_type == "lr":
                raise ValueError("Must provide ligand for ligand models.")

        if receptor is None and self.receptor_for_downstream is not None:
            receptor = self.receptor_for_downstream
        else:
            if self.mod_type == "lr":
                raise ValueError("Must provide receptor for lr models.")

        if sender_cell_type is None and self.sender_ct_for_downstream is not None:
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
                idx = coeffs.columns.index(ligand_coeff_label)
            elif self.mod_type == "lr":
                lr_coeff_label = f"b_{ligand}-{receptor}"
                idx = coeffs.columns.index(lr_coeff_label)

            coeff = coeffs.iloc[:, idx].values.reshape(1, -1)
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
            # Weight each column by the coefficient magnitude and finally by the indicator for expression/no expression
            # of the target and store as sparse array:
            sig_interm = sent_potential.multiply(coeff)
            sig_interm.eliminate_zeros()
            effect_potential = sig_interm.multiply(target_indicator)
            effect_potential.eliminate_zeros()

        effect_potential_sum_sender = np.array(effect_potential.sum(axis=1)).reshape(-1)
        normalized_effect_potential_sum_sender = (effect_potential_sum_sender - np.min(effect_potential_sum_sender)) / (
            np.max(effect_potential_sum_sender) - np.min(effect_potential_sum_sender)
        )
        effect_potential_sum_receiver = np.array(effect_potential.sum(axis=0)).reshape(-1)
        normalized_effect_potential_sum_receiver = (
            effect_potential_sum_receiver - np.min(effect_potential_sum_receiver)
        ) / (np.max(effect_potential_sum_receiver) - np.min(effect_potential_sum_receiver))

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
            if len(valid_lr_combos) < 2:
                raise ValueError(
                    f"Pathway effect potential computation for pathway {pathway} is unsuitable for this model, "
                    f"since there are fewer than two valid ligand-receptor pairs in the pathway that were "
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

    def sender_receiver_effect_deg_detection(
        self,
        target: Optional[str] = None,
        diff_sending_or_receiving: Literal["sending", "receiving"] = "sending",
        ligand: Optional[str] = None,
        receptor: Optional[str] = None,
        pathway: Optional[str] = None,
        sender_cell_type: Optional[str] = None,
        receiver_cell_type: Optional[str] = None,
        no_cell_type_markers: bool = False,
    ):
        """Computes differential expression of genes in cells with high or low sent signaling effect potential,
        or differential expression of genes in cells with high or low received signaling effect potential.

        Args:
            target: Target to use for differential expression analysis. If None, will use the first listed target.
            diff_sending_or_receiving: Whether to compute differential expression of genes in cells with high or low
                sending effect potential ("sending cells") or high or low receiving effect potential ("receiving
                cells").
            ligand: Ligand to use for differential expression analysis. Will take precedent over sender/receiver cell
                type if also provided.
            receptor: Optional receptor to use for differential expression analysis. Needed if
                'diff_sending_or_receiving' is 'receiving'. Will take precedent over sender/receiver cell type if
                also provided.
            pathway: Optional pathway to use for differential expression analysis. Will use ligands and receptors in
                these pathways to collectively compute signaling potential score. Will take precedent over
                ligand/receptor and sender/receiver cell type if provided.
            sender_cell_type: Sender cell type to use for differential expression analysis.
            receiver_cell_type: Receiver cell type to use for differential expression analysis.
            no_cell_type_markers: Whether to consider cell type markers during differential expression testing,
                as these are least likely to be interesting patterns. Defaults to False, and if True will first
                perform differential expression testing for each cell type group, removing significant results.

        Returns:
            GAM_adata: AnnData object where each entry in .var contains a gene that has been modeled, with results of
                statistical testing added
            bs: BSplines object containing information about the basis splines
        """

        if diff_sending_or_receiving not in ["sending", "receiving"]:
            raise ValueError(
                f"diff_sending_or_receiving must be either 'sending' or 'receiving', not {diff_sending_or_receiving}."
            )

        if pathway is None and ligand is None and sender_cell_type is None:
            raise ValueError("Must provide at least one pathway, ligand, or sender_cell_type.")

        if pathway is not None:
            if self.mod_type == "lr":
                send_key = f"norm_sum_sent_effect_potential_{pathway}_lr_for_{target}"
                receive_key = f"norm_sum_received_effect_potential_{pathway}_lr_for_{target}"
            elif self.mod_type == "ligand":
                send_key = f"norm_sum_sent_effect_potential_{pathway}_ligands_for_{target}"
                receive_key = f"norm_sum_received_effect_potential_{pathway}_ligands_for_{target}"
            else:
                raise ValueError(f"No signaling effects with mod_type {self.mod_type}.")

            # Check for already computed pathway effect potential, and if not existing, compute it:
            if send_key not in self.adata.obsm_keys():
                self.logger.info(
                    f"Ligand-receptor effect potential for {target} via {pathway}. " f"Computing effect potential now."
                )
                _, _, _ = self.get_pathway_potential(pathway, target, spatial_weights=None, store_summed_potential=True)

            # Key for AnnData storage of the source signal:
            if diff_sending_or_receiving == "sending":
                source_key = f"Sent potential {pathway}"
            else:
                source_key = f"Received potential {pathway}"

        elif ligand is not None:
            if receptor is not None:
                if self.mod_type != "lr":
                    raise ValueError(
                        f"With mod_type {self.mod_type}, no receptors can be specified because the "
                        f"initial model did not use them."
                    )

                send_key = f"norm_sum_sent_effect_potential_{ligand}_for_{target}_via_{receptor}"
                receive_key = f"norm_sum_received_effect_potential_{ligand}_for_{target}_via_{receptor}"
                # Check for already computed ligand-receptor effect potential, and if not existing, compute it:
                if send_key not in self.adata.obsm_keys():
                    self.logger.info(
                        f"Ligand-receptor effect potential for {target} via {ligand}-{receptor}. "
                        f"Computing effect potential now."
                    )
                    _, _, _ = self.get_effect_potential(
                        ligand, receptor, spatial_weights=None, store_summed_potential=True
                    )

                # Key for AnnData storage of the source signal:
                if diff_sending_or_receiving == "sending":
                    source_key = f"Sent potential {ligand}-via-{receptor}"
                else:
                    source_key = f"Received potential {ligand}-via-{receptor}"

            else:
                if self.mod_type != "ligand":
                    raise ValueError(
                        f"Only ligand was provided to compute effect-associated DEGs, but the initial "
                        f"{self.mod_type} model did not use ligands or uses ligands in addition to "
                        f"receptors."
                    )

                send_key = f"norm_sum_sent_effect_potential_{ligand}_for_{target}"
                receive_key = f"norm_sum_received_effect_potential_from_{ligand}_for_{target}"
                # Check for already computed ligand effect potential, and if not existing, compute it:
                self.logger.info(
                    f"Ligand-receptor effect potential for {target} via {ligand}. " f"Computing effect potential now."
                )
                if send_key not in self.adata.obsm_keys():
                    _, _, _ = self.get_effect_potential(
                        ligand, target, spatial_weights=None, store_summed_potential=True
                    )

                # Key for AnnData storage of the source signal:
                if diff_sending_or_receiving == "sending":
                    source_key = f"Sent potential {ligand}"
                else:
                    source_key = f"Received potential {ligand}"

        elif sender_cell_type is not None:
            if self.mod_type != "niche":
                raise ValueError(
                    f"Only sender_cell_type was provided to compute effect-associated DEGs, but the "
                    f"initial {self.mod_type} model does not use cell type identity."
                )

            if receiver_cell_type is not None:
                send_key = f"norm_sum_sent_{sender_cell_type}_effect_potential_to_{receiver_cell_type}_for_{target}"
                receive_key = (
                    f"norm_sum_{receiver_cell_type}_received_effect_potential_from_{sender_cell_type}_for_" f"{target}"
                )

                # Key for AnnData storage of the source signal:
                if diff_sending_or_receiving == "sending":
                    source_key = f"Sent potential {sender_cell_type}"
                else:
                    source_key = f"Received potential {sender_cell_type}"

            else:
                send_key = f"norm_sum_sent_effect_potential_{sender_cell_type}_for_{target}"
                receive_key = f"norm_sum_received_effect_potential_from_{sender_cell_type}_for_{target}"

                # Key for AnnData storage of the source signal:
                if diff_sending_or_receiving == "sending":
                    source_key = f"Sent potential {sender_cell_type}-via-{receiver_cell_type}"
                else:
                    source_key = f"Received potential {sender_cell_type}-via-{receiver_cell_type}"

        if diff_sending_or_receiving == "sending":
            effect_potential = self.adata.obsm[send_key]
        else:
            effect_potential = self.adata.obsm[receive_key]

        # For sending analyses, restrict the differential analysis to transcription factors- these indicate ultimate
        # upstream regulators of the signaling:
        if diff_sending_or_receiving == "sending":
            if self.cci_dir is None:
                raise ValueError("With 'diff_sending_or_receiving' set to 'sending', please provide :attr `cci_dir`.")

            if self.species == "human":
                grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
                rna_bp_db = pd.read_csv(os.path.join(self.cci_dir, "human_RBP_db.csv"), index_col=0)
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            elif self.species == "mouse":
                grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
                rna_bp_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_RBP_db.csv"), index_col=0)
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)

            self.logger.info(
                "Selecting transcription factors and RNA-binding proteins for analysis of differential " "expression."
            )
            all_TFs = list(grn.columns)
            all_RBPs = list(rna_bp_db["Gene_Name"].values)
            # First filter to genes that were actually measured in the data:
            all_TFs = [tf for tf in all_TFs if tf in self.adata.var_names]
            all_RBPs = [r for r in all_RBPs if r in self.adata.var_names]

            # Further subset list of TFs to those that are implicated in signaling patterns of interest and also
            # are expressed in at least n% of the cells that are nonzero for sending potential (use the user input
            # 'target_expr_threshold'):
            nz_sending = np.any(effect_potential != 0, axis=1)
            adata_subset = self.adata[nz_sending, :]
            n_cells_threshold = int(self.target_expr_threshold * adata_subset.n_obs)

            # Get TFs known to regulate the ligand/pathway of interest:
            if pathway is not None:
                lr_db_subset = lr_db[lr_db["pathway"] == pathway]
                all_senders = list(set(lr_db_subset["from"]))

                sender_regulators = list(grn.columns[grn.loc[all_senders].eq(1).any()])
            elif ligand is not None:
                sender_regulators = list(grn.columns[grn.loc[ligand].eq(1).any()])
            else:
                # Set sender regulators to all TFs:
                sender_regulators = all_TFs

            sender_regulators.extend(all_RBPs)

            if scipy.sparse.issparse(self.adata.X):
                nnz_counts = np.array(adata_subset[:, sender_regulators].X.getnnz(axis=0)).flatten()
            else:
                nnz_counts = np.array(self.adata[:, sender_regulators].X.getnnz(axis=0)).flatten()

            to_keep = list(np.array(sender_regulators)[nnz_counts >= n_cells_threshold])
            counts = self.adata[:, to_keep].X

        else:
            # For receiving analyses, identify gene expression signatures associated with the signaling effects:
            # Filter to the top 3000 highly variable genes, and exclude cell type markers and genes without
            # sufficient expression levels:
            (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(self.adata.X, numgenes=3000)
            high_variance_genes_filter = list(self.adata.var.index[gene_counts_stats.high_var.values])
            adata_hvg = self.adata[:, high_variance_genes_filter]

            nz_sending = np.any(effect_potential != 0, axis=1)
            adata_subset = adata_hvg[nz_sending, :]
            n_cells_threshold = int(self.target_expr_threshold * adata_subset.n_obs)

            if scipy.sparse.issparse(self.adata.X):
                nnz_counts = np.array(adata_subset.X.getnnz(axis=0)).flatten()
            else:
                nnz_counts = np.array(adata_subset.X.getnnz(axis=0)).flatten()

            genes_to_keep = list(np.array(adata_subset.var_names)[nnz_counts >= n_cells_threshold])

            if no_cell_type_markers:
                # Remove cell type markers (more likely to be reflective than responsive to signaling) using a series of
                # binomial models- each column is the p-value for a particular cell type group compared to all other groups
                p_values = pd.DataFrame(
                    index=adata_hvg.var_names,
                    columns=[
                        f"p-value {cat}_v_all_other_groups" for cat in self.adata.obs[self.group_key].cat.categories
                    ],
                )

                with Pool() as pool:
                    for cat in self.adata.obs[self.group_key].cat.categories:
                        group1 = self.adata[self.adata.obs[self.group_key] == cat]
                        group2 = self.adata[self.adata.obs[self.group_key] != cat]

                        # For each gene, compute Mann-Whitney U test:
                        gene_p_vals = pool.starmap(
                            compute_gene_groups_p_val, [(gene, group1, group2) for gene in genes_to_keep]
                        )
                        genes, pvals = zip(*gene_p_vals)
                        qvals = multitesting_correction(pvals, method="fdr_bh")

                        for gene, qval in zip(genes, qvals):
                            p_values.loc[gene, f"p-value {cat}_v_all_other_groups"] = qval

                significant_p_values = p_values[p_values < 0.05]
                # Collect significant genes for each column (group)
                significant_genes_per_group = [
                    significant_p_values.index[significant_p_values[col].notna()].tolist()
                    for col in significant_p_values.columns
                ]
                # Flatten the list of lists and take the set to get unique significant genes across all groups
                significant_markers = list(set([gene for sublist in significant_genes_per_group for gene in sublist]))
                genes_to_keep = list(set(genes_to_keep) - set(significant_markers))

            counts = self.adata[:, genes_to_keep].X

        GAM_adata, bs = fit_DE_GAM(
            counts,
            var=effect_potential,
            genes=genes_to_keep,
            cells=self.sample_names,
        )

        GAM_adata.uns["predictor_key"] = source_key

        # Compute q-values for each gene:
        GAM_adata = DE_GAM_test(GAM_adata, bs)
        return GAM_adata, bs

    def group_sender_receiver_effect_degs(
        self,
        GAM_adata: anndata.AnnData,
        bs_obj: BSplines,
        offset: Optional[np.ndarray] = None,
        n_points: int = 50,
        num_pcs: int = 10,
        num_neighbors: int = 5,
        leiden_resolution: float = 1.0,
        q_val_cutoff: float = 0.05,
        top_n_genes: Optional[int] = None,
    ):
        """Given differential expression of genes in cells with high or low sent signaling effect potential,
        or differential expression of genes in cells with high or low received signaling effect potential,
        cluster genes by these observed patterns.

        Args:
            GAM_adata: AnnData object containing results of GAM models
            bs_obj: BSplines object containing information about the basis functions used for the model
            offset: Optional offset term of shape [n_samples, ]
            n_points: Number of points to sample when constructing the linear predictor, to enable evaluation of the
                fitted GAM for each model
            num_pcs: Number of PCs when performing PCA to cluster differential gene expression patterns
            num_neighbors: Number of neighbors when constructing the KNN graph for leiden clustering
            leiden_resolution: Resolution parameter for leiden clustering
            q_val_cutoff: Adjusted p-value cutoff to select genes to include in clustering analyses
            top_n_genes: Optional integer- if provided, cluster only the top n genes by Wald statistic that represent
                the genes that are most positively and most negatively enriched in relation to signaling effect
                potential

        Returns:
            GAM_adata: AnnData object where each entry in .var contains a gene that has been modeled, with Leiden
                partitioning results added. Note that this may be a subset of the original AnnData object
        """
        if "predictor_key" not in GAM_adata.uns_keys():
            raise RuntimeError(
                "Must run :func `sender_receiver_effect_deg_detection` before running :func "
                "`group_sender_receiver_effect_degs`."
            )

        design_matrix = GAM_adata.obsm["var"]

        # Sample points for which to construct the linear predictor:
        # Min and max values for predictor:
        n_curves = design_matrix.shape[1]
        min_val = np.min(design_matrix)
        max_val = np.max(design_matrix)

        if n_curves == 1:
            data_points = np.linspace(min_val, max_val, num=n_points).reshape(-1, 1)
            # Compute linear predictor at the selected points:
            exog_smooth_interp = bs_obj.transform(data_points)
            exog_smooth_pred = np.hstack((data_points, exog_smooth_interp))
        else:
            raise RuntimeError("Operability with multiple predictors not yet implemented.")

        # Get genes that are below p-value cutoff:
        adata_subset = GAM_adata[:, GAM_adata.var["qvals"] < q_val_cutoff]

        # To predict expression given the linear predictor and the coefficients:
        betaAll = GAM_adata.uns["beta"]

        scaled_yhat_df = pd.DataFrame(index=adata_subset.var_names, columns=adata_subset.obs_names)

        for gene in adata_subset.var_names:
            if GAM_adata.var["converged"][gene]:
                beta_gene = betaAll.loc[gene].values
                yhat = np.dot(exog_smooth_pred, beta_gene)
                if GAM_adata.uns["family"] == "nb" or GAM_adata.uns["family"] == "poisson":
                    yhat = np.exp(yhat)
                if offset is not None:
                    yhat += offset

                scaled_yhat_df[gene] = yhat

        # Scale the overall predicted expression values for PCA:
        scaler = StandardScaler()
        scaled_yhat_df = scaler.fit_transform(scaled_yhat_df.values)

        GAM_adata.varm["pred_y"] = scaled_yhat_df

        if top_n_genes is not None:
            # Check if there are enough genes to cluster:
            top_n_genes = min(top_n_genes, scaled_yhat_df.shape[1])

            # Get top n genes by Wald statistic:
            top_n_genes = GAM_adata.var.sort_values("wald_stats").head(top_n_genes).index.tolist()
            adata_subset = adata_subset[:, top_n_genes]
            scaled_yhat_df = scaled_yhat_df[top_n_genes]
        else:
            # Check if there are too many genes to easily represent on a plot:
            if scaled_yhat_df.shape[1] > 200:
                top_n_genes = 200

                # Get top n genes by Wald statistic:
                top_n_genes = GAM_adata.var.sort_values("wald_stats").head(top_n_genes).index.tolist()
                adata_subset = adata_subset[:, top_n_genes]
                scaled_yhat_df = scaled_yhat_df[top_n_genes]

        pca_obj = PCA(n_components=num_pcs, svd_solver="full")
        x_pca = pca_obj.fit_transform(scaled_yhat_df.values)
        cluster_labels = calculate_leiden_partition(
            x_pca, num_neighbors=num_neighbors, resolution=leiden_resolution, graph_type="distance"
        )

        adata_subset.var["cluster"] = cluster_labels
        adata_subset.uns["__type"] = "UMI"
        return adata_subset

    def calc_and_group_sender_receiver_effect_degs(
        self,
        target: Optional[str] = None,
        diff_sending_or_receiving: Literal["sending", "receiving"] = "sending",
        ligand: Optional[str] = None,
        receptor: Optional[str] = None,
        pathway: Optional[str] = None,
        sender_cell_type: Optional[str] = None,
        receiver_cell_type: Optional[str] = None,
        no_cell_type_markers: Optional[bool] = None,
        n_points: int = 50,
        num_pcs: int = 10,
        num_neighbors: int = 5,
        leiden_resolution: float = 0.5,
        top_n_genes: Optional[int] = None,
    ):
        """Wrapper to find differential expression of genes in cells with high or low sent signaling effect potential,
        or differential expression of genes in cells with high or low received signaling effect potential, and
        cluster genes by these observed patterns.

        Args:
            target: Target to use for differential expression analysis. If None, will use the first listed target.
            diff_sending_or_receiving: Whether to compute differential expression of genes in cells with high or low
                sending effect potential ("sending cells") or high or low receiving effect potential ("receiving
                cells").
            ligand: Ligand to use for differential expression analysis. Will take precedent over sender/receiver cell
                type if also provided.
            receptor: Optional receptor to use for differential expression analysis. Needed if
                'diff_sending_or_receiving' is 'receiving'. Will take precedent over sender/receiver cell type if
                also provided.
            pathway: Optional pathway to use for differential expression analysis. Will use ligands and receptors in
                these pathways to collectively compute signaling potential score. Will take precedent over
                ligand/receptor and sender/receiver cell type if provided.
            sender_cell_type: Sender cell type to use for differential expression analysis.
            receiver_cell_type: Receiver cell type to use for differential expression analysis.
            no_cell_type_markers: Whether to consider cell type markers during differential expression testing,
                as these are least likely to be interesting patterns. Defaults to False, and if True will first
                perform differential expression testing for each cell type group, removing significant results.
            n_points: Number of points to sample when constructing the linear predictor, to enable evaluation of the
                fitted GAM for each model
            num_pcs: Number of PCs when performing PCA to cluster differential gene expression patterns
            num_neighbors: Number of neighbors when constructing the KNN graph for leiden clustering
            leiden_resolution: Resolution parameter for leiden clustering
            top_n_genes: Optional integer- if provided, cluster only the top n genes by Wald statistic that represent
                the genes that are most positively and most negatively enriched in relation to signaling effect
                potential
        """

        # Input checks- if not manually provided, check argparse inputs (make sure an input was given from somewhere,
        # though):
        if self.diff_sending_or_receiving is not None:
            diff_sending_or_receiving = self.diff_sending_or_receiving
        if self.n_GAM_points is not None:
            n_points = self.n_GAM_points
        if self.n_leiden_pcs is not None:
            num_pcs = self.n_leiden_pcs
        if self.n_leiden_neighbors is not None:
            num_neighbors = self.n_leiden_neighbors
        if self.leiden_resolution is not None:
            leiden_resolution = self.leiden_resolution
        if self.top_n_DE_genes is not None:
            top_n_genes = self.top_n_DE_genes
        if self.no_cell_type_markers is not None:
            no_cell_type_markers = self.no_cell_type_markers

        if ligand is None and self.ligand_for_downstream is not None:
            ligand = self.ligand_for_downstream
        else:
            if self.mod_type == "ligand" or self.mod_type == "lr":
                raise ValueError("Must provide ligand for ligand models.")

        if receptor is None and self.receptor_for_downstream is not None:
            receptor = self.receptor_for_downstream
        else:
            if self.mod_type == "lr":
                raise ValueError("Must provide receptor for lr models.")

        if sender_cell_type is None and self.sender_ct_for_downstream is not None:
            sender_cell_type = self.sender_ct_for_downstream
        else:
            if self.mod_type == "niche":
                raise ValueError("Must provide sender cell type for niche models.")

        if receiver_cell_type is None and self.receiver_ct_for_downstream is not None:
            receiver_cell_type = self.receiver_ct_for_downstream

        GAM_adata, bs_obj = self.sender_receiver_effect_deg_detection(
            target=target,
            diff_sending_or_receiving=diff_sending_or_receiving,
            ligand=ligand,
            receptor=receptor,
            pathway=pathway,
            sender_cell_type=sender_cell_type,
            receiver_cell_type=receiver_cell_type,
            no_cell_type_markers=no_cell_type_markers,
        )

        GAM_adata_subset = self.group_sender_receiver_effect_degs(
            GAM_adata=GAM_adata,
            bs_obj=bs_obj,
            n_points=n_points,
            num_pcs=num_pcs,
            num_neighbors=num_neighbors,
            leiden_resolution=leiden_resolution,
            top_n_genes=top_n_genes,
        )

        # Save final AnnData object- different naming convention depending on which signaling molecules or cell types
        # were supplied:
        if ligand is not None:
            if receptor is not None:
                GAM_adata_subset.write_h5ad(
                    os.path.join(os.path.dirname(self.adata_path), f"{ligand}-{receptor}_effect_on_{target}_deg.h5ad")
                )
            else:
                GAM_adata_subset.write_h5ad(
                    os.path.join(os.path.dirname(self.adata_path), f"{ligand}_effect_on_{target}_deg.h5ad")
                )
        elif pathway is not None:
            GAM_adata_subset.write_h5ad(
                os.path.join(os.path.dirname(self.adata_path), f"{pathway}_effect_on_{target}_deg.h5ad")
            )
        elif sender_cell_type is not None:
            if receiver_cell_type is not None:
                GAM_adata_subset.write_h5ad(
                    os.path.join(
                        os.path.dirname(self.adata_path),
                        f"{sender_cell_type}_effect_on_{target}_in_{receiver_cell_type}_deg.h5ad",
                    )
                )
            else:
                GAM_adata_subset.write_h5ad(
                    os.path.join(os.path.dirname(self.adata_path), f"{sender_cell_type}_effect_on_{target}_deg.h5ad")
                )

    def compute_cell_type_coupling(
        self,
        targets: Optional[Union[str, List[str]]] = None,
    ):
        """Generates heatmap of spatially differentially-expressed features for each pair of sender and receiver
        categories- if :attr `mod_type` is "niche", this directly averages the effects for each neighboring cell type
        for each observation. If :attr `mod_type` is "lr" or "ligand", this correlates cell type prevalence with the
        size of the predicted effect on downstream expression for each L:R pair.

        Args:
            targets: Optional string or list of strings to select targets from among the genes used to fit the model
                to compute signaling effects for. If not given, will use all targets.
            filter_targets: Whether to filter targets based on the :attr:`filter_target_threshold` value. This
                threshold is based on the Pearson correlation w/ the true expression and should be between 0 and 1.
            filter_target_threshold: Only used if 'filter_targets' is True. Threshold to use for filtering targets
                based on the Pearson correlation of the reconstruction w/ the true expression

        Returns:
            ct_coupling: 3D array summarizing cell type coupling in terms of effect on downstream expression
            ct_coupling_significance: 3D array summarizing significance of cell type coupling in terms of effect on
                downstream expression
        """

        # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver:
        spatial_weights = self._compute_all_wi(self.search_bw, bw_fixed=self.bw_fixed, exclude_self=True).toarray()

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
                    ligand = col.split(":")[0]
                    receptor = col.split(":")[1]

                    effect_potential, _, _ = self.get_effect_potential(
                        target=target, ligand=ligand, receptor=receptor, spatial_weights=spatial_weights
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
                            avg_effect_potential,
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
                            avg_effect_potential,
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


# NOTE TO SELF: ADD WRAPPER FUNC HERE THAT ALLOWS FOR MULTIPLE TARGETS, LIGANDS/RECEPTORS, ETC. TO BE SPECIFIED,
# AND THE DOWNSTREAM FUNCTION WILL RUN USING ALL OF THE PASSED VALUES
