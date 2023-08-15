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
import glob
import itertools
import math
import os
import re
from collections import Counter
from typing import List, Literal, Optional, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import rcParams
from mpi4py import MPI
from pysal import explore, lib
from scipy.stats import pearsonr, spearmanr, ttest_1samp, zscore
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import normalize

from ...configuration import config_spateo_rcParams
from ...logging import logger_manager as lm
from ...plotting.static.utils import save_return_show_fig_utils
from ..dimensionality_reduction import (
    find_optimal_n_umap_components,
    umap_conn_indices_dist_embedding,
)
from .MuSIC import MuSIC
from .regression_utils import multitesting_correction, permutation_testing, wald_test
from .SWR_mpi import define_spateo_argparse


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

        self.k = self.arg_retrieve.top_k_receivers

        # Coefficients:
        if not self.set_up:
            self.logger.info(
                "Running :func `SWR._set_up_model()` to organize predictors and targets for downstream "
                "analysis now..."
            )
            self._set_up_model()
            # self.logger.info("Finished preprocessing, getting fitted coefficients and standard errors.")

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
            columns = [col for col in coef.columns if col.startswith("b_") and "intercept" not in col]
            coef = coef[[columns]]
            coef = self.comm.bcast(coef, root=0)
            se = self.standard_errors[target]
            se = self.comm.bcast(se, root=0)

            # Parallelize computations over observations and features:
            local_p_values_all = np.zeros((len(self.x_chunk), self.n_features))

            # Compute p-values for local observations and features
            for i, obs_index in enumerate(self.x_chunk):
                for j in range(self.n_features):
                    if se.iloc[obs_index, j] == 0:
                        local_p_values_all[i, j] = 1
                    else:
                        local_p_values_all[i, j] = wald_test(coef.iloc[obs_index, j], se.iloc[obs_index, j])

            # Collate p-values from all processes:
            p_values_all = self.comm.gather(local_p_values_all, root=0)

            if self.comm.rank == 0:
                p_values_all = np.concatenate(p_values_all, axis=0)
                p_values_df = pd.DataFrame(p_values_all, index=self.sample_names, columns=columns)
                # Multiple testing correction for each observation:
                qvals = np.zeros_like(p_values_all)
                for i in range(p_values_all.shape[0]):
                    qvals[i, :] = multitesting_correction(
                        p_values_all[i, :], method=method, alpha=significance_threshold
                    )
                q_values_df = pd.DataFrame(qvals, index=self.sample_names, columns=columns)

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

        # Pearson and Spearman dictionary for all cells:
        pearson_dict = {}
        spearman_dict = {}
        # Pearson and Spearman dictionary for only the expressing subset of cells:
        nz_pearson_dict = {}
        nz_spearman_dict = {}

        for i, gene in enumerate(all_genes):
            y = self.adata[:, gene].X.toarray().reshape(-1)
            music_results_target = pred_vals[:, i]

            # Remove index of the largest predicted value (to mitigate sensitivity of these metrics to outliers):
            outlier_index = np.where(np.max(music_results_target))[0]
            music_results_target_to_plot = np.delete(music_results_target, outlier_index)
            y_plot = np.delete(y, outlier_index)

            # Indices where target is nonzero:
            nonzero_indices = y_plot != 0

            rp, _ = pearsonr(y_plot, music_results_target_to_plot)
            r, _ = spearmanr(y_plot, music_results_target_to_plot)

            rp_nz, _ = pearsonr(y_plot[nonzero_indices], music_results_target_to_plot[nonzero_indices])
            r_nz, _ = spearmanr(y_plot[nonzero_indices], music_results_target_to_plot[nonzero_indices])

            pearson_dict[gene] = rp
            spearman_dict[gene] = r
            nz_pearson_dict[gene] = rp_nz
            nz_spearman_dict[gene] = r_nz

        # Mean of diagnostic metrics:
        mean_pearson = sum(pearson_dict.values()) / len(pearson_dict.values())
        mean_spearman = sum(spearman_dict.values()) / len(spearman_dict.values())
        mean_nz_pearson = sum(nz_pearson_dict.values()) / len(nz_pearson_dict.values())
        mean_nz_spearman = sum(nz_spearman_dict.values()) / len(nz_spearman_dict.values())

        data = []
        for gene in pearson_dict.keys():
            data.append(
                {
                    "Gene": gene,
                    "Pearson coefficient": pearson_dict[gene],
                    "Spearman coefficient": spearman_dict[gene],
                    "Pearson coefficient (expressing cells)": nz_pearson_dict[gene],
                    "Spearman coefficient (expressing cells)": nz_spearman_dict[gene],
                }
            )
        # Color palette:
        colors = {
            "Pearson coefficient": "#FF7F00",
            "Spearmann coefficient": "#87CEEB",
            "Pearson coefficient (expressing cells)": "#0BDA51",
            "Spearmann coefficient (expressing cells)": "#FF6961",
        }
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
        plt.figure(figsize=(width, 6))
        plt.xticks(rotation="vertical")
        ax = sns.barplot(
            data=df,
            x="Gene",
            y="Spearman coefficient",
            palette=colors["Spearman coefficient"],
            edgecolor="black",
            dodge=True,
        )

        # Mean line:
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

        # Plot Pearson correlation barplot (expressing cells):
        plt.figure(figsize=(width, 6))
        plt.xticks(rotation="vertical")
        ax = sns.barplot(
            data=df,
            x="Gene",
            y="Pearson coefficient (expressing cells)",
            palette=colors["Pearson coefficient (expressing cells)"],
            edgecolor="black",
            dodge=True,
        )

        # Mean line:
        ax.axhline(mean_nz_pearson, color="black", linestyle=line_style, linewidth=line_thickness)

        # Update legend:
        legend_label = f"Mean: {mean_nz_pearson}"
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="black", linewidth=line_thickness, linestyle=line_style))
        labels.append(legend_label)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.title(f"Pearson correlation (expressing cells) {file_name}")
        plt.tight_layout()
        plt.show()

        # Plot Spearman correlation barplot (expressing cells):
        plt.figure(figsize=(width, 6))
        plt.xticks(rotation="vertical")
        ax = sns.barplot(
            data=df,
            x="Gene",
            y="Spearman coefficient (expressing cells)",
            palette=colors["Spearman coefficient (expressing cells)"],
            edgecolor="black",
            dodge=True,
        )

        # Mean line:
        ax.axhline(mean_nz_spearman, color="black", linestyle=line_style, linewidth=line_thickness)

        # Update legend:
        legend_label = f"Mean: {mean_nz_spearman}"
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="black", linewidth=line_thickness, linestyle=line_style))
        labels.append(legend_label)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.title(f"Spearman correlation (expressing cells) {file_name}")
        plt.tight_layout()
        plt.show()

    def visualize_enriched_interactions(
        self,
        target_subset: Optional[List[str]] = None,
        cell_types: Optional[List[str]] = None,
        metric: Literal["number", "proportion", "specificity", "mean", "fc", "fc_qvals"] = "fc",
        normalize: bool = True,
        plot_significant: bool = False,
        metric_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "Reds",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """Given the target gene of interest, identify interaction features that are enriched for particular targets.
        Visualized in heatmap form.

        Args:
            target_subset: List of targets to consider. If None, will use all targets used in model fitting.
            cell_types: Can be used to restrict the enrichment analysis to only cells of a particular type. If given,
                will search for cell types in "group_key" attribute from model initialization.
            metric: Metric to display on plot. For all plot variants, the color will be determined by a combination
            of the size & magnitude of the effect. Options:
                - "number": Number of cells for which the interaction is predicted to have nonzero effect
                - "proportion": Percentage of interactions predicted to have nonzero effect over the number of cells
                    that express each target.
                - "specificity": Number of target-expressing cells for which a particular interaction is predicted to
                    have nonzero effect over the total number of cells for which a particular interaction is
                    present in (including target-expressing and non-expressing cells). Essentially, measures the
                    degree to which an interaction is coupled to a particular target.
                - "mean": Average effect size over all target-expressing cells.
                - "fc": Fold change in mean expression of target-expressing cells with and without each specified
                    interaction. Way of inferring that interaction may actually be repressive rather than activatory.
                - "fc_qvals": Log-transformed significance of the fold change.
            normalize: Whether to minmax scale the metric values. If True, will apply this scaling over all elements
                of the array. Only used for 'metric' = "number", "proportion" or "specificity".
            plot_significant: Whether to include only significant predicted interactions in the plot and metric
                calculation.
            metric_threshold: Optional threshold for 'metric' used to filter plot elements. Any interactions below
                this threshold will not be color coded. Will use 0.05 by default. Should be between 0 and 1. For
                'metric' = "fc", this threshold will be interpreted as a distance from a fold-change of 1.
            cut_pvals: For metric = "fc_qvals", the q-values are log-transformed. Any log10-transformed q-value that is
                below this will be clipped to this value.
            fontsize: Size of font for x and y labels.
            figsize: Size of figure.
            cmap: Colormap to use for heatmap. If metric is "number", "proportion", "specificity", the bottom end of
                the range is 0. It is recommended to use a sequential colormap (e.g. "Reds", "Blues", "Viridis",
                etc.). For metric = "fc", if a divergent colormap is not provided, "seismic" will automatically be
                used.
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()
        # But set display DPI to 300:
        plt.rcParams["figure.dpi"] = 300

        # Check inputs:
        if metric not in ["number", "proportion", "specificity", "mean", "fc", "fc_qvals"]:
            raise ValueError(
                f"Unrecognized metric {metric}. Options are 'number', 'proportion', 'specificity', 'mean', "
                f"'fc' or 'fc_qvals'."
            )

        if cell_types is None:
            adata = self.adata.copy()
        else:
            adata = self.adata[self.adata.obs[self.group_key].isin(cell_types)].copy()

        all_targets = list(self.coeffs.keys())
        targets = all_targets if target_subset is None else target_subset
        feature_names = [feat for feat in self.feature_names if feat != "intercept"]
        df = pd.DataFrame(0, index=feature_names, columns=targets)
        # For metric = fold change, significance of the fold-change:
        if metric == "fc" or metric == "fc_qvals":
            df_pvals = pd.DataFrame(1, index=feature_names, columns=targets)
            # For fold-change, colormap should be divergent-
            if metric == "fc":
                diverging_colormaps = [
                    "PiYG",
                    "PRGn",
                    "BrBG",
                    "PuOr",
                    "RdGy",
                    "RdBu",
                    "RdYlBu",
                    "RdYlGn",
                    "Spectral",
                    "coolwarm",
                    "bwr",
                    "seismic",
                ]
                if cmap not in diverging_colormaps:
                    logger.info("For metric fold-change, colormap should be divergent: using 'seismic'.")
                    cmap = "seismic"
        if metric != "fc":
            sequential_colormaps = [
                "Blues",
                "BuGn",
                "BuPu",
                "GnBu",
                "Greens",
                "Greys",
                "Oranges",
                "OrRd",
                "PuBu",
                "PuBuGn",
                "PuRd",
                "Purples",
                "RdPu",
                "Reds",
                "YlGn",
                "YlGnBu",
                "YlOrBr",
                "YlOrRd",
                "afmhot",
                "autumn",
                "bone",
                "cool",
                "copper",
                "gist_heat",
                "gray",
                "hot",
                "pink",
                "spring",
                "summer",
                "winter",
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
            ]
            if cmap not in sequential_colormaps:
                logger.info(f"For metric {metric}, colormap should be sequential: using 'viridis'.")
                cmap = "viridis"

        if fontsize is None:
            fontsize = rcParams.get("font.size")
        if figsize is None:
            # Set figure size based on the number of interaction features and targets:
            m = len(feature_names) * 40 / 300
            n = len(targets) * 40 / 300
            figsize = (n, m)

        for target in self.coeffs.keys():
            # Get coefficients for this key
            coef = self.coeffs[target]
            columns = [col for col in coef.columns if col.startswith("b_") and "intercept" not in col]
            feat_names_target = [col.split("_")[1] for col in columns]

            # For fold-change, significance will be incorporated post-calculation:
            if plot_significant and metric != "fc":
                # Adjust coefficients array to include only the significant coefficients:
                # Try to load significance matrix, and if not found, compute it:
                try:
                    parent_dir = os.path.dirname(self.output_path)
                    is_significant_df = pd.read_csv(
                        os.path.join(parent_dir, "significance", f"{target}_is_significant.csv")
                    )
                except:
                    self.logger.info(
                        "Could not find significance matrix. Computing it now with the "
                        "Benjamini-Hochberg correction and significance threshold of 0.05..."
                    )
                    self.compute_coeff_significance()
                    parent_dir = os.path.dirname(self.output_path)
                    is_significant_df = pd.read_csv(
                        os.path.join(parent_dir, "significance", f"{target}_is_significant.csv")
                    )
                # Convolve coefficients with significance matrix:
                coef = coef * is_significant_df.values

            if metric == "number":
                # Compute number of nonzero interactions for each feature:
                n_nonzero_interactions = np.sum(coef != 0, axis=0)
                # Compute proportion:
                df.loc[feat_names_target, target] = n_nonzero_interactions
            elif metric == "proportion":
                # Compute total number of target-expressing cells, and the indices of target-expressing cells:
                target_expr_cells_indices = np.where(adata[:, target].X.toarray() != 0)[0]
                # Compute total number of target-expressing cells:
                n_target_expr_cells = len(target_expr_cells_indices)
                # Extract only the rows of coef that correspond to target-expressing cells:
                coef_target_expr = coef.iloc[target_expr_cells_indices, :]
                # Compute number of cells for which each interaction is inferred to be present from among the
                # target-expressing cells:
                n_nonzero_interactions = np.sum(coef_target_expr != 0, axis=0)
                # Compute proportion:
                df.loc[feat_names_target, target] = n_nonzero_interactions / n_target_expr_cells
            elif metric == "specificity":
                # Compute total number of target-expressing cells, and the indices of target-expressing cells:
                target_expr_cells_indices = np.where(adata[:, target].X.toarray() != 0)[0]
                # Intersection of each interaction w/ target-expressing cells to determine the numerator for the
                # proportion:
                intersections = {}
                for col in columns:
                    nz_indices = np.where(coef[col].values != 0)[0]
                    intersections[col] = np.intersect1d(target_expr_cells_indices, nz_indices)
                intersections = np.array(list(intersections.values()))

                # Compute number of cells for which each interaction is inferred to be present to determine the
                # denominator for the proportion:
                n_nonzero_interactions = np.sum(coef != 0, axis=0)

                # Ratio of intersections to total number of nonzero values:
                df.loc[feat_names_target, target] = intersections / n_nonzero_interactions
            elif metric == "mean":
                df.loc[:, target] = np.mean(coef, axis=0)
            elif metric == "fc":
                # Get indices of zero effect and predicted positive effect:
                for col in columns:
                    feat = col.split("_")[1]
                    nz_effect_indices = np.where(coef[col].values != 0)[0]
                    zero_effect_indices = np.where(coef[col].values == 0)[0]

                    # Compute mean target expression for both subsets:
                    mean_target_nonzero = adata[nz_effect_indices, target].X.mean()
                    mean_target_zero = adata[zero_effect_indices, target].X.mean()

                    # Compute fold-change:
                    df.loc[feat, target] = mean_target_nonzero / mean_target_zero
                    # Compute p-value:
                    _, pval = scipy.stats.ranksums(
                        adata[nz_effect_indices, target].X.toarray(), adata[zero_effect_indices, target].X.toarray()
                    )
                    df_pvals.loc[feat, target] = pval

        # For metric = fold change, significance of the fold-change:
        if metric == "fc":
            # Multiple testing correction for each target using the Benjamin-Hochberg method:
            for col in df_pvals.columns:
                df_pvals[col] = multitesting_correction(df_pvals[col], method="fdr_bh")

            # Optionally, for plotting, retain fold-changes w/ significant corrected p-values:
            if plot_significant:
                df[df_pvals > 0.05] = 0.0

        # Plot preprocessing:
        # For metric = fold change q-values, compute the log-transformed fold change:
        if metric == "fc_qvals":
            df_log = np.log10(df_pvals.values)
            df_log[df_log < cut_pvals] = cut_pvals
            df_pvals = pd.DataFrame(df_log, index=df_pvals.index, columns=df_pvals.columns)
            # Adjust cmap such that the value typically corresponding to the minimum is the max- the max p-value is
            # the least significant:
            cmap = f"{cmap}_r"
            label = "$\log_{10}$ FDR-corrected pvalues"
            title = "Significance of target gene expression fold-change for each interaction"
        elif metric == "fc":
            # Set values below cutoff to 1 (no fold-change):
            df[(df < metric_threshold + 1) & (df > 1 - metric_threshold)] = 1.0
            vmax = np.max(np.abs(df.values))
            vmin = -vmax
            label = "Fold-change"
            title = "Target gene expression fold-change for each interaction"
        elif metric == "mean":
            df[np.abs(df) < metric_threshold] = 0.0
            vmax = np.max(df.values)
            vmin = np.min(df.values)
            label = "Mean effect size"
            title = "Mean effect size for each interaction on each target"
        elif metric in ["number", "proportion", "specificity"]:
            if normalize:
                df = (df - df.min()) / (df.max() - df.min() + 1e-8)
            df[df < metric_threshold] = 0.0
            vmax = np.max(np.abs(df.values))
            vmin = 0
            if metric == "number":
                label = "Number of cells"
                title = "Number of cells w/ predicted effect on target for each interaction"
            elif metric == "proportion":
                label = "Proportion of cells"
                title = "Proportion of cells w/ predicted effect on target for each interaction"
            elif metric == "specificity":
                label = "Specificity"
                title = "Exclusivity of predicted effect on target for each interaction"

        # Plot heatmap:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        if metric == "fc_qvals":
            qv = sns.heatmap(
                df_pvals,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": label, "location": "top"},
                cmap=cmap,
                vmin=cut_pvals,
                vmax=0,
                ax=ax,
            )

            # Outer frame:
            for _, spine in qv.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.75)
        else:
            m = sns.heatmap(
                df,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": label, "location": "top"},
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )

            # Outer frame:
            for _, spine in m.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.75)

        plt.xlabel("Target gene", fontsize=fontsize)
        plt.ylabel("Interaction", fontsize=fontsize)
        plt.title(title, fontsize=fontsize + 4)
        plt.tight_layout()

        # Use the saved name for the AnnData object to define part of the name of the saved file:
        base_name = os.path.basename(self.adata_path)
        adata_id = os.path.splitext(base_name)[0]
        prefix = f"{adata_id}_{metric}"
        # Save figure:
        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix=prefix,
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def moran_i_signaling_effects(
        self,
        targets: Optional[Union[str, List[str]]] = None,
        k: int = 10,
        weighted: Literal["kernel", "knn"] = "knn",
        permutations: int = 1000,
        n_jobs: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Computes spatial enrichment of signaling effects.

        Args:
            targets: Can optionally specify a subset of the targets to compute this on. If not given, will use all
                targets that were specified in model fitting.
            k: Number of k-nearest neighbors to use for Moran's I computation
            weighted: Whether to use a kernel-weighted or k-nearest neighbors approach to calculate spatial weights
            permutations: Number of random permutations for calculation of pseudo-p_values.
            n_jobs: Number of jobs to use for parallelization. If -1, all available CPUs are used. If 1 is given,
                no parallel computing code is used at all.

        Returns:
            signaling_moran_df: DataFrame with Moran's I scores for each target.
            signaling_moran_pvals: DataFrame with p-values for each Moran's I score.
        """
        if weighted != "kernel" and weighted != "knn":
            raise ValueError("Invalid argument given to 'weighted' parameter. Must be 'kernel' or 'knn'.")

        # Check inputs:
        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            elif not isinstance(targets, list):
                raise ValueError(f"targets must be a list or string, not {type(targets)}.")

        # Get Moran's I scores for each target:
        feat_names = [feat for feat in self.feature_names if feat != "intercept"]
        signaling_moran_df = pd.DataFrame(0, index=feat_names, columns=targets)
        signaling_moran_pvals = pd.DataFrame(1, index=feat_names, columns=targets)
        coords = self.coords
        if weighted == "kernel":
            kw = lib.weights.Kernel(coords, k, function="gaussian")
            W = lib.weights.W(kw.neighbors, kw.weights)
        else:
            kd = lib.cg.KDTree(coords)
            nw = lib.weights.KNN(kd, k)
            W = lib.weights.W(nw.neighbors, nw.weights)

        # Moran I for a single interaction:
        def _single(interaction, X_df, W, permutations):
            cur_X = X_df[interaction].values
            mbi = explore.esda.moran.Moran(cur_X, W, permutations=permutations, two_tailed=False)
            Moran_I = mbi.I
            p_value = mbi.p_sim
            statistics = mbi.z_sim
            return [Moran_I, p_value, statistics]

        for target in targets:
            # Get coefficients for this key
            coef = self.coeffs[target]
            effects = coef[[col for col in coef.columns if col.startswith("b_") and "intercept" not in col]]
            effects.columns = [col.split("_")[1] for col in effects.columns]

            # Parallel computation of Moran's I for all interactions for this target:
            res = Parallel(n_jobs)(delayed(_single)(interaction, effects, W, permutations) for interaction in effects)
            res = pd.DataFrame(res, columns=["moran_i", "moran_p_val", "moran_z"], index=effects.columns)
            res["moran_q_val"] = multitesting_correction(res["moran_p_val"], method="fdr_bh")
            signaling_moran_df.loc[effects.columns, target] = res["moran_i"]
            signaling_moran_pvals.loc[effects.columns, target] = res["moran_q_val"]

        return signaling_moran_df, signaling_moran_pvals

    def visualize_combinatorial_effects(self):
        """For future work!"""

    def get_effect_potential(
        self,
        target: Optional[str] = None,
        ligand: Optional[str] = None,
        receptor: Optional[str] = None,
        sender_cell_type: Optional[str] = None,
        receiver_cell_type: Optional[str] = None,
        spatial_weights_membrane_bound: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        spatial_weights_secreted: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        spatial_weights_niche: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
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

        if spatial_weights_membrane_bound is None:
            # Try to load spatial weights, else re-compute them:
            membrane_bound_path = os.path.join(
                os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_membrane_bound.npz"
            )
            try:
                spatial_weights_membrane_bound = scipy.sparse.load_npz(membrane_bound_path)
            except:
                # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver.
                # Note: this is the same process used in model setup.
                spatial_weights_membrane_bound = self._compute_all_wi(
                    self.n_neighbors_membrane_bound, bw_fixed=False, exclude_self=True, verbose=False
                )
        if spatial_weights_secreted is None:
            secreted_path = os.path.join(
                os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_secreted.npz"
            )
            try:
                spatial_weights_secreted = scipy.sparse.load_npz(secreted_path)
            except:
                spatial_weights_secreted = self._compute_all_wi(
                    self.n_neighbors_secreted, bw_fixed=False, exclude_self=True, verbose=False
                )

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

            # Check if ligand is membrane-bound or secreted:
            matching_rows = self.lr_db[self.lr_db["from"] == ligand]
            if (
                matching_rows["type"].str.contains("Secreted Signaling").any()
                or matching_rows["type"].str.contains("ECM-Receptor").any()
            ):
                spatial_weights = spatial_weights_secreted
            else:
                spatial_weights = spatial_weights_membrane_bound

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

            if spatial_weights_niche is None:
                niche_weights_path = os.path.join(
                    os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_niche.npz"
                )
                try:
                    spatial_weights_niche = scipy.sparse.load_npz(niche_weights_path)
                except:
                    spatial_weights_niche = self._compute_all_wi(
                        self.n_neighbors_secreted, bw_fixed=False, exclude_self=True, verbose=False
                    )

            sender_cell_type = self.cell_categories[sender_cell_type].values.reshape(-1, 1)
            # Get sending cells only of the specified type:
            sent_potential = spatial_weights_niche.multiply(sender_cell_type)
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
        spatial_weights_secreted: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        spatial_weights_membrane_bound: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        store_summed_potential: bool = True,
    ):
        """For each cell, computes the 'pathway effect potential', which is an aggregation of the effect potentials
        of all pathway member ligand-receptor pairs (or all pathway member ligands, for ligand-only models).

        Args:
            pathway: Name of pathway to compute pathway effect potential for.
            target: Optional string to select target from among the genes used to fit the model to compute signaling
                effects for. Note that this function takes only one target at a time. If not given, will take the
                first name from among all targets.
            spatial_weights_secreted: Optional pairwise spatial weights matrix for secreted factors
            spatial_weights_membrane_bound: Optional pairwise spatial weights matrix for membrane-bound factors
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
                    spatial_weights_secreted=spatial_weights_secreted,
                    spatial_weights_membrane_bound=spatial_weights_membrane_bound,
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
                    spatial_weights_secreted=spatial_weights_secreted,
                    spatial_weights_membrane_bound=spatial_weights_membrane_bound,
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
        # Try to load spatial weights for membrane-bound and secreted ligands, compute if not found:
        membrane_bound_path = os.path.join(
            os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_membrane_bound.npz"
        )
        secreted_path = os.path.join(
            os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_secreted.npz"
        )

        try:
            spatial_weights_membrane_bound = scipy.sparse.load_npz(membrane_bound_path)
            spatial_weights_secreted = scipy.sparse.load_npz(secreted_path)
        except:
            bw_mb = (
                self.n_neighbors_membrane_bound
                if self.distance_membrane_bound is None
                else self.distance_membrane_bound
            )
            bw_fixed = True if self.distance_membrane_bound is not None else False
            spatial_weights_membrane_bound = self._compute_all_wi(
                bw=bw_mb,
                bw_fixed=bw_fixed,
                exclude_self=True,
                verbose=False,
            )
            self.logger.info(f"Saving spatial weights for membrane-bound ligands to {membrane_bound_path}.")
            scipy.sparse.save_npz(membrane_bound_path, spatial_weights_membrane_bound)

            bw_s = self.n_neighbors_membrane_bound if self.distance_secreted is None else self.distance_secreted
            bw_fixed = True if self.distance_secreted is not None else False
            # Autocrine signaling is much easier with secreted signals:
            spatial_weights_secreted = self._compute_all_wi(
                bw=bw_s,
                bw_fixed=bw_fixed,
                exclude_self=False,
                verbose=False,
            )
            self.logger.info(f"Saving spatial weights for secreted ligands to {secreted_path}.")
            scipy.sparse.save_npz(secreted_path, spatial_weights_secreted)

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
                        set(
                            self.lr_db.loc[
                                (self.lr_db["from"] == ligand) & (self.lr_db["to"] == receptor), "pathway"
                            ].values
                        )
                    )
                    pathways.extend(col_pathways)
                elif self.mod_type == "ligand":
                    col_pathways = list(set(self.lr_db.loc[self.lr_db["from"] == query, "pathway"].values))
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
                        ) = self.get_pathway_potential(
                            target=target,
                            pathway=query,
                            spatial_weights_secreted=spatial_weights_secreted,
                            spatial_weights_membrane_bound=spatial_weights_membrane_bound,
                        )
                    else:
                        ligand = query.split(":")[0]
                        receptor = query.split(":")[1]
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_effect_potential(
                            target=target,
                            ligand=ligand,
                            receptor=receptor,
                            spatial_weights_secreted=spatial_weights_secreted,
                            spatial_weights_membrane_bound=spatial_weights_membrane_bound,
                        )
                else:
                    if compute_pathway_effect:
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_pathway_potential(
                            target=target,
                            pathway=query,
                            spatial_weights_secreted=spatial_weights_secreted,
                            spatial_weights_membrane_bound=spatial_weights_membrane_bound,
                        )
                    else:
                        (
                            effect_potential,
                            normalized_effect_potential_sum_sender,
                            normalized_effect_potential_sum_receiver,
                        ) = self.get_effect_potential(
                            target=target,
                            ligand=query,
                            spatial_weights_secreted=spatial_weights_secreted,
                            spatial_weights_membrane_bound=spatial_weights_membrane_bound,
                        )

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
    def CCI_deg_detection_setup(
        self,
        sender_or_receiver_degs: Literal["sender", "receiver"] = "sender",
        use_ligands: bool = True,
        use_receptors: bool = False,
        use_pathways: bool = False,
        use_cell_types: bool = False,
    ):
        """Computes differential expression signatures of cells with various levels of ligand expression.

        Args:
            sender_or_receiver_degs: Only makes a difference if 'use_pathways' or 'use_cell_types' is specified.
                Determines whether to compute DEGs for sender or receiver cells. If 'use_pathways' is True,
                the value of this argument will determine whether ligands or receptors are used to define the model.
                Note that in either case, differential expression of TFs, binding factors, etc. will be computed in
                association w/ ligands/receptors.
            use_ligands: Use ligand array for differential expression analysis. Will take precedent over
                sender/receiver cell type if also provided.
            use_pathways: Use pathway array for differential expression analysis. Will use ligands in these pathways
                to collectively compute signaling potential score. Will take precedent over sender cell types if
                also provided.
            use_cell_types: Use cell types to use for differential expression analysis. If given,
                will preprocess/construct the necessary components to initialize cell type-specific models.

        Returns:
            None
        """

        if use_ligands and use_receptors:
            self.logger.info(
                "Both 'use_ligands' and 'use_receptors' are given as function inputs. Note that "
                "'use_ligands' will take priority."
            )

        # Check if the array of additional molecules to query has already been created:
        parent_dir = os.path.dirname(self.output_path)
        file_name = os.path.basename(self.adata_path).split(".")[0]
        if use_ligands:
            targets_path = os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_all_ligands.txt")
        elif use_receptors:
            targets_path = os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_all_receptors.txt")
        elif use_pathways:
            targets_path = os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_all_pathways.txt")
        elif use_cell_types:
            targets_folder = os.path.join(parent_dir, "cci_deg_detection")

        if not os.path.exists(os.path.join(parent_dir, "cci_deg_detection")):
            os.makedirs(os.path.join(parent_dir, "cci_deg_detection"))

        # Check for existing processed downstream-task AnnData object:
        if os.path.exists(os.path.join(parent_dir, "cci_deg_detection", f"{file_name}.h5ad")):
            # Load files in case they are already existent:
            counts_plus = anndata.read_h5ad(os.path.join(parent_dir, "cci_deg_detection", f"{file_name}.h5ad"))
            if use_ligands or use_pathways or use_receptors:
                with open(targets_path, "r") as file:
                    targets = file.readlines()
            else:
                targets = pd.read_csv(targets_path, index_col=0)
            self.logger.info(
                "Found existing files for downstream analysis- skipping processing. Can proceed by running "
                ":func ~`self.CCI_sender_deg_detection()`."
            )
        else:
            self.logger.info("Generating and saving AnnData object for downstream analysis...")
            if self.cci_dir is None:
                raise ValueError("Please provide :attr `cci_dir`.")

            if self.species == "human":
                grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
                rna_bp_db = pd.read_csv(os.path.join(self.cci_dir, "human_RBP_db.csv"), index_col=0)
                cof_db = pd.read_csv(os.path.join(self.cci_dir, "human_cofactors.csv"), index_col=0)
                tf_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_TF_TF_db.csv"), index_col=0)
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            elif self.species == "mouse":
                grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
                rna_bp_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_RBP_db.csv"), index_col=0)
                cof_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_cofactors.csv"), index_col=0)
                tf_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_TF_TF_db.csv"), index_col=0)
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)

            # Subset GRN and other databases to only include TFs that are in the adata object:
            grn = grn[[col for col in grn.columns if col in self.adata.var_names]]
            cof_db = cof_db[[col for col in cof_db.columns if col in self.adata.var_names]]
            tf_tf_db = tf_tf_db[[col for col in tf_tf_db.columns if col in self.adata.var_names]]

            analyze_pathway_ligands = sender_or_receiver_degs == "sender" and use_pathways
            analyze_pathway_receptors = sender_or_receiver_degs == "receiver" and use_pathways
            analyze_celltype_ligands = sender_or_receiver_degs == "sender" and use_cell_types
            analyze_celltype_receptors = sender_or_receiver_degs == "receiver" and use_cell_types

            if use_ligands or analyze_pathway_ligands or analyze_celltype_ligands:
                database_ligands = list(set(lr_db["from"]))
                l_complexes = [elem for elem in database_ligands if "_" in elem]
                # Get individual components if any complexes are included in this list:
                ligand_set = [l for item in database_ligands for l in item.split("_")]
                ligand_set = [l for l in ligand_set if l not in self.adata.var_names]
            elif use_receptors or analyze_pathway_receptors or analyze_celltype_receptors:
                database_receptors = list(set(lr_db["to"]))
                r_complexes = [elem for elem in database_receptors if "_" in elem]
                # Get individual components if any complexes are included in this list:
                receptor_set = [r for item in database_receptors for r in item.split("_")]
                receptor_set = [r for r in receptor_set if r not in self.adata.var_names]

            signal = {}
            subsets = {}

            if use_ligands:
                if self.mod_type != "ligand" and self.mod_type != "lr":
                    raise ValueError(
                        "Sent signal from ligands cannot be used because the original specified 'mod_type' "
                        "does not use ligand expression."
                    )
                # Sent signal from ligand- use non-lagged version b/c the upstream factor effects on ligand expression
                # are an intrinsic property:
                # Some of the columns in the ligands dataframe may be complexes- identify the single genes that
                # compose these complexes:
                sig_df = self.ligands_expr_nonlag
                for col in sig_df.columns:
                    if col in l_complexes:
                        sig_df = sig_df.drop(col, axis=1)
                        for l in col.split("_"):
                            if scipy.sparse.issparse(self.adata.X):
                                gene_expr = self.adata[:, l].X.A
                            else:
                                gene_expr = self.adata[:, l].X
                            sig_df[l] = gene_expr

                signal["all"] = sig_df
                subsets["all"] = self.adata
            elif use_receptors:
                if self.mod_type != "receptor" and self.mod_type != "lr":
                    raise ValueError(
                        "Sent signal from receptors cannot be used because the original specified 'mod_type' "
                        "does not use receptor expression."
                    )
                # Received signal from receptor:
                # Some of the columns in the receptors dataframe may be complexes- identify the single genes that
                # compose these complexes:
                sig_df = self.receptors_expr
                for col in sig_df.columns:
                    if col in r_complexes:
                        sig_df = sig_df.drop(col, axis=1)
                        for r in col.split("_"):
                            if scipy.sparse.issparse(self.adata.X):
                                gene_expr = self.adata[:, r].X.A
                            else:
                                gene_expr = self.adata[:, r].X
                            sig_df[r] = gene_expr

                signal["all"] = sig_df
                subsets["all"] = self.adata
            elif use_pathways and sender_or_receiver_degs == "sender":
                if self.mod_type != "ligand" and self.mod_type != "lr":
                    raise ValueError(
                        "Sent signal from ligands cannot be used because the original specified 'mod_type' "
                        "does not use ligand expression."
                    )
                # Groupby pathways and take the arithmetic mean of the ligands/interactions in each relevant pathway:
                lig_to_pathway_map = lr_db.set_index("from")["pathway"].drop_duplicates().to_dict()
                mapped_ligands = self.ligands_expr_nonlag.copy()
                mapped_ligands.columns = self.ligands_expr_nonlag.columns.map(lig_to_pathway_map)
                signal["all"] = mapped_ligands.groupby(by=mapped_ligands.columns, axis=1).sum()
                subsets["all"] = self.adata
            elif use_pathways and sender_or_receiver_degs == "receiver":
                if self.mod_type != "receptor" and self.mod_type != "lr":
                    raise ValueError(
                        "Received signal from receptors cannot be used because the original specified 'mod_type' "
                        "does not use receptor expression."
                    )
                # Groupby pathways and take the arithmetic mean of the receptors/interactions in each relevant pathway:
                rec_to_pathway_map = lr_db.set_index("to")["pathway"].drop_duplicates().to_dict()
                mapped_receptors = self.receptors_expr.copy()
                mapped_receptors.columns = self.receptors_expr.columns.map(rec_to_pathway_map)
                signal["all"] = mapped_receptors.groupby(by=mapped_receptors.columns, axis=1).sum()
                subsets["all"] = self.adata
            elif use_cell_types:
                if self.mod_type != "niche":
                    raise ValueError(
                        "Cell categories cannot be used because the original specified 'mod_type' does not "
                        "consider cell type. Change 'mod_type' to 'niche' if desired."
                    )

                # For downstream analysis through the lens of cell type, we can aid users in creating a downstream
                # effects model for each cell type:
                for cell_type in self.cell_categories.columns:
                    ct_subset = self.adata[self.adata.obs[self.group_key] == cell_type, :].copy()
                    subsets[cell_type] = ct_subset

                    mols = ligand_set if sender_or_receiver_degs == "sender" else receptor_set
                    ct_signaling = ct_subset[:, mols].copy()
                    # Find the set of ligands/receptors that are expressed in at least n% of the cells of this cell type
                    sig_expr_percentage = (
                        np.array((ct_signaling.X > 0).sum(axis=0)).squeeze() / ct_signaling.shape[0] * 100
                    )
                    ct_signaling = ct_signaling.var.index[sig_expr_percentage > self.target_expr_threshold]

                    sig_expr = pd.DataFrame(
                        self.adata[:, ct_signaling].X.A
                        if scipy.sparse.issparse(self.adata.X)
                        else self.adata[:, ct_signaling].X,
                        index=self.sample_names,
                        columns=ct_signaling,
                    )
                    signal[cell_type] = sig_expr

            else:
                raise ValueError(
                    "All of 'use_ligands', 'use_receptors', 'use_pathways', and 'use_cell_types' are False. Please set "
                    "at least one to True."
                )

            for subset_key in signal.keys():
                signal_values = signal[subset_key].values
                adata = subsets[subset_key]

                self.logger.info(
                    "Selecting transcription factors, cofactors and RNA-binding proteins for analysis of differential "
                    "expression."
                )

                # Further subset list of additional factors to those that are expressed in at least n% of the cells
                # that are nonzero in sending cells (use the user input 'target_expr_threshold'):
                indices = np.any(signal_values != 0, axis=0).nonzero()[0]
                nz_signal = list(self.sample_names[indices])
                adata_subset = adata[nz_signal, :]
                n_cells_threshold = int(self.target_expr_threshold * adata_subset.n_obs)

                all_TFs = list(grn.columns)
                all_TFs = [tf for tf in all_TFs if tf in cof_db.columns and tf in tf_tf_db.columns]
                if scipy.sparse.issparse(adata.X):
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

                # Subset to cofactors for which enough signal is present- filter to those expressed in at least n% of
                # the cells that express at least one of the TFs associated with the cofactor:
                all_cofactors = []
                for cofactor in cof_subset:
                    cof_row = cof_db.loc[cofactor, :]
                    cof_TFs = cof_row[cof_row == 1].index
                    tfs_expr_subset_indices = np.where(adata_subset[:, cof_TFs].X.sum(axis=1) > 0)[0]
                    tf_subset_cells = adata_subset[tfs_expr_subset_indices, :]
                    n_cells_threshold = int(self.target_expr_threshold * tf_subset_cells.n_obs)
                    if scipy.sparse.issparse(adata.X):
                        nnz_counts = np.array(tf_subset_cells[:, cofactor].X.getnnz(axis=0)).flatten()
                    else:
                        nnz_counts = np.array(tf_subset_cells[:, cofactor].X.getnnz(axis=0)).flatten()

                    if nnz_counts >= n_cells_threshold:
                        all_cofactors.append(cofactor)

                # And extend the set of transcription factors using interacting pairs that may also be present in the
                # same cells upstream transcription factors are:
                all_interacting_tfs = []
                for tf in intersecting_tf_subset:
                    tf_row = tf_tf_db.loc[tf, :]
                    tf_TFs = tf_row[tf_row == 1].index
                    tfs_expr_subset_indices = np.where(adata_subset[:, tf_TFs].X.sum(axis=1) > 0)[0]
                    tf_subset_cells = adata_subset[tfs_expr_subset_indices, :]
                    n_cells_threshold = int(self.target_expr_threshold * tf_subset_cells.n_obs)
                    if scipy.sparse.issparse(adata.X):
                        nnz_counts = np.array(tf_subset_cells[:, tf].X.getnnz(axis=0)).flatten()
                    else:
                        nnz_counts = np.array(tf_subset_cells[:, tf].X.getnnz(axis=0)).flatten()

                    if nnz_counts >= n_cells_threshold:
                        all_interacting_tfs.append(tf)

                # Do the same for RNA-binding proteins:
                all_RBPs = list(rna_bp_db["Gene_Name"].values)
                all_RBPs = [r for r in all_RBPs if r in self.feature_names]
                if len(all_RBPs) > 0:
                    if scipy.sparse.issparse(adata.X):
                        nnz_counts = np.array(adata_subset[:, all_RBPs].X.getnnz(axis=0)).flatten()
                    else:
                        nnz_counts = np.array(adata_subset[:, all_RBPs].X.getnnz(axis=0)).flatten()
                    all_RBPs = [tf for tf, nnz in zip(all_RBPs, nnz_counts) if nnz >= n_cells_threshold]
                    # Remove RBPs if any happen to be TFs or cofactors:
                    all_RBPs = [
                        r
                        for r in all_RBPs
                        if r not in all_TFs and r not in all_interacting_tfs and r not in all_cofactors
                    ]

                self.logger.info(f"For this dataset, marked {len(all_TFs)} of interest.")
                self.logger.info(
                    f"For this dataset, marked {len(all_cofactors)} transcriptional cofactors of interest."
                )
                if len(all_RBPs) > 0:
                    self.logger.info(f"For this dataset, marked {len(all_RBPs)} RNA-binding proteins of interest.")

                # Get feature names- for the singleton factors:
                regulator_features = all_TFs + all_interacting_tfs + all_cofactors + all_RBPs

                # Take subset of AnnData object corresponding to these regulators:
                counts = adata[:, regulator_features].copy()

                # Convert to dataframe, append signal (ligand/receptor) dataframe:
                counts_df = pd.DataFrame(counts.X.toarray(), index=counts.obs_names, columns=counts.var_names)
                combined_df = pd.concat([counts_df, signal[subset_key]], axis=1)

                # Convert back to AnnData object, save to file path:
                counts_plus = anndata.AnnData(combined_df.values)
                counts_plus.obs_names = combined_df.index
                counts_plus.var_names = combined_df.columns
                # Make note that certain columns are pathways and not individual molecules that can be found in the
                # AnnData object:
                if use_pathways:
                    counts_plus.uns["target_type"] = "pathway"
                elif use_ligands:
                    counts_plus.uns["target_type"] = "ligands"
                elif use_receptors:
                    counts_plus.uns["target_type"] = "receptors"

                # Optionally, can use dimensionality reduction to aid in computing the nearest neighbors for the model (
                # cells that are nearby in dimensionally-reduced TF space will be neighbors in this scenario)
                # Compute latent representation of the AnnData subset ("counts"):
                # Minmax scale interaction features for the purpose of dimensionality reduction:
                sender_deg_predictors_scaled = counts_df.apply(
                    lambda column: (column - column.min()) / (column.max() - column.min())
                )

                # Compute the ideal number of UMAP components to use- use half the number of features as the
                # max possible number of components:
                self.logger.info("Computing optimal number of UMAP components ...")
                n_umap_components = find_optimal_n_umap_components(sender_deg_predictors_scaled)

                # Perform UMAP reduction with the chosen number of components, store in AnnData object:
                _, _, _, _, X_umap = umap_conn_indices_dist_embedding(
                    sender_deg_predictors_scaled, n_neighbors=30, n_components=n_umap_components
                )
                counts_plus.obsm["X_umap"] = X_umap

                targets = signal[subset_key].columns
                # Add each target to AnnData .obs field:
                for col in signal[subset_key].columns:
                    counts_plus.obs[f"regulator_{col}"] = signal[subset_key][col].values

                if "targets_path" in locals():
                    # Save to .txt file:
                    with open(targets_path, "w") as file:
                        for t in targets:
                            file.write(t + "\n")
                else:
                    if use_ligands or (use_cell_types and sender_or_receiver_degs == "sender"):
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_ligands.txt")
                    elif use_receptors or (use_cell_types and sender_or_receiver_degs == "receiver"):
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_receptors.txt")
                    elif use_pathways:
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_pathways.txt")
                    with open(targets_path, "w") as file:
                        for t in targets:
                            file.write(t + "\n")

                self.logger.info(
                    "'CCI_sender_deg_detection'- saving regulatory molecules to test as .h5ad file to the "
                    "directory of the original AnnData object..."
                )
                counts_plus.write_h5ad(os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_{subset_key}.h5ad"))

    def CCI_deg_detection(
        self,
        cci_dir_path: str,
        sender_or_receiver_degs: Literal["sender", "receiver"] = "sender",
        use_ligands: bool = True,
        use_receptors: bool = False,
        use_pathways: bool = False,
        cell_type: Optional[str] = None,
        **kwargs,
    ) -> MuSIC:
        """Downstream method that when called, creates a separate instance of :class `MuSIC` specifically designed
        for the downstream task of detecting differentially expressed genes associated w/ ligand expression.

        Args:
            cci_dir_path: Path to directory containing all Spateo databases
            sender_or_receiver_degs: Whether to compute DEGs for sender or receiver cells. Note that 'use_ligands' is
                only an option if this is set to 'sender', whereas 'use_receptors' is only an option if this is set to
                'receiver'. If 'use_pathways' is True, the value of this argument will determine whether ligands or
                receptors are used to define the pathway.
            use_ligands: Use ligand array for differential expression analysis. Will take precedent over
                sender/receiver cell type if also provided. Should match the input to :func
                `CCI_sender_deg_detection_setup`.

            use_pathways: Use pathway array for differential expression analysis. Will use ligands in these pathways
                to collectively compute signaling potential score. Will take precedent over sender cell types if also
                provided. Should match the input to :func `CCI_sender_deg_detection_setup`.
            cell_type: Cell type to use to use for differential expression analysis. If given, will use the
                ligand/receptor subset obtained from :func ~`CCI_sender_deg_detection_setup` and cells of the chosen
                cell type in the model.
            kwargs: Keyword arguments for any of the Spateo argparse arguments. Should not include 'adata_path',
                'custom_ligands_path' & 'ligand' or 'custom_pathways_path' & 'pathway' (depending on whether ligands or
                pathways are being used for the analysis), and should not include 'output_path' (which will be
                determined by the output path used for the main model). Should also not include any of the other
                arguments for this function

        Returns:
            downstream_model: Fitted model instance that can be used for further downstream applications
        """
        logger = lm.get_main_logger()

        kwargs["mod_type"] = "downstream"
        kwargs["cci_dir_path"] = cci_dir_path

        # Use the same output directory as the main model, add folder demarcating results from downstream task:
        output_dir = os.path.dirname(self.output_path)
        output_file_name = os.path.basename(self.output_path)
        if not os.path.exists(os.path.join(output_dir, "cci_deg_detection")):
            os.makedirs(os.path.join(output_dir, "cci_deg_detection"))

        if use_ligands or use_receptors or use_pathways:
            parent_dir = os.path.dirname(self.adata_path)
            file_name = os.path.basename(self.adata_path).split(".")[0]
            if use_ligands:
                file_id = "ligand_analysis"
            elif use_receptors:
                file_id = "receptor_analysis"
            elif use_pathways and sender_or_receiver_degs == "sender":
                file_id = "pathway_analysis_ligands"
            elif use_pathways and sender_or_receiver_degs == "receiver":
                file_id = "pathway_analysis_receptors"
            if not os.path.exists(os.path.join(output_dir, "cci_deg_detection", file_id)):
                os.makedirs(os.path.join(output_dir, "cci_deg_detection", file_id))
            output_path = os.path.join(output_dir, "cci_deg_detection", file_id, output_file_name)
            kwargs["output_path"] = output_path

            logger.info(
                f"Using AnnData object stored at "
                f"{os.path.join(parent_dir, 'cci_deg_detection', f'{file_name}_all.h5ad')}."
            )
            kwargs["adata_path"] = os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_all.h5ad")
            if use_ligands:
                kwargs["custom_ligands_path"] = os.path.join(
                    parent_dir, "cci_deg_detection", f"{file_name}_ligands.txt"
                )
                logger.info(f"Using ligands stored at {kwargs['custom_ligands_path']}.")
            elif use_receptors:
                kwargs["custom_receptors_path"] = os.path.join(
                    parent_dir, "cci_deg_detection", f"{file_name}_receptors.txt"
                )
            elif use_pathways:
                kwargs["custom_pathways_path"] = os.path.join(
                    parent_dir, "cci_deg_detection", f"{file_name}_pathways.txt"
                )
                logger.info(f"Using pathways stored at {kwargs['custom_pathways_path']}.")
            else:
                raise ValueError("One of 'use_ligands', 'use_receptors' or 'use_pathways' must be True.")

            # Create new instance of MuSIC:
            comm, parser, args_list = define_spateo_argparse(**kwargs)
            downstream_model = MuSIC(comm, parser, args_list)
            downstream_model._set_up_model()
            downstream_model.fit()
            downstream_model.predict_and_save()

        elif cell_type is not None:
            # For each cell type, fit a different model:
            parent_dir = os.path.dirname(self.adata_path)
            file_name = os.path.basename(self.adata_path).split(".")[0]

            # create output sub-directory for this model:
            if sender_or_receiver_degs == "sender":
                file_id = "ligand_analysis"
            elif sender_or_receiver_degs == "receiver":
                file_id = "receptor_analysis"
            if not os.path.exists(os.path.join(output_dir, "cci_deg_detection", cell_type, file_id)):
                os.makedirs(os.path.join(output_dir, "cci_deg_detection", cell_type, file_id))
            subset_output_dir = os.path.join(output_dir, "cci_deg_detection", cell_type, file_id)
            # Check if directory already exists, if not create it
            if not os.path.exists(subset_output_dir):
                self.logger.info(f"Output folder for cell type {cell_type} does not exist, creating it now.")
                os.makedirs(subset_output_dir)
            output_path = os.path.join(subset_output_dir, output_file_name)
            kwargs["output_path"] = output_path

            kwargs["adata_path"] = os.path.join(parent_dir, "cci_deg_detection", f"{file_name}_{cell_type}.h5ad")
            logger.info(f"Using AnnData object stored at {kwargs['adata_path']}.")
            if sender_or_receiver_degs == "sender":
                kwargs["custom_ligands_path"] = os.path.join(
                    parent_dir, "cci_deg_detection", f"{file_name}_{cell_type}_ligands.txt"
                )
                logger.info(f"Using ligands stored at {kwargs['custom_ligands_path']}.")
            elif sender_or_receiver_degs == "receiver":
                kwargs["custom_receptors_path"] = os.path.join(
                    parent_dir, "cci_deg_detection", f"{file_name}_{cell_type}_receptors.txt"
                )
                logger.info(f"Using receptors stored at {kwargs['custom_receptors_path']}.")

            # Create new instance of MuSIC:
            comm, parser, args_list = define_spateo_argparse(**kwargs)
            downstream_model = MuSIC(comm, parser, args_list)
            downstream_model._set_up_model()
            downstream_model.fit()
            downstream_model.predict_and_save()

        else:
            raise ValueError("'use_ligands' and 'use_pathways' are both False, and 'cell_type' was not given.")

        return downstream_model

    # ---------------------------------------------------------------------------------------------------
    # Permutation testing
    # ---------------------------------------------------------------------------------------------------
    def permutation_test(self, gene: str, n_permutations: int = 100, permute_nonzeros_only: bool = False, **kwargs):
        """Sets up permutation test for determination of statistical significance of model diagnostics. Can be used
        to identify true/the strongest signal-responsive expression patterns.

        Args:
            gene: Target gene to perform permutation test on.
            n_permutations: Number of permutations of the gene expression to perform. Default is 100.
            permute_nonzeros_only: Whether to only perform the permutation over the gene-expressing cells
            kwargs: Keyword arguments for any of the Spateo argparse arguments. Should not include 'adata_path',
                'target_path', or 'output_path' (which will be determined by the output path used for the main
                model). Also should not include 'custom_lig_path', 'custom_rec_path', 'mod_type', 'bw_fixed' or 'kernel'
                (which will be determined by the initial model instantiation).
        """

        # Set up storage folder:
        # Check if the array of additional molecules to query has already been created:
        parent_dir = os.path.dirname(self.adata_path)
        file_name = os.path.basename(self.adata_path).split(".")[0]

        if not os.path.exists(os.path.join(parent_dir, "permutation_test")):
            os.makedirs(os.path.join(parent_dir, "permutation_test"))
        if not os.path.exists(os.path.join(parent_dir, "permutation_test_inputs")):
            os.makedirs(os.path.join(parent_dir, "permutation_test_inputs"))
        if not os.path.exists(os.path.join(parent_dir, f"permutation_test_outputs_{gene}")):
            os.makedirs(os.path.join(parent_dir, f"permutation_test_outputs_{gene}"))

        gene_idx = self.adata.var_names.tolist().index(gene)
        gene_data = np.array(self.adata.X[:, gene_idx].todense())

        permuted_data_list = [gene_data]
        perm_names = [f"{gene}_nonpermuted"]

        # Set save name for AnnData object and output file depending on whether all cells or only gene-expressing
        # cells are permuted:
        if permute_nonzeros_only:
            adata_path = os.path.join(
                parent_dir, "permutation_test", f"{file_name}_{gene}_permuted_expressing_subset.h5ad"
            )
            output_path = os.path.join(
                parent_dir, "permutation_test_outputs", f"{file_name}_{gene}_permuted_expressing_subset.csv"
            )
            self.permuted_nonzeros_only = True
        else:
            adata_path = os.path.join(parent_dir, "permutation_test", f"{file_name}_{gene}_permuted.h5ad")
            output_path = os.path.join(parent_dir, "permutation_test_outputs", f"{file_name}_{gene}_permuted.csv")
            self.permuted_nonzeros_only = False

        if permute_nonzeros_only:
            self.logger.info("Performing permutation by scrambling expression for all cells...")
            for i in range(n_permutations):
                perm_name = f"{gene}_permuted_{i}"
                permuted_data = np.random.permutation(gene_data)
                # Convert to sparse matrix
                permuted_data_sparse = scipy.sparse.csr_matrix(permuted_data)

                # Store back in the AnnData object
                permuted_data_list.append(permuted_data_sparse)
                perm_names.append(perm_name)
        else:
            self.logger.info(
                "Performing permutation by scrambling expression only for the subset of cells that "
                "express the gene of interest..."
            )
            for i in range(n_permutations):
                perm_name = f"{gene}_permuted_{i}"
                # Separate non-zero rows and zero rows:
                nonzero_indices = np.where(gene_data != 0)[0]
                zero_indices = np.where(gene_data == 0)[0]

                non_zero_rows = gene_data[gene_data != 0]
                zero_rows = gene_data[gene_data == 0]

                # Permute non-zero rows:
                permuted_non_zero_rows = np.random.permutation(non_zero_rows)

                # Recombine permuted non-zero rows and zero rows:
                permuted_gene_data = np.zeros_like(gene_data)
                permuted_gene_data[nonzero_indices] = permuted_non_zero_rows.reshape(-1, 1)
                permuted_gene_data[zero_indices] = zero_rows.reshape(-1, 1)
                # Convert to sparse matrix
                permuted_gene_data_sparse = scipy.sparse.csr_matrix(permuted_gene_data)

                # Store back in the AnnData object
                permuted_data_list.append(permuted_gene_data_sparse)
                perm_names.append(perm_name)

        # Concatenate the original and permuted data:
        all_data_sparse = scipy.sparse.hstack([self.adata.X] + permuted_data_list)
        all_data_sparse = all_data_sparse.tocsr()
        all_names = list(self.adata.var_names.tolist() + perm_names)

        # Create new AnnData object, keeping the cell type annotations, original "__type" entry, and all .obsm
        # entries (including spatial coordinates):
        adata_permuted = anndata.AnnData(X=all_data_sparse)
        adata_permuted.obs_names = self.adata.obs_names
        adata_permuted.var_names = all_names
        adata_permuted.obsm = self.adata.obsm
        adata_permuted.obs[self.group_key] = self.adata.obs[self.group_key]
        adata_permuted.obs["__type"] = self.adata.obs["__type"]

        # Save list of targets:
        targets = [v for v in adata_permuted.var_names if "permuted" in v]
        target_path = os.path.join(parent_dir, "permutation_test_inputs", f"{gene}_permutation_targets.txt")
        with open(target_path, "w") as f:
            for target in targets:
                f.write(f"{target}\n")
        # Save the permuted AnnData object:
        adata_permuted.write(adata_path)

        # Fitting permutation model:
        kwargs["adata_path"] = adata_path
        kwargs["output_path"] = output_path
        kwargs["cci_dir"] = self.cci_dir
        if hasattr(self, "custom_receptors_path") and self.mod_type.isin(["receptor", "lr"]):
            kwargs["custom_rec_path"] = self.custom_receptors_path
        elif hasattr(self, "custom_pathways_path") and self.mod_type.isin(["receptor", "lr"]):
            kwargs["custom_pathways_path"] = self.custom_pathways_path
        else:
            raise ValueError("For permutation testing, receptors/pathways must be given from .txt file.")
        if hasattr(self, "custom_ligands_path") and self.mod_type.isin(["ligand", "lr"]):
            kwargs["custom_lig_path"] = self.custom_ligands_path
        elif hasattr(self, "custom_pathways_path") and self.mod_type.isin(["ligand", "lr"]):
            kwargs["custom_pathways_path"] = self.custom_pathways_path
        else:
            raise ValueError("For permutation testing, ligands/pathways must be given from .txt file.")

        kwargs["targets_path"] = target_path
        kwargs["mod_type"] = self.mod_type
        kwargs["distance_secreted"] = self.distance_secreted
        kwargs["distance_membrane_bound"] = self.distance_membrane_bound
        kwargs["bw_fixed"] = self.bw_fixed
        kwargs["kernel"] = self.kernel

        comm, parser, args_list = define_spateo_argparse(**kwargs)
        permutation_model = MuSIC(comm, parser, args_list)
        permutation_model._set_up_model()
        permutation_model.fit()
        permutation_model.predict_and_save()

    def eval_permutation_test(self, gene: str):
        """Evaluation function for permutation tests. Will compute multiple metrics (correlation coefficients,
        F1 scores, AUROC in the case that all cells were permuted, etc.) to compare true and model-predicted gene
        expression vectors.

        Args:
            gene: Target gene for which to evaluate permutation test
        """

        parent_dir = os.path.dirname(self.adata_path)
        file_name = os.path.basename(self.adata_path).split(".")[0]

        output_dir = os.path.join(parent_dir, f"permutation_test_outputs_{gene}")
        if not os.path.exists(os.path.join(output_dir, "diagnostics")):
            os.makedirs(os.path.join(output_dir, "diagnostics"))

        if self.permuted_nonzeros_only:
            adata_permuted = anndata.read_h5ad(
                os.path.join(parent_dir, "permutation_test", f"{file_name}_{gene}_permuted_expressing_subset.h5ad")
            )
        else:
            adata_permuted = anndata.read_h5ad(
                os.path.join(parent_dir, "permutation_test", f"{file_name}_{gene}_permuted.h5ad")
            )

        predictions = pd.read_csv(os.path.join(output_dir, "predictions.csv"), index_col=0)
        original_column_names = predictions.columns.tolist()

        # Create a dictionary to map integer column names to new permutation names
        column_name_map = {}
        for column_name in original_column_names:
            if column_name != "nonpermuted":
                column_name_map[column_name] = f"permutation_{column_name}"

        # Rename the columns in the dataframe using the created dictionary
        predictions.rename(columns=column_name_map, inplace=True)

        if not self.permuted_nonzeros_only:
            # Instantiate metric storage variables:
            all_pearson_correlations = {}
            all_spearman_correlations = {}
            all_f1_scores = {}
            all_auroc_scores = {}
            all_rmse_scores = {}

        # For the nonzero subset- will be used both in the case that permutation occurred across all cells and
        # across only gene-expressing cells:
        all_pearson_correlations_nz = {}
        all_spearman_correlations_nz = {}
        all_f1_scores_nz = {}
        all_auroc_scores_nz = {}
        all_rmse_scores_nz = {}

        for col in predictions.columns:
            if "_" in col:
                perm_no = col.split("_")[1]
                y = adata_permuted[:, f"{gene}_permuted_{perm_no}"].X.toarray().reshape(-1)
            else:
                y = adata_permuted[:, f"{gene}_{col}"].X.toarray().reshape(-1)
            y_binary = (y > 0).astype(int)

            y_pred = predictions[col].values.reshape(-1)
            y_pred_binary = (y_pred > 0).astype(int)

            # Compute metrics for the subset of rows that are nonzero:
            nonzero_indices = np.nonzero(y)[0]
            y_nonzero = y[nonzero_indices]
            y_pred_nonzero = y_pred[nonzero_indices]

            y_binary_nonzero = y_binary[nonzero_indices]
            y_pred_binary_nonzero = y_pred_binary[nonzero_indices]

            rp, _ = pearsonr(y_nonzero, y_pred_nonzero)
            r, _ = spearmanr(y_nonzero, y_pred_nonzero)
            f1 = f1_score(y_binary_nonzero, y_pred_binary_nonzero)
            auroc = roc_auc_score(y_binary_nonzero, y_pred_binary_nonzero)
            rmse = mean_squared_error(y_nonzero, y_pred_nonzero, squared=False)

            all_pearson_correlations_nz[col] = rp
            all_spearman_correlations_nz[col] = r
            all_f1_scores_nz[col] = f1
            all_auroc_scores_nz[col] = auroc
            all_rmse_scores_nz[col] = rmse

            # Additionally calculate metrics for all cells if permutation occurred over all cells:
            if not self.permuted_nonzeros_only:
                rp, _ = pearsonr(y, y_pred)
                r, _ = spearmanr(y, y_pred)
                f1 = f1_score(y_binary, y_pred_binary)
                auroc = roc_auc_score(y_binary, y_pred_binary)
                rmse = mean_squared_error(y, y_pred, squared=False)

                all_pearson_correlations[col] = rp
                all_spearman_correlations[col] = r
                all_f1_scores[col] = f1
                all_auroc_scores[col] = auroc
                all_rmse_scores[col] = rmse

        # Collect all diagnostics in dataframe form:
        if not self.permuted_nonzeros_only:
            results = pd.DataFrame(
                {
                    "Pearson correlation": all_pearson_correlations,
                    "Spearman correlation": all_spearman_correlations,
                    "F1 score": all_f1_scores,
                    "AUROC": all_auroc_scores,
                    "RMSE": all_rmse_scores,
                    "Pearson correlation (expressing subset)": all_pearson_correlations_nz,
                    "Spearman correlation (expressing subset)": all_spearman_correlations_nz,
                    "F1 score (expressing subset)": all_f1_scores_nz,
                    "AUROC (expressing subset)": all_auroc_scores_nz,
                    "RMSE (expressing subset)": all_rmse_scores_nz,
                }
            )
            # Without nonpermuted scores:
            results_permuted = results.loc[[r for r in results.index if r != "nonpermuted"], :]

            self.logger.info("Average permutation metrics for all cells: ")
            self.logger.info(f"Average Pearson correlation: {results_permuted['Pearson correlation'].mean()}")
            self.logger.info(f"Average Spearman correlation: {results_permuted['Spearman correlation'].mean()}")
            self.logger.info(f"Average F1 score: {results_permuted['F1 score'].mean()}")
            self.logger.info(f"Average AUROC: {results_permuted['AUROC'].mean()}")
            self.logger.info(f"Average RMSE: {results_permuted['RMSE'].mean()}")
            self.logger.info("Average permutation metrics for expressing cells: ")
            self.logger.info(
                f"Average Pearson correlation: " f"{results_permuted['Pearson correlation (expressing subset)'].mean()}"
            )
            self.logger.info(
                f"Average Spearman correlation: "
                f"{results_permuted['Spearman correlation (expressing subset)'].mean()}"
            )
            self.logger.info(f"Average F1 score: {results_permuted['F1 score (expressing subset)'].mean()}")
            self.logger.info(f"Average AUROC: {results_permuted['AUROC (expressing subset)'].mean()}")
            self.logger.info(f"Average RMSE: {results_permuted['RMSE (expressing subset)'].mean()}")

            diagnostic_path = os.path.join(output_dir, "diagnostics", f"{gene}_permutations_diagnostics.csv")
        else:
            results = pd.DataFrame(
                {
                    "Pearson correlation (expressing subset)": all_pearson_correlations_nz,
                    "Spearman correlation (expressing subset)": all_spearman_correlations_nz,
                    "F1 score (expressing subset)": all_f1_scores_nz,
                    "AUROC (expressing subset)": all_auroc_scores_nz,
                    "RMSE (expressing subset)": all_rmse_scores_nz,
                }
            )
            # Without nonpermuted scores:
            results_permuted = results.loc[[r for r in results.index if r != "nonpermuted"], :]

            self.logger.info("Average permutation metrics for expressing cells: ")
            self.logger.info(
                f"Average Pearson correlation: " f"{results_permuted['Pearson correlation (expressing subset)'].mean()}"
            )
            self.logger.info(
                f"Average Spearman correlation: "
                f"{results_permuted['Spearman correlation (expressing subset)'].mean()}"
            )
            self.logger.info(f"Average F1 score: {results_permuted['F1 score (expressing subset)'].mean()}")
            self.logger.info(f"Average AUROC: {results_permuted['AUROC (expressing subset)'].mean()}")
            self.logger.info(f"Average RMSE: {results_permuted['RMSE (expressing subset)'].mean()}")

            diagnostic_path = os.path.join(
                output_dir, "diagnostics", f"{gene}_nonzero_subset_permutations_diagnostics.csv"
            )

        # Significance testing:
        nonpermuted_values = results.loc["nonpermuted"]

        # Create dictionaries to store the t-statistics, p-values, and significance indicators:
        t_statistics, pvals, significance = {}, {}, {}

        # Iterate over the columns of the DataFrame
        for col in results_permuted.columns:
            column_data = results_permuted[col]
            # Perform one-sample t-test:
            t_stat, pval = ttest_1samp(column_data, nonpermuted_values[col])
            # Store the t-statistic, p-value, and significance indicator
            t_statistics[col] = t_stat
            pvals[col] = pval
            significance[col] = "yes" if pval < 0.05 else "no"

        # Store the t-statistics, p-values, and significance indicators in the results DataFrame:
        results.loc["t-statistic"] = t_statistics
        results.loc["p-value"] = pvals
        results.loc["significant"] = significance

        # Save results:
        results.to_csv(diagnostic_path)

    # ---------------------------------------------------------------------------------------------------
    # In silico perturbation of signaling effects
    # ---------------------------------------------------------------------------------------------------
    def predict_perturbation_effect(
        self,
        ligand: Optional[str] = None,
        receptor: Optional[str] = None,
        regulator: Optional[str] = None,
        cell_type: Optional[str] = None,
    ):
        """Basic & theoretical in silico perturbation, to depict the effect of changing expression level of a given
        signaling molecule or upstream regulator. In silico perturbation will set the level of the specified
        regulator to 0.

        Args:
            ligand: Expression of this ligand will be set to 0
            receptor: Expression of this receptor will be set to 0
            regulator: Expression of this regulator will be set to 0. Examples of "regulators" are transcription
                factors or cofactors, anything that comprises the independent variable array found by :func `
            cell_type:

        Returns:

        """
        # For ligand or L:R models, recompute neighborhood ligand level if perturbing a ligand:
        if ligand is not None and self.mod_type in ["ligand", "lr"]:
            "filler"

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

        if effect_strength_threshold is not None:
            effect_strength_threshold = 0.2
            self.logger.info(
                f"Computing cell type coupling for cells in which predicted sent/received effect score "
                f"is higher than {effect_strength_threshold * 100}th percentile score."
            )

        if not self.mod_type != "receptor":
            raise ValueError("Knowledge of the source is required to sent effect potential.")

        if self.mod_type in ["lr", "ligand"]:
            # Get spatial weights given bandwidth value- each row corresponds to a sender, each column to a receiver:
            # Try to load spatial weights for membrane-bound and secreted ligands, compute if not found:
            membrane_bound_path = os.path.join(
                os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_membrane_bound.npz"
            )
            secreted_path = os.path.join(
                os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_secreted.npz"
            )

            try:
                spatial_weights_membrane_bound = scipy.sparse.load_npz(membrane_bound_path)
                spatial_weights_secreted = scipy.sparse.load_npz(secreted_path)
            except:
                bw_mb = (
                    self.n_neighbors_membrane_bound
                    if self.distance_membrane_bound is None
                    else self.distance_membrane_bound
                )
                bw_fixed = True if self.distance_membrane_bound is not None else False
                spatial_weights_membrane_bound = self._compute_all_wi(
                    bw=bw_mb,
                    bw_fixed=bw_fixed,
                    exclude_self=True,
                    verbose=False,
                )
                self.logger.info(f"Saving spatial weights for membrane-bound ligands to {membrane_bound_path}.")
                scipy.sparse.save_npz(membrane_bound_path, spatial_weights_membrane_bound)

                bw_s = self.n_neighbors_membrane_bound if self.distance_secreted is None else self.distance_secreted
                bw_fixed = True if self.distance_secreted is not None else False
                # Autocrine signaling is much easier with secreted signals:
                spatial_weights_secreted = self._compute_all_wi(
                    bw=bw_s,
                    bw_fixed=bw_fixed,
                    exclude_self=False,
                    verbose=False,
                )
                self.logger.info(f"Saving spatial weights for secreted ligands to {secreted_path}.")
                scipy.sparse.save_npz(secreted_path, spatial_weights_secreted)
        else:
            niche_path = os.path.join(
                os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_niche.npz"
            )

            try:
                spatial_weights_niche = scipy.sparse.load_npz(niche_path)
            except:
                spatial_weights_niche = self._compute_all_wi(
                    bw=self.n_neighbors_niche, bw_fixed=False, exclude_self=False, kernel="bisquare"
                )
                self.logger.info(f"Saving spatial weights for niche to {niche_path}.")
                scipy.sparse.save_npz(niche_path, spatial_weights_niche)

        # Compute signaling potential for each target (mediated by each of the possible signaling patterns-
        # ligand/receptor or cell type/cell type pair):
        # Columns consist of the spatial weights of each observation- convolve with expression of each ligand to
        # get proxy of ligand signal "sent", weight by the local coefficient value to get a proxy of the "signal
        # functionally received" in generating the downstream effect and store in .obsp.
        if targets is None:
            targets = self.coeffs.keys()
        elif isinstance(targets, str):
            targets = [targets]

        # Can optionally restrict to targets that are well-predicted by the model
        if self.filter_targets:
            pearson_dict = {}
            for target in targets:
                observed = self.adata[:, target].X.toarray().reshape(-1, 1)
                predicted = self.predictions[target].reshape(-1, 1)

                # Remove index of the largest predicted value (to mitigate sensitivity of these metrics to outliers):
                outlier_index = np.where(np.max(predicted))[0]
                predicted = np.delete(predicted, outlier_index)
                observed = np.delete(observed, outlier_index)

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
                        target=target,
                        ligand=ligand,
                        receptor=receptor,
                        spatial_weights_membrane_bound=spatial_weights_membrane_bound,
                        spatial_weights_secreted=spatial_weights_secreted,
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
                        target=target,
                        ligand=col,
                        spatial_weights_membrane_bound=spatial_weights_membrane_bound,
                        spatial_weights_secreted=spatial_weights_secreted,
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
                        spatial_weights_niche=spatial_weights_niche,
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
