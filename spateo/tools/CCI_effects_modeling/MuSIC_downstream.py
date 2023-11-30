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
import collections
import math
import os
from collections import Counter
from itertools import product
from typing import List, Literal, Optional, Tuple, Union

import anndata
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import scipy.cluster.hierarchy as sch
import scipy.sparse
import scipy.stats
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_ind
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

import spateo.tools.find_neighbors

from ...configuration import config_spateo_rcParams
from ...logging import logger_manager as lm
from ...plotting.static.networks import plot_network
from ...plotting.static.utils import save_return_show_fig_utils
from ..dimensionality_reduction import find_optimal_pca_components, pca_fit
from ..utils import compute_corr_ci
from .MuSIC import MuSIC
from .regression_utils import multitesting_correction, wald_test
from .SWR import define_spateo_argparse


# ---------------------------------------------------------------------------------------------------
# Statistical testing, correlated differential expression analysis
# ---------------------------------------------------------------------------------------------------
class MuSIC_Interpreter(MuSIC):
    """
    Interpretation and downstream analysis of spatially weighted regression models.

    Args:
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.
        args_list: If parser is provided by function call, the arguments to parse must be provided as a separate
            list. It is recommended to use the return from :func `define_spateo_argparse()` for this.
    """

    def __init__(self, parser: argparse.ArgumentParser, args_list: Optional[List[str]] = None):
        super().__init__(parser, args_list, verbose=False)

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
        self.coeffs, self.standard_errors = self.return_outputs(downstream=True)
        # Design matrix:
        self.design_matrix = pd.read_csv(
            os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"), index_col=0
        )

        # If predictions of an L:R model have been computed, load these as well:
        if os.path.exists(os.path.join(os.path.dirname(self.output_path), "predictions.csv")):
            self.predictions = pd.read_csv(
                os.path.join(os.path.dirname(self.output_path), "predictions.csv"), index_col=0
            )

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

        self.logger.info(
            "Computing significance for all coefficients, note this may take a long time for large "
            "datasets (> 10k cells)..."
        )

        for target in self.coeffs.keys():
            # Check for existing file:
            parent_dir = os.path.dirname(self.output_path)
            if os.path.exists(os.path.join(parent_dir, "significance", f"{target}_is_significant.csv")):
                self.logger.info(f"Significance already computed for target {target}, moving to the next...")
                continue

            # Get coefficients and standard errors for this key
            coef = self.coeffs[target]
            columns = [col for col in coef.columns if col.startswith("b_") and "intercept" not in col]
            coef = coef[columns]
            se = self.standard_errors[target]
            se_feature_match = [c.replace("se_", "") for c in se.columns]

            def compute_p_value(cell_name, feat):
                return wald_test(coef.loc[cell_name, f"b_{feat}"], se.loc[cell_name, f"se_{feat}"])

            filtered_tasks = [
                (cell_name, feat)
                for cell_name, feat in product(self.sample_names, self.feature_names)
                if feat in se_feature_match
                and se.loc[cell_name, f"se_{feat}"] != 0
                and coef.loc[cell_name, f"b_{feat}"] != 0
            ]

            # Parallelize computations for filtered tasks
            results = Parallel(n_jobs=-1)(
                delayed(compute_p_value)(cell_name, feat)
                for cell_name, feat in tqdm(filtered_tasks, desc=f"Processing for target {target}")
            )

            # Convert results to a DataFrame
            results_df = pd.DataFrame(
                results, index=pd.MultiIndex.from_tuples(filtered_tasks, names=["sample", "feature"])
            )
            p_values_all = pd.DataFrame(1, index=self.sample_names, columns=self.feature_names)
            p_values_all.update(results_df.unstack(level="feature").droplevel(0, axis=1))

            # Multiple testing correction for each observation:
            qvals = np.zeros_like(p_values_all.values)
            for i in range(p_values_all.shape[0]):
                qvals[i, :] = multitesting_correction(
                    p_values_all.iloc[i, :], method=method, alpha=significance_threshold
                )
            q_values_df = pd.DataFrame(qvals, index=self.sample_names, columns=self.feature_names)

            # Significance:
            is_significant_df = q_values_df < significance_threshold

            # Save dataframes:
            parent_dir = os.path.dirname(self.output_path)
            p_values_all.to_csv(os.path.join(parent_dir, "significance", f"{target}_p_values.csv"))
            q_values_df.to_csv(os.path.join(parent_dir, "significance", f"{target}_q_values.csv"))
            is_significant_df.to_csv(os.path.join(parent_dir, "significance", f"{target}_is_significant.csv"))

            self.logger.info(f"Finished computing significance for target {target}.")

    def compute_and_visualize_diagnostics(
        self, type: Literal["correlations", "confusion", "rmse"], n_genes_per_plot: int = 20
    ):
        """
        For true and predicted gene expression, compute and generate either: confusion matrices,
        or correlations, including the Pearson correlation, Spearman correlation, or root mean-squared-error (RMSE).

        Args:
            type: Type of diagnostic to compute and visualize. Options: "correlations" for Pearson & Spearman
                correlation, "confusion" for confusion matrix, "rmse" for root mean-squared-error.
            n_genes_per_plot: Only used if "type" is "confusion". Number of genes to plot per figure. If there are
                more than this number of genes, multiple figures will be generated.
        """
        # Plot title:
        file_name = os.path.splitext(os.path.basename(self.adata_path))[0]

        parent_dir = os.path.dirname(self.output_path)
        pred_path = os.path.join(parent_dir, "predictions.csv")

        predictions = pd.read_csv(pred_path, index_col=0)
        all_genes = predictions.columns
        width = 0.5 * len(all_genes)
        pred_vals = predictions.values

        if type == "correlations":
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
            line_style = "--"
            line_thickness = 2
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

        elif type == "confusion":
            confusion_matrices = {}

            for i, gene in enumerate(all_genes):
                y = self.adata[:, gene].X.toarray().reshape(-1)
                music_results_target = pred_vals[:, i]
                predictions_binary = (music_results_target > 0).astype(int)
                y_binary = (y > 0).astype(int)
                confusion_matrices[gene] = confusion_matrix(y_binary, predictions_binary)

            total_figs = int(math.ceil(len(all_genes) / n_genes_per_plot))

            for fig_index in range(total_figs):
                start_index = fig_index * n_genes_per_plot
                end_index = min(start_index + n_genes_per_plot, len(all_genes))
                genes_to_plot = all_genes[start_index:end_index]

                fig, axs = plt.subplots(1, len(genes_to_plot), figsize=(width, width / 5))
                axs = axs.flatten()

                for i, gene in enumerate(genes_to_plot):
                    sns.heatmap(
                        confusion_matrices[gene],
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        ax=axs[i],
                        cbar=False,
                        xticklabels=["Predicted \nnot expressed", "Predicted \nexpressed"],
                        yticklabels=["Actual \nnot expressed", "Actual \nexpressed"],
                    )
                    axs[i].set_title(gene)

                # Hide any unused subplots on the last figure if total genes don't fill up the grid
                for j in range(len(genes_to_plot), len(axs)):
                    axs[j].axis("off")

                plt.tight_layout()

                # Save confusion matrices:
                parent_dir = os.path.dirname(self.output_path)
                plt.savefig(os.path.join(parent_dir, f"confusion_matrices_{fig_index}.png"), bbox_inches="tight")

        elif type == "rmse":
            rmse_dict = {}
            nz_rmse_dict = {}

            for i, gene in enumerate(all_genes):
                y = self.adata[:, gene].X.toarray().reshape(-1)
                music_results_target = pred_vals[:, i]
                rmse_dict[gene] = np.sqrt(mean_squared_error(y, music_results_target))

                # Indices where target is nonzero:
                nonzero_indices = y != 0

                nz_rmse_dict[gene] = np.sqrt(
                    mean_squared_error(y[nonzero_indices], music_results_target[nonzero_indices])
                )

            mean_rmse = sum(rmse_dict.values()) / len(rmse_dict.values())
            mean_nz_rmse = sum(nz_rmse_dict.values()) / len(nz_rmse_dict.values())

            data = []
            for gene in rmse_dict.keys():
                data.append({"Gene": gene, "RMSE": rmse_dict[gene], "RMSE (expressing cells)": mean_nz_rmse[gene]})
            # Color palette:
            colors = {"RMSE": "#FF7F00", "RMSE (expressing cells)": "#87CEEB"}
            df = pd.DataFrame(data)

            # Plot RMSE barplot:
            sns.set(font_scale=2)
            sns.set_style("white")
            plt.figure(figsize=(width, 6))
            plt.xticks(rotation="vertical")
            ax = sns.barplot(
                data=df,
                x="Gene",
                y="RMSE",
                palette=colors["RMSE"],
                edgecolor="black",
                dodge=True,
            )

            # Mean line:
            line_style = "--"
            line_thickness = 2
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

            # Plot RMSE barplot (expressing cells):
            plt.figure(figsize=(width, 6))
            plt.xticks(rotation="vertical")
            ax = sns.barplot(
                data=df,
                x="Gene",
                y="RMSE (expressing cells)",
                palette=colors["RMSE (expressing cells)"],
                edgecolor="black",
                dodge=True,
            )

            # Mean line:
            ax.axhline(mean_nz_rmse, color="black", linestyle=line_style, linewidth=line_thickness)

            # Update legend:
            legend_label = f"Mean: {mean_nz_rmse}"
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([0], [0], color="black", linewidth=line_thickness, linestyle=line_style))
            labels.append(legend_label)
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

            plt.title(f"RMSE (expressing cells) {file_name}")
            plt.tight_layout()
            plt.show()

    def visualize_effect_specificity(
        self,
        denominator: Literal["target", "interaction"] = "target",
        n_cells_threshold: int = 200,
        cell_type_specific: bool = False,
        target_subset: Optional[List[str]] = None,
        interaction_subset: Optional[List[str]] = None,
        ct_subset: Optional[List[str]] = None,
        group_key: Optional[str] = None,
        effect_threshold: float = 0.05,
        use_significant: bool = False,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        center: Optional[float] = None,
        cmap: str = "Reds",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
        save_df: bool = False,
    ):
        """Computes and visualizes the specificity of the effect of each interaction on each target. Specificity can
        be defined as the proportion of cells expressing the target that also express the interaction,
        or the proportion of cells expressing the interaction for which the interaction is predicted to effect a
        particular target.

        Args:
            denominator: Either "target" or "interaction". If "target", specificity is defined as the proportion of
                cells expressing the target that also express the interaction. If "interaction", specificity is defined
                as the proportion of cells expressing the interaction for which the interaction is predicted to effect
                a particular target.
            n_cells_threshold: Minimum number of cells expressing the target or interaction, over all cells or over
                cells of each type (if cell_type_specific is True). For interactions/targets below this threshold,
                value will be set to 0.
            cell_type_specific: Whether to factor in cell type-specific expression of each target
            target_subset: List of targets to consider. If None, will use all targets used in model fitting.
            interaction_subset: List of interactions to consider. If None, will use all interactions used in model.
                Is necessary if "plot_type" is "barplot", since the barplot is only designed to accomodate up to three
                interactions at once.
            ct_subset: Only used if "cell_type_specific" is True. If given, will search for cell types in
                "group_key" attribute from model initialization. Recommended to use to subset to cell types with
                sufficient numbers. If not given, will use all cell types.
            group_key: Can be used to specify entry in adata.obs that contains cell type groupings. If None,
                will use :attr `group_key` from model initialization.
            effect_threshold: Threshold minimum effect size to consider an effect for further analysis, as an absolute
                value. Defaults to 0.05.
            use_significant: Whether to use only significant effects in computing the specificity. If True,
                will filter to cells + interactions where the interaction is significant for the target. Only valid
                if :func `compute_coeff_significance()` has been run.
            fontsize: Size of font for x and y labels.
            figsize: Size of figure.
            center: Optional, determines position of the colormap center. Between 0 and 1.
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
            save_df: Set True to save the metric dataframe in the end
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()
        # But set display DPI to 300:
        plt.rcParams["figure.dpi"] = 300

        if denominator not in ["target", "interaction"]:
            raise ValueError(f"Input to 'denominator' must be either 'target' or 'interaction'")

        # If cell type-specific, save the output rather than plotting all of them:
        if cell_type_specific:
            logger.info("Cell type-specific analysis: saving output rather than plotting.")
            save_show_or_return = "save"
            figure_folder = os.path.join(os.path.dirname(self.output_path), "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)
        if save_df:
            output_folder = os.path.join(os.path.dirname(self.output_path), "analyses")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # Colormap should be sequential:
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
            logger.info("Colormap should be sequential: using 'viridis'.")
            cmap = "viridis"

        if cell_type_specific:
            if group_key is None:
                group_key = self.group_key

            if isinstance(ct_subset, str):
                ct_subset = [ct_subset]
            if ct_subset is None:
                adata = self.adata.copy()
            else:
                adata = self.adata[self.adata.obs[group_key].isin(ct_subset)].copy()
            cell_types = adata.obs[group_key].unique()
        else:
            adata = self.adata.copy()

        all_targets = list(self.coeffs.keys())
        all_feature_names = [feat for feat in self.feature_names if feat != "intercept"]
        if isinstance(interaction_subset, str):
            interaction_subset = [interaction_subset]
        feature_names = all_feature_names if interaction_subset is None else interaction_subset
        # Filter out extremely uncommon interactions:
        feature_names = [feature for feature in feature_names if np.sum(self.X_df[feature] > 0) > n_cells_threshold]

        if isinstance(target_subset, str):
            target_subset = [target_subset]
        targets = all_targets if target_subset is None else target_subset
        # Filter out extremely uncommon targets:
        targets = [target for target in targets if np.sum(adata[:, target].X > 0) > n_cells_threshold]

        if fontsize is None:
            fontsize = rcParams.get("font.size")

        if figsize is None:
            # Set figure size based on the number of interaction features and cell type-target combos:
            m = len(targets) * 40 / 300
            n = len(feature_names) * 40 / 300
            figsize = (n, m)

        dfs = {}
        if cell_type_specific:
            # Create a dictionary to store information about which cells belong to which cell types:
            cell_types_dict = {}

            for ct in cell_types:
                cell_type_mask = adata.obs[self.group_key] == ct
                cell_type_mask = adata.obs[cell_type_mask].index
                cell_types_dict[ct] = cell_type_mask
                dfs[ct] = pd.DataFrame(0, index=targets, columns=feature_names)
        else:
            dfs["all"] = pd.DataFrame(0, index=targets, columns=feature_names)

        expr = pd.DataFrame(adata[:, targets].X.toarray(), index=adata.obs.index, columns=targets)

        # Fill in all dataframes:
        for dict_name, df in dfs.items():
            if denominator == "target":
                # Get the number of cells expressing each target that are also affected by each interaction:
                if dict_name == "all":
                    cells_expressing_targets = expr > 0
                    title = (
                        "Number of target-expressing cells affected by interaction \n"
                        "vs. total number of target-expressing cells"
                    )
                else:
                    cell_type = dict_name
                    cells_expressing_targets = expr.loc[cell_types_dict[cell_type], targets] > 0
                    title = (
                        f"Number of target-expressing cells affected by interaction \nvs. total "
                        f"number of target-expressing cells of type {cell_type}"
                    )

                for i, target in enumerate(targets):
                    cells_expressing_target = cells_expressing_targets.iloc[:, i]
                    coeffs = (
                        self.coeffs[target]
                        if dict_name == "all"
                        else self.coeffs[target].loc[cell_types_dict[cell_type]]
                    )
                    if use_significant:
                        parent_dir = os.path.dirname(self.output_path)
                        sig = pd.read_csv(
                            os.path.join(parent_dir, "significance", f"{target}_is_significant.csv"), index_col=0
                        )
                        coeffs *= sig

                    for j, feature in enumerate(feature_names):
                        if f"b_{feature}" in coeffs.columns:
                            cells_expressing_target_and_feature = cells_expressing_target & (
                                coeffs[f"b_{feature}"] > effect_threshold
                            )
                            intersection = np.sum(cells_expressing_target_and_feature)
                            denom = np.sum(cells_expressing_target)
                            if denom != 0:
                                df.loc[target, feature] = intersection / denom
                            else:
                                df.loc[target, feature] = 0
                        else:
                            df.loc[target, feature] = 0

            else:
                # Get the number of cells
                "filler"

            # Hiearchical clustering of both targets and interactions:
            col_linkage = sch.linkage(df.transpose(), method="ward", metric="euclidean")
            col_dendro = sch.dendrogram(col_linkage, no_plot=True)
            col_clustered_order = col_dendro["leaves"]
            df = df.iloc[:, col_clustered_order]

            row_linkage = sch.linkage(df, method="ward")
            row_dendro = sch.dendrogram(row_linkage, no_plot=True)
            row_clustered_order = row_dendro["leaves"]
            df = df.iloc[row_clustered_order, :]

            dfs[dict_name] = df

            # Set up heatmap for visualization:
            if self.mod_type == "lr":
                x_label = "L:R Interaction"
            elif self.mod_type == "ligand":
                x_label = "Ligand"
            elif self.mod_type == "receptor":
                x_label = "Receptor"

            vmin = 0
            vmax = 1
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            thickness = 0.3 * figsize[0] / 10
            mask = df == 0
            m = sns.heatmap(
                df,
                square=True,
                linecolor="grey",
                linewidths=thickness,
                cbar_kws={"label": "Proportion", "location": "top"},
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                ax=ax,
            )

            # Outer frame:
            for _, spine in m.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(thickness * 2.5)

            # Adjust colorbar label font size
            cbar = m.collections[0].colorbar
            cbar.set_label("Proportion", fontsize=fontsize * 1.5, labelpad=20)
            # Adjust colorbar tick font size
            cbar.ax.tick_params(labelsize=fontsize * 1.25)
            cbar.ax.set_aspect(0.033)

            ax.set_xlabel(x_label, fontsize=fontsize * 1.25)
            ax.set_ylabel("Cell Type-Specific Target", fontsize=fontsize * 1.25)
            ax.tick_params(axis="x", labelsize=fontsize, rotation=90)
            ax.tick_params(axis="y", labelsize=fontsize)
            ax.set_title(title, fontsize=fontsize * 1.5, pad=20)

            # Use the saved name for the AnnData object to define part of the name of the saved file:
            base_name = os.path.basename(self.adata_path)
            adata_id = os.path.splitext(base_name)[0]
            prefix = f"{adata_id}_{dict_name}_{denominator}_specificity_of_effects"

            save_kwargs["ext"] = "png"
            save_kwargs["dpi"] = 300
            save_kwargs["path"] = figure_folder
            save_return_show_fig_utils(
                save_show_or_return=save_show_or_return,
                show_legend=False,
                background="white",
                prefix=prefix,
                save_kwargs=save_kwargs,
                total_panels=1,
                fig=fig,
                axes=ax,
                return_all=False,
                return_all_list=None,
            )

            if save_df:
                df.to_csv(os.path.join(output_folder, f"{prefix}.csv"))

    def explore_interaction_effect_coverage(
        self,
        target: str,
        reference_interactions: Optional[List[str]] = None,
        effect_threshold: float = 0.05,
        use_significant: bool = False,
        upper_threshold: float = 0.5,
        lower_cooccurence_threshold: float = 0.1,
        lower_specificity_threshold: float = 0.1,
    ) -> anndata.AnnData:
        """Given a set of reference interaction(s), finds cells expressing the target in the absence of any reference
        interactions and determines interaction effects that are enriched in the remainder of the cells,
        in ranked order (with two lists- ranking determined first by highest enrichment in the remaining cells and
        also by least amount of overlap with the reference interactions).

        Args:
            target: Target gene of interest
            reference_interactions: List of reference interactions- will find cells for which the reference
                interaction is predicted to have an effect and then find interactions that are enriched in the rest.
            effect_threshold: Threshold for the effect size of an interaction/effect to be considered for analysis;
                only used if "to_plot" is "percentage". Defaults to 0.0.
            use_significant: Whether to use only significant effects in computing the specificity. If True,
                will filter to cells + interactions where the interaction is significant for the target. Only valid
                if :func `compute_coeff_significance()` has been run.
            upper_threshold: Upper threshold for the proportion of cells affected by a particular interaction for the
                interaction to be considered for this analysis. Defaults to 0.5.
            lower_cooccurence_threshold: Lower threshold for the proportion of target-expressing cells that are
                affected by a particular interaction to be included in the printed output. Defaults to 0.1.
            lower_specificity_threshold: Lower threshold for the proportion of cells affected by a particular
                interaction

        Returns:
            adata: Will store results in .obs entry "interaction_effect_coverage", which will label cells accordingly
                based on the given reference interactions.
        """
        logger = lm.get_main_logger()
        adata = self.adata.copy()
        coef_target = self.coeffs[target]

        if use_significant:
            parent_dir = os.path.dirname(self.output_path)
            sig = pd.read_csv(os.path.join(parent_dir, "significance", f"{target}_is_significant.csv"), index_col=0)
            coef_target *= sig

        # Filter out interactions present in at least 50% of cells- although important they are less likely to be
        # especially interesting:
        coef_target = coef_target.loc[:, np.sum(coef_target > 0, axis=0) < upper_threshold * coef_target.shape[0]]

        coeff_target_filter = coef_target > effect_threshold
        coef_target *= coeff_target_filter
        interaction_rows = coef_target[reference_interactions].any(axis=1)
        cells_expressing_target = pd.DataFrame(adata[:, target].X.toarray().reshape(-1) > 0, index=adata.obs_names)
        coef_ref_interaction_target = interaction_rows & cells_expressing_target[target]

        # Cells expressing target in absence of reference interactions:
        cells_expressing_target_no_ref = cells_expressing_target[target] & ~coef_ref_interaction_target
        # Rank the most prevalent co-occurring interactions:
        coef_non_ref = coef_target[[col for col in coef_target.columns if col not in reference_interactions]]
        coef_non_ref_interaction_target = coef_non_ref.loc[cells_expressing_target_no_ref].astype(bool)
        interaction_cooccurrence = coef_non_ref_interaction_target.sum() / cells_expressing_target_no_ref.sum()
        interaction_cooccurrence.sort_values(ascending=False, inplace=True)
        interaction_cooccurrence = interaction_cooccurrence[interaction_cooccurrence > lower_cooccurence_threshold]
        logger.info(f"Top interactions affecting {target} in absence of reference interactions, ranked: ")
        print(interaction_cooccurrence)

        # And the co-occurring interactions that are most specific to these cells:
        interactions_df = self.X_df.loc[cells_expressing_target_no_ref, coef_non_ref_interaction_target.columns].astype(
            bool
        )
        interaction_specificity = coef_non_ref_interaction_target.sum() / interactions_df.sum()
        interaction_specificity = interaction_specificity[interaction_specificity > lower_specificity_threshold]
        if len(interaction_specificity) > 0:
            logger.info(
                f"No interactions are highly specific to target-expressing cells that are not affected by "
                f"{reference_interactions}. Try lowering threshold from {lower_specificity_threshold}."
            )
        else:
            interaction_specificity.sort_values(ascending=False, inplace=True)
            logger.info(
                f"Top interactions specific to {target}-expressing cells that are not affected by "
                f"{reference_interactions}, ranked: "
            )
            print(interaction_specificity)

        # Define AnnData .obs to label cells by their reference interactions:
        # For each reference interaction, normalize the effect size by minmax scaling:
        coef_target_mod = coef_target * coef_ref_interaction_target
        percentile_997 = coef_target_mod.quantile(0.997)
        coef_target_mod = coef_target_mod.clip(upper=percentile_997, axis=1)
        coef_target_mod = (coef_target_mod - coef_target_mod.min()) / (coef_target_mod.max() - coef_target_mod.min())
        max_interaction = coef_target_mod.idxmax(axis=1)
        # If the second largest interaction effect is half the max, label as "Multiple interactions"
        max_value = coef_target_mod.max(axis=1)
        half_max_value = max_value / 2
        mask_multiple_interactions = coef_target_mod.apply(
            lambda row: row.nlargest(2).values[-1] >= half_max_value[row.name], axis=1
        )
        labels = [f"{interaction} effect region" for interaction in max_interaction]
        labels = np.where(mask_multiple_interactions, "Multiple interactions", labels)
        adata.obs[f"interaction_effect_on_{target}_coverage"] = labels
        return adata

    def visualize_neighborhood(
        self,
        target: str,
        interaction: str,
        use_significant: bool = False,
        n_anchors: int = 100,
        n_neighbors_expressing: int = 20,
        display_plot: bool = True,
    ) -> anndata.AnnData:
        """Sets up AnnData object for visualization of interaction effects- cells will be colored by expression of
        the target gene, potentially conditioned on receptor expression, and neighboring cells will be colored by
        ligand expression.

        Args:
            target: Target gene of interest
            interaction: Interaction feature to visualize, given in the same form as in the design matrix (if model
                is a ligand-based model or receptor-based model, this will be of form "Col4a1". If model is a
                ligand-receptor based model, this will be of form "Col4a1:Itgb1", for example).
            use_significant: Whether to use only significant effects in computing the specificity. If True,
                will filter to cells + interactions where the interaction is significant for the target. Only valid
                if :func `compute_coeff_significance()` has been run.
            n_anchors: Number of target gene-expressing cells to use as anchors for visualization. Will be selected
                randomly from the set of target gene-expressing cells.
            n_neighbors_expressing: Filters the set of cells that can be selected as anchors based on the number of
                their neighbors that express the chosen ligand. Only used for models that incorporate ligand expression.
            display_plot: Whether to save a plot. If False, will return the AnnData object without doing
                anything else- this can then be visualized e.g. using spateo-viewer.

        Returns:
            adata: Modified AnnData object containing the expression information for the target gene and neighboring
                ligand expression.
        """
        # Compute connectivity matrix if not already existing- only needed for ligand and L:R models:
        from ..find_neighbors import neighbors

        logger = lm.get_main_logger()

        if display_plot:
            figure_folder = os.path.join(os.path.dirname(self.output_path), "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)
            path = os.path.join(figure_folder, f"{target}_example_neighborhoods.html")
            logger.info(f"Saving plot to {path}")

        if self.mod_type != "lr" and self.mod_type != "ligand":
            logger.info("This function is only applicable for ligand-based models.")

        if "spatial_connectivities" in self.adata.obsp.keys():
            conn = self.adata.obsp["spatial_connectivities"]
        else:
            logger.info("Spatial graph not found, computing...")
            conn = None

        if conn is None:
            _, adata = neighbors(
                self.adata,
                n_neighbors=self.n_neighbors_membrane_bound * 2,
                basis="spatial",
                spatial_key=self.coords_key,
                n_neighbors_method="ball_tree",
            )
            conn = adata.obsp["spatial_connectivities"]

        adata = self.adata.copy()
        coef_target = self.coeffs[target]
        if use_significant:
            parent_dir = os.path.dirname(self.output_path)
            sig = pd.read_csv(os.path.join(parent_dir, "significance", f"{target}_is_significant.csv"), index_col=0)
            coef_target *= sig

        # Compute the multiple possible masks that can be used to subset to the cells of interest:
        # Get the target gene expression:
        target_expression = adata[:, target].X.toarray().reshape(-1)
        # Get the interaction effect:
        interaction_effect = coef_target[f"b_{interaction}"].values

        # Get the cells expressing the target gene:
        target_expressing_mask = target_expression > 0
        target_expressing_cells = adata.obs_names[target_expressing_mask]
        # Cells with significant interaction effect on target:
        interaction_mask = interaction_effect > 0
        interaction_cells = adata.obs_names[interaction_mask]

        # If applicable, split the interaction feature and get the ligand and receptor- for features w/ multiple
        # ligands or multiple receptors, process accordingly:
        to_check = interaction.split(":")[0] if ":" in interaction else interaction
        if "/" in to_check:
            genes = to_check.split("/")
            separator = "/"
        elif "_" in to_check:
            genes = to_check.split("_")
            separator = "_"
        else:
            genes = [to_check]
            separator = None

        if separator == "/":
            # Cells expressing any of the genes
            ligand_expr_mask = np.zeros(len(adata), dtype=bool)
            for gene in genes:
                ligand_expr_mask |= adata[:, gene].X.toarray().squeeze() > 0
        elif separator == "_":
            # Cells expressing all of the genes
            ligand_expr_mask = np.ones(len(adata), dtype=bool)
            for gene in genes:
                ligand_expr_mask &= adata[:, gene].X.toarray().squeeze() > 0
        else:
            # Single gene
            ligand_expr_mask = adata[:, to_check].X.toarray().squeeze() > 0

        # Check how many cells have sufficient number of neighbors expressing the ligand:
        neighbor_counts = np.zeros(len(adata))
        for i in range(len(adata)):
            # Get neighbors
            neighbors = conn[i].nonzero()[1]
            neighbor_counts[i] = np.sum(ligand_expr_mask[neighbors])

        # Get the cells with sufficient number of neighbors expressing the ligand:
        cells_meeting_neighbor_ligand_threshold = adata.obs_names[neighbor_counts > n_neighbors_expressing]

        if self.mod_type == "lr":
            to_check = interaction.split(":")[1] if ":" in interaction else interaction
            if "_" in to_check:
                genes = to_check.split("_")
                separator = "_"
            else:
                genes = [to_check]
                separator = None

            if separator == "_":
                # Cells expressing all of the genes
                receptor_expr_mask = np.ones(len(adata), dtype=bool)
                for gene in genes:
                    receptor_expr_mask &= adata[:, gene].X.toarray().squeeze() > 0
            else:
                # Single gene
                receptor_expr_mask = adata[:, to_check].X.toarray().squeeze() > 0
            # Get the cells expressing the receptor, to further subset the target-expressing cells to also :
            receptor_expressing_cells = adata.obs_names[receptor_expr_mask]

        if self.mod_type == "lr":
            # Get the intersection of cells expressing target, predicted to be affected by interaction,
            # with sufficient number of neighbors expressing the chosen ligand and expressing receptor:
            adata_mask = (
                target_expressing_cells
                & interaction_cells
                & cells_meeting_neighbor_ligand_threshold
                & receptor_expressing_cells
            )
        else:
            # Get the intersection of cells expressing target, predicted to be affected by interaction,
            # and with sufficient number of neighbors expressing the chosen ligand:
            adata_mask = target_expressing_cells & interaction_cells & cells_meeting_neighbor_ligand_threshold
        adata_sub = adata[adata_mask].copy()

        # Randomly choose a subset of target cells to use as anchors:
        target_expressing_selected = np.random.choice(adata_sub.obs_names, size=n_anchors, replace=False)
        selected_indices = [np.where(adata.obs_names == string)[0][0] for string in target_expressing_selected]
        # Find the neighbors of these anchor cells:
        neighbors = conn[selected_indices].nonzero()[1]
        neighbors = np.unique(neighbors)
        # Remove the anchor cells from the neighbors:
        neighbors = neighbors[~np.isin(neighbors, selected_indices)]
        neighbors_selected = adata.obs_names[neighbors]

        # Target expression in the selected cells:
        target_expression = adata_sub[target_expressing_selected, target].X.toarray().squeeze()
        # Ligand expression in the neighbors:
        ligand = interaction.split(":")[0] if ":" in interaction else interaction
        genes = ligand.split("/")
        gene_values = adata[neighbors_selected, genes].X.toarray()
        if separator == "/":
            # Arithmetic mean of the genes
            ligand_expression = np.mean(gene_values, axis=1)
        elif separator == "_":
            # Geometric mean of the genes
            # Replace zeros with np.nan
            gene_values[gene_values == 0] = np.nan
            # Compute product along the rows
            products = np.nanprod(gene_values, axis=1)
            # Count non-nan values in each row for nth root calculation
            non_nan_counts = np.sum(~np.isnan(gene_values), axis=1)
            # Avoid division by zero
            non_nan_counts[non_nan_counts == 0] = np.nan
            ligand_expression = np.power(products, 1 / non_nan_counts)
        else:
            ligand_expression = adata[neighbors_selected, ligand].X.toarray().squeeze()

        adata.obs[f"{interaction}_{target}_example_points"] = 0.0
        adata.obs.loc[target_expressing_selected, f"{interaction}_{target}_example_points"] = target_expression
        adata.obs.loc[neighbors_selected, f"{interaction}_{target}_example_points"] = ligand_expression

        if display_plot:
            # plotly to create 3D scatter plot:
            spatial_coords = adata.obsm[self.coords_key]
            if spatial_coords.shape[1] == 2:
                x, y = spatial_coords[:, 0], spatial_coords[:, 1]
                z = np.zeros(len(x))
            else:
                x, y, z = spatial_coords[:, 0], spatial_coords[:, 1], spatial_coords[:, 2]

            # Color assignment:
            default_color = "#D3D3D3"
            # Apply 'cool' colormap to target_expression
            target_cmap = plt.cm.get_cmap("cool")
            target_colors = target_cmap(
                adata.obs.loc[target_expressing_selected, f"{interaction}" f"_{target}_example_points"]
            )
            scatter_target = go.Scatter3d(
                x=x[selected_indices],
                y=y[selected_indices],
                z=z[selected_indices],
                mode="markers",
                marker=dict(color=target_colors, size=1.5, colorbar=dict(title=f"{target} Expression")),
            )

            # Apply 'hot' colormap to ligand_expression
            ligand_cmap = plt.cm.get_cmap("hot")
            ligand_colors = ligand_cmap(adata.obs.loc[neighbors_selected, f"{interaction}_{target}_example_points"])
            scatter_ligand = go.Scatter3d(
                x=x[neighbors],
                y=y[neighbors],
                z=z[neighbors],
                mode="markers",
                marker=dict(color=ligand_colors, size=1.5, colorbar=dict(title=f"{ligand} Expression")),
            )

            rest_indices = list(set(range(len(x))) - set(selected_indices) - set(neighbors))
            scatter_rest = go.Scatter3d(
                x=x[rest_indices],
                y=y[rest_indices],
                z=z[rest_indices],
                mode="markers",
                marker=dict(color=default_color, size=1.5),
            )

            # Create the figure and add the scatter plots
            fig = go.Figure(data=[scatter_rest, scatter_target, scatter_ligand])

            # Turn off the grid
            fig.update_layout(
                scene=dict(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False)),
                margin=dict(l=0, r=0, b=0, t=30),  # Adjust margins to minimize spacing
                title=dict(text=f"Target: {target}, Ligand: {ligand}", y=0.9, yanchor="top", x=0.5, xanchor="center"),
            )
            fig.write_html(path)

        return adata

    def cell_type_specific_interactions(
        self,
        to_plot: Literal["mean", "percentage"] = "mean",
        plot_type: Literal["heatmap", "barplot"] = "heatmap",
        group_key: Optional[str] = None,
        ct_subset: Optional[List[str]] = None,
        target_subset: Optional[List[str]] = None,
        interaction_subset: Optional[List[str]] = None,
        lower_threshold: float = 0.3,
        upper_threshold: float = 1.0,
        effect_threshold: float = 0.0,
        use_significant: bool = False,
        row_normalize: bool = False,
        col_normalize: bool = False,
        normalize_targets: bool = False,
        hierarchical_cluster_y: bool = False,
        group_y_cell_type: bool = False,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        center: Optional[float] = None,
        cmap: str = "Reds",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
        save_df: bool = False,
    ):
        """Map interactions and interaction effects that are specific to particular cell type groupings. Returns a
        heatmap representing the enrichment of the interaction/effect within cells of that grouping (if "to_plot" is
        effect, this will be enrichment of the effect on cell type-specific expression). Enrichment determined by
        mean effect size or expression.

        Args:
            to_plot: Whether to plot the mean effect size or the proportion of cells in a cell type w/ effect on
                target. Options are "mean" or "percentage".
            plot_type: Whether to plot the results as a heatmap or barplot. Options are "heatmap" or "barplot". If
                "barplot", must provide a subset of up to four interactions to visualize.
            group_key: Can be used to specify entry in adata.obs that contains cell type groupings. If None,
                will use :attr `group_key` from model initialization.
            ct_subset: Can be used to restrict the enrichment analysis to only cells of a particular type. If given,
                will search for cell types in "group_key" attribute from model initialization. Recommended to use to
                subset to cell types with sufficient numbers.
            target_subset: List of targets to consider. If None, will use all targets used in model fitting.
            interaction_subset: List of interactions to consider. If None, will use all interactions used in model.
                Is necessary if "plot_type" is "barplot", since the barplot is only designed to accomodate up to three
                interactions at once.
            lower_threshold: Lower threshold for the proportion of cells in a cell type group that must express a
                particular interaction/effect for it to be colored on the plot, as a proportion of the max value.
                Threshold will be applied to the non-normalized values (if normalization is applicable). Defaults to
                0.3.
            upper_threshold: Upper threshold for the proportion of cells in a cell type group that must express a
                particular interaction/effect for it to be colored on the plot, as a proportion of the max value.
                Threshold will be applied to the non-normalized values (if normalization is applicable). Defaults to
                1.0 (the max value).
            effect_threshold: Threshold for the effect size of an interaction/effect to be considered for analysis;
                only used if "to_plot" is "percentage". Defaults to 0.0.
            use_significant: Whether to use only significant effects in computing the specificity. If True,
                will filter to cells + interactions where the interaction is significant for the target. Only valid
                if :func `compute_coeff_significance()` has been run.
            row_normalize: Whether to minmax scale the metric values by row (i.e. for each interaction/effect). Helps
                to alleviate visual differences that result from scale rather than differences in mean value across
                cell types.
            col_normalize: Whether to minmax scale the metric values by column (i.e. for each interaction/effect). Helps
                to alleviate visual differences that result from scale rather than differences in mean value across
                cell types.
            normalize_targets: Whether to minmax scale the metric values by column for each target (i.e. for each
                interaction/effect), to remove differences that occur as a result of scale of expression. Provides a
                clearer picture of enrichment for each target.
            hierarchical_cluster_y: Whether to cluster the y-axis (target gene in cell type) using hierarchical
                clustering. If False, will order the y-axis by the order of the target genes for organization purposes.
            group_y_cell_type: Whether to group the y-axis (target gene in cell type) by cell type. If False,
                will group by target gene instead. Defaults to False.
            fontsize: Size of font for x and y labels.
            figsize: Size of figure.
            center: Optional, determines position of the colormap center. Between 0 and 1.
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
            save_df: Set True to save the metric dataframe in the end
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()
        # But set display DPI to 300:
        plt.rcParams["figure.dpi"] = 300

        if to_plot not in ["mean", "percentage"]:
            raise ValueError("Unrecognized input for plotting. Options are 'mean' or 'percentage'.")

        if plot_type == "barplot" and interaction_subset is None:
            raise ValueError("Must provide a subset of interactions to visualize if 'plot_type' is 'barplot'.")
        if plot_type == "barplot" and len(interaction_subset) > 4:
            raise ValueError(
                "Can only visualize up to four interactions at once with 'barplot' (for practical/plot "
                "readability reasons)."
            )

        if self.mod_type not in ["lr", "ligand", "receptor"]:
            raise ValueError("Model type must be one of 'lr', 'ligand', or 'receptor'.")

        if save_show_or_return in ["save", "both", "all"]:
            figure_folder = os.path.join(os.path.dirname(self.output_path), "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)

        if save_df:
            output_folder = os.path.join(os.path.dirname(self.output_path), "analyses")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # Colormap should be sequential:
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
            logger.info(f"For option {to_plot}, colormap should be sequential: using 'viridis'.")
            cmap = "viridis"

        if group_key is None:
            group_key = self.group_key

        # Get appropriate adata:
        if isinstance(ct_subset, str):
            ct_subset = [ct_subset]

        if ct_subset is None:
            adata = self.adata.copy()
        else:
            adata = self.adata[self.adata.obs[group_key].isin(ct_subset)].copy()
        cell_types = adata.obs[group_key].unique()

        all_targets = list(self.coeffs.keys())
        all_feature_names = [feat for feat in self.feature_names if feat != "intercept"]
        if isinstance(interaction_subset, str):
            interaction_subset = [interaction_subset]
        feature_names = all_feature_names if interaction_subset is None else interaction_subset

        if fontsize is None:
            fontsize = rcParams.get("font.size")

        if isinstance(target_subset, str):
            target_subset = [target_subset]
        targets = all_targets if target_subset is None else target_subset
        combinations = product(cell_types, targets)
        combinations = [f"{ct}-{target}" for ct, target in combinations]

        if figsize is None:
            if plot_type == "heatmap":
                # Set figure size based on the number of interaction features and cell type-target combos:
                m = len(combinations) * 40 / 300
                n = len(feature_names) * 40 / 300
            else:
                # Set figure size based on the number of cell type-target combos:
                n = len(combinations) * 40 / 300
                m = (n / 3) * len(interaction_subset)
            figsize = (n, m)

        df = pd.DataFrame(0, index=combinations, columns=feature_names)
        for ct in cell_types:
            cell_type_mask = adata.obs[self.group_key] == ct
            cell_type_mask = adata.obs[cell_type_mask].index

            # Get appropriate coefficient arrays:
            for target in targets:
                if to_plot == "mean":
                    mean_effects = []
                    coef_target = self.coeffs[target]
                    if use_significant:
                        parent_dir = os.path.dirname(self.output_path)
                        sig = pd.read_csv(
                            os.path.join(parent_dir, "significance", f"{target}_is_significant.csv"), index_col=0
                        )
                        coef_target *= sig

                    for feat in feature_names:
                        if f"b_{feat}" in coef_target.columns:
                            # Get mean effect size for each interaction feature in each cell type:
                            mean_effects.append(coef_target.loc[cell_type_mask, f"b_{feat}"].values.mean())
                        else:
                            mean_effects.append(0)
                    df.loc[f"{ct}-{target}", :] = mean_effects
                elif to_plot == "percentage":
                    percentages = []
                    coef_target = self.coeffs[target]

                    for feat in feature_names:
                        if f"b_{feat}" in coef_target.columns:
                            # Get percentage of cells in each cell type that express each interaction feature:
                            percentages.append(
                                (coef_target.loc[cell_type_mask, f"b_{feat}"].values > effect_threshold).mean()
                            )
                        else:
                            percentages.append(0)
                    df.loc[f"{ct}-{target}", :] = percentages

        # Apply metric threshold (do this in a grouped manner, for each target):
        # Split the index to get the targets portion of the tuples
        grouping_element = df.index.map(lambda x: x.split("-")[1])
        # Compute the maximum (and optionally used minimum) for each group
        group_max = df.groupby(grouping_element).max()
        # Apply the threshold in a grouped fashion
        for group in group_max.index:
            # Select the rows belonging to the current group
            group_rows = df.index[df.index.str.contains(f"-{group}$")]

            # Apply the lower threshold specific to this group
            df.loc[group_rows] = df.loc[group_rows].where(
                df.loc[group_rows].ge(lower_threshold * group_max.loc[group]), 0
            )

            if normalize_targets:
                # Take 0 to be the min. value in all cases:
                df.loc[group_rows] = df.loc[group_rows] / group_max.loc[group]

        if upper_threshold != 1.0:
            df[df >= upper_threshold * df.max().max()] = df.max().max()

        # Optionally, normalize each row by minmax scaling (to get an idea of the top effects for each target),
        # or each column by minmax scaling:
        if row_normalize or col_normalize or normalize_targets:
            normalize = True
        else:
            normalize = False

        if row_normalize:
            # Calculate row-wise min and max
            row_min = df.min(axis=1).values.reshape(-1, 1)
            row_max = df.max(axis=1).values.reshape(-1, 1)

            df = (df - row_min) / (row_max - row_min)
        elif col_normalize:
            df = (df - df.min()) / (df.max() - df.min())
        df.fillna(0, inplace=True)
        # Hierarchical clustering- first to group interactions w/ similar patterns across cell types:
        col_linkage = sch.linkage(df.transpose(), method="ward")
        col_dendro = sch.dendrogram(col_linkage, no_plot=True)
        col_clustered_order = col_dendro["leaves"]
        df = df.iloc[:, col_clustered_order]

        # Then to group cell types w/ similar interaction patterns, if specified:
        if hierarchical_cluster_y:
            row_linkage = sch.linkage(df, method="ward")
            row_dendro = sch.dendrogram(row_linkage, no_plot=True)
            row_clustered_order = row_dendro["leaves"]
            df = df.iloc[row_clustered_order, :]
        else:
            # Sort by target:
            # Create a temporary MultiIndex
            df.index = pd.MultiIndex.from_tuples(df.index.str.split("-").map(tuple), names=["first", "second"])
            if group_y_cell_type:
                # Sort by the first element, then the second
                df.sort_index(level=["first", "second"], inplace=True)
            else:
                # Sort by the second element, then the first
                df.sort_index(level=["second", "first"], inplace=True)
            # Revert to the original index format
            df.index = df.index.map("-".join)
        df = df.loc[:, ~(df == 0).all()]

        if normalize and to_plot == "mean":
            if plot_type == "heatmap":
                label = (
                    "Normalized avg. effect per cell type"
                    if not normalize_targets
                    else "Normalized avg. effect per cell type (normalized within target)"
                )
            else:
                label = (
                    "Normalized avg. effect\n per cell type"
                    if not normalize_targets
                    else "Normalized avg. effect\n per cell type \n(normalized within target)"
                )
        elif normalize and to_plot == "percentage":
            if plot_type == "heatmap":
                label = (
                    "Normalized enrichment of effect per cell type"
                    if not normalize_targets
                    else "Normalized enrichment of effect per cell type (normalized within target)"
                )
            else:
                label = (
                    "Normalized enrichment of\n effect per cell type"
                    if not normalize_targets
                    else "Normalized enrichment of\n effect per cell type\n (normalized within target)"
                )
        elif not normalize and to_plot == "mean":
            label = "Avg. effect per cell type" if plot_type == "heatmap" else "Avg. effect\n per cell type"
        else:
            label = (
                "Enrichment of effect per cell type"
                if plot_type == "heatmap"
                else "Enrichment of effect\n per cell type"
            )

        if self.mod_type == "lr":
            x_label = "Interaction"
            title = "Enrichment of L:R interaction in each cell type"
        elif self.mod_type == "ligand":
            x_label = "Neighboring ligand expression"
            title = "Enrichment of neighboring ligand expression in each cell type for each target"
        elif self.mod_type == "receptor":
            x_label = "Receptor expression"
            title = "Enrichment of receptor expression in each cell type"

        # Formatting color legend:
        if group_y_cell_type:
            group_labels = [idx.split("-")[0] for idx in df.index]
        else:
            group_labels = [idx.split("-")[1] for idx in df.index]

        target_colors = plt.cm.get_cmap("tab20").colors
        if group_y_cell_type:
            color_mapping = {
                annotation: target_colors[i % len(target_colors)] for i, annotation in enumerate(set(cell_types))
            }
        else:
            color_mapping = {
                annotation: target_colors[i % len(target_colors)] for i, annotation in enumerate(set(targets))
            }
        max_annotation_length = max([len(annotation) for annotation in color_mapping.keys()])
        if max_annotation_length > 30:
            ax2_size = "30%"
        elif max_annotation_length > 20:
            ax2_size = "20%"
        else:
            ax2_size = "10%"

        if plot_type == "heatmap":
            # Plot heatmap:
            vmin = 0
            vmax = 1 if normalize else df.max().max()

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            divider = make_axes_locatable(ax)
            ax2 = divider.append_axes("right", size=ax2_size, pad=0)

            # Keep track of groups:
            current_group = None
            group_start = None

            # Color legend:
            for i, annotation in enumerate(group_labels):
                if annotation != current_group:
                    if current_group is not None:
                        group_center = len(df) - ((group_start + i - 1) / 2) - 1
                        ax2.text(0.22, group_center, current_group, va="center", ha="left", fontsize=fontsize)

                    current_group = annotation
                    group_start = i

                color = color_mapping[annotation]
                ax2.add_patch(plt.Rectangle((0, i), 0.2, 1, color=color))
            # Add label for the last group:
            group_center = len(df) - ((group_start + len(df) - 1) / 2) - 1
            ax2.text(0.22, group_center, current_group, va="center", ha="left", fontsize=fontsize)
            ax2.set_ylim(0, len(df.index))
            ax2.axis("off")

            thickness = 0.3 * figsize[0] / 10
            mask = df == 0
            m = sns.heatmap(
                df,
                square=True,
                linecolor="grey",
                linewidths=thickness,
                cbar_kws={"label": label, "location": "top", "pad": 0},
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                ax=ax,
            )

            # Outer frame:
            for _, spine in m.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(thickness * 2.5)

            # Adjust colorbar label font size
            cbar = m.collections[0].colorbar
            cbar.set_label(label, fontsize=fontsize * 1.5, labelpad=10)
            # Adjust colorbar tick font size
            cbar.ax.tick_params(labelsize=fontsize * 1.25)
            cbar.ax.set_aspect(0.033)

            ax.set_xlabel(x_label, fontsize=fontsize * 1.25)
            ax.set_ylabel("Cell Type-Specific Target", fontsize=fontsize * 1.25)
            ax.tick_params(axis="x", labelsize=fontsize, rotation=90)
            ax.tick_params(axis="y", labelsize=fontsize)
            ax.set_title(title, fontsize=fontsize * 1.5, pad=20)

            # Use the saved name for the AnnData object to define part of the name of the saved file:
            base_name = os.path.basename(self.adata_path)
            adata_id = os.path.splitext(base_name)[0]
            prefix = f"{adata_id}_{to_plot}_enrichment_cell_type"
        else:
            fig, axes = plt.subplots(nrows=len(interaction_subset), ncols=1, figsize=figsize)
            colormap = plt.cm.get_cmap(cmap)
            # Determine the order of the plot based on averaging over the chosen interactions (if there is more than
            # one):
            df_sub = df[interaction_subset]
            df_sub["Group"] = group_labels
            # Ranks within each group:
            grouped_ranked_df = df_sub.groupby("Group").rank(ascending=False)
            grouped_ranked_df.drop("Group", axis=1, inplace=True)
            # Average rank across groups:
            avg_ranked_df = grouped_ranked_df.mean()
            # Sort by average rank:
            sorted_features = avg_ranked_df.sort_values().index.tolist()
            df = df[sorted_features]

            # Color legend:
            divider = make_axes_locatable(axes[0])
            ax2 = divider.append_axes("top", size=ax2_size, pad=0)

            current_group = None
            group_start = None

            for i, annotation in enumerate(group_labels):
                if annotation != current_group:
                    if current_group is not None:
                        group_center = (group_start + i - 1) / 2
                        ax2.text(group_center, 0.22, current_group, va="bottom", ha="center", fontsize=fontsize)

                    current_group = annotation
                    group_start = i

                color = color_mapping[annotation]
                ax2.add_patch(plt.Rectangle((i, 0.5), 1, 0.2, color=color))

            # Add label for the last group:
            group_center = (group_start + len(df) - 1) / 2
            ax2.text(group_center, 0.22, current_group, va="bottom", ha="center", fontsize=fontsize)
            ax2.set_xlim(0, len(df.index))
            ax2.axis("off")

            for i, ax in enumerate(axes):
                # From the larger dataframe, get the column for the chosen interaction as a series:
                interaction = interaction_subset[i]
                interaction_series = df[interaction]

                vmin = 0
                vmax = 1 if normalize else interaction_series.max()
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                colors = [colormap(norm(val)) for val in interaction_series]
                sns.barplot(
                    x=interaction_series.index,
                    y=interaction_series.values,
                    edgecolor="black",
                    linewidth=1,
                    palette=colors,
                    ax=ax,
                )
                ax.set_xlabel("Cell Type-Specific Target", fontsize=fontsize * 1.25)
                ax.set_ylabel(label, fontsize=fontsize * 1.25)
                ax.set_title(interaction, fontsize=fontsize * 1.5, pad=20)

                if ax is axes[-1]:
                    ax.tick_params(axis="x", labelsize=fontsize, rotation=90)

            # Use the saved name for the AnnData object to define part of the name of the saved file:
            base_name = os.path.basename(self.adata_path)
            adata_id = os.path.splitext(base_name)[0]
            prefix = f"{adata_id}_{to_plot}_enrichment_cell_type"

        # Save figure:
        save_kwargs["ext"] = "png"
        save_kwargs["dpi"] = 300
        save_kwargs["path"] = figure_folder
        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=False,
            background="white",
            prefix=prefix,
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

        if save_df:
            df.to_csv(os.path.join(output_folder, f"{prefix}.csv"))

    def cell_type_interaction_fold_change(
        self,
        ref_ct: str,
        query_ct: str,
        group_key: Optional[str] = None,
        target_subset: Optional[List[str]] = None,
        interaction_subset: Optional[List[str]] = None,
        to_plot: Literal["mean", "percentage"] = "mean",
        plot_type: Literal["volcano", "barplot"] = "barplot",
        source_data: Literal["interaction", "effect", "target"] = "effect",
        top_n_to_plot: Optional[int] = None,
        significance_cutoff: float = 1.3,
        fold_change_cutoff: float = 1.5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "seismic",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
        save_df: bool = False,
    ):
        """Computes fold change in predicted interaction effects between two cell types, and visualizes result.

        Args:
            ref_ct: Label of the first cell type to consider. Fold change will be computed with respect to the level
                in this cell type.
            query_ct: Label of the second cell type to consider
            group_key: Name of the key in .obs containing cell type information. If not given, will use
                :attr `group_key` from model initialization.
            target_subset: List of targets to consider. If None, will use all targets used in model fitting.
            interaction_subset: List of interactions to consider. If None, will use all interactions used in model.
            to_plot: Whether to plot the mean effect size or the proportion of cells in a cell type w/ effect on
                target. Options are "mean" or "percentage".
            plot_type: Whether to plot the results as a volcano plot or barplot. Options are "volcano" or "barplot".
            source_data: Selects what to use in computing fold changes. Options:
                - "interaction": will use the design matrix (e.g. neighboring ligand expression or L:R mapping)
                - "effect": will use the coefficient arrays for each target
                - "target": will use the target gene expression
            top_n_to_plot: If given, will only include the top n features in the visualization. Recommended if
                "source_data" is "effect", as all combinations of interaction and target will be considered in this
                case.
            significance_cutoff: Cutoff for negative log-10 q-value to consider an interaction/effect significant. Only
                used if "plot_type" is "volcano". Defaults to 1.3 (corresponding to an approximate q-value of 0.05).
            fold_change_cutoff: Cutoff for fold change to consider an interaction/effect significant. Only used if
                "plot_type" is "volcano". Defaults to 1.5.
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
            save_df: Set True to save the metric dataframe in the end
        """
        config_spateo_rcParams()
        # But set display DPI to 300:
        plt.rcParams["figure.dpi"] = 300

        parent_dir = os.path.dirname(self.output_path)
        if save_show_or_return in ["save", "both", "all"]:
            figure_folder = os.path.join(parent_dir, "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)

        if save_df:
            output_folder = os.path.join(os.path.dirname(self.output_path), "analyses")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        if fontsize is None:
            fontsize = rcParams.get("font.size")

        if group_key is None:
            group_key = self.group_key

        if target_subset is None:
            target_subset = self.targets_expr.columns
        if interaction_subset is None:
            interaction_subset = self.feature_names

        ref_names = self.adata[self.adata.obs[group_key] == ref_ct].obs_names
        query_names = self.adata[self.adata.obs[group_key] == query_ct].obs_names

        def assign_significance(row):
            if row["qval"] < 0.0001:
                return "****"
            elif row["qval"] < 0.001:
                return "***"
            elif row["qval"] < 0.01:
                return "**"
            elif row["qval"] < 0.05:
                return "*"
            else:
                return ""

        # Series/dataframes for each group:
        if source_data == "interaction":
            ref_data = self.X_df.loc[ref_names, interaction_subset]
            query_data = self.X_df.loc[query_names, interaction_subset]
        elif source_data == "effect":
            # Coefficients for all targets in subset:
            for target in target_subset:
                if target not in self.coeffs.keys():
                    raise ValueError(f"Target {target} not found in model.")
                else:
                    coef_target = self.coeffs[target]
                    coef_target.columns = coef_target.columns.str.replace("b_", "")
                    coef_target = coef_target[[col for col in coef_target.columns if col != "intercept"]]
                    coef_target.columns = [replace_col_with_collagens(col) for col in coef_target.columns]
                    coef_target.columns = [f"{col}-> target {target}" for col in coef_target.columns]
                    if "effect_df" not in locals():
                        effect_df = self.coeffs[target].loc[:, interaction_subset]
                    else:
                        effect_df = pd.concat([effect_df, coef_target.loc[:, interaction_subset]], axis=1)

            ref_data = effect_df.loc[ref_names, :]
            query_data = effect_df.loc[query_names, :]
        elif source_data == "target":
            ref_data = self.targets_expr.loc[ref_names, target_subset]
            query_data = self.targets_expr.loc[query_names, target_subset]

        else:
            raise ValueError(
                f"Unrecognized input for source_data: {source_data}. Options are 'interaction', 'effect', or 'target'."
            )

        # Compute significance for each column:
        pvals = []
        for col in ref_data.columns:
            pvals.append(ttest_ind(ref_data[col], query_data[col])[1])
        # Correct for multiple hypothesis testing:
        qvals = multitesting_correction(pvals, method="fdr_bh")
        results = pd.DataFrame(qvals, index=ref_data.columns, columns=["qval"])
        results["Significance"] = qvals.apply(assign_significance, axis=1)
        # Negative log q-value (in the case of volcano plot):
        results["-log10(qval)"] = -np.log10(qvals)
        # Cut off values larger than 8:
        results.loc[results["-log10(qval)"] > 8, "-log10(qval)"] = 8

        if to_plot == "mean":
            ref_data = ref_data.mean(axis=0)
            query_data = query_data.mean(axis=0)
        elif to_plot == "percentage":
            ref_data = (ref_data > 0).mean(axis=0)
            query_data = (query_data > 0).mean(axis=0)

        if source_data == "effect":
            x_label = f"$\\log_2$(Fold change effect on {target} \n{ref_ct} and {query_ct})"
            title = f"Fold change effect on {target} \n{ref_ct} and {query_ct}"
            if self.mod_type == "lr":
                y_label = f"L:R effect on {target}"
            elif self.mod_type == "ligand":
                y_label = f"Ligand effect on {target}"
        elif source_data == "interaction":
            x_label = f"$\\log_2$(Fold change interaction enrichment \n {ref_ct} and {query_ct})"
            title = f"Fold change interaction enrichment \n{ref_ct} and {query_ct}"
            if self.mod_type == "lr":
                y_label = "L:R interaction"
            elif self.mod_type == "ligand":
                y_label = "Ligand"
        elif source_data == "target":
            x_label = f"$\\log_2$(Fold change target expression \n {ref_ct} and {query_ct})"
            title = f"Fold change target expression \n {ref_ct} and {query_ct}"
            y_label = "Target"

        # Compute fold change:
        fold_change = query_data / ref_data
        results["Fold Change"] = fold_change
        # Take the log of the fold change:
        results["Fold Change"] = np.log2(results["Fold Change"])
        results = results.sort_values("Fold Change")
        if top_n_to_plot is not None:
            results = results.iloc[:top_n_to_plot, :]

        # Plot:
        if figsize is None:
            # Set figure size based on the number of interaction features and targets:
            m = len(results) / 2
            n = m / 2
            figsize = (n, m)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        cmap = plt.cm.get_cmap(cmap)
        # Center colormap at 0:
        max_distance = max(abs(results["Fold Change"]).max(), abs(results["Fold Change"]).min())
        norm = plt.Normalize(-max_distance, max_distance)
        colors = cmap(norm(results["Fold Change"]))

        if plot_type == "barplot":
            barplot = sns.barplot(
                x="Fold Change",
                y=results.index,
                data=results,
                orient="h",
                palette=colors,
                edgecolor="black",
                linewidth=1,
                ax=ax,
            )
            for index, row in results.iterrows():
                ax.text(row["Fold Change"], index, f"{row['Significance']}", color="black", ha="right")

            ax.axvline(x=0, color="grey", linestyle="--", linewidth=2)
            ax.set_xlim(-max_distance * 1.1, max_distance * 1.1)
            ax.set_xticklabels(results.index, fontsize=fontsize)
            ax.set_yticklabels(["{:.2f}".format(y) for y in ax.get_yticks()], fontsize=fontsize)
            ax.set_xlabel(x_label, fontsize=fontsize * 1.25)
            ax.set_ylabel(y_label, fontsize=fontsize * 1.25)
            ax.set_title(title, fontsize=fontsize * 1.5)
        elif plot_type == "volcano":
            if len(results) > 20:
                size = 20
            else:
                size = 40

            significant = results["-log10(qval)"] > significance_cutoff
            significant_up = results["Fold Change"] > fold_change_cutoff
            significant_down = results["Fold Change"] < -fold_change_cutoff

            sns.scatterplot(
                x=results["Fold Change"][significant & significant_up],
                y=results["-log10(qval)"][significant & significant_up],
                hue=results["Fold Change"][significant & significant_up],
                palette="Reds",
                edgecolor="black",
                ax=ax,
                s=size,
            )

            sns.scatterplot(
                x=results["Fold Change"][significant & significant_down],
                y=results["-log10(qval)"][significant & significant_down],
                hue=results["Fold Change"][significant & significant_down],
                palette="Blues",
                edgecolor="black",
                ax=ax,
                s=size,
            )

            sns.scatterplot(
                x=results["Fold Change"][~(significant & (significant_up | significant_down))],
                y=results["-log10(qval)"][~(significant & (significant_up | significant_down))],
                color="grey",
                edgecolor="black",
                ax=ax,
                s=size,
            )

            ax.axhline(y=significance_cutoff, color="grey", linestyle="--", linewidth=1.5)
            ax.axvline(x=fold_change_cutoff, color="grey", linestyle="--", linewidth=1.5)
            ax.axvline(x=-fold_change_cutoff, color="grey", linestyle="--", linewidth=1.5)
            ax.set_xlim(-max_distance * 1.1, max_distance * 1.1)
            ax.set_xticklabels(["{:.2f}".format(x) for x in ax.get_xticks()], fontsize=fontsize)
            ax.set_yticklabels(["{:.2f}".format(y) for y in ax.get_yticks()], fontsize=fontsize)
            ax.set_xlabel(x_label, fontsize=fontsize * 1.25)
            ax.set_ylabel(r"$-log_10$(qval)", fontsize=fontsize * 1.25)
            ax.set_title(title, fontsize=fontsize * 1.5)

        # Use the saved name for the AnnData object to define part of the name of the saved file:
        base_name = os.path.basename(self.adata_path)
        adata_id = os.path.splitext(base_name)[0]
        prefix = f"{adata_id}_fold_changes_{source_data}_{ref_ct}_{query_ct}"
        save_kwargs["ext"] = "png"
        save_kwargs["dpi"] = 300
        save_kwargs["path"] = figure_folder
        # Save figure:
        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=False,
            background="white",
            prefix=prefix,
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def enriched_interactions_barplot(
        self,
        interactions: Optional[Union[str, List[str]]] = None,
        targets: Optional[Union[str, List[str]]] = None,
        effect_size_threshold: float = 0.0,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "Reds",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """Visualize the top predicted effect sizes for each interaction on particular target gene(s).

        Args:
            interactions: Optional subset of interactions to focus on, given in the form ligand(s):receptor(s),
                following the formatting in the design matrix. If not given, will consider all interactions that were
                specified in model fitting.
            targets: Can optionally specify a subset of the targets to compute this on. If not given, will use all
                targets that were specified in model fitting. If multiple targets are given, "save_show_or_return"
                should be "save" (and provide appropriate keyword arguments for saving using "save_kwargs"), otherwise
                only the last target will be shown.
            effect_size_threshold: Lower bound for average effect size to include a particular interaction in the
                barplot
            fontsize: Size of font for x and y labels
            figsize: Size of figure
            cmap: Colormap to use for heatmap. If metric is "number", "proportion", "specificity", the bottom end of
                the range is 0. It is recommended to use a sequential colormap (e.g. "Reds", "Blues", "Viridis",
                etc.). For metric = "fc", if a divergent colormap is not provided, "seismic" will automatically be
                used.
            save_show_or_return: Whether to save, show or return the figure
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        config_spateo_rcParams()
        # But set display DPI to 300:
        plt.rcParams["figure.dpi"] = 300

        if fontsize is None:
            fontsize = rcParams.get("font.size")

        if interactions is None:
            interactions = self.X_df.columns.tolist()
        elif isinstance(interactions, str):
            interactions = [interactions]
        elif not isinstance(interactions, list):
            raise TypeError(f"Interactions must be a list or string, not {type(interactions)}.")

        # Predictions:
        parent_dir = os.path.dirname(self.output_path)
        pred_path = os.path.join(parent_dir, "predictions.csv")
        predictions = pd.read_csv(pred_path, index_col=0)

        if save_show_or_return in ["save", "both", "all"]:
            figure_folder = os.path.join(parent_dir, "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)

        if targets is None:
            targets = self.custom_targets
        elif isinstance(targets, str):
            targets = [targets]
        elif not isinstance(targets, list):
            raise TypeError(f"targets must be a list or string, not {type(targets)}.")

        for target in targets:
            # Get coefficients for this key
            coef = self.coeffs[target]
            effects = coef[[col for col in coef.columns if col.startswith("b_") and "intercept" not in col]]
            effects.columns = [col.split("_")[1] for col in effects.columns]
            # Subset to only the interactions of interest:
            interactions = [interaction for interaction in interactions if interaction in effects.columns]
            effects = effects[interactions]

            # Subset to cells expressing the target that are predicted to be expressing the target:
            target_expr = self.adata[:, target].X.toarray().squeeze() > 0
            target_pred = predictions[target].values.astype(bool)
            target_true_pos_indices = np.where(target_expr & target_pred)[0]
            target_expr_sub = self.adata[target_true_pos_indices, :].copy()

            # Subset effects dataframe to same subset:
            effects_sub = effects.loc[target_expr_sub.obs_names, :]
            # Compute average for each column:
            average_effects = effects_sub.mean(axis=0)

            # Filter based on the threshold
            average_effects = average_effects[average_effects > effect_size_threshold]
            # Sort the average_expression in descending order
            average_effects = average_effects.sort_values(ascending=False)
            if self.mod_type != "ligand":
                average_effects.index = [replace_col_with_collagens(idx) for idx in average_effects.index]
                average_effects.index = [replace_hla_with_hlas(idx) for idx in average_effects.index]

            # Plot:
            if figsize is None:
                # Set figure size based on the number of interaction features and targets:
                m = len(average_effects) / 2
                figsize = (m, 5)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

            palette = sns.color_palette(cmap, n_colors=len(average_effects))
            sns.barplot(
                x=average_effects.index,
                y=average_effects.values,
                edgecolor="black",
                linewidth=1,
                palette=palette,
                ax=ax,
            )
            ax.set_xticklabels(average_effects.index, rotation=90, fontsize=fontsize)
            ax.set_yticklabels(["{:.2f}".format(y) for y in ax.get_yticks()], fontsize=fontsize)
            ax.set_xlabel("Interaction (ligand(s):receptor(s))", fontsize=fontsize)
            ax.set_ylabel("Mean Coefficient \nMagnitude", fontsize=fontsize)
            ax.set_title(f"Average Predicted Interaction Effects on {target}", fontsize=fontsize)

            # Use the saved name for the AnnData object to define part of the name of the saved file:
            base_name = os.path.basename(self.adata_path)
            adata_id = os.path.splitext(base_name)[0]
            prefix = f"{adata_id}_interaction_barplot_{target}"
            save_kwargs["ext"] = "png"
            save_kwargs["dpi"] = 300
            save_kwargs["path"] = figure_folder
            # Save figure:
            save_return_show_fig_utils(
                save_show_or_return=save_show_or_return,
                show_legend=False,
                background="white",
                prefix=prefix,
                save_kwargs=save_kwargs,
                total_panels=1,
                fig=fig,
                axes=ax,
                return_all=False,
                return_all_list=None,
            )

    def partial_correlation_interactions(
        self,
        interactions: Optional[Union[str, List[str]]] = None,
        targets: Optional[Union[str, List[str]]] = None,
        method: Literal["pearson", "spearman"] = "pearson",
        filter_interactions_proportion_threshold: Optional[float] = None,
        plot_zero_threshold: Optional[float] = None,
        ignore_outliers: bool = True,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        center: Optional[float] = None,
        cmap: str = "Reds",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
        save_df: bool = False,
    ):
        """Repression is more difficult to infer from single-cell data- this function computes semi-partial correlations
        to shed light on interactions that may be overall repressive. In this case, for a given interaction-target pair,
        all other interactions are used as covariates in a semi-partial correlation (to account for their effects on
        the target, but not the other interactions which should be more independent of each other compared to the
        target).

        Args:
            interactions: Optional, given in the form ligand(s):receptor(s), following the formatting in the design
                matrix. If not given, will use all interactions that were specified in model fitting.
            targets: Can optionally specify a subset of the targets to compute this on. If not given, will use all
                targets that were specified in model fitting.
            method: Correlation type, options:
                - Pearson :math:`r` product-moment correlation
                - Spearman :math:`\\rho` rank-order correlation
            filter_interactions_proportion_threshold: Optional, if given, will filter out interactions that are
                predicted to occur in below this proportion of cells beforehand (to reduce the number of computations)
            plot_zero_threshold: Optional, if given, will mask out values below this threshold in the heatmap (will
                keep the interactions in the dataframe, just will not color the elements in the plot). Can also be
                used together with filter_interactions_proportion_threshold.
            ignore_outliers: Whether to ignore extremely high values for target gene expression when computing partial
                correlations
            alternative: Defines the alternative hypothesis, or tail of the partial correlation. Must be one of
                "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
                p-value. "greater" tests against the alternative hypothesis that the partial correlation is
                positive (greater than zero), "less" tests against the hypothesis that the partial
                correlation is negative.
            fontsize: Size of font for x and y labels
            figsize: Size of figure
            center: Optional, determines position of the colormap center. Between 0 and 1.
            cmap: Colormap to use for heatmap. If metric is "number", "proportion", "specificity", the bottom end of
                the range is 0. It is recommended to use a sequential colormap (e.g. "Reds", "Blues", "Viridis",
                etc.). For metric = "fc", if a divergent colormap is not provided, "seismic" will automatically be
                used.
            save_show_or_return: Whether to save, show or return the figure
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
            save_df: Set True to save the metric dataframe in the end
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()
        # But set display DPI to 300:
        plt.rcParams["figure.dpi"] = 300

        assert method in [
            "pearson",
            "spearman",
        ], 'only "pearson" and "spearman" are supported for partial correlation.'

        if save_show_or_return in ["save", "both", "all"]:
            figure_folder = os.path.join(os.path.dirname(self.output_path), "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)

        if save_df:
            output_folder = os.path.join(os.path.dirname(self.output_path), "analyses")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # Filter interactions based on prevalence, if specified:
        if filter_interactions_proportion_threshold is not None:
            interaction_proportions = (self.X_df != 0).sum() / self.X_df.shape[0]
            X_df = self.X_df.loc[:, interaction_proportions >= filter_interactions_proportion_threshold]
        else:
            X_df = self.X_df

        if interactions is None:
            interactions = X_df.columns.tolist()
        elif isinstance(interactions, str):
            interactions = [interactions]
        elif not isinstance(interactions, list):
            raise TypeError(f"Interactions must be a list or string, not {type(interactions)}.")

        if not all([c in X_df.columns for c in interactions]):
            logger.warning(
                "Some columns given in 'interactions' are not in dataframe. If "
                "'filter_interactions_proportion_threshold' is given, these may have gotten filtered out. Filtering to "
                "provided interactions that can be found in the dataframe."
            )
            interactions = [c for c in interactions if c in X_df.columns]

        if targets is None:
            targets = self.custom_targets
        elif isinstance(targets, str):
            targets = [targets]
        elif not isinstance(targets, list):
            raise TypeError(f"targets must be a list or string, not {type(interactions)}.")

        # Predictions:
        parent_dir = os.path.dirname(self.output_path)
        pred_path = os.path.join(parent_dir, "predictions.csv")
        predictions = pd.read_csv(pred_path, index_col=0)

        interactions_df = X_df.loc[:, interactions]
        targets_df = pd.DataFrame(self.adata[:, targets].X.toarray(), columns=targets, index=self.adata.obs_names)

        # Check that columns are numeric
        assert all([interactions_df[c].dtype.kind in "bfiu" for c in interactions])
        assert all([targets_df[c].dtype.kind in "bfiu" for c in targets])

        df = pd.concat([interactions_df, targets_df], axis=1)
        n = df.shape[0]  # Number of samples

        # Optionally log-transform to remove the effect of outliers:
        if ignore_outliers:
            df = df.apply(np.log1p)

        # Get all unique combinations of interactions and targets:
        combinations = [f"{i}-{j}" for i in interactions for j in targets]
        all_stats = pd.DataFrame(0, index=combinations, columns=["n_samples", "r", "CI95%"])
        all_stats["n_samples"] = n

        # Compute partial correlations for each interaction-target pair:
        for interaction in interactions:
            other_interactions = [c for c in interactions if c != interaction]
            for target in targets:
                # Subset to cells expressing the target that are predicted to be expressing the target:
                target_expr = self.adata[:, target].X.toarray().squeeze() > 0
                target_pred = predictions[target].values.astype(bool)
                target_true_pos_indices = np.where(target_expr & target_pred)[0]
                target_true_pos_labels = df.index[target_true_pos_indices]

                # The dataframe to compute correlations from will consist of the interaction, the target and all other
                # interactions as covariates:
                data = df.loc[target_true_pos_labels, [interaction, target] + other_interactions]
                k = data.shape[1] - 2  # Number of covariates

                # Compute partial correlation:
                if method == "spearman":
                    # Convert the data to rank, similar to R cov()
                    V = data.rank(na_option="keep").cov()
                else:
                    V = data.cov()

                # Inverse covariance matrix:
                Vi = np.linalg.pinv(V, hermitian=True)
                Vi_diag = Vi.diagonal()
                D = np.diag(np.sqrt(1 / Vi_diag))
                pcor = -1 * (D @ Vi @ D)

                # Semi-partial correlation:
                with np.errstate(divide="ignore"):
                    spcor = (
                        pcor
                        / np.sqrt(np.diag(V))[..., None]
                        / np.sqrt(np.abs(Vi_diag - Vi**2 / Vi_diag[..., None])).T
                    )

                # Remove x covariates
                r = spcor[1, 0]

                # Two-sided confidence interval:
                ci = compute_corr_ci(r, (n - k), confidence=95, decimals=6, alternative=alternative)
                all_stats.loc[f"{interaction}-{target}", "r"] = r
                all_stats.loc[f"{interaction}-{target}", "CI95%"] = ci
        all_stats["Interaction"] = [i.split("-")[0] for i in all_stats.index]
        all_stats["Target"] = [i.split("-")[1] for i in all_stats.index]

        base_name = os.path.basename(self.adata_path)
        adata_id = os.path.splitext(base_name)[0]
        prefix = f"{adata_id}_semipartial_correlations"

        if save_df:
            all_stats.to_csv(os.path.join(output_folder, f"{prefix}_stats.csv"))

        all_stats_to_plot = all_stats.pivot(index="Interaction", columns="Target", values="r")
        if figsize is None:
            # Set figure size based on the number of interaction features and targets:
            m = len(all_stats_to_plot.index) * 40 / 300
            n = len(all_stats_to_plot.columns) * 40 / 300
            figsize = (n, m)

        # Plot heatmap:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        if plot_zero_threshold is not None:
            mask = all_stats_to_plot.abs() < plot_zero_threshold
        else:
            mask = all_stats_to_plot == 0

        vmax = np.max(np.abs(all_stats_to_plot.values))
        m = sns.heatmap(
            all_stats_to_plot,
            square=True,
            linecolor="grey",
            linewidths=0.3,
            cbar_kws={"label": "Partial correlation", "location": "top"},
            cmap=cmap,
            center=center,
            vmin=-vmax,
            vmax=vmax,
            mask=mask,
            ax=ax,
        )

        # Outer frame:
        for _, spine in m.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.75)

        # Adjust colorbar label font size
        cbar = m.collections[0].colorbar
        cbar.set_label("Partial correlation", fontsize=fontsize * 1.1)
        # Adjust colorbar tick font size
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.set_aspect(0.05)

        plt.xlabel("Target gene", fontsize=fontsize * 1.1)
        plt.ylabel("Interaction", fontsize=fontsize * 1.1)
        plt.title("Partial correlation", fontsize=fontsize * 1.25)
        plt.tight_layout()

        # Save figure:
        save_kwargs["ext"] = "png"
        save_kwargs["dpi"] = 300
        save_kwargs["path"] = figure_folder
        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=False,
            background="white",
            prefix=prefix,
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

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
                if "/" in ligand:
                    multi_interaction = True
                    lr_pair = f"{ligand}:{receptor}"
                else:
                    multi_interaction = False
                    lr_pair = (ligand, receptor)
                if lr_pair not in self.lr_pairs and lr_pair not in self.feature_names:
                    raise ValueError(
                        "Invalid ligand-receptor pair given. Check that input to 'lr_pair' is given in "
                        "the form of a tuple."
                    )
            else:
                multi_interaction = False

            # Check if ligand is membrane-bound or secreted:
            matching_rows = self.lr_db[self.lr_db["from"].isin(ligand.split("/"))]
            if (
                matching_rows["type"].str.contains("Secreted Signaling").any()
                or matching_rows["type"].str.contains("ECM-Receptor").any()
            ):
                spatial_weights = spatial_weights_secreted
            else:
                spatial_weights = spatial_weights_membrane_bound

            # Use the non-lagged ligand expression to construct ligand indicator array:
            if not multi_interaction:
                ligand_expr = self.ligands_expr_nonlag[ligand].values.reshape(-1, 1)
            else:
                all_multi_ligands = ligand.split("/")
                ligand_expr = self.ligands_expr_nonlag[all_multi_ligands].mean(axis=1).values.reshape(-1, 1)

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
                if "/" in ligand:
                    ligand = replace_col_with_collagens(ligand)
                    ligand = replace_hla_with_hlas(ligand)

                self.adata.obs[
                    f"norm_sum_sent_effect_potential_{ligand}_for_{target}"
                ] = normalized_effect_potential_sum_sender

                self.adata.obs[
                    f"norm_sum_received_effect_potential_from_{ligand}_for_{target}"
                ] = normalized_effect_potential_sum_receiver

            elif self.mod_type == "lr":
                if "/" in ligand:
                    ligand = replace_col_with_collagens(ligand)
                    ligand = replace_hla_with_hlas(ligand)

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
        all_receivers = list(set(lr_db_subset["to"]))
        all_senders = list(set(lr_db_subset["from"]))

        if self.mod_type == "lr":
            self.logger.info(
                "Computing pathway effect potential for ligand-receptor pairs in pathway, since :attr "
                "`mod_type` is 'lr'."
            )

            # Get ligand-receptor combinations in the pathway from our model:
            valid_lr_combos = []
            for col in self.design_matrix.columns:
                parts = col.split(":")
                if parts[1] in all_receivers:
                    valid_lr_combos.append((parts[0], parts[1]))

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
        max_val: float = 0.05,
    ):
        """Given the pairwise effect potential array, computes the effect vector field.

        Args:
            effect_potential: Sparse array containing computed effect potentials- output from
                :func:`get_effect_potential`
            normalized_effect_potential_sum_sender: Array containing the sum of the effect potentials sent by each
                cell. Output from :func:`get_effect_potential`.
            normalized_effect_potential_sum_receiver: Array containing the sum of the effect potentials received by
                each cell. Output from :func:`get_effect_potential`.
            max_val: Constrains the size of the vector field vectors. Recommended to set within the order of
                magnitude of 1/100 of the desired plot dimensions.
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
        sending_vf = np.clip(sending_vf, -max_val, max_val)

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
        receiving_vf = np.clip(receiving_vf, -max_val, max_val)

        del effect_potential

        # Shorten names if collagens/HLA in "sig":
        sig = replace_col_with_collagens(sig)
        sig = replace_hla_with_hlas(sig)

        self.adata.obsm[f"spatial_effect_sender_vf_{sig}_{target}"] = sending_vf
        self.adata.obsm[f"spatial_effect_receiver_vf_{sig}_{target}"] = receiving_vf

    # ---------------------------------------------------------------------------------------------------
    # Constructing gene regulatory networks
    # ---------------------------------------------------------------------------------------------------
    def CCI_deg_detection_setup(
        self,
        group_key: Optional[str] = None,
        custom_tfs: Optional[List[str]] = None,
        sender_receiver_or_target_degs: Literal["sender", "receiver", "target"] = "sender",
        use_ligands: bool = True,
        use_receptors: bool = False,
        use_pathways: bool = False,
        use_targets: bool = False,
        use_cell_types: bool = False,
        compute_dim_reduction: bool = False,
    ):
        """Computes differential expression signatures of cells with various levels of ligand expression.

        Args:
            group_key: Key to add to .obs of the AnnData object created by this function, containing cell type labels
                for each cell. If not given, will use :attr `group_key`.
            custom_tfs: Optional list of transcription factors to make sure to be included in analysis. If given,
                these TFs will be included among the regulators regardless of the expression-based thresholding done in
                preprocessing.
            sender_receiver_or_target_degs: Only makes a difference if 'use_pathways' or 'use_cell_types' is specified.
                Determines whether to compute DEGs for ligands, receptors or target genes. If 'use_pathways' is True,
                the value of this argument will determine whether ligands or receptors are used to define the model.
                Note that in either case, differential expression of TFs, binding factors, etc. will be computed in
                association w/ ligands/receptors/target genes (only valid if 'use_cell_types' and not 'use_pathways'
                is specified.
            use_ligands: Use ligand array for differential expression analysis. Will take precedent over
                sender/receiver cell type if also provided.
            use_receptors: Use receptor array for differential expression analysis. Will take precedent over
                sender/receiver cell type if also provided.
            use_pathways: Use pathway array for differential expression analysis. Will use ligands in these pathways
                to collectively compute signaling potential score. Will take precedent over sender cell types if
                also provided.
            use_targets: Use target array for differential expression analysis.
            use_cell_types: Use cell types to use for differential expression analysis. If given,
                will preprocess/construct the necessary components to initialize cell type-specific models. Note-
                should be used alongside 'use_ligands', 'use_receptors', 'use_pathways' or 'use_targets' to select
                which molecules to investigate in each cell type.
            compute_dim_reduction: Whether to compute PCA representation of the data subsetted to targets.
        """

        if use_pathways and self.species != "human":
            raise ValueError("Pathway analysis is only available for human samples.")

        if group_key is None:
            group_key = self.group_key

        if (use_ligands and use_receptors) or (use_ligands and use_targets) or (use_receptors and use_targets):
            self.logger.info(
                "Multiple of 'use_ligands', 'use_receptors', 'use_targets' are given as function inputs. Note that "
                "'use_ligands' will take priority."
            )
        if sender_receiver_or_target_degs == "target" and use_pathways:
            raise ValueError("`sender_receiver_or_target_degs` cannot be 'target' if 'use_pathways' is True.")

        # Check if the array of additional molecules to query has already been created:
        output_dir = os.path.dirname(self.output_path)
        file_name = os.path.basename(self.adata_path).split(".")[0]
        if use_ligands:
            targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_ligands.txt")
        elif use_receptors:
            targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_receptors.txt")
        elif use_pathways:
            targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_pathways.txt")
        elif use_targets:
            targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_target_genes.txt")
        elif use_cell_types:
            targets_folder = os.path.join(output_dir, "cci_deg_detection")

        if not os.path.exists(os.path.join(output_dir, "cci_deg_detection")):
            os.makedirs(os.path.join(output_dir, "cci_deg_detection"))

        # Check for existing processed downstream-task AnnData object:
        if os.path.exists(os.path.join(output_dir, "cci_deg_detection", f"{file_name}.h5ad")):
            # Load files in case they are already existent:
            counts_plus = anndata.read_h5ad(os.path.join(output_dir, "cci_deg_detection", f"{file_name}.h5ad"))
            if use_ligands or use_pathways or use_receptors or use_targets:
                with open(targets_path, "r") as file:
                    targets = file.readlines()
            else:
                targets = pd.read_csv(targets_path, index_col=0)
            self.logger.info(
                "Found existing files for downstream analysis- skipping processing. Can proceed by running "
                ":func ~`self.CCI_sender_deg_detection()`."
            )
        # Else generate the necessary files:
        else:
            self.logger.info("Generating and saving AnnData object for downstream analysis...")
            if self.cci_dir is None:
                raise ValueError("Please provide :attr `cci_dir`.")

            if self.species == "human":
                grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            elif self.species == "mouse":
                grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)

            # Subset GRN and other databases to only include TFs that are in the adata object:
            grn = grn[[col for col in grn.columns if col in self.adata.var_names]]

            analyze_pathway_ligands = sender_receiver_or_target_degs == "sender" and use_pathways
            analyze_pathway_receptors = sender_receiver_or_target_degs == "receiver" and use_pathways
            analyze_celltype_ligands = sender_receiver_or_target_degs == "sender" and use_cell_types
            analyze_celltype_receptors = sender_receiver_or_target_degs == "receiver" and use_cell_types
            analyze_celltype_targets = sender_receiver_or_target_degs == "target" and use_cell_types

            if use_ligands or analyze_pathway_ligands or analyze_celltype_ligands:
                database_ligands = list(set(lr_db["from"]))
                l_complexes = [elem for elem in database_ligands if "_" in elem]
                # Get individual components if any complexes are included in this list:
                ligand_set = [l for item in database_ligands for l in item.split("_")]
                ligand_set = [l for l in ligand_set if l in self.adata.var_names]
            elif use_receptors or analyze_pathway_receptors or analyze_celltype_receptors:
                database_receptors = list(set(lr_db["to"]))
                r_complexes = [elem for elem in database_receptors if "_" in elem]
                # Get individual components if any complexes are included in this list:
                receptor_set = [r for item in database_receptors for r in item.split("_")]
                receptor_set = [r for r in receptor_set if r in self.adata.var_names]
            elif use_targets or analyze_celltype_targets:
                target_set = self.targets_expr.columns

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
            elif use_pathways and sender_receiver_or_target_degs == "sender":
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
            elif use_pathways and sender_receiver_or_target_degs == "receiver":
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
            elif use_targets:
                if self.targets_path is not None:
                    with open(self.targets_path, "r") as f:
                        targets = f.read().splitlines()
                else:
                    targets = self.custom_targets
                # Check that all targets can be found in the source AnnData object:
                targets = [t for t in targets if t in self.adata.var_names]
                targets_expr = pd.DataFrame(
                    self.adata[:, targets].X.A if scipy.sparse.issparse(self.adata.X) else self.adata[:, targets].X,
                    index=self.adata.obs_names,
                    columns=targets,
                )
                signal["all"] = targets_expr
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

                    if "ligand_set" in locals():
                        mols = ligand_set
                    elif "receptor_set" in locals():
                        mols = receptor_set
                    elif "target_set" in locals():
                        mols = target_set
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

                self.logger.info("Selecting transcription factors for analysis of differential expression.")

                # Further subset list of additional factors to those that are expressed in at least n% of the cells
                # that are nonzero in cells of interest (use the user input 'target_expr_threshold'):
                n_cells_threshold = int(self.target_expr_threshold * adata.n_obs)

                all_TFs = list(grn.columns)
                if scipy.sparse.issparse(adata.X):
                    nnz_counts = np.array(adata[:, all_TFs].X.getnnz(axis=0)).flatten()
                else:
                    nnz_counts = np.array(adata[:, all_TFs].X.getnnz(axis=0)).flatten()
                all_TFs = [tf for tf, nnz in zip(all_TFs, nnz_counts) if nnz >= n_cells_threshold]
                if custom_tfs is not None:
                    all_TFs.extend(custom_tfs)

                # Also add all TFs that can bind these TFs:
                primary_tf_rows = grn.loc[all_TFs]
                secondary_TFs = primary_tf_rows.columns[(primary_tf_rows == 1).any()].tolist()
                if scipy.sparse.issparse(adata.X):
                    nnz_counts = np.array(adata[:, secondary_TFs].X.getnnz(axis=0)).flatten()
                else:
                    nnz_counts = np.array(adata[:, secondary_TFs].X.getnnz(axis=0)).flatten()
                secondary_TFs = [
                    tf for tf, nnz in zip(secondary_TFs, nnz_counts) if nnz >= int(0.5 * n_cells_threshold)
                ]
                secondary_TFs = [tf for tf in secondary_TFs if tf not in all_TFs]

                regulator_features = all_TFs + secondary_TFs
                # Prioritize those that are most coexpressed with at least one target:
                regulator_expr = pd.DataFrame(
                    adata[:, regulator_features].X.A, index=signal[subset_key].index, columns=regulator_features
                )
                # Dataframe containing target expression:
                ds_targets_df = signal[subset_key]

                def intersection_ratio(signal_col, regulator_col):
                    signal_nonzero = set(signal_col[signal_col != 0].index)
                    regulator_nonzero = set(regulator_col[regulator_col != 0].index)
                    intersection = signal_nonzero.intersection(regulator_nonzero)
                    if len(regulator_nonzero) == 0:
                        return 0
                    return len(intersection) / len(regulator_nonzero)

                # Calculating the intersection ratios and finding top 10 regulators for each signal
                top_regulators = {}

                for signal_column in ds_targets_df.columns:
                    ratios = {
                        regulator_column: intersection_ratio(
                            ds_targets_df[signal_column], regulator_expr[regulator_column]
                        )
                        for regulator_column in regulator_expr.columns
                    }
                    sorted_regulators = sorted(ratios, key=ratios.get, reverse=True)[:10]
                    top_regulators[signal_column] = sorted_regulators

                # Final set of top regulators:
                regulator_features = list(set(reg for regs in top_regulators.values() for reg in regs))

                # If custom TFs were filtered out this way, re-add them:
                if custom_tfs is not None:
                    regulator_features.extend(custom_tfs)
                    regulator_features = list(set(regulator_features))

                self.logger.info(
                    f"For this dataset, marked {len(regulator_features)} transcription factors of interest."
                )

                # Take subset of AnnData object corresponding to these regulators:
                counts = adata[:, regulator_features].copy()

                # Convert to dataframe:
                counts_df = pd.DataFrame(counts.X.toarray(), index=counts.obs_names, columns=counts.var_names)
                # combined_df = pd.concat([counts_df, signal[subset_key]], axis=1)

                # Store the targets (ligands/receptors) to AnnData object, save to file path:
                counts_targets = anndata.AnnData(scipy.sparse.csr_matrix(signal[subset_key].values))
                counts_targets.obs_names = signal[subset_key].index
                counts_targets.var_names = signal[subset_key].columns
                targets = signal[subset_key].columns
                # Make note that certain columns are pathways and not individual molecules that can be found in the
                # AnnData object:
                if use_pathways:
                    counts_targets.uns["target_type"] = "pathway"
                elif use_ligands or (use_cell_types and sender_receiver_or_target_degs == "sender"):
                    counts_targets.uns["target_type"] = "ligands"
                elif use_receptors or (use_cell_types and sender_receiver_or_target_degs == "receiver"):
                    counts_targets.uns["target_type"] = "receptors"
                elif use_targets or (use_cell_types and sender_receiver_or_target_degs == "target"):
                    counts_targets.uns["target_type"] = "target_genes"

                if compute_dim_reduction:
                    # To compute PCA, first need to standardize data:
                    sig_sub_df = signal[subset_key]
                    sig_sub_df = np.log1p(sig_sub_df)
                    sig_sub_df = (sig_sub_df - sig_sub_df.mean()) / sig_sub_df.std()

                    # Optionally, can use dimensionality reduction to aid in computing the nearest neighbors for the
                    # model (cells that are nearby in dimensionally-reduced signaling space will be neighbors in
                    # this scenario).
                    # Compute latent representation of the AnnData subset:

                    # Compute the ideal number of UMAP components to use- use half the number of features as the
                    # max possible number of components:
                    self.logger.info("Computing optimal number of PCA components ...")
                    n_pca_components = find_optimal_pca_components(sig_sub_df.values, TruncatedSVD)

                    # Perform UMAP reduction with the chosen number of components, store in AnnData object:
                    _, X_pca = pca_fit(sig_sub_df.values, TruncatedSVD, n_components=n_pca_components)
                    counts_targets.obsm["X_pca"] = X_pca
                    self.logger.info("Computed dimensionality reduction for gene expression targets.")

                # Compute the "Jaccard array" (recording expressed/not expressed):
                counts_targets.obsm["X_jaccard"] = np.where(signal[subset_key].values > 0, 1, 0)
                cell_types = self.adata.obs.loc[signal[subset_key].index, self.group_key]
                counts_targets.obs[group_key] = cell_types

                if self.total_counts_key is not None:
                    counts_targets.obs[self.total_counts_key] = self.adata.obs.loc[
                        signal[subset_key].index, self.total_counts_key
                    ]

                # Iterate over regulators:
                regulators = counts_df.columns
                # Add each target to AnnData .obs field:
                for reg in regulators:
                    counts_targets.obs[f"regulator_{reg}"] = counts_df[reg].values

                if "targets_path" in locals():
                    # Save to .txt file:
                    with open(targets_path, "w") as file:
                        for t in targets:
                            file.write(t + "\n")
                else:
                    if use_ligands or (use_cell_types and sender_receiver_or_target_degs == "sender"):
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_ligands.txt")
                    elif use_receptors or (use_cell_types and sender_receiver_or_target_degs == "receiver"):
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_receptors.txt")
                    elif use_pathways:
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_pathways.txt")
                    elif use_targets or (use_cell_types and sender_receiver_or_target_degs == "target"):
                        targets_path = os.path.join(targets_folder, f"{file_name}_{subset_key}_target_genes.txt")
                    with open(targets_path, "w") as file:
                        for t in targets:
                            file.write(t + "\n")

                if "ligand_set" in locals():
                    id = "ligand_regulators"
                elif "receptor_set" in locals():
                    id = "receptor_regulators"
                elif "target_set" in locals():
                    id = "target_gene_regulators"

                self.logger.info(
                    "'CCI_sender_deg_detection'- saving regulatory molecules to test as .h5ad file to the "
                    "directory of the output..."
                )
                counts_targets.write_h5ad(
                    os.path.join(output_dir, "cci_deg_detection", f"{file_name}_{subset_key}_{id}.h5ad")
                )

    def CCI_deg_detection(
        self,
        group_key: str,
        cci_dir_path: str,
        sender_receiver_or_target_degs: Literal["sender", "receiver", "target"] = "sender",
        use_ligands: bool = True,
        use_receptors: bool = False,
        use_pathways: bool = False,
        use_targets: bool = False,
        cell_type: Optional[str] = None,
        use_dim_reduction: bool = False,
        **kwargs,
    ):
        """Downstream method that when called, creates a separate instance of :class `MuSIC` specifically designed
        for the downstream task of detecting differentially expressed genes associated w/ ligand expression.

        Args:
            group_key: Key in `adata.obs` that corresponds to the cell type (or other grouping) labels
            cci_dir_path: Path to directory containing all Spateo databases
            sender_receiver_or_target_degs: Only makes a difference if 'use_pathways' or 'use_cell_types' is specified.
                Determines whether to compute DEGs for ligands, receptors or target genes. If 'use_pathways' is True,
                the value of this argument will determine whether ligands or receptors are used to define the model.
                Note that in either case, differential expression of TFs, binding factors, etc. will be computed in
                association w/ ligands/receptors/target genes (only valid if 'use_cell_types' and not 'use_pathways'
                is specified.
            use_ligands: Use ligand array for differential expression analysis. Will take precedent over receptors and
                sender/receiver cell types if also provided. Should match the input to :func
                `CCI_sender_deg_detection_setup`.
            use_receptors: Use receptor array for differential expression analysis.
            use_pathways: Use pathway array for differential expression analysis. Will use ligands in these pathways
                to collectively compute signaling potential score. Will take precedent over sender cell types if also
                provided. Should match the input to :func `CCI_sender_deg_detection_setup`.
            use_targets: Use target genes array for differential expression analysis.
            cell_type: Cell type to use to use for differential expression analysis. If given, will use the
                ligand/receptor subset obtained from :func ~`CCI_deg_detection_setup` and cells of the chosen
                cell type in the model.
            use_dim_reduction: Whether to use PCA representation of the data to find nearest neighbors. If False,
                will instead use the Jaccard distance. Defaults to False. Note that this will ultimately fail if
                dimensionality reduction was not performed in :func ~`CCI_deg_detection_setup`.
            kwargs: Keyword arguments for any of the Spateo argparse arguments. Should not include 'adata_path',
                'custom_lig_path' & 'ligand' or 'custom_pathways_path' & 'pathway' (depending on whether ligands or
                pathways are being used for the analysis), and should not include 'output_path' (which will be
                determined by the output path used for the main model). Should also not include any of the other
                arguments for this function

        Returns:
            downstream_model: Fitted model instance that can be used for further downstream applications
        """
        logger = lm.get_main_logger()

        if use_pathways and self.species != "human":
            raise ValueError("Pathway analysis is only available for human samples.")

        kwargs["mod_type"] = "downstream"
        kwargs["cci_dir"] = cci_dir_path
        kwargs["species"] = self.species
        kwargs["group_key"] = group_key
        kwargs["coords_key"] = "X_pca" if use_dim_reduction else "X_jaccard"
        kwargs["bw_fixed"] = True
        kwargs["total_counts_threshold"] = self.total_counts_threshold
        kwargs["total_counts_key"] = self.total_counts_key

        # Use the same output directory as the main model, add folder demarcating results from downstream task:
        output_dir = os.path.dirname(self.output_path)
        output_file_name = os.path.basename(self.output_path)
        if not os.path.exists(os.path.join(output_dir, "cci_deg_detection")):
            os.makedirs(os.path.join(output_dir, "cci_deg_detection"))

        if use_ligands or use_receptors or use_pathways or use_targets:
            file_name = os.path.basename(self.adata_path).split(".")[0]
            if use_ligands:
                id = "ligand_regulators"
                file_id = "ligand_analysis"
            elif use_receptors:
                id = "receptor_regulators"
                file_id = "receptor_analysis"
            elif use_pathways and sender_receiver_or_target_degs == "sender":
                id = "ligand_regulators"
                file_id = "pathway_analysis_ligands"
            elif use_pathways and sender_receiver_or_target_degs == "receiver":
                id = "receptor_regulators"
                file_id = "pathway_analysis_receptors"
            elif use_targets:
                id = "target_gene_regulators"
                file_id = "target_gene_analysis"
            if not os.path.exists(os.path.join(output_dir, "cci_deg_detection", file_id)):
                os.makedirs(os.path.join(output_dir, "cci_deg_detection", file_id))
            output_path = os.path.join(output_dir, "cci_deg_detection", file_id, output_file_name)
            kwargs["output_path"] = output_path

            logger.info(
                f"Using AnnData object stored at "
                f"{os.path.join(output_dir, 'cci_deg_detection', f'{file_name}_all_{id}.h5ad')}."
            )
            kwargs["adata_path"] = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_{id}.h5ad")
            if use_ligands:
                kwargs["custom_lig_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_all_ligands.txt"
                )
                logger.info(f"Using ligands stored at {kwargs['custom_lig_path']}.")
            elif use_receptors:
                kwargs["custom_rec_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_all_receptors.txt"
                )
            elif use_pathways:
                kwargs["custom_pathways_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_all_pathways.txt"
                )
                logger.info(f"Using pathways stored at {kwargs['custom_pathways_path']}.")
            elif use_targets:
                kwargs["targets_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_all_target_genes.txt"
                )
                logger.info(f"Using target genes stored at {kwargs['targets_path']}.")
            else:
                raise ValueError("One of 'use_ligands', 'use_receptors', 'use_pathways' or 'use_targets' must be True.")

            # Create new instance of MuSIC:
            parser, args_list = define_spateo_argparse(**kwargs)
            downstream_model = MuSIC(parser, args_list)
            downstream_model._set_up_model()
            downstream_model.fit()
            downstream_model.predict_and_save()

        elif cell_type is not None:
            # For each cell type, fit a different model:
            file_name = os.path.basename(self.adata_path).split(".")[0]

            # create output sub-directory for this model:
            if sender_receiver_or_target_degs == "sender":
                file_id = "ligand_analysis"
            elif sender_receiver_or_target_degs == "receiver":
                file_id = "receptor_analysis"
            elif sender_receiver_or_target_degs == "target":
                file_id = "target_gene_analysis"
            if not os.path.exists(os.path.join(output_dir, "cci_deg_detection", cell_type, file_id)):
                os.makedirs(os.path.join(output_dir, "cci_deg_detection", cell_type, file_id))
            subset_output_dir = os.path.join(output_dir, "cci_deg_detection", cell_type, file_id)
            # Check if directory already exists, if not create it
            if not os.path.exists(subset_output_dir):
                self.logger.info(f"Output folder for cell type {cell_type} does not exist, creating it now.")
                os.makedirs(subset_output_dir)
            output_path = os.path.join(subset_output_dir, output_file_name)
            kwargs["output_path"] = output_path

            kwargs["adata_path"] = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_{cell_type}.h5ad")
            logger.info(f"Using AnnData object stored at {kwargs['adata_path']}.")
            if sender_receiver_or_target_degs == "sender":
                kwargs["custom_lig_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_ligands.txt"
                )
                logger.info(f"Using ligands stored at {kwargs['custom_lig_path']}.")
            elif sender_receiver_or_target_degs == "receiver":
                kwargs["custom_rec_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_receptors.txt"
                )
                logger.info(f"Using receptors stored at {kwargs['custom_rec_path']}.")
            elif sender_receiver_or_target_degs == "target":
                kwargs["targets_path"] = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_target_genes.txt"
                )
                logger.info(f"Using target genes stored at {kwargs['targets_path']}.")

            # Create new instance of MuSIC:
            parser, args_list = define_spateo_argparse(**kwargs)
            downstream_model = MuSIC(parser, args_list)
            downstream_model._set_up_model()
            downstream_model.fit()
            downstream_model.predict_and_save()

        else:
            raise ValueError("'use_ligands' and 'use_pathways' are both False, and 'cell_type' was not given.")

    def visualize_CCI_degs(
        self,
        plot_mode: Literal["proportion", "average"] = "proportion",
        sender_receiver_or_target_degs: Literal["sender", "receiver", "target"] = "sender",
        use_ligands: bool = True,
        use_receptors: bool = False,
        use_pathways: bool = False,
        use_targets: bool = False,
        cell_type: Optional[str] = None,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "seismic",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """Visualize the result of downstream model that maps TFs/other regulatory genes to target genes.

        Args:
            plot_mode: Specifies what gets plotted.
                Options:
                    - "proportion": elements of the plot represent the proportion of total target-expressing cells
                        for which the given factor is predicted to have a nonzero effect
                    - "average": elements of the plot represent the average effect size across all target-expressing
                        cells
            sender_receiver_or_target_degs: Only makes a difference if 'use_pathways' or 'use_cell_types' is specified.
                Determines whether to compute DEGs for ligands, receptors or target genes. If 'use_pathways' is True,
                the value of this argument will determine whether ligands or receptors are used to define the model.
                Note that in either case, differential expression of TFs, binding factors, etc. will be computed in
                association w/ ligands/receptors/target genes (only valid if 'use_cell_types' and not 'use_pathways'
                is specified.
            use_ligands: Set True if this was True for the original model. Used to find the correct output location.
            use_receptors: Set True if this was True for the original model. Used to find the correct output location.
            use_pathways: Set True if this was True for the original model. Used to find the correct output location.
            use_targets: Set True if this was True for the original model. Used to find the correct output location.
            cell_type: Cell type of interest- should be the same as was provided to :func `CCI_deg_detection`.
            figsize: Width and height of plotting window
            cmap: Name of matplotlib colormap specifying colormap to use
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        output_dir = os.path.dirname(self.output_path)
        file_name = os.path.basename(self.adata_path).split(".")[0]
        if save_show_or_return in ["save", "both", "all"]:
            figure_folder = os.path.join(output_dir, "cci_deg_detection", "figures")
            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)

        if use_pathways and self.species != "human":
            raise ValueError("Pathway analysis is only available for human samples.")

        # Load files for all targets:
        if use_ligands or use_receptors or use_pathways or use_targets:
            if use_ligands:
                file_id = "ligand_analysis"
                adata_id = "ligand_regulators"
                plot_id = "Target Ligand"
                title_id = "ligand"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_ligands.txt")
            elif use_receptors:
                file_id = "receptor_analysis"
                adata_id = "receptor_regulators"
                plot_id = "Target Receptor"
                title_id = "receptor"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_receptors.txt")
            elif use_targets:
                file_id = "target_gene_analysis"
                adata_id = "target_gene_regulators"
                plot_id = "Target Gene"
                title_id = "target gene"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_target_genes.txt")
            elif use_pathways and sender_receiver_or_target_degs == "sender":
                file_id = "pathway_analysis_ligands"
                adata_id = "ligand_regulators"
                plot_id = "Target Ligand"
                title_id = "ligand"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_pathways.txt")
            elif use_pathways and sender_receiver_or_target_degs == "receiver":
                file_id = "pathway_analysis_receptors"
                adata_id = "receptor_regulators"
                plot_id = "Target Receptor"
                title_id = "receptor"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_pathways.txt")
        elif cell_type is not None:
            if sender_receiver_or_target_degs == "sender":
                file_id = "ligand_analysis"
                adata_id = "ligand_regulators"
                plot_id = "Target Ligand"
                title_id = "ligand"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_ligands.txt")
            elif sender_receiver_or_target_degs == "receiver":
                file_id = "receptor_analysis"
                adata_id = "receptor_regulators"
                plot_id = "Target Receptor"
                title_id = "receptor"
                targets_path = os.path.join(output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_receptors.txt")
            elif sender_receiver_or_target_degs == "target":
                file_id = "target_analysis"
                adata_id = "target_gene_regulators"
                plot_id = "Target Gene"
                title_id = "target"
                targets_path = os.path.join(
                    output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_target_genes.txt"
                )
        else:
            raise ValueError(
                "'use_ligands', 'use_receptors', 'use_pathways' are all False, and 'cell_type' was not given."
            )
        contents_folder = os.path.join(output_dir, "cci_deg_detection", file_id)

        # Load list of targets:
        with open(targets_path, "r") as f:
            targets = [line.strip() for line in f.readlines()]

        # Filter targets if desired:
        if self.filter_targets:
            predictions = pd.read_csv(
                os.path.join(output_dir, "cci_deg_detection", file_id, "predictions.csv"), index_col=0
            )
            pearson_dict = {}
            for target in targets:
                if target not in predictions.columns:
                    self.logger.info(
                        f"Model for target {target} ran into errors in the fitting process, so no predictions exist."
                    )
                    continue
                observed = self.adata[:, target].X.toarray().reshape(-1)
                predicted = predictions[target].values.reshape(-1)

                rp, _ = pearsonr(observed, predicted)
                pearson_dict[target] = rp

            targets = [target for target in predictions.columns if pearson_dict[target] > self.filter_target_threshold]

        # Complete list of regulatory factors- search through .obs of the AnnData object:
        if cell_type is None:
            adata = anndata.read_h5ad(os.path.join(output_dir, "cci_deg_detection", f"{file_name}_all_{adata_id}.h5ad"))
        else:
            adata = anndata.read_h5ad(
                os.path.join(output_dir, "cci_deg_detection", f"{file_name}_{cell_type}_{adata_id}.h5ad")
            )
        regulator_cols = [col.replace("regulator_", "") for col in adata.obs.columns if "regulator_" in col]

        # Compute proportion or average coefficients for all targets:
        # Load all targets files:
        target_files = {}

        for filename in os.listdir(contents_folder):
            # Check if any of the search strings are present in the filename
            for t in targets:
                if t in filename:
                    filepath = os.path.join(contents_folder, filename)
                    target_file = pd.read_csv(filepath, index_col=0)
                    target_file = target_file[
                        [c for c in target_file.columns if "b_" in c and "intercept" not in c]
                    ].copy()
                    target_file.columns = [c.replace("b_", "") for c in target_file.columns]
                    target_files[t] = target_file

        # Plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        if plot_mode == "proportion":
            all_proportions = pd.DataFrame(0, index=regulator_cols, columns=targets)
            for t, target_df in target_files.items():
                nz_cells = np.where(adata[:, t].X.toarray() > 0)[0]
                proportions = (target_df.iloc[nz_cells] != 0).mean()
                all_proportions.loc[proportions.index, t] = proportions

            if all_proportions.shape[0] < all_proportions.shape[1]:
                to_plot = all_proportions.T
                xlabel = "Regulatory factor"
                ylabel = plot_id
            else:
                to_plot = all_proportions
                xlabel = plot_id
                ylabel = "Regulatory factor"

            mask = to_plot < 0.1
            hmap = sns.heatmap(
                to_plot,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": "Proportion of cells", "location": "top", "shrink": 0.5},
                cmap=cmap,
                center=0.3,
                vmin=0.0,
                vmax=1.0,
                mask=mask,
                ax=ax,
            )

            # Adjust colorbar label font size
            cbar = hmap.collections[0].colorbar
            cbar.set_label("Proportion of cells", fontsize=self.fontsize * 1.1)
            # Adjust colorbar tick font size
            cbar.ax.tick_params(labelsize=self.fontsize)
            cbar.ax.set_aspect(0.05)

            # Outer frame:
            for _, spine in hmap.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.75)

            plt.xlabel(xlabel, fontsize=self.fontsize * 1.1)
            plt.ylabel(ylabel, fontsize=self.fontsize * 1.1)
            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            plt.title(
                f"Preponderance of inferred \n regulatory effect on {title_id} expression",
                fontsize=self.fontsize * 1.25,
            ) if cell_type is None else plt.title(
                f"Preponderance of inferred regulatory \n effect on {title_id} expression in {cell_type}",
                fontsize=self.fontsize * 1.25,
            )
            plt.tight_layout()

        elif plot_mode == "average":
            all_averages = pd.DataFrame(0, index=regulator_cols, columns=targets)
            for t, target_df in target_files.items():
                nz_cells = np.where(adata[:, t].X.toarray() > 0)[0]
                averages = target_df.iloc[nz_cells].mean()
                all_averages.loc[averages.index, t] = averages

            if all_averages.shape[0] < all_averages.shape[1]:
                to_plot = all_averages.T
                xlabel = "Regulatory factor"
                ylabel = plot_id
            else:
                to_plot = all_averages
                xlabel = plot_id
                ylabel = "Regulatory factor"

            q40 = np.percentile(all_averages.values.flatten(), 40)
            q20 = np.percentile(all_averages.values.flatten(), 20)
            mask = to_plot < q20
            hmap = sns.heatmap(
                to_plot,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": "Average effect size", "location": "top", "shrink": 0.5},
                cmap=cmap,
                center=q40,
                vmin=0.0,
                vmax=1.0,
                mask=mask,
                ax=ax,
            )

            # Adjust colorbar label font size
            cbar = hmap.collections[0].colorbar
            cbar.set_label("Average effect size", fontsize=self.fontsize * 1.1)
            # Adjust colorbar tick font size
            cbar.ax.tick_params(labelsize=self.fontsize)
            cbar.ax.set_aspect(0.05)

            # Outer frame:
            for _, spine in hmap.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.75)

            plt.xlabel(xlabel, fontsize=self.fontsize * 1.1)
            plt.ylabel(ylabel, fontsize=self.fontsize * 1.1)
            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            plt.title(
                f"Average inferred \n regulatory effects on {title_id} expression", fontsize=self.fontsize * 1.25
            ) if cell_type is None else plt.title(
                f"Average inferred regulatory effects on {title_id} expression in {cell_type}",
                fontsize=self.fontsize * 1.25,
            )
            plt.tight_layout()

            save_kwargs["ext"] = "png"
            save_kwargs["dpi"] = 300
            save_kwargs["path"] = figure_folder
            save_return_show_fig_utils(
                save_show_or_return=save_show_or_return,
                show_legend=True,
                background="white",
                prefix=f"{plot_mode}_{file_name}_{file_id}",
                save_kwargs=save_kwargs,
                total_panels=1,
                fig=fig,
                axes=ax,
                return_all=False,
                return_all_list=None,
            )

    def visualize_intercellular_network(
        self,
        lr_model_output_dir: str,
        target_subset: Optional[Union[List[str], str]] = None,
        top_n_targets: Optional[int] = 3,
        ligand_subset: Optional[Union[List[str], str]] = None,
        receptor_subset: Optional[Union[List[str], str]] = None,
        regulator_subset: Optional[Union[List[str], str]] = None,
        include_tf_ligand: bool = False,
        include_tf_target: bool = True,
        cell_subset: Optional[Union[List[str], str]] = None,
        select_n_lr: int = 5,
        select_n_tf: int = 3,
        effect_size_threshold: float = 0.2,
        coexpression_threshold: float = 0.2,
        aggregate_method: Literal["mean", "median", "sum"] = "mean",
        cmap_neighbors: str = "autumn",
        cmap_default: str = "winter",
        scale_factor: float = 3,
        layout: Literal["random", "circular", "kamada", "planar", "spring", "spectral", "spiral"] = "planar",
        node_fontsize: int = 8,
        edge_fontsize: int = 8,
        arrow_size: int = 1,
        node_label_position: str = "middle center",
        edge_label_position: str = "middle center",
        upper_margin: float = 40,
        lower_margin: float = 20,
        left_margin: float = 50,
        right_margin: float = 50,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        save_id: Optional[str] = None,
        save_ext: str = "png",
        dpi: int = 300,
    ):
        """After fitting model, construct and visualize the inferred intercellular regulatory network. Effect sizes (
        edge values) will be averaged over cells specified by "cell_subset", otherwise all cells will be used.

        Args:
            lr_model_output_dir: Path to directory containing the outputs of the L:R model. This function will assume
                :attr `output_path` is the output path for the downstream model, i.e. connecting regulatory
                factors/TFs to ligands/receptors/targets.
            target_subset: Optional, can be used to specify target genes downstream of signaling interactions of
                interest. If not given, will use all targets used for the model.
            top_n_targets: Optional, can be used to specify the number of top targets to include in the network
                instead of providing full list of custom targets ("top" judged by fraction of the chosen subset of
                cells each target is expressed in).
            ligand_subset: Optional, can be used to specify subset of ligands. If not given, will use all ligands
                present in any of the interactions for the model.
            receptor_subset: Optional, can be used to specify subset of receptors. If not given, will use all receptors
                present in any of the interactions for the model.
            regulator_subset: Optional, can be used to specify subset of regulators (transcription factors,
                etc.). If not given, will use all regulatory molecules used in fitting the downstream model(s).
            include_tf_ligand: Whether to include TF-ligand interactions in the network. While providing more
                information, this can make it more difficult to interpret the plot. Defaults to False.
            include_tf_target: Whether to include TF-target interactions in the network. While providing more
                information, this can make it more difficult to interpret the plot. Defaults to True.
            cell_subset: Optional, can be used to specify subset of cells to use for averaging effect sizes. If not
                given, will use all cells. Can be either:
                    - A list of cell IDs (must be in the same format as the cell IDs in the adata object)
                    - Cell type label(s)
            select_n_lr: Threshold for filtering out edges with low effect sizes, by selecting up to the top n L:R
                interactions per target (fewer can be selected if the top n are all zero). Default is 5.
            select_n_tf: Threshold for filtering out edges with low effect sizes, by selecting up to the top n
                TFs. For TF-ligand edges, will select the top n for each receptor (with a theoretical maximum of n *
                number of receptors in the graph).
            coexpression_threshold: For receptor-target, TF-ligand, TF-receptor links, only draw edges if the
                molecule pairs in question are coexpressed in > threshold number of cells.
            aggregate_method: Only used when "include_tf_ligand" is True. For the TF-ligand array, each row will
                be replaced by the mean, median or sum of the neighboring rows. Defaults to "mean".
            cmap_neighbors: Colormap to use for nodes belonging to "source"/receiver cells. Defaults to
                yellow-orange-red.
            cmap_default: Colormap to use for nodes belonging to "neighbor"/sender cells. Defaults to
                purple-blue-green.
            scale_factor: Adjust to modify the size of the nodes
            layout: Used for positioning nodes on the plot. Options:
                - "random": Randomly positions nodes ini the unit square.
                - "circular": Positions nodes on a circle.
                - "kamada": Positions nodes using Kamada-Kawai path-length cost-function.
                - "planar": Positions nodes without edge intersections, if possible.
                - "spring": Positions nodes using Fruchterman-Reingold force-directed algorithm.
                - "spectral": Positions nodes using eigenvectors of the graph Laplacian.
                - "spiral": Positions nodes in a spiral layout.
            node_fontsize: Font size for node labels
            edge_fontsize: Font size for edge labels
            arrow_size: Size of the arrow for directed graphs, by default 1
            node_label_position: Position of node labels. Options: 'top left', 'top center', 'top right', 'middle left',
                'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right'
            edge_label_position: Position of edge labels. Options: 'top left', 'top center', 'top right', 'middle left',
                'middle center', 'middle right', 'bottom left', 'bottom center', 'bottom right'
            title: Optional, title for the plot. If not given, will use the AnnData object path to derive this.
            upper_margin: Margin between top of the plot and top of the figure
            lower_margin: Margin between bottom of the plot and bottom of the figure
            left_margin: Margin between left of the plot and left of the figure
            right_margin: Margin between right of the plot and right of the figure
            save_path: Optional, directory to save figure to. If not given, will save to the parent folder of the
                path provided for :attr `output_path` in the argument specification.
            save_id: Optional unique identifier that can be used in saving. If not given, will use the AnnData
                object path to derive this.
            save_ext: File extension to save figure as. Default is "png".
            dpi: Resolution to save figure at. Default is 300.


        Returns:
            G: Graph object, such that it can be separately plotted in interactive window.
            sizing_list: List of node sizes, for use in interactive window.
            color_list: List of node colors, for use in interactive window.
        """

        logger = lm.get_main_logger()
        config_spateo_rcParams()
        # Set display DPI:
        plt.rcParams["figure.dpi"] = dpi

        # Check that self.output_path corresponds to the downstream model if "regulator_subset" is given:
        downstream_model_output_dir = os.path.dirname(self.output_path)
        if (
            not os.path.exists(os.path.join(downstream_model_output_dir, "cci_deg_detection"))
            and regulator_subset is not None
        ):
            raise FileNotFoundError(
                "No downstream model was ever constructed, however this is necessary to include "
                "regulatory factors in the network."
            )

        # Check that lr_model_output_dir points to the correct folder for the L:R model- to do this check for
        # predictions file directly in the folder (for downstream models, predictions are further nested in the
        # "cci_deg_detection" derived subdirectories):
        if not os.path.exists(os.path.join(lr_model_output_dir, "predictions.csv")):
            raise FileNotFoundError(
                "Check that provided `lr_model_output_dir` points to the correct folder for the "
                "L:R model. For example, if the specified model output path is "
                "outer/folder/results.csv, this should be outer/folder."
            )
        lr_model_output_files = os.listdir(lr_model_output_dir)
        # Get L:R names from the design matrix:
        for file in lr_model_output_files:
            path = os.path.join(lr_model_output_dir, file)
            if os.path.isdir(path):
                if file not in ["analyses", "significance", "networks", ".ipynb_checkpoints"]:
                    design_mat = pd.read_csv(os.path.join(path, "design_matrix", "design_matrix.csv"), index_col=0)
        lr_to_target_feature_names = design_mat.columns.tolist()

        # Get spatial weights- only needed if "include_tf_ligand" is True:
        if include_tf_ligand:
            for file in lr_model_output_files:
                path = os.path.join(lr_model_output_dir, file)
                if os.path.isdir(path):
                    if file not in ["analyses", "significance", "networks", ".ipynb_checkpoints"]:
                        spatial_weights_membrane_bound_obj = np.load(
                            os.path.join(path, "spatial_weights", "spatial_weights_membrane_bound.npz")
                        )
                        membrane_bound_data = spatial_weights_membrane_bound_obj["data"]
                        membrane_bound_indices = spatial_weights_membrane_bound_obj["indices"]
                        membrane_bound_indptr = spatial_weights_membrane_bound_obj["indptr"]
                        membrane_bound_shape = spatial_weights_membrane_bound_obj["shape"]

                        spatial_weights_membrane_bound = scipy.sparse.csr_matrix(
                            (membrane_bound_data, membrane_bound_indices, membrane_bound_indptr),
                            shape=membrane_bound_shape,
                        )
                        # Row and column indices of nonzero elements:
                        rows, cols = spatial_weights_membrane_bound.nonzero()
                        all_pairs_membrane_bound = collections.defaultdict(list)
                        for i, j in zip(rows, cols):
                            all_pairs_membrane_bound[i].append(j)

                        spatial_weights_secreted = np.load(
                            os.path.join(path, "spatial_weights", "spatial_weights_secreted.npz")
                        )
                        secreted_data = spatial_weights_secreted["data"]
                        secreted_indices = spatial_weights_secreted["indices"]
                        secreted_indptr = spatial_weights_secreted["indptr"]
                        secreted_shape = spatial_weights_secreted["shape"]

                        spatial_weights_secreted = scipy.sparse.csr_matrix(
                            (secreted_data, secreted_indices, secreted_indptr), shape=secreted_shape
                        )
                        # Row and column indices of nonzero elements:
                        rows, cols = spatial_weights_secreted.nonzero()
                        all_pairs_secreted = collections.defaultdict(list)
                        for i, j in zip(rows, cols):
                            all_pairs_secreted[i].append(j)

        # If subset for ligands and/or receptors is not specified, use all that were included in the model:
        if ligand_subset is None:
            ligand_subset = []
        if receptor_subset is None:
            receptor_subset = []

        for lig_rec in lr_to_target_feature_names:
            lig, rec = lig_rec.split(":")
            lig_split = lig.split("/")
            for l in lig_split:
                if l not in ligand_subset:
                    ligand_subset.append(l)

            if rec not in receptor_subset:
                receptor_subset.append(rec)

        downstream_model_dir = os.path.dirname(self.output_path)

        # Get the names of target genes from the L:R-to-target model from input to lr_model_output_dir:
        all_targets = []
        target_to_file = {}
        for file in lr_model_output_files:
            if file.endswith(".csv") and "predictions" not in file:
                parts = file.split("_")
                target_str = parts[-1].replace(".csv", "")
                # And map the target to the file name:
                target_to_file[target_str] = os.path.join(lr_model_output_dir, file)
                all_targets.append(target_str)

        # Check if any downstream models exist (TF-ligand, TF-receptor, or TF-target):
        if os.path.exists(os.path.join(downstream_model_dir, "cci_deg_detection")):
            if os.path.exists(os.path.join(downstream_model_dir, "cci_deg_detection", "ligand_analysis")):
                # Get the names of target genes from the TF-to-ligand model by looking within the output directory
                # containing :attr `output_path`:
                all_modeled_ligands = []
                ligand_to_file = {}
                ligand_folder = os.path.join(downstream_model_dir, "cci_deg_detection", "ligand_analysis")
                ligand_files = os.listdir(ligand_folder)
                for file in ligand_files:
                    if file.endswith(".csv"):
                        parts = file.split("_")
                        ligand_str = parts[-1].replace(".csv", "")
                        # And map the ligand to the file name:
                        ligand_to_file[ligand_str] = os.path.join(ligand_folder, file)
                        all_modeled_ligands.append(ligand_str)

                # Get TF names from the design matrix:
                for file in ligand_files:
                    path = os.path.join(ligand_folder, file)
                    if file != ".ipynb_checkpoints":
                        if os.path.isdir(path):
                            design_mat = pd.read_csv(
                                os.path.join(path, "downstream_design_matrix", "design_matrix.csv"), index_col=0
                            )
                tf_to_ligand_feature_names = [col.replace("regulator_", "") for col in design_mat.columns]

            if os.path.exists(os.path.join(downstream_model_dir, "cci_deg_detection", "target_gene_analysis")):
                # Get the names of target genes from the TF-to-target model by looking within the output directory
                # containing :attr `output_path`:
                all_modeled_targets = []
                modeled_target_to_file = {}
                target_folder = os.path.join(downstream_model_dir, "cci_deg_detection", "target_gene_analysis")
                target_files = os.listdir(target_folder)
                for file in target_files:
                    if file.endswith(".csv"):
                        parts = file.split("_")
                        target_str = parts[-1].replace(".csv", "")
                        # And map the target to the file name:
                        modeled_target_to_file[target_str] = os.path.join(target_folder, file)
                        all_modeled_targets.append(target_str)

                # Get TF names from the design matrix:
                for file in target_files:
                    path = os.path.join(target_folder, file)
                    if file != ".ipynb_checkpoints":
                        if os.path.isdir(path):
                            design_mat = pd.read_csv(
                                os.path.join(path, "downstream_design_matrix", "design_matrix.csv"), index_col=0
                            )
                tf_to_target_feature_names = [col.replace("regulator_", "") for col in design_mat.columns]

        if save_path is not None:
            save_folder = os.path.join(os.path.dirname(save_path), "networks")
        else:
            save_folder = os.path.join(os.path.dirname(self.output_path), "networks")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if cell_subset is not None:
            if not isinstance(cell_subset, list):
                cell_subset = [cell_subset]

            if all(label in set(self.adata.obs[self.group_key]) for label in cell_subset):
                adata = self.adata[self.adata.obs[self.group_key].isin(cell_subset)].copy()
                # Get numerical indices corresponding to cells in the subset:
                cell_ids = [i for i, name in enumerate(self.adata.obs_names) if name in adata.obs_names]
            else:
                adata = self.adata[cell_subset, :].copy()
                cell_ids = [i for i, name in enumerate(self.adata.obs_names) if name in adata.obs_names]
        else:
            adata = self.adata.copy()
            cell_ids = [i for i, name in enumerate(self.adata.obs_names)]

        targets = all_targets if target_subset is None else target_subset

        # Check for existing dataframes that will be used to construct the network:
        if os.path.exists(os.path.join(save_folder, "lr_to_target.csv")):
            lr_to_target_df = pd.read_csv(os.path.join(save_folder, "lr_to_target.csv"), index_col=0)
        else:
            # Construct L:R-to-target dataframe:
            lr_to_target_df = pd.DataFrame(0, index=lr_to_target_feature_names, columns=targets)

            for target in targets:
                # Load file corresponding to this target:
                file_name = target_to_file[target]
                file_path = os.path.join(lr_model_output_dir, file_name)
                target_df = pd.read_csv(file_path, index_col=0)
                target_df = target_df.loc[:, [col for col in target_df.columns if col.startswith("b_")]]
                # Compute average predicted absolute value effect size over the chosen cell subset to populate
                # L:R-to-target dataframe:
                target_df.columns = [col.replace("b_", "") for col in target_df.columns if col.startswith("b_")]
                lr_to_target_df.loc[:, target] = target_df.iloc[cell_ids, :].abs().mean(axis=0)

            # Save L:R-to-target dataframe:
            lr_to_target_df.to_csv(os.path.join(save_folder, "lr_to_target.csv"))

        # Graph construction:
        G = nx.DiGraph()

        # Identify nodes and edges from L:R-to-target dataframe:
        for target in targets:
            top_n_lr = lr_to_target_df.nlargest(n=select_n_lr, columns=target).index.tolist()
            # Or check if any of the top n should reasonably not be included- compare to :attr target_expr_threshold
            # of the maximum in the array (because these values can be variable):
            reference_value = lr_to_target_df.max().max()
            top_n_lr = [
                lr
                for lr in top_n_lr
                if lr_to_target_df.loc[lr, target] >= (self.target_expr_threshold * reference_value)
            ]

            target_label = f"Target: {target}"
            if not G.has_node(target_label):
                G.add_node(target_label, ID=target_label)

            for lr in top_n_lr:
                ligand_receptor_pair = lr
                ligands, receptor = ligand_receptor_pair.split(":")

                # Check if ligands and receptors are in their respective subsets
                if receptor in receptor_subset and any(lig in ligand_subset for lig in ligands.split("/")):
                    # For ligands separated by "/", check expression of each individual ligand in the AnnData object,
                    # keep ligands that are sufficiently expressed in the specified cell subset:
                    for lig in ligands.split("/"):
                        num_expr = (adata[:, lig].X > 0).sum()
                        expr_percent = num_expr / adata.shape[0]
                        pass_threshold = expr_percent >= self.target_expr_threshold

                        if lig in ligand_subset and pass_threshold:
                            # For the intents of this network, the ligand refers to ligand expressed in neighboring
                            # cells:
                            lig = f"Neighbor {lig}"
                            if not G.has_node(lig):
                                G.add_node(lig, ID=lig)
                            if not G.has_node(receptor):
                                G.add_node(receptor, ID=receptor)
                            G.add_edge(lig, receptor, Type="L:R", Coexpression=None)

                    # Check if receptor and target are coexpressed enough to draw connection:
                    receptor_expression = (adata[:, receptor].X > 0).toarray().flatten()
                    target_expression = (adata[:, target].X > 0).toarray().flatten()
                    coexpression = np.sum(receptor_expression * target_expression)
                    # Threshold based on proportion of cells expressing receptor:
                    num_coexpressed_threshold = coexpression_threshold * np.sum(receptor_expression)
                    if coexpression >= num_coexpressed_threshold:
                        G.add_edge(
                            receptor,
                            target_label,
                            Type="L:R effect",
                            Coexpression=coexpression / np.sum(receptor_expression),
                        )

        # Incorporate TF-to-ligand connections based on existing L:R-target links:
        if "tf_to_ligand_feature_names" in locals() and include_tf_ligand:
            # Intersection between graph ligands and modeled ligands:
            graph_ligands = [node.replace("Neighbor ", "") for node in G.nodes if node.startswith("Neighbor")]
            valid_ligands = [lig for lig in graph_ligands if lig in all_modeled_ligands]
            if len(valid_ligands) == 0:
                raise ValueError(
                    "No modeled ligands are included among the graph ligands. We recommend "
                    "re-running the downstream model with a new set of ligands, taken from the "
                    "L:R interactions used for the L:R-to-target model."
                )

            for ligand in valid_ligands:
                ligand_expression_mask = (adata[:, ligand].X > 0).toarray().flatten()
                file_name = ligand_to_file[ligand]
                file_path = os.path.join(downstream_model_dir, "cci_deg_detection", "ligand_analysis", file_name)
                ligand_df = pd.read_csv(file_path, index_col=0)
                ligand_df = ligand_df.loc[:, [col for col in ligand_df.columns if col.startswith("b_")]]
                ligand_df.columns = [col.replace("b_", "") for col in ligand_df.columns if col.startswith("b_")]
                if regulator_subset is not None:
                    ligand_df = ligand_df.loc[:, [col for col in ligand_df.columns if col in regulator_subset]]

                # For each ligand in the L:R-to-target DF, consider the subset of cells expressing the receptor
                # the ligand is associated with-
                # Construct the TF-to-ligand DF based only on the subset of cells that are neighbors for cells
                # expressing the receptor and that also express the ligand.
                # First determine whether the ligand is secreted or membrane-bound:
                matching_rows = self.lr_db[self.lr_db["from"] == ligand]
                is_secreted = (
                    matching_rows["type"].str.contains("Secreted Signaling").any()
                    or matching_rows["type"].str.contains("ECM-Receptor").any()
                )
                # Select the appropriate dictionary of neighboring cells
                neighbors_dict = all_pairs_secreted if is_secreted else all_pairs_membrane_bound

                # Get receptors on the other end of edges corresponding to this ligand:
                receptors = set([edge[1] for edge in G.edges if edge[0] == f"Neighbor {ligand}"])
                # Find the top n TFs for each receptor (connections will be made from TF to ligand to receptor,
                # so the primary node might be slightly different for each receptor):
                for receptor in receptors:
                    # Indices of cells expressing the receptor:
                    receptor_expression_mask = (adata[:, receptor].X > 0).toarray().flatten()
                    receptor_expressing_cells_indices = np.where(receptor_expression_mask)[0]
                    mean_values_temp = pd.DataFrame(
                        index=range(len(receptor_expressing_cells_indices)), columns=ligand_df.columns
                    )

                    # Processing of ligand_df - rather than measuring TF-ligand connection for each cell,
                    # mean_values_temp will instead reflect the average TF-ligand connection over all neighbors of that
                    # cell.
                    # This is necessary because the network is an intercellular network, so the receptor-target edge
                    # and TF-ligand edge coming from the same cell would be misleading:
                    ligand_df_mod = ligand_df.iloc[receptor_expressing_cells_indices, :]
                    # Get the neighboring cells for each receptor-expressing cell:
                    for i, idx in enumerate(receptor_expressing_cells_indices):
                        # Get neighboring indices for each index from neighbors_dict
                        neighboring_indices = neighbors_dict.get(idx, [])
                        if neighboring_indices:
                            mean_vals_over_neighbors = ligand_df.loc[neighboring_indices, :].mean(axis=0)
                            mean_values_temp.iloc[i, :] = mean_vals_over_neighbors
                    # Get top TFs for this receptor:
                    # First, take mean effect size for each column:
                    mean_tf_ligand_effect_size = mean_values_temp.mean(axis=0)
                    top_n_tf = mean_tf_ligand_effect_size.sort_values(ascending=False).index[:select_n_tf]
                    # For each TF in top_n, if TF and ligand pass coexpression threshold (optional), check if node
                    # exists for TF (if not create it) and add edge from TF to ligand:
                    for tf in top_n_tf:
                        if coexpression_threshold is not None:
                            tf_expression = (adata[:, tf].X > 0).toarray().flatten()
                            coexpression = np.sum(tf_expression * ligand_expression_mask)
                            # Threshold based on proportion of cells expressing ligand:
                            num_coexpressed_threshold = coexpression_threshold * np.sum(ligand_expression_mask)
                            if coexpression < coexpression_threshold:
                                continue
                        tf = f"Neighbor TF: {tf}"
                        if not G.has_node(tf):
                            G.add_node(tf, ID=tf)
                        G.add_edge(
                            tf,
                            f"Neighbor {ligand}",
                            Type="TF:L",
                            Coexpression=coexpression / np.sum(ligand_expression_mask),
                        )

        # Add TF-to-target connections:
        if "tf_to_target_feature_names" in locals() and include_tf_target:
            graph_targets = [node.replace("Target: ", "") for node in G.nodes if node.startswith("Target")]
            valid_targets = [target for target in graph_targets if target in all_modeled_targets]
            if len(valid_targets) == 0:
                raise ValueError(
                    "No modeled targets are included among the graph targets. We recommend "
                    "re-running the downstream model with a new set of targets, taken from the "
                    "targets used for the L:R-to-target model."
                )

            # Construct TF to target dataframe:
            tf_to_target_df = pd.DataFrame(0, index=tf_to_target_feature_names, columns=valid_targets)

            for target in valid_targets:
                target_expression_mask = (adata[:, target].X > 0).toarray().flatten()
                file_name = modeled_target_to_file[target]
                file_path = os.path.join(downstream_model_dir, "cci_deg_detection", "target_gene_analysis", file_name)
                target_df = pd.read_csv(file_path, index_col=0)
                target_df = target_df.loc[:, [col for col in target_df.columns if col.startswith("b_")]]
                target_df.columns = [col.replace("b_", "") for col in target_df.columns if col.startswith("b_")]
                if regulator_subset is not None:
                    target_df = target_df.loc[:, [col for col in target_df.columns if col in regulator_subset]]

                # Compute average predicted effect size over the chosen cell subset to populate the TF-to-target
                # dataframe:
                tf_to_target_df.loc[:, target] = target_df.iloc[cell_ids, :].mean(axis=0)

                # Add edges from TFs to targets:
                top_n_tf = tf_to_target_df.nlargest(n=select_n_tf, columns=target).index.tolist()
                for tf in top_n_tf:
                    if coexpression_threshold is not None:
                        tf_expression = (adata[:, tf].X > 0).toarray().flatten()
                        coexpression = np.sum(tf_expression * target_expression_mask)
                        # Threshold based on proportion of cells expressing target:
                        num_coexpressed_threshold = coexpression_threshold * np.sum(target_expression_mask)
                        if coexpression < coexpression_threshold:
                            continue
                    if not G.has_node(f"TF: {tf}"):
                        G.add_node(f"TF: {tf}", ID=f"TF: {tf}")
                    G.add_edge(
                        f"TF: {tf}",
                        f"Target: {target}",
                        Type="TF:T",
                        Coexpression=coexpression / np.sum(target_expression_mask),
                    )

        # Set colors for nodes- for neighboring cell ligands + TFs, use a distinct colormap (and same w/ receptors,
        # targets and TFs for source cells)- color both on gradient based on number of connections:
        color_list = []
        sizing_list = []
        sizing_neighbor = {}
        sizing_nonneighbor = {}

        cmap_neighbor = plt.cm.get_cmap(cmap_neighbors)
        cmap_non_neighbor = plt.cm.get_cmap(cmap_default)

        # Calculate node degrees and set color and size based on the degree and label
        n_nodes = len(G.nodes())

        for node in G.nodes():
            degree = G.degree(node)
            # Add degree as property:
            G.nodes[node]["Connections"] = degree

            if degree < 3:
                base = 2
            else:
                base = degree

            if n_nodes < 20:
                size_and_color = base * scale_factor
            else:
                size_and_color = np.sqrt(degree) * scale_factor

            # Add size to sizing_list
            if "Neighbor" in node:
                sizing_neighbor[node] = size_and_color
            else:
                sizing_nonneighbor[node] = size_and_color

        for node in G.nodes():
            if "Neighbor" in node:
                color = matplotlib.colors.to_hex(cmap_neighbor(sizing_neighbor[node] / max(sizing_neighbor.values())))
                sizing_list.append(sizing_neighbor[node])
            else:
                color = matplotlib.colors.to_hex(
                    cmap_non_neighbor(sizing_nonneighbor[node] / max(sizing_nonneighbor.values()))
                )
                sizing_list.append(sizing_nonneighbor[node])
            color_list.append(color)

        if layout == "planar":
            is_planar, _ = nx.check_planarity(G)
            if not is_planar:
                logger.info("Graph is not planar, using spring layout instead.")
                layout = "spring"

        # Draw graph:
        if title is None:
            title = f"{os.path.basename(self.adata_path).split('.')[0]} Regulatory Correlation Network"

        f = plot_network(
            G,
            title=title,
            size_method=sizing_list,
            color_method=color_list,
            node_text=["Connections"],
            node_label="ID",
            nodefont_size=node_fontsize,
            node_label_position=node_label_position,
            edge_text=["Type", "Coexpression"],
            edge_label="Type",
            edge_label_position=node_label_position,
            edgefont_size=edge_fontsize,
            edge_thickness_attr="Coexpression",
            layout=layout,
            arrow_size=arrow_size,
            upper_margin=upper_margin,
            lower_margin=lower_margin,
            left_margin=left_margin,
            right_margin=right_margin,
            show_colorbar=False,
        )

        # Save graph:
        if save_id is None:
            save_id = os.path.basename(self.adata_path).split(".")[0]
        if save_path is None:
            save_path = save_folder
        full_save_path = os.path.join(save_path, f"{save_id}_network.{save_ext}")
        logger.info(f"Writing network to {full_save_path}...")

        fig = plotly.graph_objects.Figure(f)
        # The default is 100 DPI
        fig.write_image(full_save_path, scale=dpi / 100)

        return G, sizing_list, color_list

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
        if hasattr(self, "custom_lig_path") and self.mod_type.isin(["ligand", "lr"]):
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

        parser, args_list = define_spateo_argparse(**kwargs)
        permutation_model = MuSIC(parser, args_list)
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
# Formatting functions
# ---------------------------------------------------------------------------------------------------
def replace_col_with_collagens(string):
    # Split the string at the colon (if any)
    parts = string.split(":")
    # Split the first part of the string at slashes
    elements = parts[0].split("/")
    # Flag to check if we've encountered a "COL" element or a "Collagens" element
    encountered_col = False

    # Process each element
    for i, element in enumerate(elements):
        # If the element starts with "COL" or "b_COL", or if it is "Collagens" or "b_Collagens"
        if (
            element.startswith("COL")
            or element.startswith("b_COL")
            or element.startswith("Col")
            or element.startswith("b_Col")
            or element in ["Collagens", "b_Collagens"]
        ):
            # If we've already encountered a "COL" or "Collagens" element, remove this one
            if encountered_col:
                elements[i] = None
            # Otherwise, replace it with "Collagens" or "b_Collagens" as appropriate
            else:
                if element.startswith("b_COL") or element.startswith("b_Col") or element == "b_Collagens":
                    elements[i] = "b_Collagens"
                else:
                    elements[i] = "Collagens"
                encountered_col = True

    # Remove None elements and join the rest with slashes
    replaced_part = "/".join([element for element in elements if element is not None])
    # If there's a second part, add it back
    if len(parts) > 1:
        replaced_string = replaced_part + ":" + parts[1]
    else:
        replaced_string = replaced_part

    return replaced_string


def replace_hla_with_hlas(string):
    # Split the string at the colon (if any)
    parts = string.split(":")
    # Split the first part of the string at slashes
    elements = parts[0].split("/")
    # Flag to check if we've encountered an "HLA" element or an "HLAs" element
    encountered_hla = False

    # Process each element
    for i, element in enumerate(elements):
        # If the element starts with "HLA" or "b_HLA", or if it is "HLAs" or "b_HLAs"
        if element.startswith("HLA") or element.startswith("b_HLA") or element in ["HLAs", "b_HLAs"]:
            # If we've already encountered an "HLA" or "HLAs" element, remove this one
            if encountered_hla:
                elements[i] = None
            # Otherwise, replace it with "HLAs" or "b_HLAs" as appropriate
            else:
                if element.startswith("b_HLA") or element == "b_HLAs":
                    elements[i] = "b_HLAs"
                else:
                    elements[i] = "HLAs"
                encountered_hla = True

    # Remove None elements and join the rest with slashes
    replaced_part = "/".join([element for element in elements if element is not None])
    # If there's a second part, add it back
    if len(parts) > 1:
        replaced_string = replaced_part + ":" + parts[1]
    else:
        replaced_string = replaced_part

    return replaced_string
