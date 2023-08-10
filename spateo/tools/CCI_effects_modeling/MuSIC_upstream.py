"""
Functionalities to aid in feature selection to characterize signaling patterns from spatial transcriptomics. Given a
list of signaling molecules (ligands or receptors) and/or target genes
"""
import argparse
import multiprocessing
import os
import re
import sys
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI
from scipy.stats import mannwhitneyu
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

from ...logging import logger_manager as lm
from ...preprocessing import log1p, normalize_total
from ...preprocessing.normalize import factor_normalization
from ..find_neighbors import get_wi, neighbors
from ..gene_expression_variance import get_highvar_genes_sparse
from .MuSIC import MuSIC
from .regression_utils import multicollinearity_check, multitesting_correction, smooth
from .SWR_mpi import define_spateo_argparse


# ---------------------------------------------------------------------------------------------------
# Selection of targets and signaling regulators
# ---------------------------------------------------------------------------------------------------
class MuSIC_molecule_selector(MuSIC):
    """Various methods to select initial targets or predictors for intercellular analyses.

    Args:
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.

    Attributes:
        mod_type: The type of model that will be employed for eventual downstream modeling. Will dictate how
            predictors will be found (if applicable). Options:
                - "niche": Spatially-aware, uses categorical cell type labels as independent variables.
                - "lr": Spatially-aware, essentially uses the combination of receptor expression in the "target" cell
                    and spatially lagged ligand expression in the neighboring cells as independent variables.
                - "ligand": Spatially-aware, essentially uses ligand expression in the neighboring cells as
                    independent variables.
                - "receptor": Uses receptor expression in the "target" cell as independent variables.
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"


        adata_path: Path to the AnnData object from which to extract data for modeling
        normalize: Set True to Perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size).
        smooth: Set True to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression.
        log_transform: Set True if log-transformation should be applied to expression.
        target_expr_threshold: When selecting targets, expression above a threshold percentage of cells will be used to
            filter to a smaller subset of interesting genes. Defaults to 0.1.
        r_squared_threshold: When selecting targets, only genes with an R^2 above this threshold will be used as targets


        custom_lig_path: Optional path to a .txt file containing a list of ligands for the model, separated by
            newlines. If provided, will find targets for which this set of ligands collectively explains the most
            variance for (on a gene-by-gene basis) when taking neighborhood expression into account
        custom_ligands: Optional list of ligands for the model, can be used as an alternative to :attr
            `custom_lig_path`. If provided, will find targets for which this set of ligands collectively explains the
            most variance for (on a gene-by-gene basis) when taking neighborhood expression into account
        custom_rec_path: Optional path to a .txt file containing a list of receptors for the model, separated by
            newlines. If provided, will find targets for which this set of receptors collectively explains the most
            variance for
        custom_receptors: Optional list of receptors for the model, can be used as an alternative to :attr
            `custom_rec_path`. If provided, will find targets for which this set of receptors collectively explains the
            most variance for
        custom_pathways_path: Rather than providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model. If provided,
            will find targets for which receptors in these pathways collectively explain the most variance for
        custom_pathways: Optional list of signaling pathways for the model, can be used as an alternative to :attr
            `custom_pathways_path`. If provided, will find targets for which receptors in these pathways collectively
            explain the most variance for
        targets_path: Optional path to a .txt file containing a list of prediction target genes for the model,
            separated by newlines. If not provided, targets will be strategically selected from the given receptors.
        custom_targets: Optional list of prediction target genes for the model, can be used as an alternative to
            :attr `targets_path`.


        cci_dir: Full path to the directory containing cell-cell communication databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved


        group_key: Key in .obs of the AnnData object that contains the cell type labels, used if targeting molecules
            that have cell type-specific activity
        coords_key: Key in .obsm of the AnnData object that contains the coordinates of the cells


        n_neighbors: Number of nearest neighbors to use in the case that ligands are provided or in the case that
            ligands of interest should be found
    """

    def __init__(self, comm: MPI.Comm, parser: argparse.ArgumentParser, args_list: Optional[List[str]] = None):
        super().__init__(comm, parser, args_list, verbose=False)
        self.logger = lm.get_main_logger()

        self.parser = parser
        self.comm = MPI.COMM_WORLD

        self.parse_args()

    def parse_args(self):
        """
        Parse command line arguments for arguments pertinent to modeling.
        """
        self.arg_retrieve = self.parser.parse_args()
        self.mod_type = self.arg_retrieve.mod_type
        if self.mod_type not in ["niche", "lr", "ligand", "receptor"]:
            raise ValueError("Invalid model type provided. Must be one of 'niche', 'lr', 'ligand', or 'receptor'.")
        self.distr = self.arg_retrieve.distr

        self.adata_path = self.arg_retrieve.adata_path
        self.adata = anndata.read_h5ad(self.adata_path)

        self.cci_dir = self.arg_retrieve.cci_dir
        self.species = self.arg_retrieve.species
        self.output_path = self.arg_retrieve.output_path
        if self.output_path is None:
            raise ValueError("Must provide an output path for the results file, of the form 'path/to/file.csv'.")
        self.custom_ligands_path = self.arg_retrieve.custom_lig_path
        self.custom_ligands = self.arg_retrieve.ligand
        self.custom_receptors_path = self.arg_retrieve.custom_rec_path
        self.custom_receptors = self.arg_retrieve.receptor
        self.custom_pathways_path = self.arg_retrieve.custom_pathways_path
        self.custom_pathways = self.arg_retrieve.pathway
        self.targets_path = self.arg_retrieve.targets_path
        self.custom_targets = self.arg_retrieve.target

        self.normalize = self.arg_retrieve.normalize
        self.smooth = self.arg_retrieve.smooth
        self.log_transform = self.arg_retrieve.log_transform
        self.target_expr_threshold = self.arg_retrieve.target_expr_threshold
        self.r_squared_threshold = self.arg_retrieve.r_squared_threshold

        self.coords_key = self.arg_retrieve.coords_key
        self.group_key = self.arg_retrieve.group_key
        self.group_subset = self.arg_retrieve.group_subset

        self.n_neighbors = self.arg_retrieve.n_neighbors
        self.bw_fixed = self.arg_retrieve.bw_fixed

    def find_targets(self):
        """Find genes that may serve as interesting targets."""

    def find_predictors(
        self,
        targets: Union[str, List[str]],
        pct_coverage: Optional[float] = 0.9,
        min_n_receptors: int = 2,
        nontarget_expr_percentage_threshold: float = 0.4,
        save_id: Optional[str] = None,
        **kwargs,
    ):
        """Find ligands and receptors for modeling efforts. Will also suggest targets from among the given targets
        based on the uniqueness of the found ligand-receptor signature.

        Args:
            pct_coverage: Will adjust thresholds until this percentage of expressing cells have at least one
                interaction associated (if possible). Will include receptors in the order of largest -> smallest
                difference in the percentage of target-expressing cells with that receptor vs. the percentage of
                non-target-expressing cells with that receptor.
            min_n_receptors: Minimum number of receptors to find- if threshold percentage filter is not met for a
                sufficient number of receptors, will relax filter until threshold is met.
            nontarget_expr_percentage: Percentage of non-target-expressing cells that also express a given receptor-
                this number will be used to identify whether a particular target/set of targets is interesting for
                cell-cell communication modeling. Recommended to set above 0.2 but below 0.5.
            save_id: Optional string to append to the end of the saved file name. Will save signaling molecule names as
                "ligand_{save_id}.txt", etc.
            kwargs: Keyword arguments for any of the Spateo argparse arguments. Should not include 'output_path' (
                which will be determined by the output path used for the main model). Should also not include any of
                'mod_type', 'ligands' or 'receptors', which will be determined by this function.
        """
        # Save found ligands and receptors to the same directory as the output path:
        if self.species == "human":
            try:
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CCI resources cannot be found at {self.cci_dir}. Please check the path " f"and try again."
                )
            except IOError:
                raise IOError(
                    "Issue reading L:R database. Files can be downloaded from "
                    "https://github.com/aristoteleo/spateo-release/spateo/tools/database."
                )

            r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_receptor_TF_db.csv"), index_col=0)
            grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
        elif self.species == "mouse":
            try:
                lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CCI resources cannot be found at {self.cci_dir}. Please check the path " f"and try again."
                )
            except IOError:
                raise IOError(
                    "Issue reading L:R database. Files can be downloaded from "
                    "https://github.com/aristoteleo/spateo-release/spateo/tools/database."
                )

            r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
            grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)

        # All receptors associated with any of the given targets:
        receptors = []
        for target in targets:
            target_gene_row = grn.loc[target, :]
            target_TFs = target_gene_row[target_gene_row == 1].index.tolist()
            filtered_df = r_tf_db[r_tf_db["tf"].isin(target_TFs)]
            target_receptors = list(set(filtered_df["receptor"].tolist()))
            receptors.extend(target_receptors)

        complexes = [r for r in receptors if "_" in r]
        receptors = [r for r in receptors if all(part in self.adata.var_names for part in r.split("_")) or "_" not in r]
        # Partition AnnData for this set of targets:
        target_cells = self.adata[:, targets].X
        target_cell_indices = (target_cells > 0).sum(axis=1).nonzero()[0]
        # Subset matrix to the target cells:
        target_sub = self.adata[target_cell_indices, :]

        nontarget_cell_indices = [i for i in range(self.adata.n_obs) if i not in target_cell_indices]
        nontarget_sub = self.adata[nontarget_cell_indices, :]

        # Rank receptors given this target set- compute expression in the target-expressing subset and the subset not
        # expressing the target gene for each receptor:
        expr_percentages_target = []
        expr_percentages_nontarget = []
        for r in receptors:
            r_subset_target = target_sub[:, r].X > 0
            expressing_cells_target = r_subset_target.sum()
            percentage_target = (expressing_cells_target / target_sub.n_obs) * 100
            expr_percentages_target.append(percentage_target)

            r_subset_nontarget = nontarget_sub[:, r].X > 0
            expressing_cells_nontarget = r_subset_nontarget.sum()
            percentage_nontarget = (expressing_cells_nontarget / nontarget_sub.n_obs) * 100
            expr_percentages_nontarget.append(percentage_nontarget)

        # For complexes, take the minimum percentages of the constituent receptors:
        # Create a mapping between receptors and their corresponding expression percentages
        receptor_to_percentage_target = {r: percentage for r, percentage in zip(receptors, expr_percentages_target)}
        receptor_to_percentage_nontarget = {
            r: percentage for r, percentage in zip(receptors, expr_percentages_nontarget)
        }

        # Set to collect parts that should be removed at the end
        parts_to_remove = set()

        # Iterate through complexes
        min_percentages_target = []
        min_percentages_nontarget = []
        for complex_r in complexes:
            # Extract subparts of the complex
            parts = complex_r.split("_")
            parts_to_remove.update(parts)

            # Find the corresponding percentages for target and nontarget
            percentages_target = [
                receptor_to_percentage_target[part] for part in parts if part in receptor_to_percentage_target
            ]
            percentages_nontarget = [
                receptor_to_percentage_nontarget[part] for part in parts if part in receptor_to_percentage_nontarget
            ]

            # Compute the minimum percentage for both target and nontarget
            if percentages_target:
                min_percentages_target.append(min(percentages_target))
                receptor_to_percentage_target[complex_r] = min(percentages_target)
            if percentages_nontarget:
                min_percentages_nontarget.append(min(percentages_nontarget))
                receptor_to_percentage_nontarget[complex_r] = min(percentages_nontarget)
        parts_to_remove = set(parts_to_remove)
        # Remove the individual parts from receptor_to_percentage_target
        for part in parts_to_remove:
            receptor_to_percentage_target.pop(part, None)
            receptor_to_percentage_nontarget.pop(part, None)

        # Get difference in percentage of target-expressing cells with that receptor vs. the percentage of
        # non-target-expressing cells with that receptor
        difference_dict = {
            key: receptor_to_percentage_target[key] - receptor_to_percentage_nontarget[key]
            for key in receptor_to_percentage_target
        }
        receptors = list(receptor_to_percentage_target.keys())
        if len(receptors) < min_n_receptors:
            self.logger.info("Value for 'min_n_receptors' is too high- setting equal to the total number of receptors.")
            min_n_receptors = len(receptors)

        # Ligands that bind these receptors:
        lr_pairs = {}

        for r in receptors:
            r_ligands = []
            filtered_df = lr_db[lr_db["to"] == r]
            ligands = list(set(filtered_df["from"]))
            # For each ligand, check that all parts can be found in the dataset:
            for l in ligands:
                parts = l.split("_")
                if all(part in self.adata.var_names for part in parts):
                    r_ligands.append(l)
            lr_pairs[r] = r_ligands

        # Construct design matrix from min_n_receptors receptors, prepare model independent variable array and adjust
        # until percent coverage is met or all possible receptors have been queried:
        # Initial set of receptors:
        sorted_receptors = sorted(difference_dict, key=lambda receptor: difference_dict[receptor], reverse=True)
        top_receptors = sorted_receptors[: min_n_receptors - 1]
        # Initial set of associated ligands:
        ligands = []
        for r in top_receptors:
            ligands.extend(lr_pairs[r])

        test_pct = 0
        next_index = min_n_receptors - 1
        while test_pct < pct_coverage or len(receptors) == len(sorted_receptors):
            # Add the next receptor and associated ligands:
            next_receptor = sorted_receptors[next_index]
            top_receptors.append(next_receptor)
            ligands.extend(lr_pairs[next_receptor])

            kwargs["output_path"] = self.output_path
            kwargs["adata_path"] = self.adata_path
            kwargs["receptor"] = top_receptors
            kwargs["ligand"] = ligands
            kwargs["target"] = targets

            comm, parser, args_list = define_spateo_argparse(**kwargs)
            upstream_model = MuSIC(comm, parser)
            upstream_model.define_sig_inputs(recompute=True)

            # Load design matrix:
            X_df = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"), index_col=0
            )
            # Get coverage of these interaction features:
            X_df["contains_nonzero"] = X_df.apply(lambda row: int(row.any()), axis=1)
            # Subset to the target-expressing cells:
            X_df_target = X_df.iloc[target_cell_indices, :]
            # Get percentage of target-expressing cells with at least one nonzero interaction feature:
            test_pct = (X_df_target["contains_nonzero"].sum() / X_df.shape[0]) * 100

            next_index += 1

        # Check if the final signaling features are highly-specific to the target(s):
        X_df_nontarget = X_df.iloc[nontarget_cell_indices, :]
        test_pct = (X_df_nontarget["contains_nonzero"].sum() / X_df.shape[0]) * 100
        if test_pct < nontarget_expr_percentage_threshold:
            self.logger.info(f"Chosen set of ligands/receptors are highly-specific to chosen target(s): \n{targets}.")

        # Final list of used receptors:
        self.logger.info(f"Final set of {len(top_receptors)} receptors: \n{top_receptors}")
        # Final list of used ligands:
        self.logger.info(f"Final set of {len(ligands)} ligands: \n{ligands}")

        self.logger.info(
            f"Saving list of manually found ligands to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'ligands.txt')}"
        )
        with open(os.path.join(os.path.dirname(self.adata_path), "ligands.txt"), "w") as f:
            f.write("\n".join(ligands))

        self.logger.info(
            f"Saving list of manually found receptors to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'receptors.txt')}"
        )
        with open(os.path.join(os.path.dirname(self.adata_path), "receptors.txt"), "w") as f:
            f.write("\n".join(top_receptors))

        # At the end of this process, delete the design matrix and any other files created by MuSIC:
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "targets.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr_nonlag.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "receptors_expr.csv"))
