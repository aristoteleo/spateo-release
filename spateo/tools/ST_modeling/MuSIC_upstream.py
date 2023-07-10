"""
Functionalities to aid in feature selection to characterize signaling patterns from spatial transcriptomics. Given a
list of signaling molecules (ligands or receptors) and
"""
import argparse
import multiprocessing
import os
import re
import sys
from functools import partial
from multiprocessing import Pool
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI
from scipy.stats import mannwhitneyu
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm
from spateo.preprocessing import log1p, normalize_total
from spateo.preprocessing.normalize import factor_normalization
from spateo.tools.find_neighbors import get_wi, neighbors
from spateo.tools.gene_expression_variance import get_highvar_genes_sparse
from spateo.tools.ST_modeling.MuSIC import MuSIC
from spateo.tools.ST_modeling.regression_utils import (
    multicollinearity_check,
    multitesting_correction,
    smooth,
)


# ---------------------------------------------------------------------------------------------------
# Selection of targets and signaling regulators
# ---------------------------------------------------------------------------------------------------
class MuSIC_target_selector:
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

    def __init__(self, parser: argparse.ArgumentParser):
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

    def set_predictors_or_targets(self):
        """Unbiased identification of appropriate signaling molecules and/or target genes for modeling with
        heuristical methods."""
        any_predictors_given = any(
            x is not None
            for x in [
                self.custom_ligands_path,
                self.custom_ligands,
                self.custom_receptors_path,
                self.custom_receptors,
                self.custom_pathways_path,
                self.custom_pathways,
            ]
        )
        if (any_predictors_given and self.targets_path is not None) or (
            any_predictors_given and self.custom_targets is not None
        ):
            self.logger.info(
                "Targets were provided, but so were predictors (ligands and/or receptors). Automated "
                "selection of targets/predictors is not needed and modeling can proceed with the given "
                "targets and predictors."
            )
            sys.exit()
        elif (not any_predictors_given and self.targets_path is None) or (
            not any_predictors_given and self.custom_targets is not None
        ):
            self.auto_select_targets = True
            self.auto_select_predictors = True
            self.logger.info(
                "Targets were not provided, and neither were predictors (ligands and/or receptors). "
                "First, automatically selecting highly-variable targets. Then, selecting predictors for "
                "these targets."
            )

            # Automatically select targets- choose between the top 3000 and the top 50% of genes by variance,
            # depending on the breadth of the dataset:
            # First filter AnnData to genes that are expressed above target threshold (5% of cells by default):
            if scipy.sparse.issparse(self.adata.X):
                expr_percentage = np.array((self.adata.X > 0).sum(axis=0) / self.adata.n_obs)[0]
            else:
                expr_percentage = np.count_nonzero(self.adata.X, axis=0) / self.adata.n_obs
            valid = list(np.array(self.adata.var_names)[expr_percentage > self.target_expr_threshold])
            self.adata = self.adata[:, valid]

            n_genes = min(3000, int(self.adata.n_vars / 2))
            (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(self.adata.X, numgenes=n_genes)
            high_variance_genes_filter = list(self.adata.var.index[gene_counts_stats.high_var.values])

            all_predictors = []
            for target in high_variance_genes_filter:
                data = self.adata[:, target].X
                predictors_for_target = self.parse_predictors(data)
                all_predictors.extend(predictors_for_target)

            all_predictors = list(set(all_predictors))
            # Save the list of regulators to a file in the same directory as the AnnData file:
            save_dir = os.path.dirname(self.adata_path)
            predictors_path = os.path.join(save_dir, f"predictors_{self.species}.txt")
            with open(predictors_path, "w") as f:
                f.write("\n".join(all_predictors))
            self.logger.info(f"Check {predictors_path} for the list of predictors.")

            # Also save the list of targets to a file in the same directory as the AnnData file:
            targets_path = os.path.join(save_dir, f"targets_{self.species}.txt")
            with open(targets_path, "w") as f:
                f.write("\n".join(high_variance_genes_filter))
            self.logger.info(f"Check {targets_path} for the list of targets.")

        elif (self.targets_path is not None and not any_predictors_given) or (
            self.custom_targets is not None and not any_predictors_given
        ):
            self.auto_select_targets = False
            self.auto_select_predictors = True
            self.logger.info(
                "Targets were provided, but predictors (ligands and/or receptors) were not."
                "Finding subset of interactions/interaction molecules that is appropriate for these targets."
            )

            # Read targets and subset to target expression:
            if self.targets_path is not None:
                try:
                    with open(self.targets_path, "r") as file:
                        self.custom_targets = [x.strip() for x in file.readlines()]
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "The provided targets file could not be found. Please check the path and try again."
                    )
                except IOError:
                    raise IOError("The provided targets file could not be read. Please check the file and try again.")

            # Iterate over targets and get the set of appropriate L:R interactions:
            all_predictors = []
            for target in self.custom_targets:
                data = self.adata[:, target].X
                predictors_for_target = self.parse_predictors(data)
                all_predictors.extend(predictors_for_target)

            all_predictors = list(set(all_predictors))
            # Save the list of regulators to a file in the same directory as the AnnData file:
            save_dir = os.path.dirname(self.adata_path)
            predictors_path = os.path.join(save_dir, f"predictors_{self.species}.txt")
            with open(predictors_path, "w") as f:
                f.write("\n".join(all_predictors))
            self.logger.info(f"Check {predictors_path} for the list of predictors.")

        elif (self.targets_path is None and any_predictors_given) or (
            self.custom_targets is None and any_predictors_given
        ):
            self.auto_select_targets = True
            self.auto_select_predictors = False
            self.logger.info(
                "Predictors (ligands and/or receptors) were provided, but targets were not. "
                "Automatically selecting highly-variable targets."
            )

            # Automatically select targets- choose between the top 3000 and the top 50% of genes by variance,
            # depending on the breadth of the dataset:
            # First filter AnnData to genes that are expressed above target threshold (5% of cells by default):
            if scipy.sparse.issparse(self.adata.X):
                expr_percentage = np.array((self.adata.X > 0).sum(axis=0) / self.adata.n_obs)[0]
            else:
                expr_percentage = np.count_nonzero(self.adata.X, axis=0) / self.adata.n_obs
            valid = list(np.array(self.adata.var_names)[expr_percentage > self.target_expr_threshold])
            self.adata = self.adata[:, valid]

            n_genes = min(3000, int(self.adata.n_vars / 2))
            (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(self.adata.X, numgenes=n_genes)
            high_variance_genes_filter = list(self.adata.var.index[gene_counts_stats.high_var.values])

            # Save the list of targets to a file in the same directory as the AnnData file:
            save_dir = os.path.dirname(self.adata_path)
            targets_path = os.path.join(save_dir, f"targets_{self.species}.txt")
            with open(targets_path, "w") as f:
                f.write("\n".join(high_variance_genes_filter))
            self.logger.info(f"Check {targets_path} for the list of targets.")

    def parse_predictors(self, data: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None):
        """Unbiased identification of appropriate signaling molecules and/or target genes for modeling with
            heuristical methods.

        Args:
            data: Optional 1D array containing gene expression values. If given, will be queried to find cells that
                express the target genes, which can then be used to find appropriate predictors. If not given,
                will use :attr `adata`.
        """

        if data is not None:
            # Get the indices to search over- where the target is nonzero:
            if isinstance(data, np.ndarray):
                indices = np.nonzero(data)[0]
            elif isinstance(data, scipy.sparse.spmatrix):
                indices = data.nonzero()[0]
            subset_indices = self.adata.obs_names[indices]
            n_obs = indices.shape[0]
        else:
            # All cells will be queried:
            subset_indices = self.adata.obs_names
            n_obs = self.adata.n_obs

        # Set threshold (either user-provided percentage or enough to be expressed in 50 cells) for expression of
        # target genes:
        threshold = max(self.target_expr_threshold, 50.0 / n_obs)

        # Get list of all available predictors, filter by threshold expression to get finalized list to save to
        # file in the same directory as the AnnData file:
        self.logger.info("Finding all available predictors...")
        lr_dir = pd.read_csv(os.path.join(self.cci_dir, f"lr_db_{self.species}.csv"), index_col=0)
        # Check ligands, receptors, or both depending on the model framing:
        set_ligands = set(lr_dir["from"])
        set_receptors = list(set(lr_dir["to"]))
        l_complexes = [s for s in set_ligands if "_" in s]
        r_complexes = [s for s in set_receptors if "_" in s]

        all_ligands = []
        for lig in set_ligands:
            if "_" in lig:
                components = lig.split("_")
                all_ligands.extend(components)
                all_ligands.remove(lig)

        all_receptors = []
        for rec in set_receptors:
            if "_" in rec:
                components = rec.split("_")
                all_receptors.extend(components)
                all_receptors.remove(rec)

        if self.mod_type == "receptor" or self.mod_type == "lr":
            # Subset to receptors that are expressed in > threshold number of cells:
            if scipy.sparse.issparse(self.adata.X):
                rec_expr_percentage = np.array(
                    (self.adata[subset_indices, all_receptors].X > 0).sum(axis=0) / self.adata.n_obs
                )[0]
            else:
                rec_expr_percentage = (
                    np.count_nonzero(self.adata[subset_indices, all_receptors].X, axis=0) / self.adata.n_obs
                )
            receptors = list(np.array(all_receptors)[rec_expr_percentage > threshold])

            # Recombine complex components if all components passed the thresholding:
            complex_members = [
                rec for rec in receptors for complex_string in r_complexes if rec == complex_string.split("_")[0]
            ]
            complex_associated = [
                complex_string.split("_")[1:]
                for rec in complex_members
                for complex_string in r_complexes
                if rec == complex_string.split("_")[0]
            ]
            combined_complex = [
                f"{comp}-{'_'.join(other_comps)}" for comp, other_comps in zip(complex_members, complex_associated)
            ]
            # Drop the individual components if they are not standalone receptors as well (i.e. also in the set
            # of possible receptors by themselves):
            for comp in complex_members + complex_associated:
                if comp not in set_receptors:
                    all_receptors.remove(comp)
            all_receptors.extend(combined_complex)

        if self.mod_type == "ligand" or self.mod_type == "lr":
            # Subset to ligands that are known to interact with any of the receptors:
            "filler"

        if self.mod_type == "ligand":
            regulators = all_ligands
        elif self.mod_type == "receptor":
            regulators = all_receptors
        elif self.mod_type == "lr":
            regulators = all_ligands + all_receptors

        return regulators
