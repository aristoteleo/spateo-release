"""
Functionalities to aid in feature selection to characterize signaling patterns from spatial transcriptomics. Given a
list of signaling molecules (ligands or receptors) and
"""
import argparse
import multiprocessing
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from mpi4py import MPI
from MuSIC import MuSIC
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm
from spateo.preprocessing import log1p, normalize_total
from spateo.tools.find_neighbors import get_wi, transcriptomic_connectivity
from spateo.tools.gene_expression_variance import get_highvar_genes_sparse
from spateo.tools.ST_modeling.regression_utils import multitesting_correction, smooth


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

        self.mod_type = None
        self.species = None
        self.ligands = None
        self.receptors = None
        self.targets = None
        self.normalize = None
        self.smooth = None
        self.log_transform = None
        self.target_expr_threshold = None
        self.coords = None

        self.parse_args()
        self.load_and_process()

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
        self.cci_dir = self.arg_retrieve.cci_dir
        self.species = self.arg_retrieve.species
        self.output_path = self.arg_retrieve.output_path
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
            self.logger.info(
                "Targets were not provided, and neither were predictors (ligands and/or receptors). "
                "First, automatically selecting spatially-interesting targets."
            )

    def load_and_process(self):
        """
        Load AnnData object and subset to ligand, receptor, and/or target expression where appropriate (depending on
        inputs to "mod_type" and to the custom ligands, receptors, targets, and/or pathways arguments).
        """
        self.adata = anndata.read_h5ad(self.adata_path)
        self.n_samples = self.adata.n_obs
        self.sample_names = self.adata.obs_names
        self.coords = self.adata.obsm[self.coords_key]
        # If group_subset is given, subset the AnnData object to contain the specified groups as well as neighboring
        # cells:
        if self.group_subset is not None:
            # Set up if ligand expression is involved in the predictors or the targets:
            if self.mod_type in ["ligand", "lr"]:
                subset = self.adata.obs[self.group_key].isin(self.group_subset)
                fitted_indices = [self.sample_names.get_loc(name) for name in subset.index]
                # Add cells that are neighboring cells of the chosen type, but which are not of the chosen type:
                get_wi_partial = partial(
                    get_wi,
                    n_samples=self.n_samples,
                    coords=self.coords,
                    fixed_bw=False,
                    exclude_self=True,
                    kernel="bisquare",
                    bw=self.n_neighbors,
                    threshold=0.01,
                    sparse_array=True,
                )

                with Pool() as pool:
                    weights = pool.map(get_wi_partial, fitted_indices)
                w_subset = scipy.sparse.vstack(weights)
                rows, cols = w_subset.nonzero()
                unique_indices = set(rows)
                names_all_neighbors = self.sample_names[unique_indices]
                self.adata = self.adata[self.adata.obs[self.group_key].isin(names_all_neighbors)]

            elif self.mod_type == "receptor":
                self.adata = self.adata[self.adata.obs[self.group_key].isin(self.group_subset)]

        # Preprocess AnnData object:
        if self.distr in ["poisson", "nb"]:
            if self.normalize or self.smooth or self.log_transform:
                self.logger.info(
                    f"With a {self.distr} assumption, discrete counts are required for the response variable. "
                    f"Computing normalizations and transforms if applicable, but rounding nonintegers to nearest "
                    f"integer; original counts can be round in .layers['raw']. Log-transform should not be applied."
                )
                self.adata.layers["raw"] = self.adata.X

        if self.normalize:
            if self.distr == "gaussian":
                self.logger.info("Setting total counts in each cell to 1e4 inplace...")
                normalize_total(self.adata)
            else:
                self.logger.info("Setting total counts in each cell to 1e4 and rounding nonintegers inplace...")
                normalize_total(self.adata)
                self.adata.X = (
                    scipy.sparse.csr_matrix(np.round(self.adata.X))
                    if scipy.sparse.issparse(self.adata.X)
                    else np.round(self.adata.X)
                )

        # Smooth data if 'smooth' is True and log-transform data matrix if 'log_transform' is True:
        if self.smooth:
            # Compute connectivity matrix if not already existing:
            try:
                conn = self.adata.obsp["expression_connectivities"]
            except:
                _, adata = transcriptomic_connectivity(self.adata, n_neighbors_method="ball_tree")
                conn = adata.obsp["expression_connectivities"]

            if self.distr == "gaussian":
                self.logger.info("Smoothing gene expression inplace...")
                adata_smooth_norm, _ = smooth(self.adata.X, conn, normalize_W=True)
                self.adata.X = adata_smooth_norm

            else:
                self.logger.info("Smoothing gene expression and rounding nonintegers inplace...")
                adata_smooth_norm, _ = smooth(self.adata.X, conn, normalize_W=True, return_discrete=True)
                self.adata.X = adata_smooth_norm

        if self.log_transform:
            if self.distr == "gaussian":
                self.logger.info("Log-transforming expression inplace...")
                self.adata.X = log1p(self.adata)
            else:
                self.logger.info(
                    "For the chosen distributional assumption, log-transform should not be applied. Log-transforming "
                    "expression and storing in adata.layers['X_log1p'], but not applying inplace and not using for "
                    "modeling."
                )
                self.adata.layers["X_log1p"] = log1p(self.adata)

        self.define_predictors_and_targets()

    def define_predictors_and_targets(self):
        """Define ligand expression array, receptor expression array and target expression array, depending on the
        provided inputs."""

        # # First, define targets if need be:
        # if hasattr(self, "auto_select_targets"):
        #

        if self.species == "human":
            self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_receptor_TF_db.csv"), index_col=0)
            tf_target_db = pd.read_csv(os.path.join(self.cci_dir, "human_TF_target_db.csv"), index_col=0)
        elif self.species == "mouse":
            self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
            r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
            tf_target_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_TF_target_db.csv"), index_col=0)
        else:
            raise ValueError("Invalid species specified. Must be one of 'human' or 'mouse'.")

        database_ligands = set(self.lr_db["from"])
        database_receptors = set(self.lr_db["to"])
        database_pathways = set(r_tf_db["pathway"])

        if self.custom_ligands_path is not None or self.custom_ligands is not None:
            if self.custom_ligands_path is not None:
                with open(self.custom_ligands_path, "r") as f:
                    ligands = f.read().splitlines()
            else:
                ligands = self.custom_ligands
            ligands = [l for l in ligands if l in database_ligands]
        else:
            # Use all possible ligands:
            ligands = database_ligands

        l_complexes = [elem for elem in ligands if "_" in elem]
        # Get individual components if any complexes are included in this list:
        ligands = [l for item in ligands for l in item.split("_")]
        ligands = [l for l in ligands if l in self.adata.var_names]
        # Subset ligands based on expression in sufficient numbers of cells:
        if scipy.sparse.issparse(self.adata.X):
            ligands = [
                l for l in ligands if (self.adata[:, l].X > 0).sum() >= self.n_samples * self.target_expr_threshold
            ]
        else:
            ligands = [l for l in ligands if self.adata[:, l].X.getnnz() >= self.n_samples * self.target_expr_threshold]

        self.ligands_expr = pd.DataFrame(
            self.adata[:, ligands].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, ligands].X,
            index=self.sample_names,
            columns=ligands,
        )

        # Combine columns if they are part of a complex- eventually the individual columns should be dropped,
        # but store them in a temporary list to do so later because some may contribute to multiple complexes:
        to_drop = []
        for element in l_complexes:
            parts = element.split("_")
            if all(part in self.ligands_expr.columns for part in parts):
                # Combine the columns into a new column with the name of the hyphenated element- here we will
                # compute the geometric mean of the expression values of the complex components:
                self.ligands_expr[element] = self.ligands_expr[parts].apply(
                    lambda x: x.prod() ** (1 / len(parts)), axis=1
                )
                # Mark the individual components for removal if the individual components cannot also be
                # found as ligands:
                to_drop.extend([part for part in parts if part not in database_ligands])
            else:
                # Drop the hyphenated element from the dataframe if all components are not found in the
                # dataframe columns
                partial_components = [l for l in ligands if l in parts]
                to_drop.extend(partial_components)
                if len(partial_components) > 0:
                    self.logger.info(
                        f"Not all components from the {element} heterocomplex could be found in the " f"dataset."
                    )

        # Drop any possible duplicate ligands alongside any other columns to be dropped:
        to_drop = list(set(to_drop))
        self.ligands_expr.drop(to_drop, axis=1, inplace=True)
        first_occurrences = self.ligands_expr.columns.duplicated(keep="first")
        self.ligands_expr = self.ligands_expr.loc[:, ~first_occurrences]

        # Compute spatial lag of ligand expression:
        spatial_weights = self._compute_all_wi(bw=self.n_neighbors, bw_fixed=False, exclude_self=True)
        lagged_expr_mat = np.zeros_like(self.ligands_expr.values)
        for i, ligand in enumerate(self.ligands_expr.columns):
            expr = self.ligands_expr[ligand]
            expr_sparse = scipy.sparse.csr_matrix(expr.values.reshape(-1, 1))
            lagged_expr = spatial_weights.dot(expr_sparse).toarray().flatten()
            lagged_expr_mat[:, i] = lagged_expr
        self.ligands_expr = pd.DataFrame(lagged_expr_mat, index=self.sample_names, columns=ligands)

        if (
            self.custom_receptors_path is not None
            or self.custom_receptors is not None
            or self.custom_pathways_path is not None
            or self.custom_pathways is not None
        ):
            if self.custom_receptors_path is not None:
                with open(self.custom_receptors_path, "r") as f:
                    receptors = f.read().splitlines()
            elif self.custom_receptors is not None:
                receptors = self.custom_receptors
            elif self.custom_pathways_path is not None:
                with open(self.custom_pathways_path, "r") as f:
                    pathways = f.read().splitlines()
            else:
                pathways = self.custom_pathways

            if "pathways" in locals():
                pathways = [p for p in pathways if p in database_pathways]
                # Get all receptors associated with these pathway(s):
                r_tf_db_subset = r_tf_db[r_tf_db["pathway"].isin(pathways)]
                receptors = set(r_tf_db_subset["receptor"])
                r_complexes = [elem for elem in receptors if "_" in elem]
                # Get individual components if any complexes are included in this list:
                receptors = [r for item in receptors for r in item.split("_")]
                receptors = list(set(receptors))
        else:
            # Use all possible receptors:
            receptors = database_receptors

        receptors = [r for r in receptors if r in database_receptors]
        r_complexes = [elem for elem in receptors if "_" in elem]
        # Get individual components if any complexes are included in this list:
        receptors = [r for item in receptors for r in item.split("_")]
        receptors = [r for r in receptors if r in self.adata.var_names]
        # Subset receptors based on expression in sufficient numbers of cells:
        if scipy.sparse.issparse(self.adata.X):
            receptors = [
                r for r in receptors if (self.adata[:, r].X > 0).sum() >= self.n_samples * self.target_expr_threshold
            ]
        else:
            receptors = [
                r for r in receptors if self.adata[:, r].X.getnnz() >= self.n_samples * self.target_expr_threshold
            ]

        self.receptors_expr = pd.DataFrame(
            self.adata[:, receptors].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, receptors].X,
            index=self.sample_names,
            columns=receptors,
        )

        # Combine columns if they are part of a complex- eventually the individual columns should be dropped,
        # but store them in a temporary list to do so later because some may contribute to multiple complexes:
        to_drop = []
        for element in r_complexes:
            if "_" in element:
                parts = element.split("_")
                if all(part in self.receptors_expr.columns for part in parts):
                    # Combine the columns into a new column with the name of the hyphenated element- here we will
                    # compute the geometric mean of the expression values of the complex components:
                    self.receptors_expr[element] = self.receptors_expr[parts].apply(
                        lambda x: x.prod() ** (1 / len(parts)), axis=1
                    )
                    # Mark the individual components for removal if the individual components cannot also be
                    # found as receptors:
                    to_drop.extend([part for part in parts if part not in database_receptors])
                else:
                    # Drop the hyphenated element from the dataframe if all components are not found in the
                    # dataframe columns
                    partial_components = [r for r in receptors if r in parts]
                    to_drop.extend(partial_components)
                    if len(partial_components) > 0:
                        self.logger.info(
                            f"Not all components from the {element} heterocomplex could be found in the "
                            f"dataset, so this complex was not included."
                        )

        # Drop any possible duplicate ligands alongside any other columns to be dropped:
        to_drop = list(set(to_drop))
        self.receptors_expr.drop(to_drop, axis=1, inplace=True)
        first_occurrences = self.receptors_expr.columns.duplicated(keep="first")
        self.receptors_expr = self.receptors_expr.loc[:, ~first_occurrences]

        # Ensure there is some degree of compatibility between the selected ligands and receptors if model uses
        # both ligands and receptors:
        if self.mod_type == "lr":
            self.logger.info("Preparing data: finding matched pairs between the selected ligands and receptors.")
            starting_n_ligands = len(self.ligands_expr.columns)
            starting_n_receptors = len(self.receptors_expr.columns)

            lr_ref = self.lr_db[["from", "to"]]
            # Don't need entire dataframe, just take the first two rows of each:
            lig_melt = self.ligands_expr.iloc[[0, 1], :].melt(var_name="from", value_name="value_ligand")
            rec_melt = self.receptors_expr.iloc[[0, 1], :].melt(var_name="to", value_name="value_receptor")

            merged_df = pd.merge(lr_ref, rec_melt, on="to")
            merged_df = pd.merge(merged_df, lig_melt, on="from")
            pairs = merged_df[["from", "to"]].drop_duplicates(keep="first")
            self.lr_pairs = [tuple(x) for x in zip(pairs["from"], pairs["to"])]
            if len(self.lr_pairs) == 0:
                raise RuntimeError(
                    "No matched pairs between the selected ligands and receptors were found. If path to custom list of "
                    "ligands and/or receptors was provided, ensure ligand-receptor pairings exist among these lists, "
                    "or check data to make sure these ligands and/or receptors were measured and were not filtered out."
                )

            pivoted_df = merged_df.pivot_table(values=["value_ligand", "value_receptor"], index=["from", "to"])
            filtered_df = pivoted_df[pivoted_df.notna().all(axis=1)]
            # Filter ligand and receptor expression to those that have a matched pair:
            self.ligands_expr = self.ligands_expr[filtered_df.index.get_level_values("from").unique()]
            self.receptors_expr = self.receptors_expr[filtered_df.index.get_level_values("to").unique()]
            final_n_ligands = len(self.ligands_expr.columns)
            final_n_receptors = len(self.receptors_expr.columns)

            self.logger.info(
                f"Found {final_n_ligands} ligands and {final_n_receptors} receptors that have matched pairs. "
                f"{starting_n_ligands - final_n_ligands} ligands removed from the list and "
                f"{starting_n_receptors - final_n_receptors} receptors/complexes removed from the list due to not "
                f"having matched pairs among the corresponding set of receptors/ligands, respectively."
                f"Remaining ligands: {self.ligands_expr.columns.tolist()}."
                f"Remaining receptors: {self.receptors_expr.columns.tolist()}."
            )

            self.logger.info(f"Set of ligand-receptor pairs: {self.lr_pairs}")

        if self.targets_path is not None or self.custom_targets is not None:
            if self.targets_path is not None:
                with open(self.targets_path, "r") as f:
                    targets = f.read().splitlines()
            else:
                targets = self.custom_targets
            targets = [t for t in targets if t in self.adata.var_names]

            # Subset targets based on expression in sufficient numbers of cells:
            if scipy.sparse.issparse(self.adata.X):
                targets = [
                    t for t in targets if (self.adata[:, t].X > 0).sum() >= self.n_samples * self.target_expr_threshold
                ]
            else:
                targets = [
                    t for t in targets if self.adata[:, t].X.getnnz() >= self.n_samples * self.target_expr_threshold
                ]
        else:
            # If targets are not provided, check through all genes expressed in above a threshold proportion of cells:
            targets = [
                t
                for t in self.adata.var_names
                if (self.adata[:, t].X > 0).sum() >= self.n_samples * self.target_expr_threshold
            ]

        self.targets_expr = pd.DataFrame(
            self.adata[:, targets].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, targets].X,
            index=self.sample_names,
            columns=targets,
        )

        self.target_list = targets

    def _compute_all_wi(
        self,
        bw: Union[float, int],
        bw_fixed: bool = False,
        exclude_self: bool = True,
        kernel: str = "bisquare",
    ) -> scipy.sparse.spmatrix:
        """Compute spatial weights for all samples in the dataset given a specified bandwidth.

        Args:
            bw: Bandwidth for the spatial kernel
            fixed_bw: Whether the bandwidth considers a uniform distance for each sample (True) or a nonconstant
                distance for each sample that depends on the number of neighbors (False). If not given, will default
                to self.fixed_bw.
            exclude_self: Whether to include each sample itself as one of its nearest neighbors. If not given,
                will default to self.exclude_self.
            kernel: Kernel to use for the spatial weights. If not given, will default to self.kernel.

        Returns:
            wi: Array of weights for all samples in the dataset
        """

        # Parallelized computation of spatial weights for all samples:
        get_wi_partial = partial(
            get_wi,
            n_samples=self.n_samples,
            coords=self.coords,
            fixed_bw=bw_fixed,
            exclude_self=exclude_self,
            kernel=kernel,
            bw=bw,
            threshold=0.01,
            sparse_array=True,
        )

        with Pool() as pool:
            weights = pool.map(get_wi_partial, range(self.n_samples))
        w = scipy.sparse.vstack(weights)
        return w

    def select_features(self):
        """Feature selection using neural network importance metrics."""
        if not os.path.exists(os.path.join(self.output_path, "predictors_and_targets")):
            os.makedirs(os.path.join(self.output_path, "predictors_and_targets"))

        if self.mod_type == "ligand":
            X = self.ligands_expr.values
            self.feature_names = self.ligands_expr.columns.tolist()
        elif self.mod_type == "receptor":
            X = self.receptors_expr.values
            self.feature_names = self.receptors_expr.columns.tolist()
        elif self.mod_type == "lr":
            # Construct L:R products array:
            X = np.zeros((self.n_samples, len(self.lr_pairs)))

            for idx, lr_pair in enumerate(self.lr_pairs):
                lig, rec = lr_pair[0], lr_pair[1]
                lig_expr_values = self.ligands_expr[lig].values.reshape(-1, 1)
                rec_expr_values = self.receptors_expr[rec].values.reshape(-1, 1)

                # Communication signature b/w receptor in target and ligand in neighbors
                X[:, idx] = (lig_expr_values * rec_expr_values).flatten()
            self.feature_names = [f"{lr_pair[0]}:{lr_pair[1]}" for lr_pair in self.lr_pairs]

        pool = Pool(multiprocessing.cpu_count())
        results = pool.map(self.process_column, [(X, col_name) for col_name in self.targets_expr.columns])
        pool.close()
        pool.join()

        # Dictionary to store significant features for each target:
        sig_features_dict = {}
        for column in self.targets_expr.columns:
            sig_features_dict[column] = results[0]

        # Series to store R-squared values:
        r_squared_series = pd.Series([result[1] for result in results])

        # Based on the given inputs (the combination of links to ligand/receptor lists and target list),
        # return either the list of selected features or the list of filtered targets.
        if self.targets_path is not None or self.custom_targets is not None:
            # Minimal filtering based on reconstruction capability:
            filtered_feats = r_squared_series[r_squared_series >= self.r_squared_threshold].index.tolist()
            self.logger.info(
                f"Of the original {len(self.target_list)}, {len(filtered_feats)} features remain post-filtering."
            )
            with open(os.path.join(self.output_path, "predictors_and_targets/targets.txt"), "w") as file:
                file.write("\n".join(filtered_feats))
        else:
            all_sig_features = set(sig_features_dict.values())
            with open(os.path.join(self.output_path, "predictors_and_targets/predictors.txt"), "w") as file:
                file.write("\n".join(all_sig_features))

    def process_column(self, X: np.ndarray, col_name: str):
        y = self.targets_expr[col_name].values

        sorted_indices, r_squared = self.run_selection_model(X, y)
        return sorted_indices, r_squared

    def run_selection_model(self, X: np.ndarray, y: np.ndarray):
        # Network architecture:
        model = tf.keras.Sequential()
        layer_dims = []
        if X.shape[1] >= 128:
            layer_dims.append(128)
        else:
            layer_dims.append(X.shape[1] / 4)
        while layer_dims[-1] / 4 > 8:
            layer_dims.append(layer_dims / 4)
        layer_dims.append(8)
        layer_dims.append(1)

        model.add(tf.keras.layers.Dense(layer_dims[0], activation="linear", input_shape=(X.shape[1],)))
        for layer_dim in layer_dims[1:-1]:
            model.add(tf.keras.layers.Dense(layer_dim, activation="linear"))
        model.add(tf.keras.layers.Dense(layer_dims[-1], activation="relu"))

        model.compile(optimizer="adam", loss="poisson")

        num_epochs = 100
        batch_size = 32
        model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

        y_pred = model.predict(X)
        # Do not compute feature importances if prediction accuracy is low:
        r_squared = r2_score(y, y_pred)

        res = permutation_importance(model, X, y, n_repeats=10, n_jobs=1, random_state=42)
        importance_scores = res.importances_mean
        # sorted_indices = np.argsort(importance_scores)[::-1]
        # sorted_names = [self.feature_names[idx] for idx in sorted_indices]

        # Null distribution of permutation importance:
        random_state = check_random_state(888)
        null_importances = []
        for _ in range(100):  # number of iterations for null distribution
            y_permuted = random_state.permutation(y)
            res_null = permutation_importance(model, X, y_permuted, n_repeats=10, n_jobs=1)
            null_importances.append(res_null.importances_mean)

        # Compute p-values and significance:
        null_importances = np.array(null_importances)
        p_values = (np.sum(null_importances >= importance_scores, axis=0) + 1) / (null_importances.shape[0] + 1)
        # Multiple testing correction:
        p_values_corrected = multitesting_correction(p_values, method="fdr_bh", alpha=0.05)
        significant_scores = p_values_corrected < 0.05
        significant_indices = np.where(significant_scores)
        significant_names = [self.feature_names[idx] for idx in significant_indices[0]]

        return significant_names, r_squared

    # Depending on the model type, define predictors or targets- if targets are given and model is "ligand" or "lr",
    # will identify a good set of ligands by using ligand expression in the 10 nearest neighbors of each cell. If
    # targets are given and model is "receptor", will identify a good set of receptors- if model is "lr",
    # will additionally filter these receptors for
    # For given ligands,

    # If none of the ligands list, receptor list, or target list are given, it is assumed that a niche model will be
    # defined. The targets will be identified by spatial autocorrelation using Geary's C.


### ----------------------------------- Dimensionality reduction ----------------------------------- ###
# def compute_pca(
#     adata: AnnData,
#     subset: Optional[List[str]] = None,
#     n_pca_components: int = 30,
#     pca_key: str = "X_pca",
#     layer: Optional[str] = None,
# ):
#     """Compute PCA decomposition for gene expression data (or any biological data stored in AnnData object).
#
#     Args:
#         adata: AnnData object containing data for which to perform PCA
#         subset: Can be used to optionally subset to only genes of interest for PCA
#         n_pca_components: Number of principal components
#         pca_key: Key in adata.obsm in which to store the representation after dimensionality reduction
#         layer: Can optionally provide a layer to use for dimensionality reduction (stored in .layers). If not given,
#         will use .X.
#
#     Returns:
#         adata_copy: AnnData object post-transformation (will return a copy of AnnData in case the original is
#             subsetted).
#     """
#     logger = lm.get_main_logger()
#
#     adata_copy = adata.copy()
#
#     if layer is None:
#         if subset is not None:
#             adata = adata[:, subset]
#         expr = adata.X
#     else:
#         if "X" in layer:
#             expr = adata.X
#         elif layer in adata.layers.keys():
#             expr = adata.layers[layer]
#         elif layer in adata.obsm.keys():
#             expr = adata.obsm[layer]
#         else:
#             logger.error("Input to 'layer' could not be found among valid AnnData keys.")
#
#     cm_genesums = expr.sum(axis=0)
#     valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
#     valid_ind = np.array(valid_ind).flatten()
#
#     expr = expr[:, valid_ind]
#
#     pca = PCA(
#         n_components=min(n_pca_components, expr.shape[1] - 1),
#         svd_solver="arpack",
#         random_state=0,
#     )
#     fit = pca.fit(expr.toarray()) if scipy.sparse.issparse(expr) else pca.fit(expr)
#     X_pca = fit.transform(expr.toarray()) if scipy.sparse.issparse(expr) else fit.transform(expr)
#     adata_copy.obsm[pca_key] = X_pca
#
#     # Loadings:
#     loadings = pca.components_
#     adata_copy.uns["loadings"] = loadings
#
#     adata_copy.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_
#     adata_copy.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]
#
#     return adata_copy
