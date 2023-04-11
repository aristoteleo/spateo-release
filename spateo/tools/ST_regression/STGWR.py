"""
Modeling cell-cell communication using a regression model that is considerate of the spatial heterogeneity of (and thus
the context-dependency of the relationships of) the response variable.
"""
import argparse
import math
import os
import re
import sys
from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI
from patsy import dmatrix
from scipy.spatial.distance import cdist

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm
from spateo.preprocessing.normalize import normalize_total
from spateo.preprocessing.transform import log1p
from spateo.tools.find_neighbors import get_wi, transcriptomic_connectivity
from spateo.tools.spatial_degs import moran_i
from spateo.tools.ST_regression.distributions import Gaussian, NegativeBinomial, Poisson
from spateo.tools.ST_regression.regression_utils import (
    compute_betas_local,
    iwls,
    smooth,
)

# NOTE: set lower bound AND upper bound bandwidth much lower for membrane-bound ligands/receptors pairs

# ---------------------------------------------------------------------------------------------------
# GWR for cell-cell communication
# ---------------------------------------------------------------------------------------------------
class STGWR:
    """Spatially weighted regression on spatial omics data with parallel processing. Runs after being called
    from the command line.

    Args:
        comm: MPI communicator object initialized with mpi4py, to control parallel processing operations
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.

    Attributes:
        mod_type: The type of model that will be employed- this dictates how the data will be processed and
            prepared. Options:
                - "niche": Spatially-aware, uses spatial connections between samples as independent variables
                - "lr": Spatially-aware, uses the combination of receptor expression in the "target" cell and spatially
                    lagged ligand expression in the neighboring cells as independent variables.
                - "slice": Spatially-aware, uses a coupling of spatial category connections, ligand expression
                    and receptor expression to perform regression on select receptor-downstream genes.


        adata_path: Path to the AnnData object from which to extract data for modeling
        csv_path: Can also be used to specify path to non-AnnData .csv object. Assumes the first three columns
            contain x- and y-coordinates and then dependent variable values, in that order, with all subsequent
            columns containing independent variable values.
        normalize: Set True to Perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size). It is advisable not to do this if performing Poisson or negative binomial
            regression.
        smooth: Set True to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression. It is advisable not to do this if performing Poisson or negative binomial regression.
        log_transform: Set True if log-transformation should be applied to expression. It is advisable not to do
            this if performing Poisson or negative binomial regression.
        target_expr_threshold: Only used if :param `mod_type` is "lr" or "slice" and :param `targets_path` is not
            given. When manually selecting targets, expression above a threshold percentage of cells will be used to
            filter to a smaller subset of interesting genes. Defaults to 0.2.


        custom_lig_path: Optional path to a .txt file containing a list of ligands for the model, separated by
            newlines. Only used if :attr `mod_type` is "lr" or "slice" (and thus uses ligand/receptor expression
            directly in the inference). If not provided, will select ligands using a threshold based on expression
            levels in the data.
        custom_rec_path: Optional path to a .txt file containing a list of receptors for the model, separated by
            newlines. Only used if :attr `mod_type` is "lr" or "slice" (and thus uses ligand/receptor expression
            directly in the inference). If not provided, will select receptors using a threshold based on expression
            levels in the data.
        custom_pathways_path: Rather than  providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model. Only used if :attr `mod_type`
            is "lr" or "slice".
        targets_path: Optional path to a .txt file containing a list of prediction target genes for the model,
            separated by newlines. If not provided, targets will be strategically selected from the given receptors.
        init_betas_path: Optional path to a .npy file containing initial coefficient values for the model. Initial
            coefficients should have shape [n_features, ].


        cci_dir: Full path to the directory containing cell-cell communication databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved


        coords_key: Key in .obsm of the AnnData object that contains the coordinates of the cells
        group_key: Key in .obs of the AnnData object that contains the category grouping for each cell
        covariate_keys: Can be used to optionally provide any number of keys in .obs or .var containing a continuous
            covariate (e.g. expression of a particular TF, avg. distance from a perturbed cell, etc.)


        bw: Used to provide previously obtained bandwidth for the spatial kernel. Consists of either a distance
            value or N for the number of nearest neighbors. Can be obtained using BW_Selector or some other
            user-defined method. Pass "np.inf" if all other points should have the same spatial weight. Defaults to
            1000 if not provided.
        minbw: For use in automated bandwidth selection- the lower-bound bandwidth to test.
        maxbw: For use in automated bandwidth selection- the upper-bound bandwidth to test.


        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".


        bw_fixed: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        exclude_self: If True, ignore each sample itself when computing the kernel density estimation
        fit_intercept: Set True to include intercept in the model and False to exclude intercept
    """

    def __init__(self, comm: MPI.Comm, parser: argparse.ArgumentParser):
        self.logger = lm.get_main_logger()

        self.comm = comm
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
        self.groups = None
        self.y = None
        self.X = None

        self.bw = None
        self.minbw = None
        self.maxbw = None

        self.distr = None
        self.kernel = None
        self.n_samples = None
        self.n_features = None
        # Number of STGWR runs to go through:
        self.n_runs_all = None
        # Flag for whether model has been set up and AnnData has been processed:
        self.set_up = False

        self.parse_stgwr_args()

    def _set_up_model(self):
        if self.mod_type is None:
            self.logger.error(
                "No model type provided; need to provide a model type to fit. Options: 'niche', 'lr', " "'slice'."
            )

        # Check if the program is currently in the master process:
        if self.comm.rank == 0:
            # If AnnData object is given, process it:
            if self.adata_path is not None:
                # Ensure CCI directory is provided:
                if self.cci_dir is None:
                    self.logger.error(
                        "No CCI directory provided; need to provide a CCI directory to fit a model with "
                        "ligand/receptor expression."
                    )
                self.load_and_process()
            else:
                if self.csv_path is None:
                    self.logger.error(
                        "No AnnData path or .csv path provided; need to provide at least one of these "
                        "to provide a default dataset to fit."
                    )
                else:
                    custom_data = pd.read_csv(self.csv_path, index_col=0)
                    self.coords = custom_data.iloc[:, :2].values
                    self.target = pd.DataFrame(
                        custom_data.iloc[:, 2], index=custom_data.index, columns=[custom_data.columns[2]]
                    )
                    self.logger.info(f"Extracting target from column labeled '{custom_data.columns[2]}'.")
                    independent_variables = custom_data.iloc[:, 3:]
                    self.X = independent_variables.values
                    self.n_samples = self.X.shape[0]
                    self.n_features = self.X.shape[1]
                    self.feature_names = independent_variables.columns
                    self.sample_names = custom_data.index

            self.n_runs_all = np.arange(self.n_samples)

        # Broadcast data to other processes- gene expression variables:
        if self.adata_path is not None:
            if self.mod_type == "niche" or self.mod_type == "slice":
                self.cell_categories = self.comm.bcast(self.cell_categories, root=0)
            if self.mod_type == "lr" or self.mod_type == "slice":
                self.ligands_expr = self.comm.bcast(self.ligands_expr, root=0)
                self.receptors_expr = self.comm.bcast(self.receptors_expr, root=0)
            if hasattr(self, "targets_expr"):
                self.targets_expr = self.comm.bcast(self.targets_expr, root=0)
            elif hasattr(self, "target"):
                self.target = self.comm.bcast(self.target, root=0)

        # Broadcast data to other processes:
        self.X = self.comm.bcast(self.X, root=0)
        self.bw = self.comm.bcast(self.bw, root=0)
        self.coords = self.comm.bcast(self.coords, root=0)
        self.tolerance = self.comm.bcast(self.tolerance, root=0)
        self.max_iter = self.comm.bcast(self.max_iter, root=0)
        self.alpha = self.comm.bcast(self.alpha, root=0)
        self.n_samples = self.comm.bcast(self.n_samples, root=0)
        self.n_features = self.comm.bcast(self.n_features, root=0)
        self.n_runs_all = self.comm.bcast(self.n_runs_all, root=0)

        # Split data into chunks for each process:
        chunk_size = int(math.ceil(float(len(self.n_runs_all)) / self.comm.size))
        # Assign chunks to each process:
        self.x_chunk = self.n_runs_all[self.comm.rank * chunk_size : (self.comm.rank + 1) * chunk_size]

    def parse_stgwr_args(self):
        """
        Parse command line arguments for arguments pertinent to modeling.
        """
        self.arg_retrieve = self.parser.parse_args()
        self.mod_type = self.arg_retrieve.mod_type
        self.adata_path = self.arg_retrieve.adata_path
        self.csv_path = self.arg_retrieve.csv_path
        self.cci_dir = self.arg_retrieve.cci_dir
        self.species = self.arg_retrieve.species
        self.output_path = self.arg_retrieve.output_path
        self.custom_ligands_path = self.arg_retrieve.custom_lig_path
        self.custom_receptors_path = self.arg_retrieve.custom_rec_path
        self.custom_pathways_path = self.arg_retrieve.custom_pathways_path
        self.targets_path = self.arg_retrieve.targets_path
        self.init_betas_path = self.arg_retrieve.init_betas_path
        # Check if path to init betas is given:
        if self.init_betas_path is not None:
            self.logger.info(f"Loading initial betas from: {self.init_betas_path}")
            self.init_betas = np.load(self.init_betas_path)
        else:
            self.init_betas = None

        self.normalize = self.arg_retrieve.normalize
        self.smooth = self.arg_retrieve.smooth
        self.log_transform = self.arg_retrieve.log_transform
        self.target_expr_threshold = self.arg_retrieve.target_expr_threshold

        self.coords_key = self.arg_retrieve.coords_key
        self.group_key = self.arg_retrieve.group_key
        self.covariate_keys = self.arg_retrieve.covariate_keys

        self.bw_fixed = self.arg_retrieve.bw_fixed
        self.exclude_self = self.arg_retrieve.exclude_self
        self.distr = self.arg_retrieve.distr
        # Get appropriate distribution family based on specified:
        if self.distr == "gaussian":
            link = Gaussian.__init__.__defaults__[0]
            self.distr_obj = Gaussian(link)
        elif self.distr == "poisson":
            link = Poisson.__init__.__defaults__[0]
            self.distr_obj = Poisson(link)
        elif self.distr == "nb":
            link = NegativeBinomial.__init__.__defaults__[0]
            self.distr_obj = NegativeBinomial(link)
        self.kernel = self.arg_retrieve.kernel

        if not self.bw_fixed and self.kernel not in ["bisquare", "uniform"]:
            self.logger.error(
                "`bw_fixed` is set to False for adaptive kernel- it is assumed the chosen bandwidth is "
                "the number of neighbors for each sample. However, only the `bisquare` and `uniform` "
                "kernels perform hard thresholding and so it is recommended to use one of these kernels- "
                "the other kernels may result in different results."
            )

        self.fit_intercept = self.arg_retrieve.fit_intercept
        # Parameters related to the fitting process (tolerance, number of iterations, etc.)
        self.tolerance = self.arg_retrieve.tolerance
        self.max_iter = self.arg_retrieve.max_iter
        self.alpha = self.arg_retrieve.alpha

        if self.arg_retrieve.bw:
            if self.bw_fixed:
                self.bw = float(self.arg_retrieve.bw)
            else:
                self.bw = int(self.arg_retrieve.bw)

        if self.arg_retrieve.minbw:
            if self.bw_fixed:
                self.minbw = float(self.arg_retrieve.minbw)
            else:
                self.minbw = int(self.arg_retrieve.minbw)

        if self.arg_retrieve.maxbw:
            if self.bw_fixed:
                self.maxbw = float(self.arg_retrieve.maxbw)
            else:
                self.maxbw = int(self.arg_retrieve.maxbw)

        # Helpful messages at process start:
        if self.comm.rank == 0:
            print("-" * 60, flush=True)
            self.logger.info(f"Running STGWR on {self.comm.size} processes...")
            fixed_or_adaptive = "Fixed " if self.bw_fixed else "Adaptive "
            type = fixed_or_adaptive + self.kernel.capitalize()
            self.logger.info(f"Spatial kernel: {type}")

            if self.adata_path is not None:
                self.logger.info(f"Loading AnnData object from: {self.adata_path}")
            elif self.csv_path is not None:
                self.logger.info(f"Loading CSV file from: {self.csv_path}")
            if self.mod_type is not None:
                self.logger.info(f"Model type: {self.mod_type}")
                self.logger.info(f"Loading cell-cell interaction databases from the following folder: {self.cci_dir}")
                if self.custom_ligands_path is not None:
                    self.logger.info(f"Using list of custom ligands from: {self.custom_ligands_path}")
                if self.custom_receptors_path is not None:
                    self.logger.info(f"Using list of custom receptors from: {self.custom_receptors_path}")
                if self.targets_path is not None:
                    self.logger.info(f"Using list of target genes from: {self.targets_path}")
                self.logger.info(
                    f"Saving results to: {self.output_path}. Note that running `fit` or "
                    f"`predict_and_save` will clear the contents of this folder- copy any essential "
                    f"files beforehand."
                )

    def load_and_process(self):
        """
        Load AnnData object and process it for modeling.
        """
        self.adata = anndata.read_h5ad(self.adata_path)
        self.adata.uns["__type"] = "UMI"
        self.sample_names = self.adata.obs_names
        self.coords = self.adata.obsm[self.coords_key]
        self.n_samples = self.adata.n_obs
        # Placeholder- this will change at time of fitting:
        self.n_features = self.adata.n_vars

        if self.distr in ["poisson", "nb"]:
            if self.normalize or self.smooth or self.log_transform:
                self.logger.info(
                    f"With a {self.distr} assumption, discrete counts are required for the response variable. "
                    f"Computing normalizations and transforms if applicable, but rounding nonintegers up to nearest "
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

        # Define necessary quantities that will later be used to define the independent variable array- the one-hot
        # cell-type array, the ligand expression array and the receptor expression array:

        # One-hot cell type array (or other category):
        if self.mod_type == "niche" or self.mod_type == "slice":
            group_name = self.adata.obs[self.group_key]
            db = pd.DataFrame({"group": group_name})
            categories = np.array(group_name.unique().tolist())
            db["group"] = pd.Categorical(db["group"], categories=categories)

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)
            # Ensure columns are in order:
            self.cell_categories = X.reindex(sorted(X.columns), axis=1)

        # Ligand-receptor expression array
        if self.mod_type == "lr" or self.mod_type == "slice":
            if self.species == "human":
                self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
                r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_receptor_TF_db.csv"), index_col=0)
                tf_target_db = pd.read_csv(os.path.join(self.cci_dir, "human_TF_target_db.csv"), index_col=0)
            elif self.species == "mouse":
                self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
                r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
                tf_target_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_TF_target_db.csv"), index_col=0)
            else:
                self.logger.error("Invalid species specified. Must be one of 'human' or 'mouse'.")
            database_ligands = set(self.lr_db["from"])
            database_receptors = set(self.lr_db["to"])
            database_pathways = set(r_tf_db["pathway"])

            if self.custom_ligands_path is not None:
                with open(self.custom_ligands_path, "r") as f:
                    ligands = f.read().splitlines()
                    ligands = [l for l in ligands if l in database_ligands]
                    l_complexes = [elem for elem in ligands if "_" in elem]
                    # Get individual components if any complexes are included in this list:
                    ligands = [l for item in ligands for l in item.split("_")]
            else:
                # List of possible complexes to search through:
                l_complexes = [elem for elem in database_ligands if "_" in elem]
                # And all possible ligand molecules:
                all_ligands = [l for item in database_ligands for l in item.split("_")]

                # Get list of ligands from among the most highly spatially-variable genes, indicative of potentially
                # interesting spatially-enriched signal:
                self.logger.info(
                    "Preparing data: getting list of ligands from among the most highly " "spatially-variable genes."
                )
                m_degs = moran_i(self.adata)
                m_filter_genes = m_degs[m_degs.moran_q_val < 0.05].sort_values(by=["moran_i"], ascending=False).index
                ligands = [g for g in m_filter_genes if g in all_ligands]

                # If no significant spatially-variable ligands are found, use the top 10 most spatially-variable
                # ligands:
                if len(ligands) == 0:
                    self.logger.info(
                        "No significant spatially-variable ligands found. Using top 10 most "
                        "spatially-variable ligands."
                    )
                    m_filter_genes = m_degs.sort_values(by=["moran_i"], ascending=False).index
                    ligands = [g for g in m_filter_genes if g in all_ligands][:10]

                # If any ligands are part of complexes, add all complex components to this list:
                for element in l_complexes:
                    if "_" in element:
                        complex_members = element.split("_")
                        for member in complex_members:
                            if member in ligands:
                                other_members = [m for m in complex_members if m != member]
                                for member in other_members:
                                    ligands.append(member)
                ligands = list(set(ligands))

                self.logger.info(
                    f"Found {len(ligands)} among significantly spatially-variable genes and associated "
                    f"complex members."
                )

            ligands = [l for l in ligands if l in self.adata.var_names]
            self.ligands_expr = pd.DataFrame(
                self.adata[:, ligands].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, ligands].X,
                index=self.adata.obs_names,
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

            if self.custom_receptors_path is not None:
                with open(self.custom_receptors_path, "r") as f:
                    receptors = f.read().splitlines()
                    receptors = [r for r in receptors if r in database_receptors]
                    r_complexes = [elem for elem in receptors if "_" in elem]
                    # Get individual components if any complexes are included in this list:
                    receptors = [r for item in receptors for r in item.split("_")]

            elif self.custom_pathways_path is not None:
                with open(self.custom_pathways_path, "r") as f:
                    pathways = f.read().splitlines()
                    pathways = [p for p in pathways if p in database_pathways]
                # Get all receptors associated with these pathway(s):
                r_tf_db_subset = r_tf_db[r_tf_db["pathway"].isin(pathways)]
                receptors = set(r_tf_db_subset["receptor"])
                r_complexes = [elem for elem in receptors if "_" in elem]
                # Get individual components if any complexes are included in this list:
                receptors = [r for item in receptors for r in item.split("_")]
                receptors = list(set(receptors))

            else:
                # List of possible complexes to search through:
                r_complexes = [elem for elem in database_receptors if "_" in elem]
                # And all possible receptor molecules:
                all_receptors = [r for item in database_receptors for r in item.split("_")]

                # Get list of receptors from among the most highly spatially-variable genes, indicative of
                # potentially interesting spatially-enriched signal:
                self.logger.info(
                    "Preparing data: getting list of ligands from among the most highly spatially-variable genes."
                )
                m_degs = moran_i(self.adata)
                m_filter_genes = m_degs[m_degs.moran_q_val < 0.05].sort_values(by=["moran_i"], ascending=False).index
                receptors = [g for g in m_filter_genes if g in all_receptors]

                # If no significant spatially-variable receptors are found, use the top 10 most spatially-variable
                # receptors:
                if len(receptors) == 0:
                    self.logger.info(
                        "No significant spatially-variable receptors found. Using top 10 most "
                        "spatially-variable receptors."
                    )
                    m_filter_genes = m_degs.sort_values(by=["moran_i"], ascending=False).index
                    receptors = [g for g in m_filter_genes if g in all_receptors][:10]

                # If any receptors are part of complexes, add all complex components to this list:
                for element in r_complexes:
                    if "_" in element:
                        complex_members = element.split("_")
                        for member in complex_members:
                            if member in receptors:
                                other_members = [m for m in complex_members if m != member]
                                for member in other_members:
                                    receptors.append(member)
                receptors = list(set(receptors))

                self.logger.info(
                    f"Found {len(receptors)} among significantly spatially-variable genes and associated "
                    f"complex members."
                )

            receptors = [r for r in receptors if r in self.adata.var_names]

            self.receptors_expr = pd.DataFrame(
                self.adata[:, receptors].X.toarray()
                if scipy.sparse.issparse(self.adata.X)
                else self.adata[:, receptors].X,
                index=self.adata.obs_names,
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

            # Ensure there is some degree of compatibility between the selected ligands and receptors:
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
                self.logger.error(
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

        else:
            self.logger.error("Invalid `mod_type` specified. Must be one of 'niche', 'slice', or 'lr'.")

        # Get gene targets:
        self.logger.info("Preparing data: getting gene targets.")
        # For niche model, targets must be manually provided:
        if self.targets_path is None and self.mod_type == "niche":
            self.logger.error(
                "For niche model, `targets_path` must be provided. For slice and L:R models, targets can be "
                "automatically inferred, but ligand/receptor information does not exist for the niche model."
            )

        if self.targets_path is not None:
            with open(self.targets_path, "r") as f:
                targets = f.read().splitlines()
                targets = [t for t in targets if t in self.adata.var_names]

        # Else get targets by connecting to the targets of the L:R-downstream transcription factors:
        else:
            # Get the targets of the L:R-downstream transcription factors:
            tf_subset = r_tf_db[r_tf_db["receptor"].isin(self.receptors_expr.columns)]
            tfs = set(tf_subset["tf"])
            tfs = [tf for tf in tfs if tf in self.adata.var_names]
            # Subset to TFs that are expressed in > threshold number of cells:
            if scipy.sparse.issparse(self.adata.X):
                tf_expr_percentage = np.array((self.adata[:, tfs].X > 0).sum(axis=0) / self.adata.n_obs)[0]
            else:
                tf_expr_percentage = np.count_nonzero(self.adata[:, tfs].X, axis=0) / self.adata.n_obs
            tfs = np.array(tfs)[tf_expr_percentage > self.target_expr_threshold]

            targets_subset = tf_target_db[tf_target_db["TF"].isin(tfs)]
            targets = list(set(targets_subset["target"]))
            targets = [target for target in targets if target in self.adata.var_names]
            # Subset to targets that are expressed in > threshold number of cells:
            if scipy.sparse.issparse(self.adata.X):
                target_expr_percentage = np.array((self.adata[:, targets].X > 0).sum(axis=0) / self.adata.n_obs)[0]
            else:
                target_expr_percentage = np.count_nonzero(self.adata[:, targets].X, axis=0) / self.adata.n_obs
            targets = np.array(targets)[target_expr_percentage > self.target_expr_threshold]

        self.targets_expr = pd.DataFrame(
            self.adata[:, targets].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, targets].X,
            index=self.adata.obs_names,
            columns=targets,
        )

        # Compute initial spatial weights for all samples- use twice the min distance as initial bandwidth if not
        # provided (for fixed bw) or 10 nearest neighbors (for adaptive bw):
        if self.bw is None:
            if self.bw_fixed:
                init_bw = (
                    np.min(
                        np.array(
                            [np.min(np.delete(cdist([self.coords[i]], self.coords), 0)) for i in range(self.n_samples)]
                        )
                    )
                    * 2
                )
            else:
                init_bw = 10
        else:
            init_bw = self.bw
        self.all_spatial_weights = self._compute_all_wi(init_bw)
        self.all_spatial_weights = self.comm.bcast(self.all_spatial_weights, root=0)

    def _set_search_range(self):
        """Set the search range for the bandwidth selection procedure."""

        # Check whether the signaling types defined are membrane-bound or are composed of soluble molecules:
        if hasattr(self, "signaling_types"):
            if self.signaling_types == "Cell-Cell Contact":
                # Signaling is limited to occurring between only the nearest neighbors of each cell:
                if self.bw_fixed:
                    distances = cdist(self.coords, self.coords)
                    # Set max bandwidth to the average distance to the 20 nearest neighbors:
                    nearest_idxs_all = np.argpartition(distances, 21, axis=1)[:, 1:21]
                    nearest_distances = np.take_along_axis(distances, nearest_idxs_all, axis=1)
                    self.maxbw = np.mean(nearest_distances, axis=1)

                    if self.minbw is None:
                        # Set min bandwidth to the average distance to the 5 nearest neighbors:
                        nearest_idxs_all = np.argpartition(distances, 6, axis=1)[:, 1:6]
                        nearest_distances = np.take_along_axis(distances, nearest_idxs_all, axis=1)
                        self.minbw = np.mean(nearest_distances, axis=1)
                else:
                    self.maxbw = 20

                    if self.minbw is None:
                        self.minbw = 5

                if self.minbw >= self.maxbw:
                    self.logger.error(
                        "The minimum bandwidth must be less than the maximum bandwidth. Please adjust the `minbw` "
                        "parameter accordingly."
                    )
                return

        # If the bandwidth is defined by a fixed spatial distance:
        if self.bw_fixed:
            max_dist = np.max(np.array([np.max(cdist([self.coords[i]], self.coords)) for i in range(self.n_samples)]))
            # Set max bandwidth higher than the max distance between any two given samples:
            self.maxbw = max_dist * 2

            if self.minbw is None:
                min_dist = np.min(
                    np.array(
                        [np.min(np.delete(cdist(self.coords[[i]], self.coords), i)) for i in range(self.n_samples)]
                    )
                )
                self.minbw = min_dist / 2

        # If the bandwidth is defined by a fixed number of neighbors (and thus adaptive in terms of radius):
        else:
            if self.maxbw is None:
                self.maxbw = 100

            if self.minbw is None:
                self.minbw = 5

        if self.minbw >= self.maxbw:
            self.logger.error(
                "The minimum bandwidth must be less than the maximum bandwidth. Please adjust the `minbw` "
                "parameter accordingly."
            )

    def _compute_all_wi(self, bw: Union[float, int]) -> scipy.sparse.spmatrix:
        """Compute spatial weights for all samples in the dataset given a specified bandwidth.

        Args:
            bw: Bandwidth for the spatial kernel

        Returns:
            wi: Array of weights for all samples in the dataset
        """

        # Parallelized computation of spatial weights for all samples:
        if not self.bw_fixed:
            self.logger.info(
                "Note that 'fixed' was not selected for the bandwidth estimation. Input to 'bw' will be "
                "taken to be the number of nearest neighbors to use in the bandwidth estimation."
            )

        get_wi_partial = partial(
            get_wi,
            n_samples=self.n_samples,
            coords=self.coords,
            fixed_bw=self.bw_fixed,
            exclude_self=self.exclude_self,
            kernel=self.kernel,
            bw=bw,
            threshold=0.01,
            sparse_array=True,
        )

        with Pool() as pool:
            weights = pool.map(get_wi_partial, range(self.n_samples))
        w = scipy.sparse.vstack(weights)
        return w

    def _compute_niche_mat(self) -> Tuple[np.ndarray, List[str]]:
        """Compute the niche matrix for the dataset."""
        # Compute "presence" of each cell type in the neighborhood of each sample:
        dmat_neighbors = self.all_spatial_weights.dot(self.cell_categories.values)

        # Encode the "niche" or each sample by taking into account each sample's own cell type:
        data = {"categories": self.cell_categories, "dmat_neighbors": dmat_neighbors}
        niche_mat = np.asarray(dmatrix("categories:dmat_neighbors-1", data))
        connections_cols = list(product(self.cell_categories.columns, self.cell_categories.columns))
        connections_cols.sort(key=lambda x: x[1])
        return niche_mat, connections_cols

    def _adjust_x(self):
        """Adjust the independent variable array based on the defined bandwidth."""

        # If applicable, use the cell type category array to encode the niche of each sample:
        if self.mod_type == "niche":
            self.X, connections_cols = self._compute_niche_mat()
            # If feature names doesn't already exist, create it:
            if not hasattr(self, "feature_names"):
                self.feature_names = [f"{i[0]}-{i[1]}" for i in connections_cols]

        # If applicable, use the ligand expression array, the receptor expression array and the spatial weights array
        # to compute the ligand-receptor expression signature of each spatial neighborhood:
        elif self.mod_type == "lr":
            X_df = pd.DataFrame(
                np.zeros((self.n_samples, len(self.lr_pairs))), columns=self.feature_names, index=self.adata.obs_names
            )

            for lr_pair in self.lr_pairs:
                lig, rec = lr_pair[0], lr_pair[1]
                lig_expr_values = scipy.sparse.csr_matrix(self.ligands_expr[lig].values.reshape(-1, 1))
                rec_expr_values = scipy.sparse.csr_matrix(self.receptors_expr[rec].values.reshape(-1, 1))

                # Communication signature b/w receptor in target and ligand in neighbors:
                lr_product = np.dot(rec_expr_values, lig_expr_values.T)
                # Neighborhood mask:
                X_df[f"{lig}-{rec}"] = scipy.sparse.csr_matrix.sum(
                    scipy.sparse.csr_matrix.multiply(self.all_spatial_weights, lr_product), axis=1
                ).A.flatten()

            self.X = X_df.values
            # If feature names doesn't already exist, create it:
            if not hasattr(self, "feature_names"):
                self.feature_names = [pair[0] + "-" + pair[1] for pair in self.lr_pairs]
            # If list of L:R labels (secreted vs. membrane-bound vs. ECM) doesn't already exist, create it:
            if not hasattr(self, "self.signaling_types"):
                self.signaling_types = self.lr_db.loc[
                    (self.lr_db["from"].isin([x[0] for x in self.lr_pairs]))
                    & (self.lr_db["to"].isin([x[1] for x in self.lr_pairs])),
                    "type",
                ].tolist()

        # If applicable, combine the ideas of the above two models:
        elif self.mod_type == "slice":
            # Each ligand-receptor pair will have an associated niche matrix:
            niche_mats = {}

            for lr_pair in self.lr_pairs:
                lig, rec = lr_pair[0], lr_pair[1]
                lig_expr_values = scipy.sparse.csr_matrix(self.ligands_expr[lig].values.reshape(-1, 1))
                rec_expr_values = scipy.sparse.csr_matrix(self.receptors_expr[rec].values.reshape(-1, 1))
                # Multiply one-hot category array by the expression of select receptor within that cell:
                rec_expr = np.multiply(
                    self.cell_categories, np.tile(rec_expr_values.toarray(), self.cell_categories.shape[1])
                )
                lig_expr = np.multiply(
                    self.cell_categories, np.tile(lig_expr_values.toarray(), self.cell_categories.shape[1])
                )

                # Multiply adjacency matrix by the cell-specific expression of select ligand:
                nbhd_lig_expr = self.all_spatial_weights.dot(lig_expr)

                # Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the
                # ligand-receptor niches of each sample by documenting the cell type-specific L:R enrichment within
                # the niche:
                data = {"category_rec_expr": rec_expr, "neighborhood_lig_expr": nbhd_lig_expr}
                lr_connections = np.asarray(dmatrix("category_rec_expr:neighborhood_lig_expr-1", data))

                lr_connections_cols = list(product(self.cell_categories.columns, self.cell_categories.columns))
                lr_connections_cols.sort(key=lambda x: x[1])
                n_connections_pairs = len(lr_connections_cols)
                # Swap sending & receiving cell types because we're looking at receptor expression in the "source" cell
                # and ligand expression in the surrounding cells.
                lr_connections_cols = [f"{i[1]}-{i[0]}:{lig}-{rec}" for i in lr_connections_cols]
                niche_mats[f"{lig}-{rec}"] = pd.DataFrame(lr_connections, columns=lr_connections_cols)
                niche_mats = {key: value for key, value in sorted(niche_mats.items())}

            # Combine the niche matrices for each ligand-receptor pair:
            self.X = pd.concat(niche_mats.values(), axis=1)
            self.X.index = self.adata.obs_names
            n_cols = self.X.shape[1]

            # Drop all-zero columns (represent cell type pairs with no spatial coupled L/R expression):
            self.X = self.X.loc[:, (self.X != 0).any(axis=0)]
            self.feature_names = self.X.columns.tolist()
            self.logger.info(
                f"Dropped all-zero columns from cell type-specific signaling array, from {n_cols} to "
                f"{self.X.shape[1]}."
            )
            self.X = self.X.values
            # If list of L:R labels (secreted vs. membrane-bound vs. ECM) doesn't already exist, create it:
            if not hasattr(self, "self.signaling_types"):
                query = re.compile(r"\w+-\w+:(\w+-\w+)")
                self.signaling_types = []
                for col in self.feature_names:
                    ligrec = re.search(query, col).group(1)
                    result = self.lr_db.loc[
                        (self.lr_db["from"] == ligrec.split("-")[0]) & (self.lr_db["to"] == ligrec.split("-")[1]),
                        "type",
                    ].iloc[0]

                    self.signaling_types.append(result)

        # Optionally, add continuous covariate value for each cell:
        if self.covariate_keys is not None:
            matched_obs = []
            matched_var_names = []
            for key in self.covariate_keys:
                if key in self.adata.obs:
                    matched_obs.append(key)
                elif key in self.adata.var_names:
                    matched_var_names.append(key)
                else:
                    self.logger.info(
                        f"Specified covariate key '{key}' not found in adata.obs. Not adding this "
                        f"covariate to the X matrix."
                    )
            matched_obs_matrix = self.adata.obs[matched_obs].to_numpy()
            matched_var_matrix = self.adata[:, matched_var_names].X.toarray()
            cov_names = matched_obs + matched_var_names
            concatenated_matrix = np.concatenate((matched_obs_matrix, matched_var_matrix), axis=1)
            self.X = np.concatenate((self.X, concatenated_matrix), axis=1)
            self.feature_names += cov_names

        self.n_features = self.X.shape[1]
        # Rebroadcast the number of features to fit:
        self.n_features = self.comm.bcast(self.n_features, root=0)
        # Broadcast secreted vs. membrane-bound reference:
        if hasattr(self, "self.signaling_types"):
            # Secreted + ECM-receptor can diffuse across larger distances, but membrane-bound interactions are
            # limited by non-diffusivity. Therefore, it is not advisable to include a mixture of membrane-bound with
            # either of the other two categories in the same model.
            if (
                "Cell-Cell Contact" in set(self.signaling_types) and "Secreted Signaling" in set(self.signaling_types)
            ) or ("Cell-Cell Contact" in set(self.signaling_types) and "ECM-Receptor" in set(self.signaling_types)):
                self.logger.error(
                    "It is not advisable to include a mixture of membrane-bound with either secreted or ECM-receptor "
                    "in the same model because the valid distance scales over which they operate is different. If you "
                    "wish to include both, please run the model twice, once for each category."
                )

            self.signaling_types = set(self.signaling_types)
            if "Secred Signaling" in self.signaling_types or "ECM-Receptor" in self.signaling_types:
                self.signaling_types = "Diffusive Signaling"
            else:
                self.signaling_types = "Cell-Cell Contact"
            self.signaling_types = self.comm.bcast(self.signaling_types, root=0)

    def local_fit(
        self, i: int, y: np.ndarray, X: np.ndarray, bw: Union[float, int], final: bool = False, mgwr: bool = False
    ) -> Union[np.ndarray, List[float]]:
        """Fit a local regression model for each sample.

        Args:
            i: Index of sample for which local regression model is to be fitted
            y: Response variable
            X: Independent variable array
            bw: Bandwidth for the spatial kernel
            final: Set True to indicate that no additional parameter selection needs to be performed; the model can
                be fit and more stats can be returned.
            mgwr: Set True to fit a multiscale GWR model where the independent-dependent relationships can vary over
                different spatial scales

        Returns:
            A single output will be given for each case, and can contain either `betas` or a list w/ combinations of
            the following:
                - i: Index of sample for which local regression model was fitted
                - diagnostic: Portion of the output to be used for diagnostic purposes- for Gaussian regression,
                    this is the residual for the fitted response variable value compared to the observed value. For
                    non-Gaussian generalized linear regression, this is the fitted response variable value (which
                    will be used to compute deviance and log-likelihood later on).
                - hat_i: Row i of the hat matrix, which is the effect of deleting sample i from the dataset on the
                    estimated predicted value for sample i
                - bw_diagnostic: Output to be used for diagnostic purposes during bandwidth selection- for Gaussian
                    regression, this is the squared residual, for non-Gaussian generalized linear regression,
                    this is the fitted response variable value. One of the returns if :param `final` is False
                - betas: Estimated coefficients for sample i- if :param `mgwr` is True, betas is the only return
                - eig: Squared canonical correlation coefficients b/w the predicted values and the response variable,
                    aka the eigenvalues
        """
        # Reshape y if necessary:
        if self.n_features > 1:
            y = y.reshape(-1, 1)

        wi = get_wi(
            i, n_samples=self.n_samples, coords=self.coords, fixed_bw=self.bw_fixed, kernel=self.kernel, bw=bw
        ).reshape(-1, 1)

        if self.distr == "gaussian":
            betas, pseudoinverse = compute_betas_local(y, X, wi)
            pred_y = np.dot(X[i], betas)
            residual = y[i] - pred_y
            diagnostic = residual

            # Effect of deleting sample i from the dataset on the estimated predicted value at sample i:
            hat_i = np.dot(X[i], pseudoinverse[:, i])

        elif self.distr == "poisson" or self.distr == "nb":
            # init_betas (initial coefficients) to be incorporated at runtime:
            betas, y_hat, _, final_irls_weights, _, _, pseudoinverse = iwls(
                y,
                X,
                distr=self.distr,
                init_betas=self.init_betas,
                tol=self.tolerance,
                max_iter=self.max_iter,
                spatial_weights=wi,
                link=None,
                alpha=self.alpha,
                tau=None,
            )

            # Reshape coefficients if necessary:
            betas = betas.flatten()
            pred_y = y_hat[i]
            diagnostic = pred_y
            # Effect of deleting sample i from the dataset on the estimated predicted value at sample i:
            hat_i = np.dot(X[i], pseudoinverse[:, i]) * final_irls_weights[i][0]

        else:
            self.logger.error("Invalid `distr` specified. Must be one of 'gaussian', 'poisson', or 'nb'.")

        # Squared canonical correlation:
        eig = np.diag(np.dot(pseudoinverse, pseudoinverse.T)).reshape(-1)

        if final:
            if mgwr:
                return betas
            return np.concatenate(([i, diagnostic, hat_i], betas, eig))
        else:
            # For bandwidth optimization:
            if self.distr == "gaussian":
                bw_diagnostic = residual * residual
            elif self.distr == "poisson" or self.distr == "nb":
                # Else just return fitted value for diagnostic purposes:
                bw_diagnostic = pred_y
            return [bw_diagnostic, hat_i]

    def find_optimal_bw(self, range_lowest: float, range_highest: float, function: Callable) -> float:
        """Perform golden section search to find the optimal bandwidth.

        Args:
            range_lowest: Lower bound of the search range
            range_highest: Upper bound of the search range
            function: Function to be minimized

        Returns:
            bw: Optimal bandwidth
        """
        delta = 0.38197
        new_lb = range_lowest + delta * np.abs(range_highest - range_lowest)
        new_ub = range_highest - delta * np.abs(range_highest - range_lowest)

        score = None
        optimum_bw = None
        difference = 1.0e9
        iterations = 0
        results_dict = {}

        while np.abs(difference) > self.tolerance and iterations < self.max_iter:
            iterations += 1

            # Bandwidth needs to be discrete:
            if not self.bw_fixed:
                new_lb = np.round(new_lb)
                new_ub = np.round(new_ub)

            if new_lb in results_dict:
                lb_score = results_dict[new_lb]
            else:
                lb_score = function(new_lb)
                results_dict[new_lb] = lb_score

            if new_ub in results_dict:
                ub_score = results_dict[new_ub]
            else:
                ub_score = function(new_ub)
                results_dict[new_ub] = ub_score

            if self.comm.rank == 0:
                # Follow direction of increasing score until score stops increasing:
                if lb_score <= ub_score:
                    # Set new optimum score and bandwidth:
                    optimum_score = lb_score
                    optimum_bw = new_lb

                    # Update new max upper bound and test lower bound:
                    range_highest = new_ub
                    new_ub = new_lb
                    new_lb = range_lowest + delta * np.abs(range_highest - range_lowest)

                # Else follow direction of decreasing score until score stops decreasing:
                else:
                    # Set new optimum score and bandwidth:
                    optimum_score = ub_score
                    optimum_bw = new_ub

                    # Update new max lower bound and test upper bound:
                    range_lowest = new_lb
                    new_lb = new_ub
                    new_ub = range_highest - delta * np.abs(range_highest - range_lowest)

                difference = lb_score - ub_score
                # Update new value for score:
                score = optimum_score

            new_lb = self.comm.bcast(new_lb, root=0)
            new_ub = self.comm.bcast(new_ub, root=0)
            score = self.comm.bcast(score, root=0)
            difference = self.comm.bcast(difference, root=0)
            optimum_bw = self.comm.bcast(optimum_bw, root=0)

        return optimum_bw

    def mpi_fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        bw: Union[float, int],
        final: bool = False,
        mgwr: bool = False,
        y_label: Optional[str] = None,
    ):
        """Fit local regression model for each sample in parallel, given a specified bandwidth.

        Args:
            y: Response variable
            X: Independent variable array- if not given, will default to :attr `X`. Note that if object was initialized
                using an AnnData object, this will be overridden with :attr `X` even if a different array is given.
            bw: Bandwidth for the spatial kernel
            final: Set True to indicate that no additional parameter selection needs to be performed; the model can
                be fit and more stats can be returned.
            mgwr: Set True to fit a multiscale GWR model where the independent-dependent relationships can vary over
                different spatial scales
            y_label: Optional, can be used to provide a unique ID for the dependent variable for saving purposes
        """
        # If model to be run is a "niche", "lr" or "slice" model, update the spatial weights and then update X given
        # the current value of the bandwidth:
        if X is None:
            if hasattr(self, "adata"):
                self.all_spatial_weights = self._compute_all_wi(bw)
                self.all_spatial_weights = self.comm.bcast(self.all_spatial_weights, root=0)
                self._adjust_x()
                self.X = self.comm.bcast(self.X, root=0)
                X = self.X
            else:
                X = self.X
        if X.shape[1] != self.n_features:
            self.n_features = X.shape[1]
            self.n_features = self.comm.bcast(self.n_features, root=0)

        if final:
            if mgwr:
                local_fit_outputs = np.empty((self.x_chunk.shape[0], self.n_features), dtype=np.float64)
            else:
                local_fit_outputs = np.empty((self.x_chunk.shape[0], 2 * self.n_features + 3), dtype=np.float64)

            # Fitting for each location:
            pos = 0
            for i in self.x_chunk:
                local_fit_outputs[pos] = self.local_fit(i, y, X, bw, final=final, mgwr=mgwr)
                pos += 1

            # Gather data to the central process such that an array is formed where each sample has its own
            # measurements:
            all_fit_outputs = self.comm.gather(local_fit_outputs, root=0)
            # Column 0: Index of the sample
            # Column 1: Diagnostic (residual for Gaussian, fitted response value for Poisson/NB)
            # Column 2: Contribution of each sample to its own value
            # Column 3: Estimated coefficients
            # Column 4: Canonical correlations

            # If mgwr, do not need to fit using fixed bandwidth:
            if mgwr:
                all_fit_outputs = self.comm.bcast(all_fit_outputs, root=0)
                all_fit_outputs = np.vstack(all_fit_outputs)
                return all_fit_outputs

            if self.comm.rank == 0:
                all_fit_outputs = np.vstack(all_fit_outputs)
                self.logger.info(f"Computing metrics for GWR using bandwidth: {bw}")

                # Residual sum of squares for Gaussian model:
                if self.distr == "gaussian":
                    RSS = np.sum(all_fit_outputs[:, 1] ** 2)
                    # Total sum of squares:
                    TSS = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - RSS / TSS

                    # Note: trace of the hat matrix and effective number of parameters (ENP) will be used
                    # interchangeably:
                    ENP = np.sum(all_fit_outputs[:, 2])
                    # Residual variance:
                    sigma_squared = RSS / (self.n_samples - ENP)
                    # Corrected Akaike Information Criterion:
                    aicc = self.compute_aicc_linear(RSS, ENP)
                    # Scale the squared canonical correlation coefficients by their standard errors:
                    all_fit_outputs[:, -self.n_features :] = np.sqrt(
                        all_fit_outputs[:, -self.n_features :] * sigma_squared
                    )

                    # For saving outputs:
                    header = "name,residual,influence,"
                else:
                    r_squared = None

                if self.distr == "poisson" or self.distr == "nb":
                    if self.distr == "poisson":
                        distr = Poisson()
                    else:
                        distr = NegativeBinomial()
                    # Deviance:
                    deviance = distr.deviance(y, all_fit_outputs[:, 1])
                    # Residual deviance:
                    residual_deviance = distr.deviance_residuals(y, all_fit_outputs[:, 1])
                    # Reshape if necessary:
                    if self.n_features > 1:
                        residual_deviance = residual_deviance.reshape(-1, 1)
                    # ENP:
                    ENP = np.sum(all_fit_outputs[:, 2])
                    # Corrected Akaike Information Criterion:
                    aicc = self.compute_aicc_glm(residual_deviance, ENP)
                    # Scale the squared canonical correlation coefficients using the residual deviance:
                    all_fit_outputs[:, -self.n_features :] = np.sqrt(
                        all_fit_outputs[:, -self.n_features :] * residual_deviance
                    )

                    # For saving outputs:
                    header = "name,deviance,influence,"
                else:
                    deviance = None

                # Save results:
                varNames = self.feature_names
                if self.fit_intercept:
                    varNames = ["intercept"] + list(varNames)
                # Columns for coefficients and squared canonical coefficients:
                for x in varNames:
                    header += "b_" + x + ","
                for x in varNames:
                    header += "sq_cc_" + x + ","

                # Return output diagnostics and save result- NOTE: PASS DEVIANCE AS WELL:
                self.output_diagnostics(aicc, ENP, r_squared, deviance)
                self.save_results(all_fit_outputs, header, label=y_label)

            return

        # If not the final run:
        if self.distr == "gaussian":
            # Compute AICc using the sum of squared residuals:
            RSS = 0
            trace_hat = 0

            for i in self.x_chunk:
                fit_outputs = self.local_fit(i, y, X, bw, final=False)
                err_sq, hat_i = fit_outputs[0], fit_outputs[1]
                RSS += err_sq
                trace_hat += hat_i

            # Send data to the central process:
            RSS_list = self.comm.gather(RSS, root=0)
            trace_hat_list = self.comm.gather(trace_hat, root=0)

            if self.comm.rank == 0:
                RSS = np.sum(RSS_list)
                trace_hat = np.sum(trace_hat_list)
                aicc = self.compute_aicc_linear(RSS, trace_hat)
                if not mgwr:
                    self.logger.info(f"Bandwidth: {bw}, Linear AICc: {aicc}")
                return aicc

        elif self.distr == "poisson" or self.distr == "nb":
            # Compute AICc using the fitted and observed values:
            trace_hat = 0
            pos = 0
            y_pred = np.empty(self.x_chunk.shape[0], dtype=np.float64)

            for i in self.x_chunk:
                fit_outputs = self.local_fit(i, y, X, bw, final=False)
                y_pred_i, hat_i = fit_outputs[0], fit_outputs[1]
                y_pred[pos] = y_pred_i
                trace_hat += hat_i
                pos += 1

            # Send data to the central process:
            all_y_pred = self.comm.gather(y_pred, root=0)
            trace_hat_list = self.comm.gather(trace_hat, root=0)

            if self.comm.rank == 0:
                if self.distr == "poisson":
                    distr = Poisson()
                else:
                    distr = NegativeBinomial()

                deviance_residuals = distr.deviance_residuals(y, all_y_pred)
                trace_hat = np.sum(trace_hat_list)
                aicc = self.compute_aicc_glm(deviance_residuals, trace_hat)
                if not mgwr:
                    self.logger.info(f"Bandwidth: {bw}, GLM AICc: {aicc}")
                return aicc

        return

    def fit(
        self,
        y: Optional[pd.DataFrame] = None,
        X: Optional[pd.DataFrame] = None,
        mgwr: bool = False,
    ):
        """For each column of the dependent variable array, fit model. If given bandwidth, run :func
        `STGWR.mpi_fit()` with the given bandwidth. Otherwise, compute optimal bandwidth using :func
        `STGWR.select_optimal_bw()`, minimizing AICc.

        Args:
            y: Optional dataframe, can be used to provide dependent variable array directly to the fit function. If
                None, will use :attr `targets_expr` computed using the given AnnData object to create this (each
                individual column will serve as an independent variable).
            X: Optional dataframe, can be used to provide dependent variable array directly to the fit function. If
                None, will use :attr `X` computed using the given AnnData object and the type of the model to create.
            mgwr: Set True to indicate that a multiscale model should be fitted

        Returns:
            all_data: Dictionary containing outputs of :func `STGWR.mpi_fit()` with the chosen or determined bandwidth-
                note that this will either be None or in the case that :param `mgwr` is True, an array of shape [
                n_samples, n_features] representing the coefficients for each sample (if :param `mgwr` is False,
                these arrays will instead be saved to file).
            all_bws: Dictionary containing outputs in the case that bandwidth is not already known, resulting from
                the conclusion of the optimization process. Will also be None if :param `mgwr` is True.
        """

        if not self.set_up:
            self.logger.info("Model has not yet been set up to run, running :func `STGWR._set_up_model()` now...")
            self._set_up_model()

        if y is None:
            y_arr = self.targets_expr if hasattr(self, "targets_expr") else self.target
        else:
            y_arr = y

        # Compute fit for each column of the dependent variable array individually- store each output array (if
        # applicable) and optimal bandwidth (also if applicable):
        all_data, all_bws = {}, {}

        for target in y_arr.columns:
            y = y_arr[target].values
            y = self.comm.bcast(y, root=0)

            if self.bw is not None:
                # If bandwidth is already known, run the main fit function:
                if X is None:
                    self.mpi_fit(y, self.X, self.bw, final=True)
                else:
                    self.mpi_fit(y, X, self.bw, final=True)
                return

            if self.comm.rank == 0:
                self._set_search_range()
                if not mgwr:
                    self.logger.info(f"Calculated bandwidth range over which to search: {self.minbw}-{self.maxbw}.")
            self.minbw = self.comm.bcast(self.minbw, root=0)
            self.maxbw = self.comm.bcast(self.maxbw, root=0)

            if X is None:
                fit_function = lambda bw: self.mpi_fit(y, self.X, bw, final=False, mgwr=mgwr)
            else:
                fit_function = lambda bw: self.mpi_fit(y, X.values, bw, final=False, mgwr=mgwr)
            optimal_bw = self.find_optimal_bw(self.minbw, self.maxbw, fit_function)
            if self.bw_fixed:
                optimal_bw = round(optimal_bw, 2)

            data = self.mpi_fit(y, X, optimal_bw, final=True, mgwr=mgwr, y_label=target)
            if data is not None:
                all_data[target] = data
            if optimal_bw is not None:
                all_bws[target] = optimal_bw

        return all_data, all_bws

    def predict(
        self, input: Optional[np.ndarray] = None, coeffs: Optional[Union[np.ndarray, Dict[str, pd.DataFrame]]] = None
    ) -> pd.DataFrame:
        """Given input data and learned coefficients, predict the dependent variables.

        Args:
            input: Input data to be predicted on.
            coeffs: Coefficients to be used in the prediction. If None, will attempt to load the coefficients learned
                in the fitting process from file.
        """
        if input is None:
            input = self.X

        if coeffs is None:
            coeffs = self.return_outputs()
            # If dictionary, compute outputs for the multiple dependent variables and concatenate together:
            if isinstance(coeffs, Dict):
                all_y_pred = pd.DataFrame(index=self.sample_names, columns=coeffs.keys())
                for target in coeffs:
                    y_pred = np.sum(input * coeffs[target], axis=1)
                    if self.distr != "gaussian":
                        y_pred = self.distr_obj.predict(y_pred)
                    all_y_pred = pd.concat([all_y_pred, y_pred], axis=1)
                return all_y_pred

            else:
                y_pred = pd.DataFrame(np.sum(input * coeffs, axis=1), index=self.sample_names, columns=["y_pred"])
                return y_pred

    # ---------------------------------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------------------------------
    def compute_aicc_linear(self, RSS: float, trace_hat: float) -> float:
        """Compute the corrected Akaike Information Criterion (AICc) for the linear GWR model."""
        aicc = (
            self.n_samples * np.log(RSS / self.n_samples)
            + self.n_samples * np.log(2 * np.pi)
            + self.n_samples * (self.n_samples + trace_hat) / (self.n_samples - trace_hat - 2.0)
        )

        return aicc

    def compute_aicc_glm(self, resid_dev: np.ndarray, trace_hat: float) -> float:
        """Compute the corrected Akaike Information Criterion (AICc) for the generalized linear GWR models."""
        aicc = (
            np.sum(resid_dev**2)
            + 2.0 * trace_hat
            + 2.0 * trace_hat * (trace_hat + 1.0) / (self.n_samples - trace_hat - 1.0)
        )

        return aicc

    def output_diagnostics(
        self,
        aicc: float,
        ENP: float,
        r_squared: Optional[float] = None,
        deviance: Optional[float] = None,
        y_label: Optional[str] = None,
    ) -> None:
        """Output diagnostic information about the GWR model."""

        if y_label is None:
            y_label = self.distr

        self.logger.info(f"Corrected Akaike information criterion for {y_label} model: {aicc}")
        self.logger.info(f"Effective number of parameters for {y_label} model: {ENP}")
        # Print R-squared for Gaussian assumption:
        if self.distr == "gaussian":
            if r_squared is None:
                self.logger.error(":param `r_squared` must be provided when performing Gaussian regression.")
            self.logger.info(f"R-squared for {y_label} model: {r_squared}")
        # Else log the deviance:
        else:
            if deviance is None:
                self.logger.error(":param `deviance` must be provided when performing non-Gaussian regression.")
            self.logger.info(f"Deviance for {y_label} model: {deviance}")

    # ---------------------------------------------------------------------------------------------------
    # Save to file
    # ---------------------------------------------------------------------------------------------------
    def save_results(self, data: np.ndarray, header: str, label: Optional[str]) -> None:
        """Save the results of the GWR model to file, and return the coefficients.

        Args:
            data: Elements of data to save to .csv
            header: Column names
            label: Optional, can be used to provide unique ID to save file- notably used when multiple dependent
                variables with different names are fit during this process.

        Returns:
            betas: Model coefficients
        """
        # Check if output_path was left as the default:
        if self.output_path == "./output/stgwr_results.csv":
            if not os.path.exists("./output"):
                os.makedirs("./output")

        # If output path already has files in it, clear them:
        output_dir = os.path.dirname(self.output_path)
        if os.listdir(output_dir):
            # If there are files, delete them
            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        if label is not None:
            path = os.path.splitext(self.output_path)[0] + f"_{label}" + os.path.splitext(self.output_path)[1]
        else:
            path = self.output_path

        if self.comm.rank == 0:
            # Save to .csv:
            np.savetxt(path, data, delimiter=",", header=header[:-1], comments="")

    def predict_and_save(
        self, input: Optional[np.ndarray] = None, coeffs: Optional[Union[np.ndarray, Dict[str, pd.DataFrame]]] = None
    ):
        """Given input data and learned coefficients, predict the dependent variables and then save the output.

        Args:
            input: Input data to be predicted on.
            coeffs: Coefficients to be used in the prediction. If None, will attempt to load the coefficients learned
                in the fitting process from file.
        """
        y_pred = self.predict(input, coeffs)
        # Save to parent directory of the output path:
        parent_dir = os.path.dirname(self.output_path)
        pred_path = os.path.join(parent_dir, "predictions.csv")
        y_pred.to_csv(pred_path)

    def return_outputs(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return final coefficients for all fitted models."""
        parent_dir = os.path.dirname(self.output_path)
        all_coeffs = {}
        file_list = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]
        for file in file_list:
            all_outputs = pd.read_csv(os.path.join(parent_dir, file), index_col=0)
            betas = all_outputs[[col for col in all_outputs.columns if col.startswith("b_")]]
            betas.index = self.sample_names

            # If there were multiple dependent variables, save coefficients to dictionary:
            if len(file_list) > 1:
                all_coeffs[file.split("_")[-1][:-4]] = betas
            else:
                all_coeffs = betas

        return all_coeffs

    def return_intercepts(self) -> Union[None, np.ndarray, Dict[str, np.ndarray]]:
        """Return final intercepts for all fitted models."""
        if not self.fit_intercept:
            self.logger.info("No intercepts were fit, returning None.")
            return

        parent_dir = os.path.dirname(self.output_path)
        all_intercepts = {}
        for file in os.listdir(parent_dir):
            all_outputs = pd.read_csv(os.path.join(parent_dir, file), index_col=0)
            intercepts = all_outputs["intercept"].values

            # If there were multiple dependent variables, save coefficients to dictionary:
            if file != os.path.basename(self.output_path):
                all_intercepts[file.split("_")[-1]] = intercepts
            else:
                all_intercepts = intercepts

        return all_intercepts


# MGWR:
class STMGWR(STGWR):
    """Modified version of the spatially weighted regression on spatial omics data with parallel processing,
    enabling each feature to have its own distinct spatial scale parameter. Runs after being called from the command
    line.

    Args:
        comm: MPI communicator object initialized with mpi4py, to control parallel processing operations
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.

    Attributes:
        mod_type: The type of model that will be employed- this dictates how the data will be processed and
            prepared. Options:
                - "niche": Spatially-aware, uses spatial connections between samples as independent variables
                - "lr": Spatially-aware, uses the combination of receptor expression in the "target" cell and spatially
                    lagged ligand expression in the neighboring cells as independent variables.
                - "slice": Spatially-aware, uses a coupling of spatial category connections, ligand expression
                    and receptor expression to perform regression on select receptor-downstream genes.


        adata_path: Path to the AnnData object from which to extract data for modeling
        csv_path: Can also be used to specify path to non-AnnData .csv object. Assumes the first three columns
            contain x- and y-coordinates and then dependent variable values, in that order, with all subsequent
            columns containing independent variable values.
        normalize: Set True to Perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size). It is advisable not to do this if performing Poisson or negative binomial
            regression.
        smooth: Set True to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression. It is advisable not to do this if performing Poisson or negative binomial regression.
        log_transform: Set True if log-transformation should be applied to expression. It is advisable not to do
            this if performing Poisson or negative binomial regression.
        target_expr_threshold: Only used if :param `mod_type` is "lr" or "slice" and :param `targets_path` is not
            given. When manually selecting targets, expression above a threshold percentage of cells will be used to
            filter to a smaller subset of interesting genes. Defaults to 0.2.


        custom_lig_path: Optional path to a .txt file containing a list of ligands for the model, separated by
            newlines. Only used if :attr `mod_type` is "lr" or "slice" (and thus uses ligand/receptor expression
            directly in the inference). If not provided, will select ligands using a threshold based on expression
            levels in the data.
        custom_rec_path: Optional path to a .txt file containing a list of receptors for the model, separated by
            newlines. Only used if :attr `mod_type` is "lr" or "slice" (and thus uses ligand/receptor expression
            directly in the inference). If not provided, will select receptors using a threshold based on expression
            levels in the data.
        custom_pathways_path: Rather than  providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model. Only used if :attr `mod_type`
            is "lr" or "slice".
        targets_path: Optional path to a .txt file containing a list of prediction target genes for the model,
            separated by newlines. If not provided, targets will be strategically selected from the given receptors.
        init_betas_path: Optional path to a .npy file containing initial coefficient values for the model. Initial
            coefficients should have shape [n_features, ].


        cci_dir: Full path to the directory containing cell-cell communication databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved


        coords_key: Key in .obsm of the AnnData object that contains the coordinates of the cells
        group_key: Key in .obs of the AnnData object that contains the category grouping for each cell
        covariate_keys: Can be used to optionally provide any number of keys in .obs or .var containing a continuous
            covariate (e.g. expression of a particular TF, avg. distance from a perturbed cell, etc.)


        minbw: For use in automated bandwidth selection- the lower-bound bandwidth to test.
        maxbw: For use in automated bandwidth selection- the upper-bound bandwidth to test.


        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".


        bw_fixed: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        exclude_self: If True, ignore each sample itself when computing the kernel density estimation
        fit_intercept: Set True to include intercept in the model and False to exclude intercept
    """

    def __init__(self, comm: MPI.Comm, parser: argparse.ArgumentParser):
        super().__init__(comm, parser)

    def backfitting(self):
        """
        Backfitting algorithm for MGWR, obtains parameter estimates and variate-specific bandwidths.
        """
        if self.comm.rank == 0:
            self.logger.info("MGWR Backfitting...")

        # Initialize parameters:
        betas, bw = self.fit(mgwr=True)
