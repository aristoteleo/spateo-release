"""
Modeling putative gene regulatory networks using a regression model that is considerate of the spatial heterogeneity of
(and thus the context-dependency of the relationships of) the response variable.
"""
import argparse
import math
import os
import sys
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI
from scipy.spatial.distance import cdist

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm
from spateo.preprocessing.normalize import normalize_total
from spateo.preprocessing.transform import log1p
from spateo.tools.find_neighbors import transcriptomic_connectivity
from spateo.tools.spatial_degs import moran_i
from spateo.tools.ST_regression.distributions import NegativeBinomial, Poisson
from spateo.tools.ST_regression.regression_utils import multicollinearity_check, smooth
from spateo.tools.ST_regression.SWR import MuSIC


# ---------------------------------------------------------------------------------------------------
# GWR for inferring gene regulatory networks
# ---------------------------------------------------------------------------------------------------
class GWRGRN(MuSIC):
    """
    Construct regulatory networks, taking prior knowledge network and spatial expression patterns into account.

    Args:
        MPI_comm: MPI communicator object initialized with mpi4py, to control parallel processing operations
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.

    Attributes:
        adata_path: Path to the AnnData object from which to extract data for modeling
        normalize: Set True to Perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size). It is advisable not to do this if performing Poisson or negative binomial
            regression.
        smooth: Set True to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression. It is advisable not to do this if performing Poisson or negative binomial regression.
        log_transform: Set True if log-transformation should be applied to expression. It is advisable not to do
            this if performing Poisson or negative binomial regression.


        custom_lig_path: Optional path to a .txt file containing a list of ligands for the model, separated by
            newlines. If not provided, will select ligands using a threshold based on expression
            levels in the data.
        custom_rec_path: Optional path to a .txt file containing a list of receptors for the model, separated by
            newlines. If not provided, will select receptors using a threshold based on expression
            levels in the data.
        custom_pathways_path: Rather than  providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model.
        custom_regulators_path: Optional path to a .txt file containing a list of regulatory factors (e.g.
            transcription factors) for the model. If not provided, will select transcription factors using a
            threshold based on expression levels in the data.


        cci_dir: Full path to the directory containing cell-cell communication databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved.


        coords_key: Key in .obsm of the AnnData object that contains the coordinates of the cells
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
        super().__init__(comm, parser)

        self.regulators = None
        self.custom_regulators_path = self.arg_retrieve.custom_regulators_path

        self.grn_load_and_process()

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
        if self.subsample:
            self.run_subsample()
            # Indicate model has been subsampled:
            self.subsampled = True
        else:
            chunk_size = int(math.ceil(float(len(range(self.n_samples))) / self.comm.size))
            # Assign chunks to each process:
            self.x_chunk = np.arange(self.n_samples)[self.comm.rank * chunk_size : (self.comm.rank + 1) * chunk_size]
            self.subsampled = False

    def grn_load_and_process(self):
        """
        Load AnnData object and process data for modeling.
        """
        self.adata = anndata.read_h5ad(self.adata_path)
        self.adata.uns["__type"] = "UMI"
        self.sample_names = self.adata.obs_names
        self.coords = self.adata.obsm[self.coords_key]
        self.n_samples = self.adata.n_obs
        self.n_runs_all = np.arange(self.n_samples)
        # Placeholder- this will change at time of fitting:
        self.n_features = self.adata.n_vars

        # Only used for multiscale- number of parallel processes:
        self.multiscale_chunks = self.arg_retrieve.chunks

        # Check if path to init betas is given- if not given will use the prior network to determine suitable initial
        # parameters:
        if self.init_betas_path is not None:
            self.logger.info(f"Loading initial betas from: {self.init_betas_path}")
            self.init_betas = np.load(self.init_betas_path)

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

        # Define array of targets of interest (combination of ligands, receptors and other targets):
        if self.species == "human":
            grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
            lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_receptor_TF_db.csv"), index_col=0)
        elif self.species == "mouse":
            grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
            lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
            r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
        else:
            self.logger.error("Invalid species specified. Must be one of 'human' or 'mouse'.")
        database_ligands = set(lr_db["from"])
        database_receptors = set(lr_db["to"])
        database_pathways = set(r_tf_db["pathway"])
        database_tfs = grn.columns

        # Ligand array:
        if self.custom_ligands_path is not None:
            with open(self.custom_ligands_path, "r") as f:
                ligands = f.read().splitlines()
                ligands = [l for l in ligands if l in database_ligands]
                # Get individual components if any complexes are included in this list:
                ligands = [l for item in ligands for l in item.split("_")]
        else:
            # List of possible complexes to search through:
            l_complexes = [elem for elem in database_ligands if "_" in elem]
            # All possible ligand molecules:
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
                    "No significant spatially-variable ligands found. Using top 10 most " "spatially-variable ligands."
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
                f"Found {len(ligands)} among significantly spatially-variable genes and associated " f"complex members."
            )

        ligands = [l for l in ligands if l in self.adata.var_names]

        # Receptor array:
        if self.custom_receptors_path is not None:
            with open(self.custom_receptors_path, "r") as f:
                receptors = f.read().splitlines()
                receptors = [r for r in receptors if r in database_receptors]
                # Get individual components if any complexes are included in this list:
                receptors = [r for item in receptors for r in item.split("_")]

        elif self.custom_pathways_path is not None:
            with open(self.custom_pathways_path, "r") as f:
                pathways = f.read().splitlines()
                pathways = [p for p in pathways if p in database_pathways]
            # Get all receptors associated with these pathway(s):
            r_tf_db_subset = r_tf_db[r_tf_db["pathway"].isin(pathways)]
            receptors = set(r_tf_db_subset["receptor"])
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
                "Preparing data: getting list of receptors from among the most highly spatially-variable genes."
            )
            if "m_degs" not in locals():
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

        if self.targets_path is not None:
            with open(self.targets_path, "r") as f:
                targets = f.read().splitlines()
                targets = [t for t in targets if t in self.adata.var_names]
            all_molecule_targets = ligands + receptors + targets
        else:
            all_molecule_targets = ligands + receptors

        self.molecule_expr = pd.DataFrame(
            self.adata[:, all_molecule_targets].X.toarray()
            if scipy.sparse.issparse(self.adata.X)
            else self.adata[:, all_molecule_targets].X,
            index=self.adata.obs_names,
            columns=all_molecule_targets,
        )

        # Define array of potential regulators:
        if self.custom_regulators_path is not None:
            with open(self.custom_regulators_path, "r") as f:
                regulators = f.read().splitlines()
                regulators = [r for r in regulators if r in self.adata.var_names]
        else:
            # Get list of regulatory factors from among the most highly spatially-variable genes, indicative of
            # potentially interesting spatially-enriched signal:
            self.logger.info(
                "Preparing data: getting list of regulators from among the most highly spatially-variable genes."
            )
            if "m_degs" not in locals():
                m_degs = moran_i(self.adata)
            m_filter_genes = m_degs[m_degs.moran_q_val < 0.05].sort_values(by=["moran_i"], ascending=False).index
            regulators = [g for g in m_filter_genes if g in database_tfs]

            # If no significant spatially-variable receptors are found, use the top 100 most spatially-variable TFs:
            if len(regulators) == 0:
                self.logger.info(
                    "No significant spatially-variable regulatory factors found. Using top 100 most "
                    "spatially-variable TFs."
                )
                m_filter_genes = m_degs.sort_values(by=["moran_i"], ascending=False).index
                regulators = [g for g in m_filter_genes if g in database_tfs][:100]

        self.regulators_expr = pd.DataFrame(
            self.adata[:, regulators].X.toarray()
            if scipy.sparse.issparse(self.adata.X)
            else self.adata[:, regulators].X,
            index=self.adata.obs_names,
            columns=regulators,
        )

        # If :attr `multicollinear_threshold` is given, drop multicollinear features:
        if self.multicollinear_threshold is not None:
            self.regulators_expr = multicollinearity_check(
                self.regulators_expr, self.multicollinear_threshold, logger=self.logger
            )

        self.X = self.regulators_expr.values
        self.feature_names = list(self.regulators_expr.columns)

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

            if self.fit_intercept:
                self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
                self.feature_names = ["intercept"] + self.feature_names

        # To initialize coefficients, filter the GRN rows and columns to only include the regulators and targets-
        # note that initial betas will only be used in Poisson / negative binomial regressions:
        self.all_betas = {}
        grn = grn.loc[all_molecule_targets, regulators]
        for row in grn.index:
            self.all_betas[row] = grn.loc[row, :].values
        self.all_betas = self.comm.bcast(self.all_betas, root=0)

    def grn_fit(
        self,
        y: Optional[pd.DataFrame] = None,
        X: Optional[np.ndarray] = None,
    ):
        """For each column of the dependent variable array (in this specific case, for each gene expression vector
        for ligands/receptors/other targets), fit model. If given bandwidth, run :func `SWR.mpi_fit()` with the
        given bandwidth. Otherwise, compute optimal bandwidth using :func `SWR.select_optimal_bw()`, minimizing AICc.

        Args:
            y: Optional dataframe, can be used to provide dependent variable array directly to the fit function. If
                None, will use :attr `targets_expr` computed using the given AnnData object to create this (each
                individual column will serve as an independent variable).
            X: Optional dataframe, can be used to provide dependent variable array directly to the fit function. If
                None, will use :attr `X` computed using the given AnnData object and the type of the model to create.
            multiscale: Set True to indicate that a multiscale model should be fitted

        Returns:
            all_data: Dictionary containing outputs of :func `SWR.mpi_fit()` with the chosen or determined bandwidth-
                note that this will either be None or in the case that :param `multiscale` is True, an array of shape [
                n_samples, n_features] representing the coefficients for each sample (if :param `multiscale` is False,
                these arrays will instead be saved to file).
            all_bws: Dictionary containing outputs in the case that bandwidth is not already known, resulting from
                the conclusion of the optimization process. Will also be None if :param `multiscale` is True.
        """
        if y is None:
            y_arr = self.molecule_expr
        else:
            y_arr = y
        if X is None:
            X = self.X

        all_data, all_bws = self.fit(y_arr, X, init_betas=self.all_betas, multiscale=False)

        return all_data, all_bws

    def grn_fit_multiscale(
        self,
        y: Optional[pd.DataFrame] = None,
        X: Optional[np.ndarray] = None,
    ):
        if y is None:
            y_arr = self.molecule_expr
        else:
            y_arr = y
        if X is None:
            X = self.X

        self.multiscale_backfitting(y_arr, X, init_betas=self.all_betas)

    def test(self):
        """Use as a testing space to print out attributes, etc."""
        "filler"
