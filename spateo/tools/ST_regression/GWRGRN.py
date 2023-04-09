"""
Modeling putative gene regulatory networks using a regression model that is considerate of the spatial heterogeneity of
(and thus the context-dependency of the relationships of) the response variable.
"""
import argparse
import os
import sys

import anndata
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm
from spateo.preprocessing.normalize import normalize_total
from spateo.preprocessing.transform import log1p
from spateo.tools.find_neighbors import transcriptomic_connectivity
from spateo.tools.spatial_degs import moran_i
from spateo.tools.ST_regression.distributions import NegativeBinomial, Poisson
from spateo.tools.ST_regression.regression_utils import smooth

from .STGWR import STGWR


# ---------------------------------------------------------------------------------------------------
# GWR for inferring gene regulatory networks
# ---------------------------------------------------------------------------------------------------
class GWRGRN(STGWR):
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
        self.parse_stgwr_args()
        self.custom_regulators_path = self.arg_retrieve.custom_regulators_path

    def grn_load_and_process(self):
        """
        Load AnnData object and process data for modeling.
        """
        self.adata = anndata.read_h5ad(self.adata_path)
        self.sample_names = self.adata.obs_names
        self.coords = self.adata.obsm[self.coords_key]
        self.n_samples = self.adata.n_obs
        # Placeholder- this will change at time of fitting:
        self.n_features = self.adata.n_vars

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
            all_regulators = []
            for mol in all_molecule_targets:
                reg_tfs = list(grn.columns[grn.loc[mol] == 1])
                all_regulators.extend(reg_tfs)
            regulators = list(set(all_regulators))

        self.regulators_expr = pd.DataFrame(
            self.adata[:, regulators].X.toarray()
            if scipy.sparse.issparse(self.adata.X)
            else self.adata[:, regulators].X,
            index=self.adata.obs_names,
            columns=regulators,
        )

        # To initialize coefficients, filter the GRN rows and columns to only include the regulators and targets:
        self.all_betas = {}
        grn = grn.loc[all_molecule_targets, regulators]
        for row in grn.index:
            self.all_betas[row] = grn.loc[row, :].values

    def grn_mpi_fit(self):
        "filler"

    def grn_fit(self):
        "filler"
