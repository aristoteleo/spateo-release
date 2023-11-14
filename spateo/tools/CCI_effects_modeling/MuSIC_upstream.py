"""
Functionalities to aid in feature selection to characterize signaling patterns from spatial transcriptomics. Given a
list of signaling molecules (ligands or receptors) and/or target genes
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI
from scipy.stats import percentileofscore

from ..find_neighbors import find_bw_for_n_neighbors
from .MuSIC import MuSIC
from .regression_utils import multitesting_correction
from .SWR import define_spateo_argparse


# ---------------------------------------------------------------------------------------------------
# Selection of targets and signaling regulators
# ---------------------------------------------------------------------------------------------------
class MuSIC_Molecule_Selector(MuSIC):
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

        self.load_and_process(upstream=True)

    def find_targets(
        self,
        save_id: Optional[str] = None,
        bw_membrane_bound: Union[float, int] = 8,
        bw_secreted: Union[float, int] = 25,
        kernel: Literal["bisquare", "exponential", "gaussian", "quadratic", "triangular", "uniform"] = "bisquare",
        **kwargs,
    ):
        """Find genes that may serve as interesting targets by computing the IoU with receptor signal. Will find
            genes that are highly coexpressed with receptors or ligand:receptor signals.

        Args:
            save_id: Optional string to append to the end of the saved file name. Will save signaling molecule names as
                "ligand_{save_id}.txt", etc.
            bw_membrane_bound: Bandwidth used to compute spatial weights for membrane-bound ligands. If integer,
                will convert to appropriate distance bandwidth.
            bw_secreted: Bandwidth used to compute spatial weights for secreted ligands. If integer, will convert to
                appropriate distance bandwidth.
            kernel: Type of kernel function used to weight observations when computing spatial weights; one of
                "bisquare", "exponential", "gaussian", "quadratic", "triangular" or "uniform".
            kwargs: Keyword arguments for any of the Spateo argparse arguments. Should not include 'output_path' (
                which will be determined by the output path used for the main model). Should also not include any of
                'ligands' or 'receptors', which will be determined by this function.
        """
        self.logger.info(
            "Beginning comprehensive search for targets, ligands, and receptors. This may take a long time..."
        )

        if self.mod_type != "receptor" and self.mod_type != "lr":
            raise ValueError(
                "Unsupervised target finding can only be done using receptor and ligand/receptor-based models."
            )

        lig_id = f"ligands_{save_id}" if save_id else "ligands"
        rec_id = f"receptors_{save_id}" if save_id else "receptors"
        targets_id = f"targets_{save_id}" if save_id else "targets"

        if not os.path.exists(os.path.join(os.path.splitext(self.output_path)[0])):
            os.makedirs(os.path.join(os.path.splitext(self.output_path)[0]))
        if not os.path.exists(os.path.join(os.path.dirname(self.adata_path), "targets")):
            os.makedirs(os.path.join(os.path.dirname(self.adata_path), "targets"))

        if self.species == "human":
            try:
                self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CCI resources cannot be found at {self.cci_dir}. Please check the path and try again."
                )
            except IOError:
                raise IOError(
                    "Issue reading L:R database. Files can be downloaded from "
                    "https://github.com/aristoteleo/spateo-release/spateo/tools/database."
                )

            self.r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_receptor_TF_db.csv"), index_col=0)
            self.grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
        elif self.species == "mouse":
            try:
                self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CCI resources cannot be found at {self.cci_dir}. Please check the path and try again."
                )
            except IOError:
                raise IOError(
                    "Issue reading L:R database. Files can be downloaded from "
                    "https://github.com/aristoteleo/spateo-release/spateo/tools/database."
                )

            self.r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
            self.grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)

        adata_expr = self.adata.copy()
        # Remove all-zero genes:
        adata_expr = adata_expr[:, adata_expr.X.sum(axis=0) > 0].copy()

        if self.custom_receptors is None:
            receptors = list(set(self.lr_db["to"]))
            receptors = [
                r for r in receptors if all(part in adata_expr.var_names for part in r.split("_")) or "_" not in r
            ]
        else:
            receptors = self.custom_receptors

        if self.custom_ligands is None:
            # All cognate ligands:
            ligands = []
            cognate_ligands = list(set(self.lr_db[self.lr_db["to"].isin(receptors)]["from"]))
            # For each ligand, check that all parts can be found in the dataset:
            for l in cognate_ligands:
                parts = l.split("_")
                if all(part in self.adata.var_names for part in parts):
                    ligands.append(l)
        else:
            ligands = self.custom_ligands

        # Temporarily save initial receptors and ligands to path:
        lig_path = os.path.join(os.path.dirname(self.adata_path), f"{lig_id}.txt")
        with open(lig_path, "w") as f:
            f.write("\n".join(ligands))
        rec_path = os.path.join(os.path.dirname(self.adata_path), f"{rec_id}.txt")
        with open(rec_path, "w") as f:
            f.write("\n".join(receptors))

        # If design matrix exists, use it to identify potential targets- if not, create it using all
        # ligands/receptors present in the dataset, use it to identify potential targets, then delete it :
        try:
            X_df = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"), index_col=0
            )
            self.logger.info("Loaded existing design matrix.")
        except:
            # Construct design matrix- need output path, AnnData path, path to CCI databases, and all information to
            # compute spatial weights:
            # Bandwidths for computing spatial weights:
            if isinstance(bw_membrane_bound, int):
                bw_membrane_bound = find_bw_for_n_neighbors(
                    self.adata, target_n_neighbors=bw_membrane_bound, exclude_self=True
                )
            if isinstance(bw_secreted, int):
                bw_secreted = find_bw_for_n_neighbors(self.adata, target_n_neighbors=bw_secreted, exclude_self=False)

            kwargs["output_path"] = self.output_path
            kwargs["adata_path"] = self.adata_path
            kwargs["cci_dir"] = self.cci_dir
            kwargs["species"] = self.species
            kwargs["custom_rec_path"] = rec_path
            kwargs["custom_lig_path"] = lig_path
            # "targets" is a necessary input, but for this purpose it doesn't matter what this is.
            self.logger.info(
                f"{receptors[0]} will be used as input for 'target', but for this purpose this is not important."
            )
            kwargs["target"] = receptors[0]
            kwargs["mod_type"] = self.mod_type
            kwargs["distance_membrane_bound"] = bw_membrane_bound
            kwargs["distance_secreted"] = bw_secreted
            kwargs["bw_fixed"] = True
            kwargs["kernel"] = kernel

            self.logger.info("Constructing design matrix.")
            comm, parser, args_list = define_spateo_argparse(**kwargs)
            upstream_model = MuSIC(comm, parser, args_list)
            upstream_model.load_and_process(upstream=True)
            upstream_model.define_sig_inputs(recompute=True)
            # Load design matrix:
            X_df = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"), index_col=0
            )

        # Filter out genes that are not expressed in greater than threshold proportion of the cells predicted to
        # have an interaction:
        cells_with_interaction = X_df[(X_df != 0).any(axis=1)].index
        adata_subset = self.adata[cells_with_interaction, :].copy()
        threshold_n = int(self.target_expr_threshold * adata_subset.shape[0])
        self.logger.info(
            f"Finding genes expressed in at least {threshold_n} cells out of {adata_subset.n_obs}- to "
            f"raise/lower this threshold, raise/lower the 'target_expr_threshold' parameter."
        )
        expr_data = adata_subset.X
        if scipy.sparse.issparse(expr_data):
            genes_expressed = np.array(expr_data.getnnz(axis=0) >= threshold_n).reshape(-1)
        else:
            genes_expressed = np.count_nonzero(expr_data, axis=0) >= threshold_n
        self.adata = self.adata[:, genes_expressed]
        self.logger.info(
            f"From {self.n_features} genes, {self.adata.n_vars} are highly expressed in cells predicted "
            f"to be involved in an interaction."
        )

        # Do not include housekeeping genes in this- e.g. actin, tubulin, ribosome subunits, ubiquitination,
        # essential metabolic genes, mitochondria, elongation factors, histone proteins, etc.:
        if self.species == "human":
            exclude = [
                "ACT",
                "TUB",
                "RPL",
                "RPS",
                "UB",
                "GAPDH",
                "HK",
                "PFK",
                "PLK",
                "CS",
                "ACO",
                "IDH",
                "SDH",
                "OGD",
                "FH",
                "MDH",
                "ACA",
                "FAS",
                "CPT",
                "GLU",
                "GOT",
                "SHMT",
                "RRM",
                "DHF",
                "SNR",
                "HNRN",
                "LDHA",
                "HSP",
                "H2",
                "H3",
                "H4",
                "HMGB",
                "EEF",
                "EIF",
                "ATP",
                "COX",
                "RAN",
                "GNAI",
                "MALAT",
                "PPIA",
                "MT-",
                "YWH",
                "ELO",
                "PTM",
                "TMS",
                "MARCK",
                "NEDD",
                "FAU",
            ]
        elif self.species == "mouse":
            exclude = [
                "Act",
                "Tub",
                "Rpl",
                "Rps",
                "Ub",
                "Gapdh",
                "Hk",
                "Pfk",
                "Plk",
                "Cs",
                "Aco",
                "Idh",
                "Sdh",
                "Ogd",
                "Fh",
                "Mdh",
                "Aca",
                "Fas",
                "Cpt",
                "Glu",
                "Got",
                "Shmt",
                "Rrm",
                "Dhf",
                "Snr",
                "Hnrn",
                "Ldha",
                "Hsp",
                "H2",
                "H3",
                "H4",
                "Hmgb",
                "Eef",
                "Eif",
                "Atp",
                "Cox",
                "Ran",
                "Gnai",
                "Malat",
                "Ppia",
                "mt-",
                "Ywh",
                "Elo",
                "Ptm",
                "Tms",
                "Marck",
                "Nedd",
                "Fau",
            ]
        self.logger.info("Excluding housekeeping genes/essential genes from target search.")
        mask = ~self.adata.var_names.str.contains("|".join(exclude))
        self.adata = self.adata[:, mask]

        # Do not include receptors in this:
        mask = ~self.adata.var_names.isin(receptors)
        self.logger.info("Excluding receptors from target search.")
        self.adata = self.adata[:, mask]
        self.logger.info(f"Size of final set of genes: {self.adata.n_vars}")

        targets_path = os.path.join(os.path.dirname(self.adata_path), f"{targets_id}.txt")
        with open(targets_path, "w") as f:
            f.write("\n".join(self.adata.var_names))

        # Delete other files created during the process:
        # os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"))
        # os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "targets.csv"))
        # os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr.csv"))
        # os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr_nonlag.csv"))
        # os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "receptors_expr.csv"))
