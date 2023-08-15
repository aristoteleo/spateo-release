"""
Functionalities to aid in feature selection to characterize signaling patterns from spatial transcriptomics. Given a
list of signaling molecules (ligands or receptors) and/or target genes
"""
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy.stats import percentileofscore

from ..find_neighbors import find_bw_for_n_neighbors
from .MuSIC import MuSIC
from .regression_utils import multitesting_correction
from .SWR_mpi import define_spateo_argparse


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

    def find_targets_single(
        self, X: pd.DataFrame, col: str, method: Optional[str] = None, significance_threshold: float = 0.05
    ) -> Union[None, List[str]]:
        """For a given feature column, ranks target genes based on degree of cooccurence with the feature. Will
        assume the name of a receptor occurs somewhere in the column name.

        Args:
            X: DataFrame containing columns to query
            col: Name of the column to compare to target genes
            method: Optional method to use for multiple hypothesis correction. It is recommended not to false
                positive correct because this function is not looking for most important features, but rather all that
                are reasonably interesting. Available methods can be found in the documentation for
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
            significance_threshold: p-value (or q-value) needed to call a parameter significant. Only used if
                'method' is not None.
        """
        # start = time.time()
        binary_col = (X[col] > 0).astype(int).values.reshape(-1)
        # Null distribution of IoU values:
        iou_null = self.adata.uns["iou_null"]

        # Storage for targets of interest for this feature, and for all p-values:
        p_vals = []

        # Find the set of targets to search over for this feature:
        if ":" in col:
            rec = col.split(":")[1]
        else:
            rec = col

        # Make note of the components of this receptor:
        if "_" in rec:
            rec_components = rec.split("_")
        else:
            rec_components = [rec]

        tf_for_rec = self.r_tf_db[self.r_tf_db["receptor"] == rec]["tf"].unique()
        tf_for_rec = [tf for tf in tf_for_rec if tf in self.grn.columns]

        # Extract the target genes where at least one of the TFs has a regulatory relationship:
        target_genes_for_common_tfs = self.grn.loc[:, tf_for_rec].apply(lambda x: x == 1)
        all_targets = self.grn.index[target_genes_for_common_tfs.any(axis=1)]
        # Only consider targets measured in the dataset:
        all_targets = [target for target in all_targets if target in self.adata.var_names]
        # Only consider targets above expression threshold:
        all_targets = np.array(
            [
                target
                for target in all_targets
                if (self.adata[:, target].X > 0).sum() > self.target_expr_threshold * self.adata.n_obs
                and target not in rec_components
            ]
        )

        for target in all_targets:
            binary_gene_expr = (self.adata[:, target].X > 0).astype(int).toarray().reshape(-1)

            # Intersection-over-union w/ the query feature:
            intersection = np.sum(binary_col * binary_gene_expr)
            union = np.sum(binary_col + binary_gene_expr > 0)
            iou = intersection / union if union > 0 else 0
            p_val = 1 - percentileofscore(iou_null, iou) / 100
            p_vals.append(p_val)

        # Mark any targets for which intersection-over-union is significant compared to the null distribution
        if method is not None:
            q_vals = multitesting_correction(p_vals, method=method, alpha=significance_threshold)
            sig = np.where(q_vals < 0.05)[0]
        else:
            sig = np.where(np.array(p_vals) < significance_threshold)[0]
        sig_targets = all_targets[sig]

        # self.logger.info(f"Number of targets for {col}: {len(sig_targets)}")
        # elapsed_time = time.time() - start
        # self.logger.info(f"Time elapsed for {col}: {elapsed_time:.2f}s")

        # If there are no targets for which the intersection with receptor is larger than the intersection with
        # non-receptor expressing cells, don't include any targets with this receptor.
        if len(sig_targets) == 0:
            return None
        else:
            return list(sig_targets)

    @staticmethod
    def compute_iou(gene_combination, expressed_nonexpressed):
        """Helper function for :func ~`find_targets_ligands_and_receptors` that computes the null distribution of
        intersection-over-union"""
        gene1_idx, gene2_idx = gene_combination
        gene1 = expressed_nonexpressed[:, gene1_idx].toarray()
        gene2 = expressed_nonexpressed[:, gene2_idx].toarray()
        intersection = np.sum(gene1 * gene2)
        union = np.sum(gene1 + gene2 > 0)
        iou = intersection / union if union > 0 else 0

        if iou > 0.5:
            # Scramble expression of both genes:
            np.random.shuffle(gene1)
            np.random.shuffle(gene2)

        # Calculate intersection, union, and IoU
        intersection = np.sum(gene1 * gene2)
        union = np.sum(gene1 + gene2 > 0)
        iou = intersection / union if union > 0 else 0
        return iou

    def find_targets_ligands_and_receptors(
        self,
        save_id: Optional[str] = None,
        bw_membrane_bound: Union[float, int] = 8,
        bw_secreted: Union[float, int] = 25,
        kernel: Literal["bisquare", "exponential", "gaussian", "quadratic", "triangular", "uniform"] = "bisquare",
        method: Optional[str] = None,
        common_signal_threshold: Optional[float] = None,
        **kwargs,
    ):
        """Find genes that may serve as interesting targets. Will find genes that are highly coexpressed with
        receptors, using this to also filter receptors and ligands.

        Args:
            save_id: Optional string to append to the end of the saved file name. Will save signaling molecule names as
                "ligand_{save_id}.txt", etc.
            bw_membrane_bound: Bandwidth used to compute spatial weights for membrane-bound ligands. If integer,
                will convert to appropriate distance bandwidth.
            bw_secreted: Bandwidth used to compute spatial weights for secreted ligands. If integer, will convert to
                appropriate distance bandwidth.
            kernel: Type of kernel function used to weight observations when computing spatial weights; one of
                "bisquare", "exponential", "gaussian", "quadratic", "triangular" or "uniform".
            method: Used for optional multiple hypothesis correction
            common_signal_threshold: Whether to use features that have high overlap with particular gene. By
                default, this function searches for high degree of coexpression between prospective target gene and
                signaling feature, which has a tendency to filter out common features. If this is given,
                will instead use these common features that are coexpressed w/ the target in greater than this
                proportion of cells.
            kwargs: Keyword arguments for any of the Spateo argparse arguments. Should not include 'output_path' (
                which will be determined by the output path used for the main model). Should also not include any of
                'ligands' or 'receptors', which will be determined by this function.
        """
        self.logger.info(
            "Beginning comprehensive search for targets, ligands, and receptors. This may take a long time..."
        )

        if self.mod_type != "receptor" and self.mod_type != "lr":
            raise ValueError(
                "Unsupervised target finding can only be done using receptor and ligand/receptor-based " "models."
            )

        lig_id = f"ligands_{save_id}" if save_id else "ligands"
        rec_id = f"receptors_{save_id}" if save_id else "receptors"
        targets_id = f"targets_{save_id}" if save_id else "targets"

        if not os.path.exists(os.path.join(os.path.splitext(self.output_path)[0])):
            os.makedirs(os.path.join(os.path.splitext(self.output_path)[0]))
        if not os.path.exists(os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets")):
            os.makedirs(os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets"))

        if self.species == "human":
            try:
                self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_human.csv"), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CCI resources cannot be found at {self.cci_dir}. Please check the path " f"and try again."
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
            kwargs["custom_rec_path"] = rec_path
            kwargs["custom_lig_path"] = lig_path
            # "targets" is a necessary input, but for this purpose it doesn't matter what this is.
            kwargs["target"] = receptors[0]
            kwargs["mod_type"] = self.mod_type
            kwargs["distance_membrane_bound"] = bw_membrane_bound
            kwargs["distance_secreted"] = bw_secreted
            kwargs["bw_fixed"] = True
            kwargs["kernel"] = kernel

            comm, parser, args_list = define_spateo_argparse(**kwargs)
            upstream_model = MuSIC(comm, parser, args_list)
            upstream_model.load_and_process(upstream=True)
            upstream_model.define_sig_inputs(recompute=True)
            # Load design matrix:
            X_df = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"), index_col=0
            )

        if common_signal_threshold is not None:
            # Non-zero rows for each column of the dataframe:
            non_zero_rows = {col: X_df.index[X_df[col] != 0].tolist() for col in X_df.columns}

        if self.mod_type == "lr":
            # All unique receptors:
            receptor_set = [c.split(":")[1] for c in X_df.columns]
        else:
            receptor_set = X_df.columns

        # Null distribution for cooccurence:
        subset = (self.adata.X > 0).mean(axis=0) < 0.5
        adata_filt = self.adata[:, subset].copy()
        expressed_nonexpressed = (adata_filt.X > 0).astype(int)

        # Intersection-over-union for all pairwise combinations of genes:
        gene_combinations = combinations(range(expressed_nonexpressed.shape[1]), 2)

        # Parallelized computation to define IoU null distribution:
        with ThreadPoolExecutor() as executor:
            iou_results = list(
                executor.map(
                    lambda gene_combination: self.compute_iou(gene_combination, expressed_nonexpressed),
                    gene_combinations,
                )
            )

        # Store results in .uns
        self.logger.info(f"Average intersection-over-union of the null distribution: {np.mean(iou_results):.3f}")
        self.adata.uns["iou_null"] = np.array(iou_results)

        # For each design matrix column, find targets with significant cooccurence:
        with ThreadPoolExecutor() as executor:
            targets = list(executor.map(lambda col: self.find_targets_single(X_df, col, method, 0.05), X_df.columns))
        rem_targets = [t for t in targets if t is not None]

        # Find targets that were marked as of interest for multiple interactions:
        flat_targets = [t for t_list in rem_targets for t in t_list]
        target_counts = Counter(flat_targets)
        main_targets = [k for k, v in target_counts.items() if v > 10]
        # Associations between main targets and interaction features:
        target_column_mapping = {t: [] for t in main_targets}
        self.logger.info(f"Found {len(target_column_mapping)} notable targets of interest.")
        for t in main_targets:
            for col_idx, target_list in enumerate(targets):
                if target_list and t in target_list:
                    # If the target is present in the list for a column, add the column name to the mapping
                    column_name = X_df.columns[col_idx]
                    target_column_mapping[t].append(column_name)

        self.logger.info(
            f"Saving mapping of most notable targets to signaling features to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'predictors_and_targets',f'major_{targets_id}_map.json')} "
            f"for potential use downstream."
        )
        with open(
            os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets", f"major_{targets_id}_map.json"),
            "w",
        ) as f:
            json.dump(target_column_mapping, f)

        # If any element in the return is None, remove the receptor from the list of receptors:
        receptors = set([r for r, condition in zip(receptor_set, targets) if condition is not None])

        # If necessary, redefine the list of ligands based on the list of remaining receptors:
        if self.mod_type == "lr":
            # New set of cognate ligands:
            new_ligands = []
            cognate_ligands = list(set(self.lr_db[self.lr_db["to"].isin(receptors)]["from"]))
            # For each ligand, check that all parts can be found in the dataset:
            for l in cognate_ligands:
                parts = l.split("_")
                if all(part in self.adata.var_names for part in parts):
                    new_ligands.append(l)
            ligands = [l for l in ligands if l in new_ligands]

        # Final list of used receptors:
        self.logger.info(f"Final set of {len(receptors)} receptors: \n{receptors}")
        # Final list of used ligands:
        self.logger.info(f"Final set of {len(ligands)} ligands: \n{ligands}")

        self.logger.info(
            f"Saving list of manually found ligands to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'predictors_and_targets', f'{lig_id}.txt')} for "
            "potential use downstream."
        )
        with open(os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets", f"{lig_id}.txt"), "w") as f:
            f.write("\n".join(ligands))

        self.logger.info(
            f"Saving list of manually found receptors to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'predictors_and_targets', f'{rec_id}.txt')} for "
            f"potential use downstream."
        )
        with open(os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets", f"{rec_id}.txt"), "w") as f:
            f.write("\n".join(receptors))

        # Final list of (potentially good) targets and mapping to interaction features:
        all_targets = list(set([t for t_list in rem_targets for t in t_list]))
        # Associations between main targets and interaction features:
        target_column_mapping = {target: [] for target in all_targets}
        for t in all_targets:
            # Add common signaling features that overlap with the target, if applicable:
            if common_signal_threshold is not None:
                # Expressing cells for target:
                nz_indices = np.nonzero(self.adata[:, t].X)[0]
                nz_cells = self.adata.obs_names[nz_indices]
                # Intersection of design matrix columns with target-expressing cells:
                total_nz = len(nz_cells)
                for col, nz in non_zero_rows.items():
                    intersection_count = len(set(nz_cells).intersection(set(nz)))
                    percentage = intersection_count / total_nz
                    if percentage > common_signal_threshold:
                        target_column_mapping[t].append(col)
            else:
                for col_idx, target_list in enumerate(targets):
                    if target_list and t in target_list:
                        # If the target is present in the list for a column, add the column name to the mapping
                        column_name = X_df.columns[col_idx]
                        target_column_mapping[t].append(column_name)
            self.logger.info(f"Number of signaling features associated with {t}: {len(target_column_mapping[t])}")

        self.logger.info(
            f"Saving mapping of most notable targets to signaling features to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'predictors_and_targets', f'{targets_id}_map.json')} "
            f"for potential use downstream."
        )
        with open(
            os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets", f"{targets_id}_map.json"), "w"
        ) as f:
            json.dump(target_column_mapping, f)

        self.logger.info(
            f"Saving list of manually found targets to "
            f"{os.path.join(os.path.dirname(self.adata_path), 'predictors_and_targets', f'{targets_id}.txt')} for "
            f"potential use downstream."
        )
        with open(
            os.path.join(os.path.dirname(self.adata_path), "predictors_and_targets", f"{targets_id}.txt"), "w"
        ) as f:
            f.write("\n".join(all_targets))

        # Delete other files created during the process:
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "targets.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr_nonlag.csv"))
        os.remove(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "receptors_expr.csv"))
