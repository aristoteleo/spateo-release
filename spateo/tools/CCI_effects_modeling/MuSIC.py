"""
Modeling cell-cell communication using a regression model that is considerate of the spatial heterogeneity of (and thus
the context-dependency of the relationships of) the response variable.
"""
import argparse
import itertools
import json
import os
import re
from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy
from patsy import dmatrix
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from tqdm import tqdm

from ...logging import logger_manager as lm
from ...preprocessing.normalize import factor_normalization
from ...preprocessing.transform import log1p
from ...tools.spatial_smooth import smooth
from ..find_neighbors import find_bw_for_n_neighbors, get_wi, neighbors
from ..spatial_degs import moran_i
from .distributions import Gaussian, NegativeBinomial, Poisson
from .regression_utils import compute_betas_local, iwls, multicollinearity_check


# ---------------------------------------------------------------------------------------------------
# Multiscale Spatially-weighted Inference of Cell-cell communication (MuSIC)
# ---------------------------------------------------------------------------------------------------
class MuSIC:
    """Spatially weighted regression on spatial omics data with parallel processing. Runs after being called
    from the command line.

    Args:
        comm: MPI communicator object initialized with mpi4py, to control parallel processing operations
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.
        args_list: If parser is provided by function call, the arguments to parse must be provided as a separate
            list. It is recommended to use the return from :func `define_spateo_argparse()` for this.
        verbose: Set True to print updates to screen. Will be set False when initializing downstream analysis object,
            which inherits from this class but for which the information is generally not as useful.
        save_subsampling: Set True to save the subsampled data to a .json file. Defaults to True, recommended to set
            True for ease of access to the subsampling results.

    Attributes:
        mod_type: The type of model that will be employed- this dictates how the data will be processed and
            prepared. Options:
                - "niche": Spatially-aware, uses categorical cell type labels as independent variables.
                - "lr": Spatially-aware, essentially uses the combination of receptor expression in the "target" cell
                    and spatially lagged ligand expression in the neighboring cells as independent variables.
                - "ligand": Spatially-aware, essentially uses ligand expression in the neighboring cells as
                    independent variables.
                - "receptor": Uses receptor expression in the "target" cell as independent variables.


        adata_path: Path to the AnnData object from which to extract data for modeling
        csv_path: Can also be used to specify path to non-AnnData .csv object. Assumes the first three columns
            contain x- and y-coordinates and then dependent variable values, in that order, with all subsequent
            columns containing independent variable values.
        normalize: Set True to perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size).
        smooth: Set True to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression. It is advisable not to do this if performing Poisson or negative binomial regression.
        log_transform: Set True if log-transformation should be applied to expression. It is advisable not to do
            this if performing Poisson or negative binomial regression.
        normalize_signaling: Set True to minmax scale the final ligand expression array (for :attr `mod_type` =
            "ligand"), or the final ligand-receptor array (for :attr `mod_type` = "lr"). This is recommended to
            associate downstream expression with rarer/less prevalent signaling mechanisms.
        target_expr_threshold: Only used if :param `mod_type` is "lr" or "ligand" and :param `targets_path` is not
            given. When manually selecting targets, expression above a threshold percentage of cells will be used to
            filter to a smaller subset of interesting genes. Defaults to 0.2.
        multicollinear_threshold: Variance inflation factor threshold used to filter out multicollinear features. A
            value of 5 or 10 is recommended.


        custom_lig_path: Optional path to a .txt file containing a list of ligands for the model, separated by
            newlines. Only used if :attr `mod_type` is "lr" or "ligand" (and thus uses ligand expression directly in
            the inference). If not provided, will select ligands using a threshold based on expression
            levels in the data.
        custom_ligands: Optional list of ligands for the model, can be used as an alternative to :attr
            `custom_lig_path`. Only used if :attr `mod_type` is "lr" or "ligand".
        custom_rec_path: Optional path to a .txt file containing a list of receptors for the model, separated by
            newlines. Only used if :attr `mod_type` is "lr" (and thus uses receptor expression directly in the
            inference). If not provided, will select receptors using a threshold based on expression
            levels in the data.
        custom_receptors: Optional list of receptors for the model, can be used as an alternative to :attr
            `custom_rec_path`. Only used if :attr `mod_type` is "lr".
        custom_pathways_path: Rather than providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model. Only used if :attr `mod_type`
            is "lr".
        custom_pathways: Optional list of signaling pathways for the model, can be used as an alternative to :attr
            `custom_pathways_path`. Only used if :attr `mod_type` is "lr".
        targets_path: Optional path to a .txt file containing a list of prediction target genes for the model,
            separated by newlines. If not provided, targets will be strategically selected from the given receptors.
        custom_targets: Optional list of prediction target genes for the model, can be used as an alternative to
            :attr `targets_path`.
        init_betas_path: Optional path to a .json file or .csv file containing initial coefficient values for the model
            for each target variable. If encoded in .json, keys should be target gene names, values should be numpy
            arrays containing coefficients. If encoded in .csv, columns should be target gene names. Initial
            coefficients should have shape [n_features, ].


        cci_dir: Full path to the directory containing cell-cell communication databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved


        coords_key: Key in .obsm of the AnnData object that contains the coordinates of the cells
        group_key: Key in .obs of the AnnData object that contains the category grouping for each cell
        group_subset: Subset of cell types to include in the model (provided as a whitespace-separated list in
            command line). If given, will consider only cells of these types in modeling. Defaults to all cell types.
        covariate_keys: Can be used to optionally provide any number of keys in .obs or .var containing a continuous
            covariate (e.g. expression of a particular TF, avg. distance from a perturbed cell, etc.)
        total_counts_key: Entry in :attr:`adata` .obs that contains total counts for each cell. Required if subsetting
            by total counts.
        total_counts_threshold: Threshold for total counts to subset cells by- cells with total counts greater than
            this threshold will be retained.

        bw: Used to provide previously obtained bandwidth for the spatial kernel. Consists of either a distance
            value or N for the number of nearest neighbors. Pass "np.inf" if all other points should have the same
            spatial weight.
        minbw: For use in automated bandwidth selection- the lower-bound bandwidth to test.
        maxbw: For use in automated bandwidth selection- the upper-bound bandwidth to test.


        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".
        n_neighbors_membrane_bound: For :attr:`mod_type` "ligand" or "lr"- ligand expression will be taken from the
            neighboring cells- this defines the number of cells to use for membrane-bound ligands.
        n_neighbors_secreted: For :attr:`mod_type` "ligand" or "lr"- ligand expression will be taken from the
            neighboring cells- this defines the number of cells to use for secreted or ECM ligands.
        use_expression_neighbors: The default for finding spatial neighborhoods for the modeling process is to
            use neighbors in physical space. If this argument is provided, expression will instead be used to find
            neighbors.


        bw_fixed: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        exclude_self: If True, ignore each sample itself when computing the kernel density estimation
        fit_intercept: Set True to include intercept in the model and False to exclude intercept
    """

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        args_list: Optional[List[str]] = None,
        verbose: bool = True,
        save_subsampling: bool = True,
    ):
        self.logger = lm.get_main_logger()

        self.parser = parser
        self.args_list = args_list
        self.verbose = verbose
        self.save_subsampling = save_subsampling

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
        # Number of samples, equal to the number of SWR runs to go through:
        self.n_samples = None
        self.n_features = None
        # Flag for whether model has been set up and AnnData has been processed:
        self.set_up = False

        self.parse_stgwr_args()

        # And initialize other attributes to None:
        self.X_df = None
        self.adata = None
        self.cell_categories = None
        self.clip = None
        self.cof_db = None
        self.ct_vec = None
        self.feature_distance = None
        self.feature_names = None
        self.grn = None
        self.ligands_expr = None
        self.ligands_expr_nonlag = None
        self.lr_db = None
        self.lr_pairs = None
        self.n_samples_subsampled = None
        self.n_samples_subset = None
        self.neighboring_unsampled = None
        self.optimal_bw = None
        self.r_tf_db = None
        self.receptors_expr = None
        self.sample_names = None
        self.subsampled = None
        self.subsampled_sample_names = None
        self.subset = None
        self.subset_indices = None
        self.subset_sample_names = None
        self.targets_expr = None
        self.tf_tf_db = None
        self.x_chunk = None

    def _set_up_model(self, downstream: bool = False):
        if self.mod_type is None and self.adata_path is not None:
            raise ValueError(
                "No model type provided; need to provide a model type to fit. Options: 'niche', 'lr', "
                "'receptor', 'ligand'."
            )

        # If AnnData object is given, process it:
        if self.adata_path is not None:
            # Ensure CCI directory is provided:
            if self.cci_dir is None and self.mod_type in ["lr", "receptor", "ligand"]:
                raise ValueError(
                    "No CCI directory provided; need to provide a CCI directory to fit a model with "
                    "ligand/receptor expression."
                )
            self.load_and_process(downstream=downstream)
        else:
            if self.csv_path is None:
                raise ValueError(
                    "No AnnData path or .csv path provided; need to provide at least one of these "
                    "to provide a default dataset to fit."
                )
            else:
                try:
                    custom_data = pd.read_csv(self.csv_path, index_col=0)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Could not find file at path {self.csv_path}.")
                except IOError:
                    raise IOError(f"Error reading file at path {self.csv_path}.")

                self.coords = custom_data.iloc[:, : self.n_spatial_dim_csv].values
                # Check if the names of any columns have been given as targets:
                if self.custom_targets is None and self.targets_path is None:
                    # It is assumed the (n + 1)th column of the .csv file is the target variable, where n is the
                    # number of spatial dimensions (e.g. 2 for 2D samples, 3 for 3D, etc.):
                    self.target = pd.DataFrame(
                        custom_data.iloc[:, self.n_spatial_dim_csv],
                        index=custom_data.index,
                        columns=[custom_data.columns[2]],
                    )
                elif self.custom_targets is not None:
                    self.targets_expr = custom_data[self.custom_targets]
                else:
                    with open(self.targets_path, "r") as f:
                        targets = f.read().splitlines()
                    self.targets_expr = custom_data[targets]

                self.logger.info(f"Extracting target from column labeled '{custom_data.columns[2]}'.")
                independent_variables = custom_data.iloc[:, 3:]
                self.X_df = independent_variables
                self.X = independent_variables.values
                self.feature_names = list(independent_variables.columns)

                # Add intercept if applicable:
                if self.fit_intercept:
                    self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
                    self.feature_names = ["intercept"] + self.feature_names

                self.n_samples = self.X.shape[0]
                self.n_features = self.X.shape[1]
                self.sample_names = custom_data.index

        # Check if this AnnData object contains cells that have already been fit to, but also contains additional
        # cells that have not been fit to:
        parent_dir = os.path.dirname(self.output_path)
        file_list = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]

        if len(file_list) > 0:
            for file in file_list:
                if not "predictions" in file:
                    check = pd.read_csv(os.path.join(parent_dir, file), index_col=0)
                    if any([name not in check.index for name in self.sample_names]):
                        self.map_new_cells()
                    break

        # Perform subsampling if applicable:
        if self.group_subset:
            subset = self.adata.obs[self.group_key].isin(self.group_subset)
            self.subset_indices = [self.sample_names.get_loc(name) for name in subset.index]
            self.subset_sample_names = subset.index
            self.n_samples_subset = len(subset)
            self.subset = True
        else:
            self.subset = False

        if self.spatial_subsample or self.total_counts_threshold != 0.0 or self.group_subset is not None:
            self.run_subsample()
            # Indicate model has been subsampled:
            self.subsampled = True
        elif self.group_subset:
            self.x_chunk = np.array(self.subset_indices)
            self.subsampled = False
        else:
            self.x_chunk = np.arange(self.n_samples)
            self.subsampled = False

        # Indicate model has now been set up:
        self.set_up = True

    def parse_stgwr_args(self):
        """
        Parse command line arguments for arguments pertinent to modeling.
        """
        if self.args_list is not None:
            self.arg_retrieve = self.parser.parse_args(self.args_list)
        else:
            self.arg_retrieve = self.parser.parse_args()

        self.mod_type = self.arg_retrieve.mod_type
        self.include_unpaired_lr = self.arg_retrieve.include_unpaired_lr
        # Set flag to evenly subsample spatial data:
        self.spatial_subsample = self.arg_retrieve.spatial_subsample

        self.adata_path = self.arg_retrieve.adata_path
        self.csv_path = self.arg_retrieve.csv_path
        self.n_spatial_dim_csv = self.arg_retrieve.n_spatial_dim_csv
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
        self.init_betas_path = self.arg_retrieve.init_betas_path
        # Check if path to init betas is given:
        if self.init_betas_path is not None:
            self.logger.info(f"Loading initial betas from: {self.init_betas_path}")
            try:
                with open(self.init_betas_path, "r") as f:
                    self.init_betas = json.load(f)
            except:
                self.init_betas = pd.read_csv(self.init_betas_path, index_col=0)
        else:
            self.init_betas = None

        self.normalize = self.arg_retrieve.normalize
        self.smooth = self.arg_retrieve.smooth
        self.log_transform = self.arg_retrieve.log_transform
        self.normalize_signaling = self.arg_retrieve.normalize_signaling
        self.target_expr_threshold = self.arg_retrieve.target_expr_threshold
        self.multicollinear_threshold = self.arg_retrieve.multicollinear_threshold

        self.coords_key = self.arg_retrieve.coords_key
        self.group_key = self.arg_retrieve.group_key
        self.group_subset = self.arg_retrieve.group_subset
        self.covariate_keys = self.arg_retrieve.covariate_keys
        self.total_counts_key = self.arg_retrieve.total_counts_key
        self.total_counts_threshold = self.arg_retrieve.total_counts_threshold

        self.bw_fixed = self.arg_retrieve.bw_fixed
        self.distance_membrane_bound = self.arg_retrieve.distance_membrane_bound
        self.distance_secreted = self.arg_retrieve.distance_secreted
        self.n_neighbors_membrane_bound = self.arg_retrieve.n_neighbors_membrane_bound
        self.n_neighbors_secreted = self.arg_retrieve.n_neighbors_secreted
        self.use_expression_neighbors = self.arg_retrieve.use_expression_neighbors
        # Also use the number of neighbors for secreted signaling for niche modeling
        self.n_neighbors_niche = self.n_neighbors_secreted
        self.exclude_self = self.arg_retrieve.exclude_self
        self.distr = self.arg_retrieve.distr

        # Get appropriate distribution family based on specified distribution:
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
            raise ValueError(
                "`bw_fixed` is set to False for adaptive kernel- it is assumed the chosen bandwidth is "
                "the number of neighbors for each sample. However, only the `bisquare` and `uniform` "
                "kernels perform hard thresholding and so it is recommended to use one of these kernels- "
                "the other kernels may result in different results."
            )

        self.fit_intercept = self.arg_retrieve.fit_intercept
        # self.include_offset = self.arg_retrieve.include_offset

        # Parameters related to the fitting process (tolerance, number of iterations, etc.)
        self.tolerance = self.arg_retrieve.tolerance
        self.max_iter = self.arg_retrieve.max_iter
        self.patience = self.arg_retrieve.patience
        self.ridge_lambda = self.arg_retrieve.ridge_lambda

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
        print("-" * 60, flush=True)
        fixed_or_adaptive = "Fixed " if self.bw_fixed else "Adaptive "
        type = fixed_or_adaptive + self.kernel.capitalize()
        self.logger.info(f"Spatial kernel: {type}")

        if self.adata_path is not None:
            self.logger.info(f"Loading AnnData object from: {self.adata_path}")
        elif self.csv_path is not None:
            self.logger.info(f"Loading CSV file from: {self.csv_path}")

        if self.mod_type is not None:
            self.logger.info(f"Model type: {self.mod_type}")
            if self.mod_type in ["lr", "ligand", "receptor"]:
                self.logger.info(
                    f"Loading cell-cell interaction databases from the following folder: " f" {self.cci_dir}."
                )
                if self.custom_ligands_path is not None:
                    self.logger.info(f"Using list of custom ligands from: {self.custom_ligands_path}.")
                if self.custom_ligands is not None:
                    self.logger.info(f"Using the provided list of ligands: {self.custom_ligands}.")
                if self.custom_receptors_path is not None:
                    self.logger.info(f"Using list of custom receptors from: {self.custom_receptors_path}.")
                if self.custom_receptors is not None:
                    self.logger.info(f"Using the provided list of receptors: {self.custom_receptors}.")
            if self.targets_path is not None:
                self.logger.info(f"Using list of target genes from: {self.targets_path}.")
            if self.custom_targets is not None:
                self.logger.info(f"Using provided list of target genes: {self.custom_targets}.")
            self.logger.info(f"Saving all outputs to this directory: {os.path.dirname(self.output_path)}.")

    def load_and_process(self, upstream: bool = False, downstream: bool = False):
        """
        Load AnnData object and process it for modeling.

        Args:
            upstream: Set False if performing the actual model fitting process, True to define only the AnnData
                object for upstream purposes.
            downstream: Set True if setting up a downstream model- in this case, ligand/receptor preprocessing will
                be skipped.
        """
        try:
            self.adata = anndata.read_h5ad(self.adata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {self.adata_path}")
        except IOError:
            raise IOError(
                f"Could not read file: {self.adata_path}. Try opening with older version of AnnData and "
                f"removing the `.raw` attribute or any unnecessary entries in e.g. .uns or .obsp."
            )

        self.adata.uns["__type"] = "UMI"

        self.sample_names = self.adata.obs_names
        self.coords = self.adata.obsm[self.coords_key]
        self.n_samples = self.adata.n_obs
        # Placeholder- this will change at time of fitting:
        self.n_features = self.adata.n_vars

        # If not performing upstream tasks, only load the AnnData object:
        if not upstream:
            # For downstream, modify some of the parameters:
            if self.mod_type == "downstream":
                # Use neighbors in expression space:
                self.logger.info("Because `mod_type` is `downstream`, using expression neighbors.")
                self.use_expression_neighbors = True
                # Pathways can be used as targets, which often violates the count assumption of the other distributions-
                # redefine the model type in this case:
                if self.adata.uns["target_type"] == "pathway":
                    self.distr = "gaussian"
                    link = Gaussian.__init__.__defaults__[0]
                    self.distr_obj = Gaussian(link)

            # If group_subset is given, subset the AnnData object to contain the specified groups as well as neighboring
            # cells:
            if self.group_subset is not None:
                subset = self.adata[self.adata.obs[self.group_key].isin(self.group_subset)]
                fitted_indices = [self.sample_names.get_loc(name) for name in subset.obs_names]
                # Add cells that are neighboring cells of the chosen type, but which are not of the chosen type:
                get_wi_partial = partial(
                    get_wi,
                    n_samples=self.n_samples,
                    coords=self.coords,
                    fixed_bw=False,
                    exclude_self=True,
                    kernel="bisquare",
                    bw=self.n_neighbors_secreted,
                    threshold=0.01,
                    sparse_array=True,
                    normalize_weights=True,
                )

                with Pool() as pool:
                    weights = pool.map(get_wi_partial, fitted_indices)
                w_subset = scipy.sparse.vstack(weights)
                rows, cols = w_subset.nonzero()
                unique_indices = list(set(cols))
                names_all_neighbors = self.sample_names[unique_indices]
                self.adata = self.adata[self.adata.obs_names.isin(names_all_neighbors)].copy()
                self.group_subsampled_sample_names = self.adata.obs_names

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
                    # self.logger.info(
                    #     "Computing TMM factors and setting total counts in each cell to uniform target sum "
                    #     "inplace..."
                    # )
                    # target_sum to None to automatically determine suitable target sum:
                    # self.adata = factor_normalization(self.adata, method="TMM", target_sum=None)
                    self.logger.info("Setting total counts in each cell to uniform target sum inplace...")
                    factor_normalization(self.adata, target_sum=1e4)
                else:
                    # self.logger.info(
                    #     "Computing TMM factors, setting total counts in each cell to uniform target sum and rounding "
                    #     "nonintegers inplace..."
                    # )
                    # target_sum to None to automatically determine suitable target sum:
                    # self.adata = factor_normalization(self.adata, method="TMM", target_sum=None)
                    self.logger.info(
                        "Setting total counts in each cell to uniform target sum and rounding nonintegers inplace..."
                    )
                    factor_normalization(self.adata, target_sum=1e4)

                    # Round, except for the case where data would round down to zero-
                    if scipy.sparse.issparse(self.adata.X):
                        mask_greater_than_1 = self.adata.X >= 1
                        mask_less_than_1 = self.adata.X.multiply(mask_greater_than_1) == 0

                        mask_less_than_1_values = self.adata.X.copy()
                        mask_greater_than_1_values = self.adata.X.copy()

                        mask_less_than_1_values.data = np.ceil(mask_less_than_1_values.data)
                        mask_greater_than_1_values.data = np.round(mask_greater_than_1_values.data)
                        result = mask_less_than_1.multiply(mask_less_than_1_values) + mask_greater_than_1.multiply(
                            mask_greater_than_1_values
                        )

                        self.adata.X = scipy.sparse.csr_matrix(result)

                    else:
                        self.adata.X = np.where(self.adata.X < 1, np.ceil(self.adata.X), np.round(self.adata.X))
                    self.logger.info("Finished normalization.")

            # Smooth data if 'smooth' is True and log-transform data matrix if 'log_transform' is True:
            if self.smooth:
                # Compute connectivity matrix if not already existing:
                try:
                    conn = self.adata.obsp["spatial_connectivities"]
                except:
                    _, adata = neighbors(
                        self.adata,
                        n_neighbors=self.n_neighbors_membrane_bound * 2,
                        basis="spatial",
                        spatial_key=self.coords_key,
                        n_neighbors_method="ball_tree",
                    )
                    conn = adata.obsp["spatial_connectivities"]

                # Subsample half of the neighbors in the smoothing process:
                n_subsample = int(self.n_neighbors_membrane_bound)
                if self.distr == "gaussian":
                    self.logger.info("Smoothing gene expression inplace...")
                    adata_smooth_norm, _ = smooth(self.adata.X, conn, normalize_W=False, n_subsample=n_subsample)
                    self.adata.X = adata_smooth_norm

                else:
                    self.logger.info("Smoothing gene expression and rounding nonintegers inplace...")
                    adata_smooth_norm, _ = smooth(
                        self.adata.X, conn, normalize_W=False, n_subsample=n_subsample, return_discrete=True
                    )
                    self.adata.X = adata_smooth_norm

            if self.log_transform:
                if self.distr == "gaussian":
                    self.logger.info("Log-transforming expression inplace...")
                    log1p(self.adata)
                else:
                    self.logger.info("For the chosen distributional assumption, log-transform should not be applied.")

            # If distribution is Poisson or negative binomial, add pseudocount to each nonzero so that the min. is 2 and
            # not 1- expression of 1 indicates some interaction has a positive effect, but the linear predictor that
            # corresponds to this is 0, indicating no net effect:
            if self.distr in ["poisson", "nb"]:
                self.adata.layers["original_counts"] = self.adata.X.copy()
                if scipy.sparse.issparse(self.adata.X):
                    self.adata.X.data += 1
                else:
                    self.adata.X += 1

            if self.mod_type == "downstream":
                # For finding upstream associations with ligand
                self.setup_downstream()

            elif self.mod_type in ["ligand", "receptor", "lr", "niche"]:
                # Construct initial arrays for CCI modeling:
                self.define_sig_inputs()

    def setup_downstream(self, adata: Optional[anndata.AnnData] = None):
        """Setup for downstream tasks- namely, models for inferring signaling-associated differential expression."""
        if adata is None:
            adata = self.adata.copy()

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
            self.tf_tf_db = pd.read_csv(os.path.join(self.cci_dir, "human_TF_TF_db.csv"), index_col=0)
            self.cof_db = pd.read_csv(os.path.join(self.cci_dir, "human_cofactors.csv"), index_col=0)
            self.grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
        elif self.species == "mouse":
            try:
                self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"CCI resources cannot be found at {self.cci_dir}. Please check the path " f"and try again."
                )
            except IOError:
                raise IOError(
                    "Issue reading L:R database. Files can be downloaded from "
                    "https://github.com/aristoteleo/spateo-release/spateo/tools/database."
                )

            self.r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
            self.tf_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_TF_TF_db.csv"), index_col=0)
            self.cof_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_cofactors.csv"), index_col=0)
            self.grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
        else:
            raise ValueError("Invalid species specified. Must be one of 'human' or 'mouse'.")

        if os.path.exists(
            os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix", "design_matrix.csv")
        ):
            self.logger.info(
                f"Found existing independent variable matrix, loading from"
                f" {os.path.join(self.output_path, 'downstream_design_matrix', 'design_matrix.csv')}."
            )
            X_df = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix", "design_matrix.csv"),
                index_col=0,
            )
            self.targets_expr = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix", "targets.csv"),
                index_col=0,
            )
            if X_df.shape[0] != self.adata.n_obs:
                self.logger.info(
                    "Found existing independent variable matrix, but the given AnnData object contains "
                    "additional/different rows compared to the one used for the prior model. "
                    "Re-processing for new cells."
                )
                loaded_previously_processed = False
            else:
                loaded_previously_processed = True
        else:
            loaded_previously_processed = False

        if not loaded_previously_processed:
            # Targets = ligands
            if self.custom_ligands_path is not None or self.custom_ligands is not None:
                if self.custom_ligands_path is not None:
                    with open(self.custom_ligands_path, "r") as f:
                        targets = f.read().splitlines()
                else:
                    targets = self.custom_ligands

            # Else: targets = receptors:
            elif self.custom_receptors_path is not None or self.custom_receptors is not None:
                if self.custom_receptors_path is not None:
                    with open(self.custom_receptors_path, "r") as f:
                        targets = f.read().splitlines()
                else:
                    targets = self.custom_receptors

            # Else: targets = pathways:
            elif self.custom_pathways_path is not None or self.custom_pathways is not None:
                raise AttributeError(
                    "For downstream models, only custom sets of ligands or receptors should be provided."
                )

            # Else: targets = the targets from the initial model:
            elif self.targets_path is not None or self.custom_targets is not None:
                if self.targets_path is not None:
                    with open(self.targets_path, "r") as f:
                        targets = f.read().splitlines()
                else:
                    targets = self.custom_targets

            else:
                raise FileNotFoundError(
                    "For 'mod_type' = 'downstream', receptors, ligands or targets can be provided. Ligands must be "
                    "provided using either 'custom_lig_path' or 'ligand' arguments, receptors using 'custom_rec_path' "
                    "or 'receptor' arguments, and targets with 'targets_path' or 'target' arguments."
                )

            # Check that all targets can be found in the source AnnData object:
            targets = [t for t in targets if t in adata.var_names]
            # Check that all targets can be found in the GRN:
            targets = [t for t in targets if t in self.grn.index]
            self.logger.info(f"Found {len(targets)} genes to serve as dependent variable targets.")

            # Define ligand/receptor/pathway expression array:
            self.targets_expr = pd.DataFrame(
                adata[:, targets].X.A if scipy.sparse.issparse(adata.X) else adata[:, targets].X,
                index=adata.obs_names,
                columns=targets,
            )

            adata_orig = adata.copy()
            adata_orig.X = adata.layers["original_counts"]
            targets_expr_raw = pd.DataFrame(
                adata_orig[:, targets].X.A if scipy.sparse.issparse(adata.X) else adata_orig[:, targets].X,
                index=adata.obs_names,
                columns=targets,
            )

            self.logger.info("Searching AnnData object .obs field to construct regulator array for modeling...")
            if not any("regulator_" in obs for obs in adata.obs.columns):
                raise ValueError(
                    "No .obs fields found in AnnData object that start with 'regulator_'. These are added by the "
                    "downstream setup function- please run :class `MuSIC_Interpreter.CCI_deg_detection_setup()`."
                )
            regulator_cols = [col for col in adata.obs.columns if "regulator_" in col]
            X_df = pd.DataFrame(
                adata.obs.loc[:, regulator_cols],
                index=adata.obs_names,
                columns=regulator_cols,
            )

            # If applicable, check for multicollinearity:
            if self.multicollinear_threshold is not None:
                X_df = multicollinearity_check(X_df, self.multicollinear_threshold, logger=self.logger)

            # Log-scale to reduce the impact of "denser" neighborhoods:
            X_df = X_df.applymap(np.log1p)

            # Normalize to alleviate the impact of differences in scale:
            X_df = X_df.apply(lambda column: (column - column.min()) / (column.max() - column.min()))

            # Save design matrix and component dataframes (here, target ligands dataframe):
            if not os.path.exists(os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix")):
                os.makedirs(os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix"))
            if not os.path.exists(
                os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix", "design_matrix.csv")
            ):
                self.logger.info(
                    f"Saving design matrix to directory "
                    f"{os.path.join(os.path.splitext(self.output_path)[0], 'downstream_design_matrix')}."
                )
                X_df.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix", "design_matrix.csv")
                )

            self.logger.info(
                f"Saving targets array to "
                f"{os.path.join(os.path.splitext(self.output_path)[0], 'downstream_design_matrix', 'targets.csv')}"
            )
            targets_expr_raw.to_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "downstream_design_matrix", "targets.csv")
            )

        self.X = X_df.values
        self.feature_names = [f.replace("regulator_", "") for f in X_df.columns]
        self.logger.info(f"All possible regulatory factors: {self.feature_names}")

        # If applicable, add covariates:
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
            matched_var_matrix = self.adata[:, matched_var_names].X.A
            cov_names = matched_obs + matched_var_names
            concatenated_matrix = np.concatenate((matched_obs_matrix, matched_var_matrix), axis=1)
            self.X = np.concatenate((self.X, concatenated_matrix), axis=1)
            self.feature_names += cov_names

        # Add intercept if applicable:
        if self.fit_intercept:
            self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
            self.feature_names = ["intercept"] + self.feature_names

        # Add small amount to expression to prevent issues during regression:
        zero_rows = np.where(np.all(self.X == 0, axis=1))[0]
        for row in zero_rows:
            self.X[row, 0] += 1e-6

        self.n_features = self.X.shape[1]
        self.X_df = pd.DataFrame(self.X, columns=self.feature_names, index=self.adata.obs_names)

        # Compute distance in "signaling space":
        # Binarize design matrix to encode presence/absence of signaling pairs:
        self.feature_distance = np.where(self.X > 0, 1, 0)
        self.logger.info(f"Avg. number of TFs: {np.mean(np.sum(self.feature_distance, axis=1)):.2f}")

    def define_sig_inputs(self, adata: Optional[anndata.AnnData] = None, recompute: bool = False):
        """For signaling-relevant models, define necessary quantities that will later be used to define the independent
        variable array- the one-hot cell-type array, the ligand expression array and the receptor expression array.

        Args:
            recompute: Re-calculate all quantities and re-save even if already-existing file can be found in path
        """
        if adata is None:
            adata = self.adata.copy()

        # Load databases if applicable:
        if self.mod_type in ["lr", "ligand", "receptor"]:
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
                tf_target_db = pd.read_csv(os.path.join(self.cci_dir, "human_TF_target_db.csv"), index_col=0)
                self.grn = pd.read_csv(os.path.join(self.cci_dir, "human_GRN.csv"), index_col=0)
            elif self.species == "mouse":
                try:
                    self.lr_db = pd.read_csv(os.path.join(self.cci_dir, "lr_db_mouse.csv"), index_col=0)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"CCI resources cannot be found at {self.cci_dir}. Please check the path " f"and try again."
                    )
                except IOError:
                    raise IOError(
                        "Issue reading L:R database. Files can be downloaded from "
                        "https://github.com/aristoteleo/spateo-release/spateo/tools/database."
                    )

                self.r_tf_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_receptor_TF_db.csv"), index_col=0)
                tf_target_db = pd.read_csv(os.path.join(self.cci_dir, "mouse_TF_target_db.csv"), index_col=0)
                self.grn = pd.read_csv(os.path.join(self.cci_dir, "mouse_GRN.csv"), index_col=0)
            else:
                raise ValueError("Invalid species specified. Must be one of 'human' or 'mouse'.")

            if self.species == "human":
                database_pathways = set(self.lr_db["pathway"])

        # Check for existing design matrix:
        if (
            os.path.exists(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"))
            and not recompute
        ):
            self.logger.info(
                f"Found existing independent variable matrix, loading from"
                f" {os.path.join(self.output_path, 'design_matrix', 'design_matrix.csv')}."
            )
            X_df = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"), index_col=0
            )
            self.targets_expr = pd.read_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "targets.csv"), index_col=0
            )
            # Check if additional targets are provided compared to previously saved run- if so, update dataframe:
            if self.targets_path is not None or self.custom_targets is not None:
                if self.targets_path is not None:
                    with open(self.targets_path, "r") as f:
                        targets = f.read().splitlines()
                else:
                    targets = self.custom_targets
                targets = [t for t in targets if t in adata.var_names]
            all_targets = list(set(self.targets_expr.columns.tolist() + targets))

            if len(all_targets) > len(self.targets_expr.columns.tolist()):
                # Filter targets to those that can be found in our prior GRN:
                self.logger.info("Adding new targets to existing targets array, saving back to path.")
                all_targets = [t for t in all_targets if t in self.grn.index]

                self.targets_expr = pd.DataFrame(
                    adata[:, all_targets].X.A if scipy.sparse.issparse(adata.X) else adata[:, all_targets].X,
                    index=adata.obs_names,
                    columns=targets,
                )

                # Cap extreme numerical outliers:
                for col in self.targets_expr.columns:
                    # Calculate the 99.7th percentile for each column
                    cap_value = np.percentile(self.targets_expr[col], 99.7)
                    # Replace values above the 99.7th percentile with the cap value
                    self.targets_expr[col] = np.where(
                        self.targets_expr[col] > cap_value, cap_value, self.targets_expr[col]
                    )
                    # Round down to the nearest integer
                    self.targets_expr[col] = np.floor(self.targets_expr[col]).astype(int)

                self.logger.info(
                    f"Saving targets array back to "
                    f"{os.path.join(os.path.splitext(self.output_path)[0], 'design_matrix', 'targets.csv')}"
                )
                self.targets_expr.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "targets.csv")
                )

            if self.mod_type == "ligand" or self.mod_type == "lr":
                self.ligands_expr = pd.read_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr.csv"),
                    index_col=0,
                )
                self.ligands_expr_nonlag = pd.read_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr_nonlag.csv"),
                    index_col=0,
                )
            if self.mod_type == "receptor" or self.mod_type == "lr":
                self.receptors_expr = pd.read_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "receptors_expr.csv"),
                    index_col=0,
                )
            if self.mod_type == "niche":
                self.cell_categories = pd.read_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "cell_categories.csv"),
                    index_col=0,
                )
            if X_df.shape[0] != self.adata.n_obs:
                self.logger.info(
                    "Found existing independent variable matrix, but the given AnnData object contains "
                    "additional/different rows compared to the one used for the prior model."
                    "Re-processing for new cells."
                )
                loaded_previously_processed = False
            else:
                loaded_previously_processed = True
        else:
            loaded_previously_processed = False

        if not loaded_previously_processed:
            # One-hot cell type array (or other category):
            if self.mod_type == "niche":
                group_name = adata.obs[self.group_key]
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

            # Ligand-receptor expression arrays:
            if self.mod_type == "lr" or self.mod_type == "ligand":
                database_ligands = set(self.lr_db["from"])

                if self.custom_ligands_path is not None or self.custom_ligands is not None:
                    if self.custom_ligands_path is not None:
                        with open(self.custom_ligands_path, "r") as f:
                            ligands = f.read().splitlines()
                    else:
                        ligands = self.custom_ligands
                    ligands = [l for l in ligands if l in database_ligands]
                    # Some ligands in the mouse database are not ligands, but internal factors that interact w/ i.e.
                    # the hormone receptors:
                    ligands = [
                        l
                        for l in ligands
                        if l
                        not in [
                            "Lta4h",
                            "Fdx1",
                            "Tfrc",
                            "Trf",
                            "Lamc1",
                            "Aldh1a2",
                            "Dhcr24",
                            "Rnaset2a",
                            "Ptges3",
                            "Nampt",
                            "Trf",
                            "Fdx1",
                            "Kdr",
                            "Apoa2",
                            "Apoe",
                            "Dhcr7",
                            "Enho",
                            "Ptgr1",
                            "Agrp",
                            "Akr1b3",
                            "Daglb",
                            "Ubash3d",
                            "Psap",
                        ]
                    ]
                    l_complexes = [elem for elem in ligands if "_" in elem]
                    # Get individual components if any complexes are included in this list:
                    ligands = [l for item in ligands for l in item.split("_")]

                elif self.custom_pathways_path is not None or self.custom_pathways is not None:
                    if self.species != "human":
                        raise ValueError("Pathway information currently exists only for the human database.")

                    if self.custom_pathways_path is not None:
                        with open(self.custom_pathways_path, "r") as f:
                            pathways = f.read().splitlines()

                    else:
                        pathways = self.custom_pathways

                    pathways = [p for p in pathways if p in database_pathways]
                    # Get all ligands associated with these pathway(s):
                    lr_db_subset = self.lr_db[self.lr_db["pathway"].isin(pathways)]
                    ligands = list(set(lr_db_subset["from"]))
                    l_complexes = [elem for elem in ligands if "_" in elem]
                    # Get individual components if any complexes are included in this list:
                    ligands = [r for item in ligands for r in item.split("_")]

                else:
                    # List of possible complexes to search through:
                    l_complexes = [elem for elem in database_ligands if "_" in elem]
                    # And all possible ligand molecules:
                    all_ligands = [l for item in database_ligands for l in item.split("_")]

                    # Get list of ligands from among the most highly spatially-variable genes, indicative of potentially
                    # interesting spatially-enriched signal:
                    self.logger.info(
                        "Preparing data: no specific ligands provided- getting list of ligands from among the most "
                        "highly spatially-variable genes."
                    )
                    m_degs = moran_i(adata)
                    m_filter_genes = (
                        m_degs[m_degs.moran_q_val < 0.05].sort_values(by=["moran_i"], ascending=False).index
                    )
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

                    # In the case of using this method to find candidate ligands, save list of ligands in the same
                    # directory as the AnnData file for later access:
                    self.logger.info(
                        f"Saving list of manually found ligands to "
                        f"{os.path.join(os.path.dirname(self.adata_path), 'ligands.txt')}"
                    )

                    with open(os.path.join(os.path.dirname(self.adata_path), "ligands.txt"), "w") as f:
                        f.write("\n".join(ligands))

                ligands = [l for l in ligands if l in adata.var_names]

                self.ligands_expr = pd.DataFrame(
                    adata[:, ligands].X.A if scipy.sparse.issparse(adata.X) else adata[:, ligands].X,
                    index=adata.obs_names,
                    columns=ligands,
                )

                # Log-scale ligand expression- to reduce the impact of very large values:
                # self.ligands_expr = self.ligands_expr.applymap(np.log1p)

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
                        # found as ligands (and the complex is present enough in the data)- otherwise,
                        # keep the individual components:
                        threshold = self.n_samples * self.target_expr_threshold

                        for part in parts:
                            # If the geometric mean of the complex components is nonzero in sufficient number of
                            # cells, the individual components can be dropped.
                            if part not in database_ligands and (self.ligands_expr[part] != 0).sum() > threshold:
                                to_drop.append(part)

                        # to_drop.extend([part for part in parts if part not in database_ligands])
                    else:
                        # Drop the hyphenated element from the dataframe if all components are not found in the
                        # dataframe columns
                        partial_components = [l for l in ligands if l in parts]
                        to_drop.extend(partial_components)
                        if len(partial_components) > 0 and self.verbose:
                            self.logger.info(
                                f"Not all components from the {element} heterocomplex could be found in the dataset."
                            )

                # Drop any possible duplicate ligands alongside any other columns to be dropped:
                to_drop = list(set(to_drop))
                self.ligands_expr.drop(to_drop, axis=1, inplace=True)
                first_occurrences = self.ligands_expr.columns.duplicated(keep="first")
                self.ligands_expr = self.ligands_expr.loc[:, ~first_occurrences]
                # Save copy of non-lagged ligand expression array:
                self.ligands_expr_nonlag = self.ligands_expr.copy()

            if self.mod_type == "lr" or self.mod_type == "receptor":
                database_receptors = set(self.lr_db["to"])

                if self.custom_receptors_path is not None or self.custom_receptors is not None:
                    if self.custom_receptors_path is not None:
                        with open(self.custom_receptors_path, "r") as f:
                            receptors = f.read().splitlines()
                    else:
                        receptors = self.custom_receptors
                    receptors = [r for r in receptors if r in database_receptors]
                    r_complexes = [elem for elem in receptors if "_" in elem]
                    # Get individual components if any complexes are included in this list:
                    receptors = [r for item in receptors for r in item.split("_")]

                elif self.custom_pathways_path is not None or self.custom_pathways is not None:
                    if self.species != "human":
                        raise ValueError("Pathway information currently exists only for the human database.")

                    if self.custom_pathways_path is not None:
                        with open(self.custom_pathways_path, "r") as f:
                            pathways = f.read().splitlines()
                    else:
                        pathways = self.custom_pathways
                    pathways = [p for p in pathways if p in database_pathways]
                    # Get all receptors associated with these pathway(s):
                    lr_db_subset = self.lr_db[self.lr_db["pathway"].isin(pathways)]
                    receptors = list(set(lr_db_subset["to"]))
                    r_complexes = [elem for elem in receptors if "_" in elem]
                    # Get all individual components if any complexes are included in this list:
                    receptors = [r for item in receptors for r in item.split("_")]

                else:
                    # List of possible complexes to search through:
                    r_complexes = [elem for elem in database_receptors if "_" in elem]
                    # And all possible receptor molecules:
                    all_receptors = [r for item in database_receptors for r in item.split("_")]

                    # Get list of receptors from among the most highly spatially-variable genes, indicative of
                    # potentially interesting spatially-enriched signal:
                    self.logger.info(
                        "Preparing data: no specific receptors or pathways provided- getting list of receptors from "
                        "among the most highly spatially-variable genes."
                    )
                    m_degs = moran_i(adata)
                    m_filter_genes = (
                        m_degs[m_degs.moran_q_val < 0.05].sort_values(by=["moran_i"], ascending=False).index
                    )
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

                    # In the case of using this method to find candidate receptors, save the list of receptors in the
                    # same directory as the AnnData object for later access:
                    self.logger.info(
                        f"Saving list of manually found receptors to "
                        f"{os.path.join(os.path.dirname(self.adata_path), 'receptors.txt')}"
                    )

                    with open(os.path.join(os.path.dirname(self.adata_path), "receptors.txt"), "w") as f:
                        f.write("\n".join(receptors))

                receptors = [r for r in receptors if r in adata.var_names]

                self.receptors_expr = pd.DataFrame(
                    adata[:, receptors].X.A if scipy.sparse.issparse(adata.X) else adata[:, receptors].X,
                    index=adata.obs_names,
                    columns=receptors,
                )

                # Log-scale receptor expression (to reduce the impact of very large values):
                # self.receptors_expr = self.receptors_expr.applymap(np.log1p)

                # Normalize receptor expression if applicable:
                if self.normalize_signaling:
                    # self.receptors_expr = self.receptors_expr.apply(
                    #     lambda column: (column - column.min()) / (column.max() - column.min())
                    # )
                    self.receptors_expr = (self.receptors_expr - self.receptors_expr.min().min()) / (
                        self.receptors_expr.max().max() - self.receptors_expr.min().min()
                    )

                # Combine columns if they are part of a complex- eventually the individual columns should be dropped,
                # but store them in a temporary list to do so later because some may contribute to multiple complexes:
                to_drop = []
                for element in r_complexes:
                    if "_" in element:
                        parts = element.split("_")
                        if all(part in self.receptors_expr.columns for part in parts):
                            # Combine the columns into a new column with the name of the hyphenated element- here we
                            # will compute the geometric mean of the expression values of the complex components:
                            self.receptors_expr[element] = self.receptors_expr[parts].apply(
                                lambda x: x.prod() ** (1 / len(parts)), axis=1
                            )
                            # Mark the individual components for removal if the individual components cannot also be
                            # found as receptors (and the complex is present enough in the data)- otherwise,
                            # keep the individual components:
                            threshold = self.n_samples * self.target_expr_threshold

                            for part in parts:
                                # If the geometric mean of the complex components is nonzero in sufficient number of
                                # cells, the individual components can be dropped.
                                if (
                                    part not in database_receptors
                                    and (self.receptors_expr[part] != 0).sum() > threshold
                                ):
                                    to_drop.append(part)
                            # to_drop.extend([part for part in parts if part not in database_receptors])
                        else:
                            # Drop the hyphenated element from the dataframe if all components are not found in the
                            # dataframe columns
                            partial_components = [r for r in receptors if r in parts and r not in database_receptors]
                            to_drop.extend(partial_components)
                            if len(partial_components) > 0 and self.verbose:
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
                    if self.verbose:
                        self.logger.info(
                            "Preparing data: finding matched pairs between the selected ligands and receptors."
                        )
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
                            "No matched pairs between the selected ligands and receptors were found. If path to "
                            "custom list of ligands and/or receptors was provided, ensure ligand-receptor pairings "
                            "exist among these lists, or check data to make sure these ligands and/or receptors "
                            "were measured and were not filtered out."
                        )

                    pivoted_df = merged_df.pivot_table(values=["value_ligand", "value_receptor"], index=["from", "to"])
                    filtered_df = pivoted_df[pivoted_df.notna().all(axis=1)]
                    # Filter ligand and receptor expression to those that have a matched pair, if matched pairs are
                    # necessary:
                    if not self.include_unpaired_lr:
                        self.ligands_expr = self.ligands_expr[filtered_df.index.get_level_values("from").unique()]
                        self.receptors_expr = self.receptors_expr[filtered_df.index.get_level_values("to").unique()]
                        final_n_ligands = len(self.ligands_expr.columns)
                        final_n_receptors = len(self.receptors_expr.columns)

                        if self.verbose:
                            self.logger.info(
                                f"Found {final_n_ligands} ligands and {final_n_receptors} receptors that have matched "
                                f"pairs. {starting_n_ligands - final_n_ligands} ligands removed from the list and "
                                f"{starting_n_receptors - final_n_receptors} receptors/complexes removed from the "
                                f"list due to not having matched pairs among the corresponding set of "
                                f"receptors/ligands, respectively."
                                f"Remaining ligands: {self.ligands_expr.columns.tolist()}.\n"
                                f"Remaining receptors: {self.receptors_expr.columns.tolist()}."
                            )

                            self.logger.info(f"Set of {len(self.lr_pairs)} ligand-receptor pairs: {self.lr_pairs}")

            # ---------------------------------------------------------------------------------------------------
            # Get gene targets
            # ---------------------------------------------------------------------------------------------------
            if self.verbose:
                self.logger.info("Preparing data: getting gene targets.")
            # For niche model and ligand model, targets must be manually provided:
            if (self.targets_path is None and self.custom_targets is None) and self.mod_type in ["niche", "ligand"]:
                raise ValueError(
                    "For niche model and ligand model, `targets_path` must be provided. For L:R models, targets can be "
                    "automatically inferred, but receptor information does not exist for the other models."
                )

            if self.targets_path is not None or self.custom_targets is not None:
                if self.targets_path is not None:
                    with open(self.targets_path, "r") as f:
                        targets = f.read().splitlines()
                else:
                    targets = self.custom_targets
                targets = [t for t in targets if t in adata.var_names]

            # Else get targets by connecting to the targets of the L:R-downstream transcription factors:
            else:
                # Get the targets of the L:R-downstream transcription factors:
                tf_subset = self.r_tf_db[self.r_tf_db["receptor"].isin(self.receptors_expr.columns)]
                tfs = set(tf_subset["tf"])
                tfs = [tf for tf in tfs if tf in adata.var_names]
                # Subset to TFs that are expressed in > threshold number of cells:
                if scipy.sparse.issparse(adata.X):
                    tf_expr_percentage = np.array((adata[:, tfs].X > 0).sum(axis=0) / adata.n_obs)[0]
                else:
                    tf_expr_percentage = np.count_nonzero(adata[:, tfs].X, axis=0) / adata.n_obs
                tfs = np.array(tfs)[tf_expr_percentage > self.target_expr_threshold]

                targets_subset = tf_target_db[tf_target_db["TF"].isin(tfs)]
                targets = list(set(targets_subset["target"]))
                targets = [target for target in targets if target in adata.var_names]
                # Subset to targets that are expressed in > threshold number of cells:
                if scipy.sparse.issparse(adata.X):
                    target_expr_percentage = np.array((adata[:, targets].X > 0).sum(axis=0) / adata.n_obs)[0]
                else:
                    target_expr_percentage = np.count_nonzero(adata[:, targets].X, axis=0) / adata.n_obs
                targets = np.array(targets)[target_expr_percentage > self.target_expr_threshold]

            # Filter targets to those that can be found in our prior GRN:
            targets = [t for t in targets if t in self.grn.index]

            self.targets_expr = pd.DataFrame(
                adata[:, targets].X.A if scipy.sparse.issparse(adata.X) else adata[:, targets].X,
                index=adata.obs_names,
                columns=targets,
            )

            # Cap extreme numerical outliers (this can happen in cancer or just as technical error):
            for col in self.targets_expr.columns:
                # Calculate the 99.7th percentile for each column
                cap_value = np.percentile(self.targets_expr[col], 99.7)
                # Replace values above the 99.7th percentile with the cap value
                self.targets_expr[col] = np.where(self.targets_expr[col] > cap_value, cap_value, self.targets_expr[col])
                # Round down to the nearest integer
                self.targets_expr[col] = np.floor(self.targets_expr[col]).astype(int)

            # Compute spatial lag of ligand expression- exclude self for membrane-bound because autocrine signaling
            # is very difficult in principle:
            if self.mod_type == "lr" or self.mod_type == "ligand":
                # Path for saving spatial weights matrices:
                if not os.path.exists(os.path.join(os.path.splitext(self.output_path)[0], "spatial_weights")):
                    os.makedirs(os.path.join(os.path.splitext(self.output_path)[0], "spatial_weights"))

                # For checking for pre-computed spatial weights:
                membrane_bound_path = os.path.join(
                    os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_membrane_bound.npz"
                )
                secreted_path = os.path.join(
                    os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_secreted.npz"
                )

                # Compute separate set of spatial weights for membrane-bound and secreted ligands:
                if "spatial_weights_membrane_bound" not in locals():
                    if os.path.exists(membrane_bound_path):
                        spatial_weights_membrane_bound = scipy.sparse.load_npz(membrane_bound_path)
                        if spatial_weights_membrane_bound.shape[0] != self.adata.n_obs:
                            loaded_previously_processed = False
                        else:
                            loaded_previously_processed = True
                    else:
                        loaded_previously_processed = False

                    if not loaded_previously_processed:
                        bw = (
                            self.n_neighbors_membrane_bound
                            if self.distance_membrane_bound is None
                            else self.distance_membrane_bound
                        )
                        bw_fixed = True if self.distance_membrane_bound is not None else False
                        spatial_weights_membrane_bound = self._compute_all_wi(
                            bw=bw,
                            bw_fixed=bw_fixed,
                            exclude_self=True,
                            verbose=False,
                        )
                        self.logger.info(f"Saving spatial weights for membrane-bound ligands to {membrane_bound_path}.")
                        scipy.sparse.save_npz(membrane_bound_path, spatial_weights_membrane_bound)

                if "spatial_weights_secreted" not in locals():
                    if os.path.exists(secreted_path):
                        spatial_weights_secreted = scipy.sparse.load_npz(secreted_path)
                        if spatial_weights_secreted.shape[0] != self.adata.n_obs:
                            loaded_previously_processed = False
                        else:
                            loaded_previously_processed = True
                    else:
                        loaded_previously_processed = False

                    if not loaded_previously_processed:
                        bw = self.n_neighbors_secreted if self.distance_secreted is None else self.distance_secreted
                        bw_fixed = True if self.distance_secreted is not None else False
                        # Autocrine signaling is much easier with secreted signals:
                        spatial_weights_secreted = self._compute_all_wi(
                            bw=bw,
                            bw_fixed=bw_fixed,
                            exclude_self=False,
                            verbose=False,
                        )
                        self.logger.info(f"Saving spatial weights for secreted ligands to {secreted_path}.")
                        scipy.sparse.save_npz(secreted_path, spatial_weights_secreted)

                lagged_expr_mat = np.zeros_like(self.ligands_expr.values, dtype=float)

                for i, ligand in enumerate(self.ligands_expr.columns):
                    expr = self.ligands_expr[ligand]
                    expr_sparse = scipy.sparse.csr_matrix(expr.values.reshape(-1, 1))
                    matching_rows = self.lr_db[self.lr_db["from"] == ligand]
                    if (
                        matching_rows["type"].str.contains("Secreted Signaling").any()
                        or matching_rows["type"].str.contains("ECM-Receptor").any()
                    ):
                        lagged_expr = spatial_weights_secreted.dot(expr_sparse).A.flatten()
                    else:
                        lagged_expr = spatial_weights_membrane_bound.dot(expr_sparse).A.flatten()
                    lagged_expr_mat[:, i] = lagged_expr
                self.ligands_expr = pd.DataFrame(
                    lagged_expr_mat, index=adata.obs_names, columns=self.ligands_expr.columns
                )

                # Normalize ligand expression to be between 0 and 1:
                if self.normalize_signaling:
                    # self.ligands_expr = self.ligands_expr.apply(
                    #     lambda column: (column - column.min()) / (column.max() - column.min())
                    # )
                    self.ligands_expr = (self.ligands_expr - self.ligands_expr.min().min()) / (
                        self.ligands_expr.max().max() - self.ligands_expr.min().min()
                    )

            # Set independent variable array based on input given as "mod_type":
            if self.mod_type == "niche":
                # Path for saving spatial weights matrices:
                if not os.path.exists(os.path.join(os.path.splitext(self.output_path)[0], "spatial_weights")):
                    os.makedirs(os.path.join(os.path.splitext(self.output_path)[0], "spatial_weights"))

                # For checking for pre-computed spatial weights:
                niche_path = os.path.join(
                    os.path.splitext(self.output_path)[0], "spatial_weights", "spatial_weights_niche.npz"
                )

                # Compute spatial weights matrix- use n_neighbors and exclude_self from the argparse (defaults to 10).
                if "spatial_weights_niche" not in locals():
                    # Check for pre-computed spatial weights:
                    if "spatial_weights" in self.adata.obsp.keys():
                        self.logger.info("Spatial weights already found in AnnData object.")
                        spatial_weights_niche = self.adata.obsp["spatial_weights"]
                    else:
                        try:
                            spatial_weights_niche = scipy.sparse.load_npz(niche_path)
                        except:
                            spatial_weights_niche = self._compute_all_wi(
                                bw=self.n_neighbors_niche, bw_fixed=False, exclude_self=False, kernel="uniform"
                            )
                            self.logger.info(f"Saving spatial weights for niche to {niche_path}.")
                            scipy.sparse.save_npz(niche_path, spatial_weights_niche)
                        # Save to AnnData object, and update AnnData object in path:
                        self.adata.obsp["spatial_weights"] = spatial_weights_niche
                        self.adata.write_h5ad(self.adata_path)

                # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
                # each category are present within the neighborhood of each sample):
                dmat_neighbors = (spatial_weights_niche > 0).astype("int").dot(self.cell_categories.values)
                # If the number of cell types is low enough, incorporate the identity of each cell itself to fully
                # encode the niche:
                if len(self.cell_categories.columns) <= 10:
                    connections_data = {"categories": self.cell_categories, "dmat_neighbors": dmat_neighbors}
                    connections = np.asarray(dmatrix("categories:dmat_neighbors-1", connections_data))
                    connections[connections > 1] = 1
                    # Add categorical array indicating the cell type of each cell:
                    niche_array = np.hstack((self.cell_categories.values, connections))

                    connections_cols = list(product(self.cell_categories.columns, self.cell_categories.columns))
                    connections_cols.sort(key=lambda x: x[1])
                    connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
                    feature_names = list(self.cell_categories.columns) + connections_cols
                    X_df = pd.DataFrame(niche_array, index=self.adata.obs_names, columns=feature_names)
                else:
                    dmat_neighbors[dmat_neighbors > 1] = 1

                    neighbors_cols = [col.replace("Group", "Proxim") for col in self.cell_categories.columns]
                    # self.feature_names = list(self.cell_categories.columns) + neighbors_cols
                    feature_names = neighbors_cols
                    X_df = pd.DataFrame(dmat_neighbors, index=self.adata.obs_names, columns=feature_names)

            elif self.mod_type == "lr":
                lr_pair_labels = [f"{lr_pair[0]}:{lr_pair[1]}" for lr_pair in self.lr_pairs]

                # Use the ligand expression array and receptor expression array to compute the ligand-receptor pairing
                # array across all cells in the sample:
                X_df = pd.DataFrame(
                    np.zeros((self.n_samples, len(self.lr_pairs))),
                    columns=lr_pair_labels,
                    index=self.adata.obs_names,
                )

                for lr_pair in self.lr_pairs:
                    lig, rec = lr_pair[0], lr_pair[1]
                    lig_expr_values = self.ligands_expr[lig].values.reshape(-1, 1)
                    rec_expr_values = self.receptors_expr[rec].values.reshape(-1, 1)

                    # Communication signature b/w receptor in target and ligand in neighbors:
                    X_df[f"{lig}:{rec}"] = lig_expr_values * rec_expr_values

                # If any columns are very sparse, drop them- define sparsity as < 0.1% of cells having nonzero values:
                cols_to_drop = [col for col in X_df.columns if (X_df[col] != 0).sum() <= self.n_samples * 0.001]
                self.logger.info(
                    f"Dropping {cols_to_drop} columns due to sparsity (presence in <"
                    f"{np.round(self.n_samples * 0.001)} cells)."
                )
                X_df.drop(columns=cols_to_drop, axis=1, inplace=True)

                # Full original signaling feature array:
                X_df_full = X_df.copy()

                # If applicable, drop all-zero columns:
                X_df = X_df.loc[:, (X_df != 0).any(axis=0)]
                if len(self.lr_pairs) != X_df.shape[1]:
                    self.logger.info(
                        f"Dropped all-zero columns and sparse columns from signaling array, from {len(self.lr_pairs)} "
                        f"to {X_df.shape[1]}."
                    )

                # If applicable, check for multicollinearity:
                if self.multicollinear_threshold is not None:
                    X_df = multicollinearity_check(X_df, self.multicollinear_threshold, logger=self.logger)

                # For L:R models only- for each receptor, keep its features independent only if there is low overlap
                # between the features (all coexpressed in < 33% of the cells spanned by the interaction set). For
                # those which the overlap is high, combine all ligands into a single feature by taking the geometric
                # mean:
                ligand_receptor = X_df.columns.str.split(":", expand=True)
                ligand_receptor.columns = ["ligand", "receptor"]
                # Unique receptors:
                unique_receptors = ligand_receptor.get_level_values(1).unique()

                # For each receptor, compute the fraction of cells in which each cognate ligand is coupled to the
                # receptor:
                for receptor in unique_receptors:
                    # Find all rows (ligands) that correspond to this particular receptor
                    receptor_rows = [idx for idx in ligand_receptor if idx[1] == receptor]
                    receptor_cols = [f"{row[0]}:{row[1]}" for row in receptor_rows]
                    ligands = [row[0] for row in receptor_rows]
                    # Get relevant subset for this receptor- all cells that have evidence of any of the chosen
                    # interaction features:
                    receptor_df = X_df[(X_df[receptor_cols] != 0).any(axis=1)]
                    # Calculate overlap
                    overlap = (receptor_df[receptor_cols] != 0).all(axis=1).mean()
                    # Set threshold based on the length of receptor_cols
                    threshold = (
                        0.67
                        if len(receptor_cols) == 2
                        else 0.5
                        if len(receptor_cols) == 3
                        else 0.4
                        if len(receptor_cols) == 4
                        else 0.33
                        if len(receptor_cols) >= 5
                        else 1
                    )
                    # If overlap is greater than threshold, combine columns
                    if len(receptor_cols) > 1 and overlap > threshold:
                        combined_ligand = "/".join(ligands)
                        combined_col = f"{combined_ligand}:{receptor}"
                        # Compute arithmetic mean:
                        X_df[combined_col] = X_df[receptor_cols].mean(axis=1)
                        # X_df[combined_col] = X_df[receptor_cols].apply(lambda x: x.prod() ** (1 / len(parts)), axis=1)
                        # Drop the original columns:
                        X_df.drop(receptor_cols, axis=1, inplace=True)
                    else:
                        # If overlap is not greater than threshold for all ligands, check pairwise overlaps
                        # Calculate overlap for each pair of ligands and store them in a dictionary
                        overlaps = {}
                        ligand_combinations = list(itertools.combinations(ligands, 2))
                        for ligand1, ligand2 in ligand_combinations:
                            overlap = (
                                (receptor_df[[f"{ligand1}:{receptor}", f"{ligand2}:{receptor}"]] != 0)
                                .all(axis=1)
                                .mean()
                            )
                            overlaps[(ligand1, ligand2)] = overlap

                        # For each ligand, check if it has more than one other ligand that it exceeds the threshold
                        # with.
                        # If so, combine them into a single feature:
                        cols_to_drop = set()
                        for ligand in ligands:
                            exceeding_ligands = [
                                pair for pair in overlaps.keys() if ligand in pair and overlaps[pair] > 0.67
                            ]
                            if len(exceeding_ligands) > 1:
                                combined_ligands = set(
                                    itertools.chain(*exceeding_ligands)
                                )  # Get unique ligands in exceeding_ligands
                                combined_cols = [f"{l}:{receptor}" for l in combined_ligands]
                                # Set threshold based on the length of combined_cols
                                threshold = (
                                    0.67
                                    if len(combined_cols) == 2
                                    else 0.5
                                    if len(combined_cols) == 3
                                    else 0.4
                                    if len(combined_cols) == 4
                                    else 0.33
                                    if len(combined_cols) >= 5
                                    else 1
                                )
                                combined_receptor_df = receptor_df[(receptor_df[combined_cols] != 0).any(axis=1)]
                                # Calculate overlap for combined ligands
                                combined_overlap = (combined_receptor_df[combined_cols] != 0).all(axis=1).mean()
                                if combined_overlap > threshold:
                                    # If the combined overlap exceeds the threshold, combine all of them
                                    combined_ligand = "/".join(combined_ligands)
                                    combined_col = f"{combined_ligand}:{receptor}"
                                    # Geometric mean:
                                    X_df[combined_col] = X_df[combined_cols].mean(axis=1)
                                    cols_to_drop.update(combined_cols)
                                else:
                                    # If the combined overlap doesn't exceed the threshold, combine the ligand with
                                    # each of the other ligands separately
                                    for ligand_pair in exceeding_ligands:
                                        other_ligand = ligand_pair[0] if ligand_pair[1] == ligand else ligand_pair[1]
                                        combined_ligand = f"{ligand}/{other_ligand}"
                                        combined_col = f"{combined_ligand}:{receptor}"
                                        # Geometric mean:
                                        X_df[combined_col] = X_df[
                                            [f"{ligand}:{receptor}", f"{other_ligand}:{receptor}"]
                                        ].mean(axis=1)
                                        cols_to_drop.update([f"{ligand}:{receptor}", f"{other_ligand}:{receptor}"])

                        # Drop all columns at once
                        X_df.drop(list(cols_to_drop), axis=1, inplace=True)
                        # Final check: if multiple columns have high degree of overlap in their ligand/receptor
                        # combination, keep the most comprehensive one:
                        # Split each column into left and right parts
                        elements_left = [set(col.split(":")[0].split("/")) for col in X_df.columns]
                        elements_right = [col.split(":")[1] for col in X_df.columns]

                        cols_to_keep = []
                        for i, col in enumerate(X_df.columns):
                            # Keep column until proven otherwise:
                            keep = True
                            for j, other_col in enumerate(X_df.columns):
                                if (
                                    i != j
                                    and elements_left[i].issubset(elements_left[j])
                                    and elements_right[i] == elements_right[j]
                                ):
                                    keep = False
                                    break
                            if keep:
                                cols_to_keep.append(col)

                        X_df = X_df[cols_to_keep]

                # Add unpaired ligands and receptors to the design matrix:
                if self.include_unpaired_lr:
                    # Make note of which ligands are paired:
                    paired_ligands = [pair[0] for pair in self.lr_pairs]
                    # Add unpaired ligands to design matrix:
                    unpaired_ligands = [lig for lig in self.ligands_expr.columns if lig not in paired_ligands]
                    self.logger.info(
                        f"Adding unpaired ligands {unpaired_ligands} to L:R design matrix, conditioning "
                        f"on the presence of any present valid receptors and TFs (from database)."
                    )
                    for lig in unpaired_ligands:
                        # Mask the ligand values by a Boolean mask, where element is a 1 if valid (cognate) receptors
                        # or receptor-associated TFs are present in the cell.
                        # Get all receptors that are paired with this ligand:
                        associated_receptors = self.lr_db[self.lr_db["from"] == lig]["to"].unique().tolist()
                        associated_receptors = [
                            component for item in associated_receptors for component in item.split("_")
                        ]
                        associated_receptors = [rec for rec in associated_receptors if rec in self.adata.var_names]
                        # Check for receptors expressed in above threshold number of cells:
                        n_cell_threshold = np.min([100, self.target_expr_threshold * self.n_samples])
                        # Filter receptors expressed in more than the threshold number of cells
                        receptors_above_threshold = [
                            r for r in associated_receptors if self.adata[:, r].X.sum() > n_cell_threshold
                        ]

                        if receptors_above_threshold:
                            # If there are receptors above the threshold, only use these for checking
                            to_check = receptors_above_threshold
                        else:
                            # Get all TFs that are associated with these receptors, any TFs that are bound by these
                            # TFs, and any of the primary TFs in the MAPK/ERK, PI3K/AKT, and JAK/STAT pathways:
                            associated_tfs = (
                                self.r_tf_db[self.r_tf_db["receptor"].isin(associated_receptors)]["tf"]
                                .unique()
                                .tolist()
                            )

                            to_check = associated_receptors + associated_tfs
                            to_check = [component for item in to_check for component in item.split("_")]
                            to_check = [item for item in to_check if item in self.adata.var_names]

                        threshold = 0 if receptors_above_threshold else 3
                        mask = self.adata[:, to_check].X.sum(axis=1) > threshold
                        mask = np.array(mask).flatten()
                        X_df[lig] = self.ligands_expr[lig] * mask

                    # Make note of which receptors are paired:
                    paired_receptors = [pair[1] for pair in self.lr_pairs]
                    # Add unpaired receptors to design matrix:
                    unpaired_receptors = [rec for rec in self.receptors_expr.columns if rec not in paired_receptors]
                    self.logger.info(f"Adding unpaired receptors {unpaired_receptors} to L:R design matrix.")
                    for rec in unpaired_receptors:
                        X_df[rec] = self.receptors_expr[rec]

                # Check for unpaired columns, round to counts to represent the average ligand value in the neighborhood:
                if self.include_unpaired_lr:
                    X_df[[c for c in X_df.columns if ":" not in c]] = X_df[
                        [c for c in X_df.columns if ":" not in c]
                    ].apply(np.rint)
                # Log-scale to reduce the impact of "denser" neighborhoods:
                X_df = X_df.applymap(np.log1p)

                # Normalize the data for the L:R pairs to alleviate differences in scale induced by the
                # multiplication operation:
                X_df = X_df.apply(lambda column: (column - column.min()) / (column.max() - column.min()))
                # The above operation will tend to exaggerate the presence of ligand in sparser neighborhoods- set
                # threshold to ignore these cells/avoid potential false positives:
                X_df[X_df < 0.3] = 0

            elif self.mod_type == "ligand" or self.mod_type == "receptor":
                if self.mod_type == "ligand":
                    X_df = self.ligands_expr
                elif self.mod_type == "receptor":
                    X_df = self.receptors_expr

                # Full original signaling feature array:
                X_df_full = X_df.copy()

                # If applicable, drop all-zero columns:
                X_df = X_df.loc[:, (X_df != 0).any(axis=0)]

                if self.mod_type == "ligand":
                    ligand_to_check_dict = {}
                    for lig in X_df.columns:
                        # Mask the ligand values by a Boolean mask, where element is a 1 if valid (cognate) receptors
                        # or receptor-associated TFs are present in the cell.
                        # Get all receptors that are paired with this ligand:
                        associated_receptors = self.lr_db[self.lr_db["from"] == lig]["to"].unique().tolist()
                        associated_receptors = [
                            component for item in associated_receptors for component in item.split("_")
                        ]
                        associated_receptors = [rec for rec in associated_receptors if rec in self.adata.var_names]
                        # Check for receptors expressed in above threshold number of cells:
                        n_cell_threshold = np.min([100, self.target_expr_threshold * self.n_samples])
                        # Filter receptors expressed in more than the threshold number of cells
                        receptors_above_threshold = [
                            r for r in associated_receptors if self.adata[:, r].X.sum() > n_cell_threshold
                        ]

                        if receptors_above_threshold:
                            # If there are receptors above the threshold, only use these for checking
                            to_check = receptors_above_threshold
                        else:
                            # Get all TFs that are associated with these receptors, any TFs that are bound by these
                            # TFs, and any of the primary TFs in the MAPK/ERK, NFKB, PI3K/AKT, and JAK/STAT pathways:
                            associated_tfs = (
                                self.r_tf_db[self.r_tf_db["receptor"].isin(associated_receptors)]["tf"]
                                .unique()
                                .tolist()
                            )

                            if self.species == "mouse":
                                additional_tfs = [
                                    "Elk1",
                                    "Fos",
                                    "Myc",
                                    "Sp1",
                                    "Jun",
                                    "Atf2",
                                    "Nfkb1",
                                    "Rela",
                                    "Ets1",
                                    "Srebf1",
                                    "Srebf2",
                                    "Creb1",
                                    "Foxo1",
                                    "Foxo3",
                                    "Foxo4",
                                    "Stat1",
                                    "Stat2",
                                    "Stat3",
                                    "Stat4",
                                    "Stat5a",
                                    "Stat5b",
                                    "Stat6",
                                ]
                            elif self.species == "human":
                                additional_tfs = [
                                    "ELK1",
                                    "FOS",
                                    "MYC",
                                    "SP1",
                                    "JUN",
                                    "ATF2",
                                    # NFkB Family
                                    "NFKB1",  # NFKB1, p50/p105 subunit
                                    "RELA",  # RELA, p65 subunit
                                    "ETS1",
                                    "SREBF1",
                                    "SREBF2",
                                    "CREB1",
                                    # FOXO factors in the PI3K/AKT pathway
                                    "FOXO1",
                                    "FOXO3",
                                    "FOXO4",
                                    # STAT family TFs
                                    "STAT1",
                                    "STAT2",
                                    "STAT3",
                                    "STAT4",
                                    "STAT5A",
                                    "STAT5B",
                                    "STAT6",
                                ]
                            associated_tfs.extend(additional_tfs)

                            to_check = associated_receptors + associated_tfs
                        to_check = [component for item in to_check for component in item.split("_")]
                        to_check = [item for item in to_check if item in self.adata.var_names]
                        ligand_to_check_dict[lig] = list(set(to_check))
                        # Arbitrary threshold, but set the number of supporting receptors/TFs to be at least 3 for
                        # increased confidence:
                        threshold = 0 if receptors_above_threshold else 3
                        mask = self.adata[:, to_check].X.sum(axis=1) > threshold
                        mask = np.array(mask).flatten()
                        X_df[lig] = X_df[lig] * mask

                if self.mod_type == "ligand":
                    self.logger.info(
                        f"Dropped all-zero columns from ligand expression array, from "
                        f"{self.ligands_expr.shape[1]} to {X_df.shape[1]}."
                    )
                elif self.mod_type == "receptor":
                    self.logger.info(
                        f"Dropped all-zero columns from receptor expression array, from "
                        f"{self.receptors_expr.shape[1]} to {X_df.shape[1]}."
                    )

                # If applicable, check for multicollinearity:
                if self.multicollinear_threshold is not None:
                    X_df = multicollinearity_check(X_df, self.multicollinear_threshold, logger=self.logger)

                # Log-scale to reduce the impact of "denser" neighborhoods:
                X_df = X_df.applymap(np.log1p)
                # Normalize the data to prevent numerical overflow:
                X_df = X_df.apply(lambda column: (column - column.min()) / (column.max() - column.min()))
                # The above operation will tend to exaggerate the presence of ligand in sparser neighborhoods- set
                # threshold to ignore these cells/avoid potential false positives:
                X_df[X_df < 0.3] = 0

            else:
                raise ValueError("Invalid `mod_type` specified. Must be one of 'niche', 'lr', 'ligand' or 'receptor'.")

        # Check for NaN/infinite- if present, set to 0:
        X_df.fillna(0, inplace=True)
        X_df.replace([np.inf, -np.inf], 0, inplace=True)
        # Alphabetize the string in each column:
        feature_names_mod = [
            ":".join("/".join(sorted(part.split("/"))) for part in feat.split(":")) for feat in X_df.columns
        ]
        X_df.columns = feature_names_mod

        # Save design matrix and component dataframes:
        if not os.path.exists(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix")):
            os.makedirs(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix"))
        if not os.path.exists(
            os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv")
        ):
            self.logger.info(
                f"Saving design matrix to "
                f"{os.path.join(os.path.splitext(self.output_path)[0], 'design_matrix', 'design_matrix.csv')}."
            )
            X_df.to_csv(os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix.csv"))
            if self.mod_type != "niche":
                X_df_full.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "design_matrix_full.csv")
                )

            if self.mod_type == "ligand":
                # Save the list of receptors/TFs that were checked to allow each ligand to be included in the design:
                with open(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligand_to_check_dict.json"),
                    "w",
                ) as f:
                    json.dump(ligand_to_check_dict, f)

            if self.mod_type == "ligand" or self.mod_type == "lr":
                self.logger.info(
                    f"Saving ligand expression matrix to "
                    f"{os.path.join(os.path.splitext(self.output_path)[0], 'design_matrix', 'ligands_expr.csv')}"
                )
                self.ligands_expr.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr.csv")
                )
                self.ligands_expr_nonlag.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "ligands_expr_nonlag.csv")
                )
            if self.mod_type == "receptor" or self.mod_type == "lr":
                self.logger.info(
                    f"Saving receptor expression matrix to "
                    f"{os.path.join(os.path.splitext(self.output_path)[0], 'design_matrix', 'receptors_expr.csv')}"
                )
                self.receptors_expr.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "receptors_expr.csv")
                )
            if self.mod_type == "niche":
                self.logger.info(
                    f"Saving cell categories to "
                    f"{os.path.join(os.path.splitext(self.output_path)[0], 'design_matrix', 'cell_categories.csv')}"
                )
                self.cell_categories.to_csv(
                    os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "cell_categories.csv")
                )

            self.logger.info(
                f"Saving targets array to "
                f"{os.path.join(os.path.splitext(self.output_path)[0], 'design_matrix', 'targets.csv')}"
            )
            self.targets_expr.to_csv(
                os.path.join(os.path.splitext(self.output_path)[0], "design_matrix", "targets.csv")
            )

        self.X = X_df.values
        self.feature_names = list(X_df.columns)
        # (For interpretability in downstream analyses) update ligand names/receptor names to reflect the final
        # molecules used:
        if self.mod_type == "ligand":
            self.ligands = self.feature_names
        elif self.mod_type == "receptor":
            self.receptors = self.feature_names
        elif self.mod_type == "lr":
            # Update :attr `lr_pairs` to reflect the final L:R pairs used:
            self.lr_pairs = [
                tuple((pair.split(":")[0], pair.split(":")[1])) for pair in self.feature_names if ":" in pair
            ]
            receptors = [p[1] for p in self.lr_pairs]

        # If applicable, add covariates:
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
            matched_var_matrix = self.adata[:, matched_var_names].X.A
            cov_names = matched_obs + matched_var_names
            concatenated_matrix = np.concatenate((matched_obs_matrix, matched_var_matrix), axis=1)
            self.X = np.concatenate((self.X, concatenated_matrix), axis=1)
            self.feature_names += cov_names

        # Add intercept if applicable:
        if self.fit_intercept:
            self.X = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
            self.feature_names = ["intercept"] + self.feature_names

        # Add small amount to expression to prevent issues during regression:
        zero_rows = np.where(np.all(self.X == 0, axis=1))[0]
        for row in zero_rows:
            self.X[row, 0] += 1e-6

        # Broadcast independent variables and feature names:
        self.n_features = self.X.shape[1]
        self.X_df = X_df
        # self.X_df = pd.DataFrame(self.X, columns=self.feature_names, index=self.adata.obs_names)

        # Compute distance in "signaling space":
        if self.mod_type != "niche":
            # Binarize design matrix to encode presence/absence of signaling pairs:
            self.feature_distance = np.where(self.X > 0, 1, 0)
        else:
            self.feature_distance = None

    def run_subsample(self, y: Optional[pd.DataFrame] = None):
        """To combat computational intensiveness of this regressive protocol, subsampling will be performed in cases
        where there are >= 5000 cells or in cases where specific cell types are manually selected for fitting- local
        fit will be performed only on this subset under the assumption that discovered signals will not be
        significantly different for the subsampled data.

        New Attributes:
            subsampled_indices: Dictionary containing indices of the subsampled cells for each dependent variable
            n_samples_subsampled: Dictionary containing number of samples to be fit (not total number of samples) for
                each dependent variable
            subsampled_sample_names: Dictionary containing lists of names of the subsampled cells for each dependent
                variable
            neighboring_unsampled: Dictionary containing a mapping between each unsampled point and the closest
                sampled point
        """
        if self.mod_type == "downstream":
            parent_dir = os.path.join(os.path.dirname(self.output_path), "downstream")
        else:
            parent_dir = os.path.dirname(self.output_path)
        if not os.path.exists(os.path.join(parent_dir, "subsampling")):
            os.makedirs(os.path.join(parent_dir, "subsampling"))
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))

        # Check for already-existing subsampling results:
        _, filename = os.path.split(self.output_path)
        filename = os.path.splitext(filename)[0]
        neighboring_unsampled_path = os.path.join(parent_dir, "subsampling", f"{filename}.json")
        subsampled_sample_names_path = os.path.join(parent_dir, "subsampling", f"{filename}_cell_names.json")

        # Check if all of these files exist, and additionally if any targets are missing from the existing files:
        existing_targets = set()
        if y is None:
            y_arr = self.targets_expr if hasattr(self, "targets_expr") else self.target
        else:
            y_arr = y

        if os.path.exists(neighboring_unsampled_path) and os.path.exists(subsampled_sample_names_path):
            self.subsampled_indices = {}
            self.n_samples_subsampled = {}
            self.logger.info("Loading existing subsampling results from previous run and resuming...")
            with open(neighboring_unsampled_path, "r") as f:
                self.neighboring_unsampled = json.load(f)
            with open(subsampled_sample_names_path, "r") as f:
                self.subsampled_sample_names = json.load(f)
            existing_targets.update(self.neighboring_unsampled.keys())

            for target in self.subsampled_sample_names.keys():
                self.subsampled_indices[target] = [
                    self.sample_names.get_loc(name) for name in self.subsampled_sample_names[target]
                ]
                self.n_samples_subsampled[target] = len(self.subsampled_indices[target])

            # New targets to process
            new_targets = set(y_arr.columns) - existing_targets
            self.logger.info(f"Already processed targets: {', '.join(existing_targets)}")
            self.logger.info(f"New targets that need to be processed: {', '.join(new_targets)}")
        else:
            self.neighboring_unsampled = {}
            self.subsampled_sample_names = {}
            self.subsampled_indices = {}
            self.n_samples_subsampled = {}

        # For subsampling by point selection (otherwise, these will not be dictionaries because they are the same
        # for all targets):
        # Dictionary to store both cell labels (:attr `subsampled_sample_names`) and numerical indices (:attr
        # `subsampled_indices`) of subsampled points, :attr `n_samples_subsampled` (for setting :attr `x_chunk`
        # later on, and :attr `neighboring_unsampled` to establish a mapping between each not-sampled point and
        # the closest sampled point:

        # If :attr `group_subset` is not None, AnnData was subset to only include cell types of interest, as well as
        # their neighboring cells. However, we only want to fit the model on the cell types of interest,
        # so :attr `subsampled_indices` consists of the set of indices that don't correspond to the neighboring
        # cells:
        if self.group_subset is not None:
            adata = self.adata[self.group_subsampled_sample_names].copy()
            n_samples = adata.n_obs
            sample_names = adata.obs_names
            coords = adata.obsm[self.coords_key]
        else:
            adata = self.adata.copy()
            n_samples = self.n_samples
            sample_names = self.sample_names
            coords = self.coords

        if self.total_counts_threshold != 0.0:
            self.logger.info(f"Subsetting to cells with greater than {self.total_counts_threshold} total counts...")
            if self.total_counts_key not in self.adata.obs_keys():
                raise KeyError(f"{self.total_counts_key} not found in .obs of AnnData.")
            adata_high_qual = adata[adata.obs[self.total_counts_key] >= self.total_counts_threshold]
            sampled_coords = adata_high_qual.obsm[self.coords_key]
            sample_names_high_qual = adata_high_qual.obs_names
            y_arr_high_qual = y_arr.loc[sample_names]
            threshold_sampled_names = sample_names.intersection(sample_names_high_qual)
            if self.spatial_subsample is False:
                self.logger.info(f"For all targets subsampled from {n_samples} to {adata_high_qual.n_obs} cells.")

            for target in y_arr_high_qual.columns:
                # Skip targets that have already been processed:
                if target in existing_targets:
                    continue

                all_values = y_arr[target].loc[sample_names].values.reshape(-1, 1)
                sampled_values = y_arr_high_qual[target].loc[threshold_sampled_names].values.reshape(-1, 1)

                data = np.concatenate((sampled_coords, sampled_values), axis=1)
                num_dims = sampled_coords.shape[1]
                dim_names = [f"dim_{i + 1}" for i in range(num_dims)]
                column_names = dim_names + [target]

                sampled_df = pd.DataFrame(data, columns=column_names, index=threshold_sampled_names)

                # Define relevant dictionaries- only if further subsampling by region will not be performed:
                if self.spatial_subsample is False:
                    # closest_dict will be the same for all targets:
                    if "closest_dict" not in locals():
                        # Use all columns except the last one (target column)
                        dim_columns = sampled_df.columns[:-1]  # This excludes the target column
                        ref = sampled_df[dim_columns].values.astype(float)
                        distances = cdist(coords.astype(float), ref, "euclidean")

                        # Create a mask for non-matching expression patterns b/w sampled and close-by neighbors:
                        all_expression = (all_values != 0).flatten()
                        sampled_expression = sampled_df[target].values != 0
                        mismatch_mask = all_expression[:, np.newaxis] != sampled_expression
                        # Replace distances in the mismatch mask with a very large value:
                        large_value = np.max(distances) + 1
                        distances[mismatch_mask] = large_value

                        closest_indices = np.argmin(distances, axis=1)

                        # Dictionary where keys are indices of subsampled points and values are lists of indices of the
                        # original points closest to them:
                        closest_dict = {}
                        for i, idx in enumerate(closest_indices):
                            key = sampled_df.index[idx]
                            if key not in closest_dict:
                                closest_dict[key] = []
                            if sample_names[i] not in sampled_df.index:
                                closest_dict[key].append(sample_names[i])
                        self.logger.info("Finished compiling mapping from unsampled to sampled points...")

                    # If subsampling by total counts, the arrays of subsampled indices, number subsampled and
                    # subsampled sample names (but notably not the unsampled-to-sampled mapping) are the same,
                    # but subsequent computation will search through dictionary keys to get these, so we format
                    # them as dictionaries anyways:
                    self.subsampled_indices[target] = [
                        self.sample_names.get_loc(name) for name in threshold_sampled_names
                    ]
                    self.n_samples_subsampled[target] = len(threshold_sampled_names)
                    self.subsampled_sample_names[target] = list(threshold_sampled_names)
                    self.neighboring_unsampled[target] = closest_dict

        # Subsampling by region:
        if self.spatial_subsample:
            self.logger.info("Performing stratified subsampling from different regions of the data...")
            for target in y_arr.columns:
                # Skip targets that have already been processed:
                if target in existing_targets:
                    self.logger.info(f"Skipping already processed target: {target}")
                    continue

                # Check if target-specific files exist:
                closest_dict_path = os.path.join(parent_dir, "subsampling", f"{filename}_{target}_closest_dict.json")
                subsampled_names_path = os.path.join(
                    parent_dir, "subsampling", f"{filename}_{target}_subsampled_names.txt"
                )

                if os.path.exists(closest_dict_path) and os.path.exists(subsampled_names_path):
                    with open(closest_dict_path, "r") as file:
                        self.neighboring_unsampled[target] = json.load(file)
                    with open(subsampled_names_path, "r") as file:
                        self.subsampled_sample_names[target] = file.read().splitlines()
                        self.subsampled_indices[target] = [
                            self.sample_names.get_loc(name) for name in self.subsampled_sample_names[target]
                        ]
                        self.n_samples_subsampled[target] = len(self.subsampled_indices[target])
                    self.logger.info(f"Loaded existing subsampling results for target {target}...")

                else:
                    if self.group_subset is None:
                        values = y_arr[target].values.reshape(-1, 1)
                    else:
                        values = y_arr[target].loc[sample_names].values.reshape(-1, 1)

                    # Spatial clustering:
                    n_clust = int(0.05 * n_samples)
                    kmeans = KMeans(n_clusters=n_clust, random_state=0).fit(coords)
                    spatial_clusters = kmeans.predict(coords).astype(int).reshape(-1, 1)

                    data = np.concatenate(
                        (
                            coords,
                            spatial_clusters,
                            values,
                        ),
                        axis=1,
                    )

                    if coords.shape[1] == 2:
                        temp_df = pd.DataFrame(
                            data,
                            columns=["x", "y", "spatial_cluster", target],
                            index=sample_names,
                        )
                    else:
                        temp_df = pd.DataFrame(
                            data,
                            columns=["x", "y", "z", "spatial_cluster", target],
                            index=sample_names,
                        )

                    temp_df[f"{target}_density"] = temp_df.groupby("spatial_cluster")[target].transform(
                        lambda x: np.count_nonzero(x) / len(x)
                    )

                    # Stratified subsampling:
                    sampled_df = pd.DataFrame()
                    for stratum in temp_df["spatial_cluster"].unique():
                        if len(set(temp_df[f"{target}_density"])) == 2:
                            stratum_df = temp_df[temp_df["spatial_cluster"] == stratum]
                            # Density of node feature in this stratum
                            node_feature_density = stratum_df[f"{target}_density"].iloc[0]

                            # Set total number of cells to subsample- sample at least the number of zero cells as
                            # nonzeros:
                            # Sample size proportional to stratum size and node feature density:
                            n_sample_nonzeros = int(np.ceil((len(stratum_df) // 2) * node_feature_density))
                            n_sample_zeros = n_sample_nonzeros
                            sample_size = n_sample_zeros + n_sample_nonzeros
                            sampled_stratum_df = stratum_df.sample(n=sample_size)
                            sampled_df = pd.concat([sampled_df, sampled_stratum_df])

                        else:
                            stratum_df = temp_df[temp_df["spatial_cluster"] == stratum]
                            # Density of node feature in this stratum
                            node_feature_density = stratum_df[f"{target}_density"].iloc[0]

                            # Proportional sample size based on number of nonzeros- or three zero cells, depending
                            # on which is larger:
                            num_nonzeros = len(stratum_df[stratum_df[f"{target}_density"] > 0])
                            n_sample_nonzeros = int(np.ceil((num_nonzeros // 2) * node_feature_density))
                            n_sample_zeros = np.maximum(n_sample_nonzeros, 3)
                            sample_size = n_sample_zeros + n_sample_nonzeros

                            # Sample at least n_sample_zeros zeros if possible:
                            zero_sub = stratum_df[stratum_df[target] == 0]
                            n_zeros_sample = np.minimum(n_sample_zeros, len(zero_sub))
                            sampled_zero_stratum_df = zero_sub.sample(n=n_zeros_sample)

                            # Check if any nonzeros exist
                            stratum_nonzero_df = stratum_df[stratum_df[target] > 0]
                            if not stratum_nonzero_df.empty:
                                # Sample from nonzeros first
                                num_nonzeros_sampled = min(len(stratum_nonzero_df), sample_size - n_sample_zeros)
                                sampled_nonzero_stratum_df = stratum_nonzero_df.sample(n=num_nonzeros_sampled)

                                # Concatenate zeros and nonzeros:
                                sampled_stratum_df = pd.concat([sampled_nonzero_stratum_df, sampled_zero_stratum_df])
                            else:
                                sampled_stratum_df = sampled_zero_stratum_df

                            sampled_df = pd.concat([sampled_df, sampled_stratum_df])

                    # Check to see if counts-based subsampling was performed- if so, subset the sampled dataframe
                    # based on the subset already generated from that:
                    if "threshold_sampled_names" in locals():
                        updated_sampled_names = set(threshold_sampled_names).intersection(sampled_df.index)
                        sampled_df = sampled_df.loc[updated_sampled_names]

                    self.logger.info(f"For target {target} subsampled from {n_samples} to {len(sampled_df)} cells.")

                    # Map each non-sampled point to its closest sampled point that matches the expression pattern
                    # (zero/nonzero):
                    if coords.shape[1] == 2:
                        ref = sampled_df[["x", "y"]].values.astype(float)
                    else:
                        ref = sampled_df[["x", "y", "z"]].values.astype(float)

                    distances = cdist(coords.astype(float), ref, "euclidean")

                    # Create a mask for non-matching expression patterns b/w sampled and close-by neighbors:
                    all_expression = (values != 0).flatten()
                    sampled_expression = sampled_df[target].values != 0
                    mismatch_mask = all_expression[:, np.newaxis] != sampled_expression
                    # Replace distances in the mismatch mask with a very large value:
                    large_value = np.max(distances) + 1
                    distances[mismatch_mask] = large_value

                    closest_indices = np.argmin(distances, axis=1)

                    # Dictionary where keys are indices of subsampled points and values are lists of indices of the
                    # original points closest to them:
                    closest_dict = {}
                    for i, idx in enumerate(closest_indices):
                        key = sampled_df.index[idx]
                        if key not in closest_dict:
                            closest_dict[key] = []
                        if sample_names[i] not in sampled_df.index:
                            closest_dict[key].append(sample_names[i])

                    # Get index position in original AnnData object of each subsampled index:
                    self.subsampled_indices[target] = [self.sample_names.get_loc(name) for name in sampled_df.index]
                    self.n_samples_subsampled[target] = len(sampled_df)
                    self.subsampled_sample_names[target] = list(sampled_df.index)
                    self.neighboring_unsampled[target] = closest_dict

                    # Save target-specific files:
                    with open(closest_dict_path, "w") as file:
                        json.dump(self.neighboring_unsampled[target], file)
                    with open(subsampled_names_path, "w") as file:
                        file.write("\n".join(self.subsampled_sample_names[target]))

        # Save dictionary mapping unsampled points to nearest sampled points, indices of sampled points and names of
        # sampled points:
        _, filename = os.path.split(self.output_path)
        filename = os.path.splitext(filename)[0]

        if self.save_subsampling:
            with open(os.path.join(parent_dir, "subsampling", f"{filename}.json"), "w") as file:
                json.dump(self.neighboring_unsampled, file)
            with open(os.path.join(parent_dir, "subsampling", f"{filename}_cell_names.json"), "w") as file:
                json.dump(self.subsampled_sample_names, file)

    def map_new_cells(self):
        """There may be instances where new cells are added to an AnnData object that has already been fit to- in
        this instance, accelerate the process by using neighboring results to project model fit to the new cells.
        """
        sample_names = self.sample_names

        if self.mod_type == "downstream":
            parent_dir = os.path.join(os.path.dirname(self.output_path), "downstream")
        else:
            parent_dir = os.path.dirname(self.output_path)
        if not os.path.exists(os.path.join(parent_dir, "subsampling")):
            os.makedirs(os.path.join(parent_dir, "subsampling"))
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))

        # Check for already-existing subsampling results:
        _, filename = os.path.split(self.output_path)
        filename = os.path.splitext(filename)[0]
        neighboring_unsampled_path = os.path.join(parent_dir, "subsampling", f"{filename}.json")
        subsampled_sample_names_path = os.path.join(parent_dir, "subsampling", f"{filename}_cell_names.json")

        # Load existing subsampling results if existent- if not create them from all cells in the initial model fit:
        if os.path.exists(neighboring_unsampled_path):
            with open(neighboring_unsampled_path, "r") as f:
                self.neighboring_unsampled = json.load(f)
            with open(subsampled_sample_names_path, "r") as f:
                self.subsampled_sample_names = json.load(f)
        else:
            self.neighboring_unsampled = {}

        # Map each new point to the closest existing point that matches the expression pattern
        parent_dir = os.path.dirname(self.output_path)
        file_list = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]

        for file in file_list:
            if not "predictions" in file:
                check = pd.read_csv(os.path.join(parent_dir, file), index_col=0)
                break
        # Only check this if the initial run has already finished (i.e. postprocessing has already been performed):
        if check.index.dtype != "float64" and check.index.dtype != "float32":
            new_samples = list(set(sample_names).difference(check.index))
            self.logger.info(f"Getting mapping information for {len(new_samples)} new cells in this AnnData object...")

            if self.coords.shape[1] == 2:
                query = pd.DataFrame(
                    self.adata[new_samples, :].obsm[self.coords_key], index=new_samples, columns=["x", "y"]
                )
                ref = pd.DataFrame(
                    self.adata[check.index, :].obsm[self.coords_key], index=check.index, columns=["x", "y"]
                )
            else:
                query = pd.DataFrame(
                    self.adata[new_samples, :].obsm[self.coords_key], index=new_samples, columns=["x", "y", "z"]
                )
                ref = pd.DataFrame(
                    self.adata[check.index, :].obsm[self.coords_key], index=check.index, columns=["x", "y", "z"]
                )

            distances = cdist(query.values.astype(float), ref.values.astype(float), "euclidean")

            if hasattr(self, "targets_expr"):
                targets = self.targets_expr.columns
                y_arr = pd.DataFrame(
                    self.adata[:, targets].X.A if scipy.sparse.issparse(self.adata.X) else self.adata[:, targets].X,
                    index=self.sample_names,
                    columns=targets,
                )
            else:
                y_arr = self.target

            for target in y_arr.columns:
                ref_values = y_arr[target].loc[check.index].values
                query_values = y_arr[target].loc[new_samples].values

                # Create a mask for non-matching expression patterns b/w sampled and close-by neighbors:
                ref_expression = (ref_values != 0).flatten()
                query_expression = (query_values != 0).flatten()
                mismatch_mask = query_expression[:, np.newaxis] != ref_expression
                # Replace distances in the mismatch mask with a very large value:
                large_value = np.max(distances) + 1
                distances[mismatch_mask] = large_value

                closest_indices = np.argmin(distances, axis=1)

                # Dictionary where keys are indices of subsampled points and values are lists of indices of the
                # original points closest to them:
                closest_dict = self.neighboring_unsampled.get(target, {})
                if closest_dict == {}:
                    for i, key in enumerate(closest_indices):
                        closest_dict[key].append(new_samples[i])
                else:
                    for i, idx in enumerate(closest_indices):
                        key = ref.index[idx]
                        if key not in closest_dict:
                            closest_dict[key] = []
                        closest_dict[key].append(new_samples[i])

                self.neighboring_unsampled[target] = closest_dict
                # # Save target-specific files:
                # with open(closest_dict_path, "w") as file:
                #     json.dump(self.neighboring_unsampled[target], file)
                self.logger.info(f"Got mapping information for new cells for target {target}.")

            # Save dictionary mapping unsampled points to nearest sampled points:
            with open(os.path.join(parent_dir, "subsampling", f"{filename}.json"), "w") as file:
                json.dump(self.neighboring_unsampled, file)

    def _set_search_range(self):
        """Set the search range for the bandwidth selection procedure.

        Args:
            y: Array of dependent variable values, used to determine the search range for the bandwidth selection
        """

        if self.minbw is None:
            if self.bw_fixed:
                if self.mod_type == "downstream":
                    # Check for dimensionality reduction:
                    if "X_pca" in self.adata.obsm_keys():
                        coords_key = "X_pca"
                        initial_bw = 8
                        alpha = 0.05

                        # Use 0.2% of the total number of cells as the target number of neighbors for the lower
                        # bandwidth:
                        n_anchors = np.min((self.n_samples, 5000))
                        self.minbw = find_bw_for_n_neighbors(
                            self.adata,
                            coords_key=coords_key,
                            n_anchors=n_anchors,
                            target_n_neighbors=int(0.002 * self.n_samples),
                            verbose=False,
                            initial_bw=initial_bw,
                            alpha=alpha,
                            normalize_distances=True,
                        )

                        self.maxbw = find_bw_for_n_neighbors(
                            self.adata,
                            coords_key=coords_key,
                            n_anchors=n_anchors,
                            target_n_neighbors=int(0.005 * self.n_samples),
                            verbose=False,
                            initial_bw=initial_bw,
                            alpha=alpha,
                            normalize_distances=True,
                        )

                elif self.distance_membrane_bound is not None and self.distance_secreted is not None:
                    self.minbw = (
                        self.distance_membrane_bound * 1.5 if self.kernel != "uniform" else self.distance_membrane_bound
                    )
                    self.maxbw = self.distance_secreted * 1.5 if self.kernel != "uniform" else self.distance_secreted
                else:
                    # Set minimum bandwidth to the distance to the smallest distance between neighboring points:
                    min_dist = np.min(
                        np.array(
                            [np.min(np.delete(cdist(self.coords[[i]], self.coords), i)) for i in range(self.n_samples)]
                        )
                    )
                    # Arbitrarily chosen limits:
                    self.minbw = min_dist
                    self.maxbw = min_dist * 10

            # If the bandwidth is defined by a fixed number of neighbors (and thus adaptive in terms of radius):
            else:
                if self.maxbw is None:
                    # If kernel decays with distance, larger bandwidth to capture more neighbors:
                    self.maxbw = (
                        self.n_neighbors_secreted * 2 if self.kernel != "uniform" else self.n_neighbors_secreted
                    )

                if self.minbw is None:
                    self.minbw = self.n_neighbors_membrane_bound

        if self.minbw >= self.maxbw:
            raise ValueError(
                "The minimum bandwidth must be less than the maximum bandwidth. Please adjust the `minbw` "
                "parameter accordingly."
            )

    def _compute_all_wi(
        self,
        bw: Union[float, int],
        bw_fixed: Optional[bool] = None,
        exclude_self: Optional[bool] = None,
        kernel: Optional[str] = None,
        verbose: bool = False,
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
            verbose: Whether to display messages during runtime

        Returns:
            wi: Array of weights for all samples in the dataset
        """

        # Parallelized computation of spatial weights for all samples:
        if verbose:
            if not self.bw_fixed:
                self.logger.info(
                    "Note that 'fixed' was not selected for the bandwidth estimation. Input to 'bw' will be "
                    "taken to be the number of nearest neighbors to use in the bandwidth estimation."
                )

        if bw_fixed is None:
            bw_fixed = self.bw_fixed
        if exclude_self is None:
            exclude_self = self.exclude_self
        if kernel is None:
            kernel = self.kernel

        normalize_weights = True if self.normalize else False

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
            normalize_weights=normalize_weights,
        )

        with Pool() as pool:
            weights = pool.map(get_wi_partial, range(self.n_samples))
        w = scipy.sparse.vstack(weights)
        return w

    def local_fit(
        self,
        i: int,
        y: np.ndarray,
        X: np.ndarray,
        bw: Union[float, int],
        y_label: str,
        coords: Optional[np.ndarray] = None,
        mask_indices: Optional[np.ndarray] = None,
        feature_mask: Optional[np.ndarray] = None,
        final: bool = False,
        fit_predictor: bool = False,
    ) -> Union[np.ndarray, List[float]]:
        """Fit a local regression model for each sample.

        Args:
            i: Index of sample for which local regression model is to be fitted
            y: Response variable
            X: Independent variable array
            bw: Bandwidth for the spatial kernel
            y_label: Name of the response variable
            coords: Can be optionally used to provide coordinates for samples- used if subsampling was performed to
                maintain all original sample coordinates (to take original neighborhoods into account)
            mask_indices: Can be optionally used to provide indices of samples to mask out of the dataset
            feature_mask: Can be optionally used to provide a mask for features to mask out of the dataset
            final: Set True to indicate that no additional parameter selection needs to be performed; the model can
                be fit and more stats can be returned.
            fit_predictor: Set True to indicate that dependent variable to fit is a linear predictor rather than a
                true response variable

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
                - betas: Estimated coefficients for sample i
                - leverages: Leverages for sample i, representing the influence of each independent variable on the
                    predicted values (linear predictor for GLMs, response variable for Gaussian regression).
        """
        # Get the index in the original AnnData object for the point in question.
        sample_index = i

        if self.init_betas is not None:
            init_betas = self.init_betas[y_label]
            if not isinstance(init_betas, np.ndarray):
                init_betas = init_betas.values
            if init_betas.ndim == 1:
                init_betas = init_betas.reshape(-1, 1)
        else:
            init_betas = None

        if self.mod_type == "niche" or hasattr(self, "target"):
            if y[i] == 0:
                cov = np.where(y == 0, 1, 0).reshape(-1)
                celltype_i = self.ct_vec[i]
                ct = np.where(self.ct_vec == celltype_i, 1, 0).reshape(-1)
            else:
                cov = None
                celltype_i = self.ct_vec[i]
                ct = np.where(self.ct_vec == celltype_i, 1, 0).reshape(-1)
            expr_mat = None
        else:
            # Distance in "signaling space", conditioned on target expression and cell type:
            if y[i] == 0:
                cov = np.where(y == 0, 1, 0).reshape(-1)
                celltype_i = self.ct_vec[i]
                ct = np.where(self.ct_vec == celltype_i, 1, 0).reshape(-1)
            else:
                cov = None
                ct = None
            expr_mat = self.feature_distance
        wi = get_wi(
            i,
            n_samples=len(X),
            cov=cov,
            ct=ct,
            coords=coords,
            expr_mat=expr_mat,
            fixed_bw=self.bw_fixed,
            kernel=self.kernel,
            bw=bw,
            use_expression_neighbors=self.use_expression_neighbors,
        ).reshape(-1, 1)

        if mask_indices is not None:
            wi[mask_indices] = 0.0
        else:
            mask_indices = []

        if self.distr == "gaussian" or fit_predictor:
            betas, pseudoinverse, inv_cov = compute_betas_local(y, X, wi, clip=self.clip)
            if i in mask_indices:
                betas = np.zeros_like(betas)
                pred_y = 0.0
            else:
                pred_y = np.dot(X[i], betas)

            residual = y[i] - pred_y
            diagnostic = residual
            # Reshape coefficients if necessary:
            betas = betas.flatten()
            # Effect of deleting sample i from the dataset on the estimated predicted value at sample i:
            hat_i = np.dot(X[i], pseudoinverse[:, i])
            # Diagonals of the inverse covariance matrix (used to compute standard errors):
            inv_diag = np.diag(inv_cov)

        elif self.distr == "poisson" or self.distr == "nb":
            betas, y_hat, _, final_irls_weights, _, _, pseudoinverse, fisher_inv = iwls(
                y,
                X,
                distr=self.distr,
                init_betas=init_betas,
                tol=self.tolerance,
                clip=self.clip,
                max_iter=self.max_iter,
                spatial_weights=wi,
                i=i,
                # offset=self.offset,
                link=None,
                ridge_lambda=self.ridge_lambda,
                mask=feature_mask,
            )

            if i in mask_indices:
                betas = np.zeros_like(betas)
                pred_y = 0.0
            else:
                pred_y = y_hat[i]
                # Adjustment for the pseudocount added in preprocessing:
                pred_y -= 1
                pred_y[pred_y < 0] = 0

            diagnostic = pred_y
            if isinstance(diagnostic, np.ndarray):
                diagnostic = pred_y[0]

            # Reshape coefficients if necessary:
            betas = betas.flatten()
            # Effect of deleting sample i from the dataset on the estimated predicted value at sample i:
            hat_i = np.dot(X[i], pseudoinverse[:, i]) * final_irls_weights[i][0]
            # Diagonals of the inverse Fisher matrix (used to compute standard errors):
            inv_diag = np.diag(fisher_inv).reshape(-1)

        else:
            raise ValueError("Invalid `distr` specified. Must be one of 'gaussian', 'poisson', or 'nb'.")

        # Leverages (used to compute standard errors of prediction):
        # CCT = np.diag(np.dot(pseudoinverse, pseudoinverse.T)).reshape(-1)

        if final:
            return np.concatenate(([sample_index, diagnostic, hat_i], betas, inv_diag))
        else:
            # For bandwidth optimization:
            if self.distr == "gaussian" or fit_predictor:
                # bw_diagnostic = residual * residual
                bw_diagnostic = residual
                return [bw_diagnostic, hat_i]
            elif self.distr == "poisson" or self.distr == "nb":
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
        patience = 0
        nan_count = 0
        optimum_score_history = []
        results_dict = {}

        while (np.abs(difference) > self.tolerance and iterations < self.max_iter and patience < 3) or nan_count < 3:
            iterations += 1

            # Bandwidth needs to be discrete:
            if not self.bw_fixed:
                new_lb = np.round(new_lb)
                new_ub = np.round(new_ub)

            if new_lb in results_dict:
                lb_score = results_dict[new_lb]
            else:
                # Return score metric (e.g. AICc) for the lower bound bandwidth:
                lb_score = function(new_lb)
                results_dict[new_lb] = lb_score

            if new_ub in results_dict:
                ub_score = results_dict[new_ub]
            else:
                # Return score metric (e.g. AICc) for the upper bound bandwidth:
                ub_score = function(new_ub)
                results_dict[new_ub] = ub_score

            # Decrease bandwidth until score stops decreasing:
            if ub_score < lb_score or np.isnan(lb_score):
                # Set new optimum score and bandwidth:
                optimum_score = ub_score
                optimum_bw = new_ub

                # Update new max lower bound and test upper bound:
                range_lowest = new_lb
                new_lb = new_ub
                new_ub = range_highest - delta * np.abs(range_highest - range_lowest)

            # Else increase bandwidth until score stops increasing:
            elif lb_score <= ub_score or np.isnan(ub_score):
                # Set new optimum score and bandwidth:
                optimum_score = lb_score
                optimum_bw = new_lb

                # Update new max upper bound and test lower bound:
                range_highest = new_ub
                new_ub = new_lb
                new_lb = range_lowest + delta * np.abs(range_highest - range_lowest)

            difference = lb_score - ub_score

            # Update new value for score:
            score = optimum_score
            optimum_score_history.append(optimum_score)
            most_optimum_score = np.min(optimum_score_history)
            if iterations >= 3:
                if optimum_score_history[-2] == most_optimum_score:
                    patience += 1
                # If score is NaN for three bandwidth iterations, exit optimization:
                elif np.isnan(lb_score) or np.isnan(ub_score):
                    nan_count += 1
                else:
                    nan_count = 0
                    patience = 0
                if self.mod_type != "downstream":
                    if np.abs(optimum_score_history[-2] - optimum_score_history[-1]) <= 0.01 * most_optimum_score:
                        self.logger.info(
                            "Plateau detected (optimum score was reached at last iteration)- exiting "
                            "optimization and returning optimum score up to this point."
                        )
                        self.logger.info(f"Score from last iteration: {optimum_score_history[-2]}")
                        self.logger.info(f"Score from current iteration: {optimum_score_history[-1]}")
                        patience = 3

            # Exit once threshold number of iterations (default to 3) have passed without improvement:
            if patience == 3:
                self.logger.info(f"Returning bandwidth {optimum_bw}")
                return optimum_bw
            if nan_count == 3:
                self.logger.info("Score is NaN for three bandwidth iterations- exiting optimization.")
                return None

        return optimum_bw

    def mpi_fit(
        self,
        y: Optional[np.ndarray],
        X: Optional[np.ndarray],
        X_labels: List[str],
        y_label: str,
        bw: Union[float, int],
        coords: Optional[np.ndarray] = None,
        mask_indices: Optional[np.ndarray] = None,
        feature_mask: Optional[np.ndarray] = None,
        final: bool = False,
        fit_predictor: bool = False,
    ) -> None:
        """Fit local regression model for each sample in parallel, given a specified bandwidth.

        Args:
            y: Response variable
            X: Independent variable array- if not given, will default to :attr `X`. Note that if object was initialized
                using an AnnData object, this will be overridden with :attr `X` even if a different array is given.
            X_labels: Optional list of labels for the features in the X array. Needed if :attr `X` passed to the
                function is not identical to the dependent variable array compiled in preprocessing.
            y_label: Used to provide a unique ID for the dependent variable for saving purposes and to query keys
                from various dictionaries
            bw: Bandwidth for the spatial kernel
            coords: Coordinates of each point in the X array
            mask_indices: Optional array used to mask out indices in the fitting process
            feature_mask: Optional array used to mask out features in the fitting process
            final: Set True to indicate that no additional parameter selection needs to be performed; the model can
                be fit and more stats can be returned.
            fit_predictor: Set True to indicate that dependent variable to fit is a linear predictor rather than a
                true response variable
        """
        if X.shape[1] != self.n_features:
            n_features = X.shape[1]
            X_labels = X_labels
        else:
            n_features = self.n_features
            X_labels = self.feature_names
        n_samples = X.shape[0]

        if self.subsampled:
            true = y[self.x_chunk]
        else:
            true = y

        if final:
            local_fit_outputs = np.empty((self.x_chunk.shape[0], 2 * n_features + 3), dtype=np.float64)

            # Fitting for each location, or each location that is among the subsampled points:
            pos = 0
            # for i in self.x_chunk:
            for pos, i in enumerate(tqdm(self.x_chunk, desc="Fitting using final bandwidth...")):
                local_fit_outputs[pos] = self.local_fit(
                    i,
                    y,
                    X,
                    y_label=y_label,
                    coords=coords,
                    mask_indices=mask_indices,
                    feature_mask=feature_mask,
                    bw=bw,
                    final=final,
                    fit_predictor=fit_predictor,
                )
                pos += 1

            # Gather data to the central process such that an array is formed where each sample has its own
            # measurements:
            # For non-MGWR:
            # Column 0: Index of the sample
            # Column 1: Diagnostic (residual for Gaussian, fitted response value for Poisson/NB)
            # Column 2: Contribution of each sample to its own value
            # Columns 3-n_feats+3: Estimated coefficients
            # Columns n_feats+3-end: Placeholder for standard errors
            # All columns are betas for MGWR
            all_fit_outputs = np.vstack(local_fit_outputs)
            # self.logger.info(f"Computing metrics for GWR using bandwidth: {bw}")

            # Note: trace of the hat matrix and effective number of parameters (ENP) will be used
            # interchangeably:
            ENP = np.sum(all_fit_outputs[:, 2])

            # Residual sum of squares for Gaussian model:
            if self.distr == "gaussian":
                RSS = np.sum(all_fit_outputs[:, 1] ** 2)
                # Total sum of squares:
                TSS = np.sum((true - np.mean(true)) ** 2)
                r_squared = 1 - RSS / TSS

                # Residual variance:
                sigma_squared = RSS / (n_samples - ENP)
                # Corrected Akaike Information Criterion:
                aicc = self.compute_aicc_linear(RSS, ENP, n_samples=X.shape[0])
                # Standard errors of the predictor:
                all_fit_outputs[:, -n_features:] = np.sqrt(all_fit_outputs[:, -n_features:] * sigma_squared)

                # For saving/showing outputs:
                header = "index,residual,influence,"
                deviance = None

                varNames = X_labels
                # Columns for the possible intercept, coefficients and squared canonical coefficients:
                for x in varNames:
                    header += "b_" + x + ","
                for x in varNames:
                    header += "se_" + x + ","

                # Return output diagnostics and save result:
                self.output_diagnostics(aicc, ENP, r_squared, deviance)
                self.save_results(all_fit_outputs, header, label=y_label)

            if self.distr == "poisson" or self.distr == "nb":
                # For negative binomial, first compute the dispersion:
                if self.distr == "nb":
                    dev_resid = self.distr_obj.deviance_residuals(true, all_fit_outputs[:, 1].reshape(-1, 1))
                    residual_deviance = np.sum(dev_resid**2)
                    df = n_samples - ENP
                    self.distr_obj.variance.disp = residual_deviance / df
                # Deviance:
                deviance = self.distr_obj.deviance(true, all_fit_outputs[:, 1].reshape(-1, 1))
                # Log-likelihood:
                # Replace NaN values with 0:
                nan_indices = np.isnan(all_fit_outputs[:, 1])
                all_fit_outputs[nan_indices, 1] = 0
                ll = self.distr_obj.log_likelihood(true, all_fit_outputs[:, 1].reshape(-1, 1))
                # ENP:
                if self.fit_intercept:
                    ENP = n_features + 1
                else:
                    ENP = n_features

                # Corrected Akaike Information Criterion:
                aicc = self.compute_aicc_glm(ll, ENP, n_samples=n_samples)
                # Standard errors of the predictor:
                all_fit_outputs[:, -n_features:] = np.sqrt(all_fit_outputs[:, -n_features:])

                # For saving/showing outputs:
                header = "index,prediction,influence,"
                r_squared = None

                varNames = X_labels
                # Columns for the possible intercept, coefficients and squared canonical coefficients:
                for x in varNames:
                    header += "b_" + x + ","
                for x in varNames:
                    header += "se_" + x + ","

                # Return output diagnostics and save result:
                self.output_diagnostics(aicc, ENP, r_squared, deviance)
                self.save_results(all_fit_outputs, header, label=y_label)

            return

        # If not the final run:
        if self.distr == "gaussian" or fit_predictor:
            # Compute AICc using the sum of squared residuals:
            RSS = 0
            trace_hat = 0
            nan_count = 0

            # for i in self.x_chunk:
            for pos, i in enumerate(tqdm(self.x_chunk, desc="Fitting for each location...")):
                fit_outputs = self.local_fit(
                    i,
                    y,
                    X,
                    y_label=y_label,
                    coords=coords,
                    mask_indices=mask_indices,
                    feature_mask=feature_mask,
                    bw=bw,
                    final=False,
                    fit_predictor=fit_predictor,
                )
                # fit_outputs = np.concatenate(([sample_index, 0.0, 0.0], zero_placeholder, zero_placeholder))
                err, hat_i = fit_outputs[0], fit_outputs[1]
                RSS += err**2
                if np.isnan(hat_i):
                    nan_count += 1
                else:
                    trace_hat += hat_i

            aicc = self.compute_aicc_linear(RSS, trace_hat, n_samples=n_samples)
            self.logger.info(f"Bandwidth: {bw:.3f}, Linear AICc: {aicc:.3f}")
            return aicc

        elif self.distr == "poisson" or self.distr == "nb":
            # Compute AICc using the fitted and observed values:
            nans = np.empty(self.x_chunk.shape[0], dtype=bool)
            trace_hats = np.empty(self.x_chunk.shape[0], dtype=np.float64)
            pos = 0
            y_pred = np.empty(self.x_chunk.shape[0], dtype=np.float64)

            # for i in self.x_chunk:
            for pos, i in enumerate(tqdm(self.x_chunk, desc="Fitting for each location...")):
                fit_outputs = self.local_fit(
                    i,
                    y,
                    X,
                    y_label=y_label,
                    coords=coords,
                    mask_indices=mask_indices,
                    feature_mask=feature_mask,
                    bw=bw,
                    final=False,
                    fit_predictor=fit_predictor,
                )
                y_pred_i, hat_i = fit_outputs[0], fit_outputs[1]
                if np.isnan(hat_i) or np.isnan(y_pred_i):
                    nans[pos] = 1
                y_pred[pos] = y_pred_i
                trace_hats[pos] = hat_i
                pos += 1

            # Send data to the central process:
            all_y_pred = np.array(y_pred).reshape(-1, 1)
            all_trace_hat = np.array(trace_hats).reshape(-1, 1)
            all_nans = np.array(nans).reshape(-1, 1)

            # Diagnostics: mean nonzero value
            pred_test_val = np.mean(all_y_pred[true != 0])
            obs_test_val = np.mean(true[true != 0])

            # For diagnostics, need to ignore NaN values, but also include print statement to indicate how many
            # such elements were ignored:
            mask = ~all_nans
            num_valid = len(mask)
            number_of_nans = np.sum(~mask)
            ll = self.distr_obj.log_likelihood(true[mask], all_y_pred[mask])
            norm_ll = ll / num_valid

            trace_hat = np.sum(all_trace_hat[mask])
            norm_trace_hat = trace_hat / num_valid
            self.logger.info(f"Bandwidth: {bw:.3f}, hat matrix trace: {trace_hat}")
            aicc = self.compute_aicc_glm(norm_ll, norm_trace_hat, n_samples=n_samples)
            self.logger.info(
                f"Bandwidth: {bw:.3f}, LL: {norm_ll:.3f}, GLM AICc: {aicc:.3f}, predicted average nonzero "
                f"value: {pred_test_val:.3f}, observed average nonzero value: {obs_test_val:.3f}"
            )

            return aicc

        return

    def fit(
        self,
        y: Optional[pd.DataFrame] = None,
        X: Optional[np.ndarray] = None,
        fit_predictor: bool = False,
        verbose: bool = True,
    ) -> Optional[Tuple[Union[None, Dict[str, np.ndarray]], Dict[str, float]]]:
        """For each column of the dependent variable array, fit model. If given bandwidth, run :func
        `SWR.mpi_fit()` with the given bandwidth. Otherwise, compute optimal bandwidth using :func
        `SWR.find_optimal_bw()`, minimizing AICc.

        Args:
            y: Optional dataframe, can be used to provide dependent variable array directly to the fit function. If
                None, will use :attr `targets_expr` computed using the given AnnData object to create this (each
                individual column will serve as an independent variable). Needed to be given as a dataframe so that
                column(s) are labeled, so each result can be associated with a labeled dependent variable.
            X: Optional array, can be used to provide dependent variable array directly to the fit function. If
                None, will use :attr `X` computed using the given AnnData object and the type of the model to create.
            n_feat: Optional int, can be used to specify one column of the X array to fit to.
            init_betas: Optional dictionary containing arrays with initial values for the coefficients. Keys should
                correspond to target genes and values should be arrays of shape [n_features, 1].
            fit_predictor: Set True to indicate that dependent variable to fit is a linear predictor rather than a
                response variable
            verbose: Set True to print out information about the bandwidth selection and/or fitting process.
        """

        if not self.set_up:
            self.logger.info("Model has not yet been set up to run, running :func `SWR._set_up_model()` now...")
            self._set_up_model()

        if y is None:
            y_arr = self.targets_expr if hasattr(self, "targets_expr") else self.target
        else:
            y_arr = y
        if X is None:
            X_orig = self.X
        else:
            X_orig = X

        # Cell type indicator:
        if self.group_key is not None:
            cell_types = self.adata.obs[self.group_key]
        else:
            cell_types = pd.Series(["NA"] * len(self.adata.obs), index=self.adata.obs.index)
        cat_to_num = {k: v + 1 for v, k in enumerate(cell_types.unique())}
        self.ct_vec = cell_types.map(cat_to_num).values

        for target in y_arr.columns:
            # Check if model has already been fit for this target:
            # Flag to indicate if a matching file name is found
            found = False
            if self.mod_type == "downstream" and "downstream" not in self.output_path:
                parent_dir = os.path.join(os.path.dirname(self.output_path), "downstream")
            else:
                parent_dir = os.path.dirname(self.output_path)
            file_list = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]
            for filename in file_list:
                parts = filename.split("_")
                if len(parts) > 1 and parts[-1].startswith(target + ".csv"):
                    self.logger.info(f"Model has already been fit for target {target}. Moving on to next target.")
                    found = True
                    break

            if found:
                continue

            y = y_arr[target].values
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # If model is based on ligands/receptors: filter X based on the prior knowledge network:
            if self.mod_type in ["lr", "receptor", "ligand"]:
                if "_" in target:
                    # Presume the gene name is first in the string:
                    gene_query = target.split("_")[0]
                else:
                    gene_query = target
                target_row = self.grn.loc[gene_query]
                target_TFs = target_row[target_row == 1].index.tolist()
                # TFs that can regulate expression of the target-binding TFs:
                primary_tf_rows = self.grn.loc[[tf for tf in target_TFs if tf in self.grn.index]]
                secondary_TFs = primary_tf_rows.columns[(primary_tf_rows == 1).any()].tolist()
                target_TFs = list(set(target_TFs + secondary_TFs))
                if len(target_TFs) == 0:
                    self.logger.info(
                        f"None of the provided regulators could be found to have an association with target gene "
                        f"{target}. Skipping past this target."
                    )
                    continue

                temp = self.r_tf_db[self.r_tf_db["tf"].isin(target_TFs)]
                target_receptors = temp["receptor"].unique().tolist()
                # All of the possible ligands that are partners to any of these receptors:
                lr_db_subset = self.lr_db[self.lr_db["to"].isin(target_receptors)]
                target_ligands = lr_db_subset["from"].unique().tolist()
                target_lr = target_receptors + target_ligands

                if self.mod_type == "lr" or self.mod_type == "receptor":
                    # Keep only the columns of X that contain any of the receptors for this target, or any of the
                    # ligands that can bind to receptors of this target:
                    keep_indices = [i for i, feat in enumerate(self.feature_names) if any(m in feat for m in target_lr)]
                    X_labels = [self.feature_names[idx] for idx in keep_indices]
                    self.logger.info(
                        f"For target {target}, from {len(self.feature_names)} features, "
                        f"kept {len(keep_indices)} to fit model."
                    )
                elif self.mod_type == "ligand":
                    # Ligands that bind to the receptors for this target:
                    target_ligands = []
                    for receptor in target_receptors:
                        filtered_df = self.lr_db[self.lr_db["to"] == receptor]
                        ligands = list(set(filtered_df["from"]))
                        target_ligands.extend(ligands)

                    keep_indices = [
                        i for i, feat in enumerate(self.feature_names) if any(l in feat for l in target_ligands)
                    ]
                    X_labels = [self.feature_names[idx] for idx in keep_indices]
                    self.logger.info(
                        f"For target {target}, from {len(self.feature_names)} features, "
                        f"kept {len(keep_indices)} to fit model."
                    )

                X = X_orig[:, keep_indices]
            # If downstream analysis model, filter X based on known protein-protein interactions:
            elif self.mod_type == "downstream":
                # If target is a complex, or a pathway, look at all rows corresponding to components of the
                # multi-component target:
                if self.adata.uns["target_type"] == "pathway" or "_" in target:
                    gene_query = self.lr_db[self.lr_db["pathway"] == target]["from"].unique().tolist()
                    gene_query = [g for element in gene_query for g in element.split("_")]
                else:
                    gene_query = target

                # Transcription factors that have binding sites proximal to this target:
                target_rows = self.grn.loc[gene_query]

                if not isinstance(target_rows, pd.Series):
                    target_TFs = target_rows.columns[(target_rows == 1).any()].tolist()
                else:
                    target_TFs = target_rows[target_rows == 1].index.tolist()

                # For the given target, subset to the TFs that directly interact with receptors that can regulate the
                # target:
                self.logger.info(
                    f"For target {target}, getting subset of TFs that can interact with "
                    f"target-associated receptors."
                )

                # Access outputs from upstream model from the downstream directory- need to get parent directory of
                # "cci_deg_detection":
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.output_path)))
                if "cci_deg_detection" in parent_dir:
                    parent_dir = os.path.dirname(parent_dir)
                files = os.listdir(parent_dir)
                csv_files = [file for file in files if file.endswith(".csv") and "prediction" not in file]
                if csv_files:
                    first_csv = csv_files[0]
                    # Split the string at underscores and remove the last part (extension)
                    id_tag = first_csv.rsplit("_", 1)[0]
                else:
                    raise FileNotFoundError(
                        f"No files found in the parent directory of the upstream model: {parent_dir}."
                    )

                # For target gene analysis, we want to filter to those TFs that are actually part of the relevant
                # signaling pathway (not all TFs will interact w/ the signaling)- for ligand/receptor analysis,
                # fine to look at all TFs that can putatively regulate expression.
                path = os.path.join(parent_dir, id_tag + f"_{target}.csv")
                targets_df = pd.read_csv(os.path.join(parent_dir, id_tag, "design_matrix", "targets.csv"), index_col=0)
                if target in targets_df.columns:  # i.e. if this gene is among the target genes of the upstream model
                    target_file = pd.read_csv(path, index_col=0)
                    regulators = [c for c in target_file.columns if c.startswith("b_")]
                    regulators = [c.replace("b_", "") for c in regulators]
                    if self.mod_type == "lr":
                        ligands = [r.split(":")[0] for r in regulators]
                    else:
                        ligands = regulators

                    # All of the possible receptors that are partners to these ligands:
                    receptors_for_ligands = self.lr_db[self.lr_db["from"].isin(ligands)]["to"].unique().tolist()
                    tfs_for_receptors = (
                        self.r_tf_db[self.r_tf_db["receptor"].isin(receptors_for_ligands)]["tf"].unique().tolist()
                    )
                    # Filter to the final set of primary TFs
                    target_TFs = [tf for tf in target_TFs if tf in tfs_for_receptors]
                else:
                    if "_" in target:
                        # Presume the gene name is first in the string:
                        gene_query = target.split("_")[0]
                    else:
                        gene_query = target
                    target_row = self.grn.loc[gene_query]
                    target_TFs = target_row[target_row == 1].index.tolist()

                # TFs that can regulate expression of the target-binding TFs:
                primary_tf_rows = self.grn.loc[[tf for tf in target_TFs if tf in self.grn.index]]
                secondary_TFs = primary_tf_rows.columns[(primary_tf_rows == 1).any()].tolist()
                target_TFs = list(set(target_TFs + secondary_TFs))
                if len(target_TFs) == 0:
                    self.logger.info(
                        f"None of the provided regulators could be found to have an association with target gene "
                        f"{target}. Skipping past this target."
                    )
                    continue

                # Cofactors for these transcription factors:
                target_TFs_cof_int = [tf for tf in target_TFs if tf in self.cof_db.columns]
                cof_subset = list(self.cof_db[(self.cof_db[target_TFs_cof_int] == 1).any(axis=1)].index)
                # Other TFs that interact with these transcription factors:
                target_TFs_i_int = [tf for tf in target_TFs if tf in self.tf_tf_db.columns]
                intersecting_tf_subset = list(self.tf_tf_db[(self.tf_tf_db[target_TFs_i_int] == 1).any(axis=1)].index)

                target_regulators = target_TFs + cof_subset + intersecting_tf_subset
                # If there are no features, skip fitting for this gene and move on:
                if len(target_regulators) == 0:
                    self.logger.info(
                        "None of the provided regulators could be found to have an association with the "
                        "target gene. Skipping past this target."
                    )
                    continue

                keep_indices = [i for i, feat in enumerate(self.feature_names) if feat in target_regulators]
                self.logger.info(
                    f"For target {target}, from {len(self.feature_names)} features, kept {len(keep_indices)} to fit "
                    f"model."
                )
                X_labels = [self.feature_names[idx] for idx in keep_indices]
                X = X_orig[:, keep_indices]
            else:
                X_labels = self.feature_names
                X = X_orig.copy()

            # If none of the ligands/receptors/L:R interactions are present in more than threshold percentage of the
            # target-expressing cells, skip fitting for this target:
            if self.mod_type in ["lr", "receptor", "ligand"]:
                y_binary = (y != 0).astype(int)
                X_binary = (X != 0).astype(int)
                concurrence = X_binary & y_binary
                # Calculate the percentage of target-expressing cells for each interaction
                percentages = concurrence.sum(axis=0) / y_binary.sum()
                # Check if any of the percentages are above the threshold
                if all(p <= self.target_expr_threshold for p in percentages):
                    self.logger.info(
                        f"None of the interactions are present in more than "
                        f"{self.target_expr_threshold * 100} percent of cells expressing {target}. "
                        f"Skipping."
                    )
                    continue

            # If subsampled, define the appropriate chunk of the right subsampled array for this process:
            if self.subsampled:
                n_samples = self.n_samples_subsampled[target]
                indices = self.subsampled_indices[target]
                self.x_chunk = np.array(indices)

            feature_mask = None

            # Use y to find the initial appropriate upper and lower bounds for coefficients:
            if self.distr != "gaussian":
                lim = np.log(np.abs(y + 1e-6))
                self.clip = np.percentile(lim, 99.7)
            else:
                self.clip = np.percentile(y, 99.7)

            if self.mod_type == "downstream" and self.bw_fixed:
                if "X_pca" not in self.adata.obsm_keys():
                    self.bw = 0.3
            elif self.mod_type == "downstream" and not self.bw_fixed:
                self.bw = int(0.005 * self.n_samples)

            if self.bw is not None:
                if verbose:
                    self.logger.info(f"Starting fitting process for target {target}. Bandwidth: {self.bw}.")
                # If bandwidth is already known, run the main fit function:
                self.mpi_fit(
                    y,
                    X,
                    X_labels=X_labels,
                    y_label=target,
                    bw=self.bw,
                    coords=self.coords,
                    feature_mask=feature_mask,
                    final=True,
                )
                continue

            if verbose:
                self.logger.info(
                    f"Starting fitting process for target {target}. First finding optimal " f"bandwidth..."
                )
                self._set_search_range()
                self.logger.info(f"Calculated bandwidth range over which to search: {self.minbw}-{self.maxbw}.")

            # Searching for optimal bandwidth- set final=False to return AICc for each run of the optimization
            # function:
            fit_function = lambda bw: self.mpi_fit(
                y,
                X,
                y_label=target,
                X_labels=X_labels,
                bw=bw,
                coords=self.coords,
                feature_mask=feature_mask,
                final=False,
                fit_predictor=fit_predictor,
            )
            optimal_bw = self.find_optimal_bw(self.minbw, self.maxbw, fit_function)
            if optimal_bw is None:
                self.logger.info(f"Issue fitting for target {target}. Skipping.")
                continue
            self.optimal_bw = optimal_bw
            # self.logger.info(f"Discovered optimal bandwidth for {target}: {self.optimal_bw}")
            if self.bw_fixed:
                optimal_bw = round(optimal_bw, 2)

            self.mpi_fit(
                y,
                X,
                y_label=target,
                X_labels=X_labels,
                bw=optimal_bw,
                coords=self.coords,
                feature_mask=feature_mask,
                final=True,
                fit_predictor=fit_predictor,
            )

    def predict(
        self,
        input: Optional[pd.DataFrame] = None,
        coeffs: Optional[Union[np.ndarray, Dict[str, pd.DataFrame]]] = None,
        adjust_for_subsampling: bool = False,
    ) -> pd.DataFrame:
        """Given input data and learned coefficients, predict the dependent variables.

        Args:
            input: Input data to be predicted on.
            coeffs: Coefficients to be used in the prediction. If None, will attempt to load the coefficients learned
                in the fitting process from file.
        """
        if input is None:
            input = self.X_df

        # else:
        #     input_all = input

        if coeffs is None:
            coeffs, _ = self.return_outputs(adjust_for_subsampling=adjust_for_subsampling)

        # If dictionary, compute outputs for the multiple dependent variables and concatenate together:
        if isinstance(coeffs, Dict):
            all_y_pred = pd.DataFrame(index=self.sample_names)
            for target in coeffs:
                if input.shape[0] != coeffs[target].shape[0]:
                    raise ValueError(
                        f"Input data has {input.shape[0]} samples but coefficients for target {target} have "
                        f"{coeffs[target].shape[0]} samples."
                    )

                # Subset to the specific features that were used for this dependent variable:
                feats = [
                    col.split("b_")[1]
                    for col in coeffs[target].columns
                    if col.startswith("b_") and "intercept" not in col
                ]
                feats = [feat for feat in feats if feat in input.columns]
                coeffs[target] = coeffs[target].loc[:, [c for c in coeffs[target].columns if c.split("b_")[1] in feats]]

                y_pred = np.sum(input.loc[:, feats].values * coeffs[target].values, axis=1)
                if self.distr != "gaussian":
                    # Subtract 1 because in the case that all coefficients are zero, np.exp(linear predictor)
                    # will be 1 at minimum, though it should be zero.
                    y_pred = self.distr_obj.predict(y_pred)

                    # Subtract 1 from predictions for predictions to account for the pseudocount from model setup:
                    y_pred -= 1
                    # if self.mod_type != "downstream":
                    #     y_pred[y_pred < 1] = 0.0
                    # else:
                    y_pred[y_pred < 0] = 0.0

                    # thresh = 1.01 if self.normalize else 0
                    # y_pred[y_pred <= thresh] = 0.0

                y_pred = pd.DataFrame(y_pred, index=self.sample_names, columns=[target])
                all_y_pred = pd.concat([all_y_pred, y_pred], axis=1)
            return all_y_pred

        # If coeffs not given as a dictionary:
        else:
            # Subset to the specific features that were used for the dependent variable for which "coeffs" are
            # passed:
            feats = [col.split("_")[1] for col in coeffs.columns if col.startswith("b_") and "intercept" not in col]
            input = input.loc[:, feats]

            if self.distr == "gaussian":
                y_pred_all = input * coeffs
            else:
                y_pred_all_nontransformed = input * coeffs
                y_pred_all = self.distr_obj.predict(y_pred_all_nontransformed)
            y_pred = pd.DataFrame(np.sum(y_pred_all, axis=1), index=self.sample_names, columns=["y_pred"])
            return y_pred

    # ---------------------------------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------------------------------
    def compute_aicc_linear(self, RSS: float, trace_hat: float, n_samples: Optional[int] = None) -> float:
        """Compute the corrected Akaike Information Criterion (AICc) for the linear GWR model."""
        if n_samples is None:
            n_samples = self.n_samples

        aicc = (
            n_samples * np.log(RSS / n_samples)
            + n_samples * np.log(2 * np.pi)
            + n_samples * (n_samples + trace_hat) / (n_samples - trace_hat - 2.0)
        )

        return aicc

    def compute_aicc_glm(self, ll: float, trace_hat: float, n_samples: Optional[int] = None) -> float:
        """Compute the corrected Akaike Information Criterion (AICc) for the generalized linear GWR models. Given by:
        :math AICc = -2*log-likelihood + 2k + (2k(k+1))/(n_eff-k-1).

        Arguments:
            ll: Model log-likelihood
            trace_hat: Trace of the hat matrix
            n_samples: Number of samples model was fitted to
        """
        if n_samples is None:
            n_samples = self.n_samples
        n_eff = n_samples - trace_hat

        aicc = -2 * ll + 2 * self.n_features + (2 * self.n_features * (self.n_features + 1)) / (n_eff - 1)

        return aicc

    def output_diagnostics(
        self,
        aicc: Optional[float] = None,
        ENP: Optional[float] = None,
        r_squared: Optional[float] = None,
        deviance: Optional[float] = None,
        y_label: Optional[str] = None,
    ) -> None:
        """Output diagnostic information about the GWR model."""

        if y_label is None:
            y_label = self.distr

        if aicc is not None:
            self.logger.info(f"Corrected Akaike information criterion for {y_label} model: {aicc}")

        if ENP is not None:
            self.logger.info(f"Effective number of parameters for {y_label} model: {ENP}")

        # Print R-squared for Gaussian assumption:
        if self.distr == "gaussian":
            if r_squared is None:
                raise ValueError(":param `r_squared` must be provided when performing Gaussian regression.")
            self.logger.info(f"R-squared for {y_label} model: {r_squared}")
        # Else log the deviance:
        else:
            if deviance is None:
                raise ValueError(":param `deviance` must be provided when performing non-Gaussian regression.")
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
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))

        directory_path, filename = os.path.split(self.output_path)

        if self.mod_type == "downstream" and "downstream" not in directory_path:
            new_containing_dir = "downstream"
            new_directory_path = os.path.join(directory_path, new_containing_dir)
            # Set new output path:
            self.output_path = os.path.join(new_directory_path, filename)

            if not os.path.exists(new_directory_path):
                os.makedirs(new_directory_path)

        # If output path already has files in it, clear them:
        # output_dir = os.path.dirname(self.output_path)
        # if os.listdir(output_dir):
        #     # If there are files, delete them
        #     for file_name in os.listdir(output_dir):
        #         file_path = os.path.join(output_dir, file_name)
        #         if os.path.isfile(file_path):
        #             os.remove(file_path)

        if label is not None:
            path = os.path.splitext(self.output_path)[0] + f"_{label}" + os.path.splitext(self.output_path)[1]
        else:
            path = self.output_path

        # Save to .csv:
        np.savetxt(path, data, delimiter=",", header=header[:-1], comments="")
        self.saved = True

    def predict_and_save(
        self,
        input: Optional[np.ndarray] = None,
        coeffs: Optional[Union[np.ndarray, Dict[str, pd.DataFrame]]] = None,
        adjust_for_subsampling: bool = True,
    ):
        """Given input data and learned coefficients, predict the dependent variables and then save the output.

        Args:
            input: Input data to be predicted on.
            coeffs: Coefficients to be used in the prediction. If None, will attempt to load the coefficients learned
                in the fitting process from file.
            adjust_for_subsampling: Set True if subsampling was performed; this indicates that the coefficients for
                the subsampled points need to be extended to the neighboring non-sampled points.
        """
        y_pred = self.predict(input, coeffs, adjust_for_subsampling=adjust_for_subsampling)
        # Save to parent directory of the output path:
        parent_dir = os.path.dirname(self.output_path)
        pred_path = os.path.join(parent_dir, "predictions.csv")
        y_pred.to_csv(pred_path)

    def return_outputs(
        self,
        adjust_for_subsampling: bool = True,
        load_from_downstream: Optional[Literal["ligand", "receptor", "target_gene"]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Return final coefficients for all fitted models.

        Args:
            adjust_for_subsampling: Set True if subsampling was performed; this indicates that the coefficients for
                the subsampled points need to be extended to the neighboring non-sampled points.
            load_from_downstream: Set to "ligand", "receptor", or "target_gene" to load coefficients from downstream
                models where targets are ligands, receptors or target genes. Can be used to load downstream model
                coefficients from CCI models.

        Outputs:
            all_coeffs: Dictionary containing dataframe consisting of coefficients for each target gene
            all_se: Dictionary containing dataframe consisting of standard errors for each target gene
        """
        parent_dir = os.path.dirname(self.output_path)
        all_coeffs = {}
        all_se = {}

        if load_from_downstream is not None:
            downstream_parent_dir = os.path.dirname(os.path.splitext(self.output_path)[0])
            if load_from_downstream == "ligand":
                folder = "ligand_analysis"
            elif load_from_downstream == "receptor":
                folder = "receptor_analysis"
            elif load_from_downstream == "target_gene":
                folder = "target_gene_analysis"
            else:
                raise ValueError(
                    "Argument `load_from_downstream` must be one of 'ligand', 'receptor', or 'target_gene'."
                )
            parent_dir = os.path.join(downstream_parent_dir, "cci_deg_detection", folder, "downstream")
            if not os.path.exists(os.path.dirname(parent_dir)):
                self.logger.info(
                    f"Could not find downstream directory {parent_dir}, this type of downstream model "
                    f"may not have been fit. Returning empty dictionaries."
                )
                return {}, {}

        elif self.mod_type == "downstream" and not hasattr(self, "saved"):
            parent_dir = os.path.join(parent_dir, "downstream")
        file_list = [f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))]

        for file in file_list:
            if not "predictions" in file:
                target = file.split("_")[-1][:-4]
                all_outputs = pd.read_csv(os.path.join(parent_dir, file), index_col=0)
                betas = all_outputs[[col for col in all_outputs.columns if col.startswith("b_")]]
                feat_sub = [col.replace("b_", "") for col in betas.columns]
                if isinstance(betas.index[0], int) or isinstance(betas.index[0], float):
                    betas.index = [self.X_df.index[idx] for idx in betas.index]
                standard_errors = all_outputs[[col for col in all_outputs.columns if col.startswith("se_")]]
                if isinstance(standard_errors.index[0], int) or isinstance(standard_errors.index[0], float):
                    standard_errors.index = [self.X_df.index[idx] for idx in standard_errors.index]

                if adjust_for_subsampling:
                    # If subsampling was performed, extend coefficients to non-sampled neighboring points (only if
                    # subsampling is not done by cell type group):
                    _, filename = os.path.split(self.output_path)
                    filename = os.path.splitext(filename)[0]

                    if os.path.exists(os.path.join(parent_dir, "subsampling", f"{filename}.json")):
                        with open(os.path.join(parent_dir, "subsampling", f"{filename}.json"), "r") as dict_file:
                            neighboring_unsampled = json.load(dict_file)

                        sampled_to_nonsampled_map = neighboring_unsampled[target]
                        betas = betas.reindex(self.sample_names, columns=betas.columns, fill_value=0)
                        standard_errors = standard_errors.reindex(
                            self.sample_names, columns=standard_errors.columns, fill_value=0
                        )
                        for sampled_idx, nonsampled_idxs in sampled_to_nonsampled_map.items():
                            for nonsampled_idx in nonsampled_idxs:
                                betas.loc[nonsampled_idx] = betas.loc[sampled_idx]
                                standard_errors.loc[nonsampled_idx] = standard_errors.loc[sampled_idx]
                                standard_errors.loc[nonsampled_idx] = standard_errors.loc[sampled_idx]

                        # If this cell does not express the receptor(s) or doesn't have the ligand in neighborhood,
                        # mask out the relevant element in "betas" and standard errors- specifically for ligand and
                        # receptor models, do not infer expression in cells that do not express the target because it
                        # is unknown whether the ligand/receptor (the half of the interacting pair that is missing) is
                        # present in the neighborhood of these cells:
                        if self.mod_type in ["receptor", "ligand", "downstream"]:
                            mask_matrix = (self.adata[:, target].X != 0).toarray().astype(int)
                            betas *= mask_matrix
                            standard_errors *= mask_matrix
                        mask_df = (self.X_df != 0).astype(int)
                        mask_df = mask_df.loc[:, [g for g in mask_df.columns if g in feat_sub]]
                        for col in betas.columns:
                            if col.replace("b_", "") not in mask_df.columns:
                                mask_df[col] = 0
                        # Make sure the columns are in the same order:
                        betas_columns = [col.replace("b_", "") for col in betas.columns]
                        mask_df = mask_df.reindex(columns=betas_columns)
                        mask_matrix = mask_df.values
                        betas *= mask_matrix
                        standard_errors *= mask_matrix

                    # Concatenate coefficients and standard errors to re-associate each row with its name in the AnnData
                    # object, save back to file path:
                    all_outputs = pd.concat([betas, standard_errors], axis=1)
                    all_outputs.to_csv(os.path.join(parent_dir, file))

                # Save coefficients and standard errors to dictionary:
                all_coeffs[target] = betas
                all_se[target] = standard_errors

        return all_coeffs, all_se

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
