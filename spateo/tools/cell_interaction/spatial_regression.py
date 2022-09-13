"""
Suite of tools for spatially-aware as well as spatially-lagged linear regression

Also performs downstream characterization following spatially-informed regression to characterize niche impact on gene
expression

Developer note: may change from OLS to TSLS (the nonspatial version through spreg...) in the future. At least for now
will stick w/ the OLS version, though
Developer note: haven't been able to find a good network for Drosophila and Zebrafish yet, so spatially lagged models
are restricted to human, mouse and axolotl. Might also be making a lot of assumptions about the axolotl,
but the axolotl LR network has columns for human equivalents for all LR so I assumed there's enough homology there
Developer note: for the sender/receiver effect functions, each row in pvalues, fold_change, etc. is a receiver,
each column is a sender
"""
import os
import time
from itertools import product
from random import sample
from typing import List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from anndata import AnnData
from dynamo.tools.moments import calc_1nd_moment
from joblib import Parallel, delayed
from matplotlib import rcParams
from patsy import dmatrix
from pysal.lib import weights
from pysal.model import spreg

from ...configuration import SKM, config_spateo_rcParams, set_pub_style
from ...plotting.static.utils import save_return_show_fig_utils

from ...logging import logger_manager as lm
from ...preprocessing.normalize import normalize_total
from ...preprocessing.transform import log1p
from ...tools.find_neighbors import construct_pairwise, transcriptomic_connectivity
from ...tools.utils import update_dict
from .regression_utils import compute_wald_test, get_fisher_inverse, ols_fit_predict


# --------------------------------------- Wrapper classes for model running --------------------------------------- #
class BaseInterpreter:
    """
    Basis class for all spatially-aware and spatially-lagged regression models that can be implemented through this
    toolkit. Includes necessary methods for data loading and preparation, computation of spatial weights matrices,
    computation of evaluation metrics and more.

    Args:
        adata : class `anndata.AnnData`
        group_key : str
            Key in .obs where group (e.g. cell type) information can be found
        spatial_key : str, default "spatial"
            Key in .obsm where x- and y-coordinates are stored
        genes : optional list
            Subset to genes of interest: will be used as dependent variables in non-ligand-based regression analyses,
            will be independent variables/exogenous variables
        drop_dummy : optional str
            Name of the category to be dropped (the "dummy variable") in the regression. The dummy category can aid
            in interpretation as all model coefficients can be taken to be in reference to the dummy category. If
            None, will randomly select a few samples to constitute the dummy group.
        layer : optional str
            Entry in .layers to use instead of .X
        cci_dir : optional str
            Full path to the directory containing cell-cell communication databases. Only used in the case of models
            that use ligands for prediction.
        smooth : bool, default False
            To correct for dropout effects, leverage gene expression neighborhoods to smooth expression
        log_transform : bool, default False
            Set True if log-transformation should be applied to expression (otherwise, will assume
            preprocessing/log-transform was computed beforehand)
        weights_mode : str, default "knn"
            Options "knn", "kernel", "band"; sets whether to use K-nearest neighbors, a kernel-based method, or distance
            band to compute spatial weights, respectively.
        data_id : optional str
            If given, will save pairwise distance arrays & nearest neighbor arrays to folder in the working
            directory, under './neighbors/{data_id}_distance.csv' and './neighbors/{data_id}_adj.csv'. Will also
            check for existing files under these names to avoid re-computing these arrays. If not given, will not save.
        kwargs : optional dict
            Provides additional spatial weight-finding arguments. Note that these must specifically match the name
            that the function will look for (case sensitive). For reference:
                - n_neighbors : int
                    Number of nearest neighbors for KNN
                - p : int
                    Minkowski p-norm for KNN and distance band methods
                - distance_metric : str
                    Pairwise distance metric for KNN
                - bandwidth : float or array-like of floats
                    Sets kernel width for kernel method
                - fixed : bool
                    Allow bandwidth to vary across observations for kernel method
                - n_neighbors_bandwidth : int
                    Number of nearest neighbors for determining bandwidth for kernel method
                - kernel_function : str
                    "triangular", "uniform", "quadratic", "quartic" or "gaussian". Rule for setting how spatial
                    weight decays with distance
                - threshold : float
                    Distance for which to consider spots "neighbors" for each spot in distance band method (typically
                    in units of pixels)
                - alpha : float
                    Should be less than 0; can be used to set weights to decay with distance for distance band method
    """

    def __init__(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        group_key: Union[None, str] = None,
        genes: Union[None, List] = None,
        drop_dummy: Union[None, str] = None,
        layer: Union[None, str] = None,
        cci_dir: Union[None, str] = None,
        smooth: bool = False,
        log_transform: bool = False,
        weights_mode: str = "knn",
        data_id: Union[None, str] = None,
        **kwargs,
    ):
        self.adata = adata
        self.cell_names = self.adata.obs_names
        # Sort cell type categories (to keep order consistent for downstream applications):
        self.celltype_names = sorted(list(set(adata.obs[group_key])))

        self.spatial_key = spatial_key
        self.group_key = group_key
        self.genes = genes
        self.drop_dummy = drop_dummy
        self.layer = layer
        self.cci_dir = cci_dir
        self.smooth = smooth
        self.log_transform = log_transform
        self.weights_mode = weights_mode
        self.data_id = data_id

        # Kwargs can be used to adjust how spatial weights/spatial neighbors are found.
        # Default values for all possible spatial weight/spatial neighbor parameters:
        self.sp_kwargs = {
            "n_neighbors": 10,
            "p": 2,
            "distance_metric": "euclidean",
            "bandwidth": None,
            "fixed": True,
            "n_neighbors_bandwidth": 6,
            "kernel_function": "triangular",
            "threshold": 50,
            "alpha": -1.0,
        }

        # Update using user input:
        self.sp_kwargs = update_dict(self.sp_kwargs, kwargs)

        self.logger = lm.get_main_logger()

        # Define reconstruction metrics:
        self.metrics = [mae, mse, nll, r_squared]

    def prepare_data(
            self,
            mod_type: str = "category",
            ligands: Union[None, List[str]] = None,
            receiving_genes: Union[None, List[str]] = None,
            species: Literal["human", "mouse", "axolotl"] = "human"
    ):
        """
        Handles any necessary data preparation, starting from given source AnnData object

        Args:
            mod_type : str, default "category"
                The type of model that will be employed- this dictates how the data will be processed and prepared.
                Options:
                    - category: spatially-aware, for each sample, computes category prevalence within the spatial
                    neighborhood and uses these as independent variables
                    - connections: spatially-aware, uses spatial connections between samples as independent variables
                    - niche: spatially-aware, uses both categories and spatial connections between samples as
                    independent variables
                    - lag_category: (NOT YET IMPLEMENTED) spatially-lagged *auto*regressive model- considers neighbor
                    expression of the dependent variable, uses neighborhood category prevalence for the samples as
                    independent variables
                    - het_lag: (NOT YET IMPLEMENTED) spatially-lagged, but considers neighbor expression of other
                    variables that are not the dependent
                    - ligand_lag: spatially-lagged, from database uses select ligand genes to perform regression on
                    select receptor and/or receptor-downstream genes, and additionally considers neighbor expression of
                    the ligands
                    - ligand_connection_lag: spatially-lagged, uses a combination of spatial connections,
                    ligand genes and ligand gene expression of the neighbors to perform regression on select receptor
                    and/or receptor-downstream genes
                    - ligand_niche_lag: spatially-lagged, uses a combination of categories, spatial connections,
                    ligand genes and ligand gene expression of the neighbors to perform regression on select
                    receptor and/or receptor-downstream genes
            ligands : optional list of str
                Only used if 'mod_type' contains "ligand". Provides the list of ligands to use as predictors. If not
                given, will attempt to subset self.genes
            receiving_genes : optional list of str
                Only used if 'mod_type' contains "ligand". Provides the list of receptor and/or receptor-downstream
                genes to investigate. If not given, will search through self.genes (that was provided on
                instantiating the object) for genes that correspond to the provided genes from 'ligands'.
            species : str, default "human"
                Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse", "axolotl" (EVENTUALLY, WILL ALSO INCLUDE DROSOPHILA/ZEBRAFISH)
        """
        if "category" in mod_type:
            self.logger.info(f"Using {self.group_key} values to predict feature expression...")
            # First, convert groups/categories into one-hot array:
            group_num = self.adata.obs[self.group_key].value_counts()
            max_group, min_group, min_group_ncells = (
                group_num.index[0],
                group_num.index[-1],
                group_num.values[-1],
            )

            group_name = self.adata.obs[self.group_key]
            db = pd.DataFrame({"group": group_name})
            categories = np.array(self.adata.obs[self.group_key].unique().tolist() + ["others"])
            db["group"] = pd.Categorical(db["group"], categories=categories)

            # Solve the dummy variable trap by dropping dummy category (deleting rows to avoid issues with
            # multicollinearity):
            if self.drop_dummy is None:
                # Leave some samples from every group intact
                db.iloc[sample(np.arange(self.adata.n_obs).tolist(), min_group_ncells), :] = "others"
            elif self.drop_dummy in categories:
                group_inds = np.where(db["group"] == self.drop_dummy)[0]
                db.iloc[group_inds, :] = "others"
                db = db["group"].cat.remove_unused_categories()
            else:
                raise ValueError(
                    f"Dummy category ({self.drop_dummy}) provided is not in the " f"adata.obs[{self.group_key}]."
                )
            drop_columns = ["group_others"]

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)
            # Compute adjacency matrix- use the KNN value in 'sp_kwargs' (which may have been passed as an
            # argument when initializing the interpreter):
            if self.data_id is not None:
                self.logger.info(f"Checking for pre-computed adjacency matrix for dataset {self.data_id}...")
                try:
                    self.adata.obsm["adj"] = pd.read_csv(
                        os.path.join(os.getcwd(), f"neighbors/{self.data_id}_neighbors.csv"), index_col=0
                    ).values
                    self.logger.info(f"Adjacency matrix loaded from file.")
                except:
                    self.logger.info(f"Pre-computed adjacency matrix not found. Computing adjacency matrix.")
                    start = time.time()
                    construct_pairwise(
                        self.adata,
                        spatial_key=self.spatial_key,
                        n_neighbors=self.sp_kwargs["n_neighbors"],
                        exclude_self=True,
                    )
                    self.logger.info(f"Computed adjacency matrix, time elapsed: {time.time()-start}s.")
            adj = self.adata.obsm["adj"]

            # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
            # each category are present within the neighborhood of each sample):
            dmat_neighbors = (adj > 0).astype("int").dot(X.values)
            self.X = pd.DataFrame(dmat_neighbors, columns=X.columns)
            self.X = self.X.reindex(sorted(self.X.columns), axis=1)
            self.n_features = self.X.shape[1]

            # To index all but the dummy column when fitting model:
            self.variable_names = self.X.columns.difference(drop_columns).to_list()

            # Get the names of all remaining groups:
            self.param_labels, group_name = (
                set(group_name).difference([self.drop_dummy]),
                group_name.to_list(),
            )

            self.param_labels = list(np.sort(list(self.param_labels)))

        elif mod_type == "connections" or mod_type == "niche":
            # Convert groups/categories into one-hot array:
            group_name = self.adata.obs[self.group_key]
            db = pd.DataFrame({"group": group_name})
            categories = np.array(self.adata.obs[self.group_key].unique().tolist())
            db["group"] = pd.Categorical(db["group"], categories=categories)

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)

            # Compute adjacency matrix- use the KNN value in 'sp_kwargs' (which may have been passed as an
            # argument when initializing the interpreter):
            construct_pairwise(
                self.adata, spatial_key=self.spatial_key, n_neighbors=self.sp_kwargs["n_neighbors"], exclude_self=True
            )
            adj = self.adata.obsm["adj"]

            # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
            # each category are present within the neighborhood of each sample):
            dmat_neighbors = (adj > 0).astype("int").dot(X.values)

            # Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the niche of
            # each sample by documenting the category-category spatial connections within the niche- specifically,
            # for each sample, records the category identity of its neighbors in space):
            data = {"categories": X, "dmat_neighbours": dmat_neighbors}
            connections = np.asarray(dmatrix("categories:dmat_neighbours-1", data))

            # If mod_type is 'connections', use only the connections matrix as independent variables in the regression:
            if mod_type == "connections":
                connections_cols = list(product(X.columns, X.columns))
                connections_cols.sort(key=lambda x: x[1])
                connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
                self.X = pd.DataFrame(connections, columns=connections_cols)
            # Otherwise if 'niche', combine two arrays:
            # connections, encoding pairwise *spatial adjacencies/spatial prevalence* between categories for each
            # sample, and
            # categories, encoding *identity* of the niche components
            elif mod_type == "niche":
                category_df = pd.DataFrame(categories, columns=X.columns)

                connections_cols = list(product(X.columns, X.columns))
                connections_cols.sort(key=lambda x: x[1])
                connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
                connections_df = pd.DataFrame(connections, columns=connections_cols)
                self.X = pd.concat([category_df, connections_df], axis=1)

            # Set self.param_labels to reflect inclusion of the interactions in the case that connections are used:
            uniq_cats = set(group_name)
            self.param_labels = list(np.sort(list(uniq_cats))) + connections_cols

        elif mod_type == "ligand_lag":
            # Load signaling network file and find appropriate subset (if 'species' is axolotl, use the human portion
            # of the network):
            signet = pd.read_csv(os.path.join(self.cci_dir, "human_mouse_signaling_network.csv"), index_col=0)
            if species not in ["human", "mouse", "axolotl"]:
                self.logger.error("Invalid input to 'species'. Options: 'human', 'mouse', 'axolotl'.")
            if species == "axolotl":
                species = "human"
            sig_set = signet[signet['species'] == species.title()]

            # Set predictors and target- for consistency with field conventions, set ligands and ligand-downstream
            # gene names to uppercase (the AnnData object is assumed to follow this convention as well):
            # Use the argument provided to 'ligands' to set the predictor block:
            if ligands is None:
                ligands = [g.upper() for g in self.genes if g in sig_set['src']]
                ligands = self.adata[:, ligands].X.toarray() if scipy.sparse.issparse(self.adata.X)
                self.X = pd.DataFrame(self.adata.X)

            if receiving_genes is None:
                "filler"


            # Set

            # Notes to self:
            # self.genes = array containing all variables to be used as dependent- to put this together,
            # compute spatial lag
            # self.param_labels = if niche lag model, to get this list, combine connections column names and ligand
            # names
            # self.X = array containing all independent variables

        elif mod_type == "ligand_connections_lag":
            "filler"
        else:
            self.logger.error("Invalid argument to 'mod_type'.")

        # Save model type as an attribute so it can be accessed by other methods:
        self.mod_type = mod_type

        # Normalize to size factor:
        normalize_total(self.adata)

        # Smooth data if 'smooth' is True and log-transform data matrix if 'log_transform' is True:
        if self.smooth:
            self.logger.info("Smoothing gene expression...")
            # Compute connectivity matrix if not already existing:
            try:
                conn = self.adata.obsp['connectivities']
            except:
                _, adata = transcriptomic_connectivity(self.adata, n_neighbors_method="ball_tree")
                conn = adata.obsp['connectivities']
            adata_smooth_norm, _ = calc_1nd_moment(self.adata.X, conn, normalize_W=True)
            self.adata.layers["M_s"] = adata_smooth_norm

            # Use smoothed layer for downstream processing:
            self.adata.layers["raw"] = self.adata.X
            self.adata.X = self.adata.layers["M_s"]

        # Filter gene names if specific gene names are provided. If not, use all genes referenced in .X:
        if self.genes is not None:
            self.genes = self.adata.var.index.intersection(self.genes)
        else:
            self.genes = self.adata.var.index
        self.adata = self.adata[:, self.genes]

        if self.log_transform:
            if self.layer is None:
                log1p(self.adata)
            else:
                log1p(self.adata.layers[self.layer])
        else:
            self.logger.warn(
                "Linear regression models are not well suited to the distributional characteristics of "
                "raw transcriptomic data- it is recommended to perform a log-transformation first."
            )

    def compute_spatial_weights(self):
        """
        Generates matrix of pairwise spatial distances, used in spatially-lagged models
        """
        # Choose how weights are computed:
        if self.weights_mode == "knn":
            self.w = weights.distance.KNN.from_array(self.adata.obsm[self.spatial_key], k=self.sp_kwargs["n_neighbors"])
        elif self.weights_mode == "kernel":
            self.w = weights.distance.Kernel.from_array(
                self.adata.obsm[self.spatial_key],
                bandwidth=self.sp_kwargs["bandwidth"],
                fixed=self.sp_kwargs["fixed"],
                k=self.sp_kwargs["n_neighbors_bandwidth"],
                function=self.sp_kwargs["kernel_function"],
            )
        elif self.weights_mode == "band":
            self.w = weights.distance.DistanceBand.from_array(
                self.adata.obsm[self.spatial_key], threshold=self.sp_kwargs["threshold"], alpha=self.sp_kwargs["alpha"]
            )
        else:
            self.logger.error("Invalid input to 'weights_mode'. Options: 'knn', 'kernel', 'band'.")

        # Row standardize spatial weights matrix:
        self.w.transform = "R"

    # ------------------------------ Parameters for spatially-aware and lagged models ------------------------------ #
    def run_OLS(self, n_jobs: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Wrapper for ordinary least squares regression.

        n_jobs : int, default 30
            For parallel processing, number of tasks to run at once

        Returns:
            coeffs : pd.DataFrame, shape [n_features, n_params]
                Contains fitted parameters for each feature
            reconst: pd.DataFrame, shape [n_samples, n_features]
                Contains predicted expression for each feature
        """
        results = Parallel(n_jobs)(
            delayed(ols_fit_predict)(self.X, self.adata, self.variable_names, cur_g) for cur_g in self.genes
        )
        coeffs = [item[0] for item in results]
        reconst = [item[1] for item in results]

        coeffs = pd.DataFrame(coeffs, index=self.genes, columns=self.X.columns)
        for cn in coeffs.columns:
            self.adata.var.loc[:, cn] = coeffs[cn]
        # Nested list transforms into dataframe rows- instantiate and transpose to get to correct shape:
        reconst = pd.DataFrame(reconst, index=self.genes, columns=self.cell_names).T
        return coeffs, reconst

    def run_GM_lag(self, n_jobs: int = 30):
        """Runs spatially lagged two-stage least squares model"""

        def _single(
            cur_g: str,
            X: pd.DataFrame,
            X_variable_names: List[str],
            uniq_g: List[str],
            adata: AnnData,
            w: np.ndarray,
            layer: Union[None, str] = None,
        ):
            """
            Defines model run process for a single feature- not callable by the user, all arguments populated by
            arguments passed on instantiation of Estimator.

            Args:
                cur_g : str
                    Name of the feature to regress on
                X : pd.DataFrame
                    Values used for the regression
                X_variable_names : list of str
                    Names of the variables used for the regression
                uniq_g : list of str
                    Names of categories- each computed parameter corresponds to a single element in uniq_g
                adata : class `anndata.AnnData`
                    AnnData object to store results in
                w : np.ndarray
                    Spatial weights array
                layer : optional str
                    Specifies layer in AnnData to use- if None, will use .X.

            Returns:
                coeffs : pd.DataFrame
                    Coefficients for each categorical group for each feature
                pred : pd.DataFrame
                    Predicted values from regression for each feature
                resid : pd.DataFrame
                    Residual values from regression for each feature
            """
            if layer is None:
                X["log_expr"] = adata[:, cur_g].X.A
            else:
                X["log_expr"] = adata[:, cur_g].layers[layer].A

            try:
                model = spreg.GM_Lag(
                    X[["log_expr"]].values,
                    X[X_variable_names].values,
                    w=w,
                    name_y="log_expr",
                    name_x=X_variable_names,
                )
                self.logger.info("Printing model summary: \n", model.summary)
                y_pred = model.predy
                resid = model.u

                # Coefficients for each cell type:
                a = pd.DataFrame(model.betas, model.name_x + ["W_log_exp"], columns=["coef"])
                b = pd.DataFrame(
                    model.z_stat,
                    model.name_x + ["W_log_exp"],
                    columns=["z_stat", "p_val"],
                )

                df = a.merge(b, left_index=True, right_index=True)

                for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                    adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = df.iloc[ind, 0]
                    adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = df.iloc[ind, 1]
                    adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = df.iloc[ind, 2]

            except:
                y_pred = np.full((X.shape[0],), np.nan)
                resid = np.full((X.shape[0],), np.nan)

                for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                    adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = np.nan
                    adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = np.nan
                    adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = np.nan

            # Outputs for a single gene:
            return adata.var.loc[cur_g, :].values, y_pred, resid

        # Wrap regressions for all single genes:
        results = Parallel(n_jobs)(
            delayed(_single)(
                cur_g, self.X, self.variable_names, self.param_labels, self.adata, self.w, self.layer
            )
            for cur_g in self.genes
        )

        coeffs = [item[0] for item in results]
        pred = [item[1] for item in results]
        resid = [item[2] for item in results]

        # Coefficients and their significance:
        coeffs = pd.DataFrame(coeffs, index=self.genes)
        coeffs.columns = self.adata.var.loc[self.genes, :].columns

        pred = pd.DataFrame(np.hstack(pred), index=self.adata.obs_names, columns=self.genes)
        resid = pd.DataFrame(np.hstack(resid), index=self.adata.obs_names, columns=self.genes)

        # Update AnnData object:
        self.adata.obsm["ypred"] = pred
        self.adata.obsm["resid"] = resid

        for cn in coeffs.columns:
            self.adata.var.loc[:, cn] = coeffs[cn]

        return coeffs, pred, resid


    # ------------------------- Downstream interpretation (mostly for interaction models) ------------------------- #
    def visualize_params(self, coeffs: pd.DataFrame):
        """
        Generates heatmap of parameter values for visualization

        coeffs :
            Contains coefficients from regression for each variable
        """

    def get_sender_receiver_effects(self, coeffs: pd.DataFrame, significance_threshold: float = 0.05):
        """
        For each predictor and each feature, determine if the influence of said predictor in predicting said feature is
        significant.

        Additionally, if the connections b/w categories are used as variables for regression,
        for each feature and each sender-receiver category pair, determines the log fold-change that the sender
        induces in the feature for the receiver.

        Only valid if the model specified uses the connections between categories as variables for the regression.

        Args:
            coeffs : pd.DataFrame
                Contains coefficients from regression for each variable
                            significance_threshold : float, default 0.5
                p-value needed to call a sender-receiver relationship significant
        """
        if not "connections" in self.mod_type or not "niche" in self.mod_type:
            self.logger.error(
                "Type coupling analysis only valid if connections between categories are used as the "
                "predictor variable."
            )

        # Return only the numerical coefficients:
        coeffs = coeffs[[col for col in coeffs.columns if "coeff" in col]].values
        if "lag" in self.mod_type:
            # Remove the first column (the intercept):
            coeffs = coeffs[:, 1:]

        # Get inverse Fisher information matrix, with the y block containing all features that were used in regression):
        y = self.adata[:, self.genes].X
        inverse_fisher = get_fisher_inverse(self.X.values, y)

        # Compute significance for each parameter:
        is_significant, pvalues, qvalues = compute_wald_test(
            params=coeffs, fisher_inv=inverse_fisher, significance_threshold=significance_threshold
        )

        # If niche-based model, extract the portion that corresponds to the interaction terms:
        if self.mod_type == "niche":
            interaction_shape = np.int(self.n_features**2)
            is_significant = is_significant[self.n_features : interaction_shape + self.n_features, :]
            pvalues = pvalues[self.n_features : interaction_shape + self.n_features, :]
            qvalues = qvalues[self.n_features : interaction_shape + self.n_features, :]

            # Compute the fold-change induced in the receiver by the sender for the case model type is "niche":
            interaction_params = coeffs[:, self.n_features : interaction_shape + self.n_features]
            # Split array such that an nxn matrix is created, where n is 'n_features' (the number of cell type
            # categories)
            self.fold_change = np.concatenate(
                np.expand_dims(
                    np.split(interaction_params.T, indices_or_sections=np.sqrt(interaction_params.T.shape[0]), axis=0),
                    axis=0,
                ),
                axis=0,
            )
        # Else if connection-based model, all regression coefficients already correspond to the interaction terms:
        else:
            self.fold_change = np.concatenate(
                np.expand_dims(np.split(coeffs.T, indices_or_sections=np.sqrt(coeffs.T.shape[0]), axis=0), axis=0),
                axis=0,
            )

        # Split array such that an nxn matrix is created, where n is 'n_features' (the number of cell type
        # categories)
        self.pvalues = np.concatenate(
            np.expand_dims(np.split(pvalues, indices_or_sections=np.sqrt(pvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.qvalues = np.concatenate(
            np.expand_dims(np.split(qvalues, indices_or_sections=np.sqrt(qvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.is_significant = np.concatenate(
            np.expand_dims(
                np.split(is_significant, indices_or_sections=np.sqrt(is_significant.shape[0]), axis=0), axis=0
            ),
            axis=0,
        )

    def type_coupling_analysis(
        self,
        cmap: str = "Reds",
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[int, int]] = None,
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
        save_id: Union[None, str] = None,
        save_kwargs: dict = {},
    ):
        """
        Generates heatmap of spatially differentially-expressed features for each pair of sender and receiver
        categories. Only valid if the model specified uses the connections between categories as variables for the
        regression.

        A high number of differentially-expressed genes between a given sender-receiver pair means that the sender
        being in the neighborhood of the receiver tends to correlate with differential expression levels of many of
        the genes within the selection- much of the cellular variation in the receiver cell type can be attributed to
        being in proximity with the sender.

        Args:
            cmap : str, default "Reds"
                Name of Matplotlib color map to use
            fontsize : optional int
                Size of figure title and axis labels
            figsize : optional tuple of form (int, int)
                Width and height of plotting window
            save_show_or_return : str
            Options: "save", "show", "return", "both", "all"
                - "both" for save and show
            save_id : optional str
                Name of the saved figure, without the extension
            save_kwargs : dict
                A dictionary that will passed to the save_fig function. By default it is an empty dictionary and the
                save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf',
                "transparent": True, "close": True, "verbose": True} as its parameters. But to change any of these
                parameters, this dictionary can be used to do so.
        """
        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        path = os.path.join(os.getcwd(), "/figures", save_id)
        # Update save_kwargs with save path:
        save_kwargs["path"] = path

        if not hasattr(self, "is_sign"):
            self.logger.warn("Significance dataframe does not exist- please run `get_sender_receiver_effects` first.")

        sig_df = pd.DataFrame(np.sum(self.is_sign, axis=-1), columns=self.cell_names, index=self.cell_names)

        np.fill_diagonal(sig_df.values, 0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.heatmap(
            sig_df,
            square=True,
            linecolor='grey',
            linewidths=0.3,
            cmap=cmap
        )
        plt.xlabel("Sender")
        plt.ylabel("Receiver")
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="type_coupling",
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def sender_effect(
        self,
        receiver: str,
        plot_mode: str = "fold_change",
        gene_subset: Union[None, List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = 'seismic',
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {}
    ):
        """
        Evaluates and visualizes the effect that each sender cell type has on specific genes in the receiver
        cell type.

        Args:
            receiver : str
                Receiver cell type label
            plot_mode : str, default "fold_change"
                Options:
                    - "qvals": elements of the plot represent statistical significance of the interaction
                    - "fold_change": elements of the plot represent fold change induced in the receiver by the sender
            gene_subset : optional list of str
                Names of genes to subset for plot. If None, will use all genes that were used in the regression.
            significance_threshold : float, default 0.05
                Set non-significant fold changes to zero, where the threshold is given here
            cut_pvals : float, default -5
                Minimum allowable log10(pval)- anything below this will be clipped to this value
            fontsize : optional int
                Size of figure title and axis labels
            figsize : optional tuple of form (int, int)
                Width and height of plotting window
            cmap : str, default "seismic"
                Name of matplotlib colormap specifying colormap to use
            save_show_or_return: Literal["save", "show", "return", "both", "all"], default "show"
                Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs : optional dict
                A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        receiver_idx = self.celltype_names.index(receiver)

        if plot_mode == 'qvals':
            # In the analysis process, the receiving cell types become aligned along the column axis:
            arr = np.log(self.qvalues[receiver_idx, :, :].copy())
            arr[arr < cut_pvals] = cut_pvals
            df = pd.DataFrame(
                arr,
                index=self.celltype_names,
                columns=self.genes
            )
            if gene_subset:
                df = df.drop(index=receiver)[gene_subset]

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor='grey',
                linewidths=0.3,
                cbar_kws={'label': "$\log_{10}$ FDR-corrected pvalues"},
                cmap=cmap, vmin=-5, vmax=0.
            )
        elif plot_mode == 'fold_change':
            arr = self.fold_change[receiver_idx, :, :].copy()
            arr[np.where(self.qvalues[receiver_idx, :, :] > significance_threshold)] = 0
            df = pd.DataFrame(
                arr,
                index=self.celltype_names,
                columns=self.genes
            )
            if gene_subset:
                df = df.drop(index=receiver)[gene_subset]
            vmax = np.max(np.abs(df.values))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor='grey',
                linewidths=0.3,
                cbar_kws={'label': "$\ln$ fold change",
                          "location": "top"},
                cmap=cmap, vmin=-vmax, vmax=vmax,
            )
        else:
            logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'fold_change'.")

        plt.xlabel("Sender cell type")
        plt.title("Sender Effects on " + receiver)
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="sender_effects_on_{}".format(receiver),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def receiver_effect(
        self,
        sender: str,
        plot_mode: str = "fold_change",
        gene_subset: Union[None, List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = 'seismic',
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {}
    ):
        """
        Evaluates and visualizes the effect that one specific sender cell type has on select genes in all possible
        receiver cell types.

        Args:
            sender : str
                Sender cell type label
            plot_mode : str, default "fold_change"
                Options:
                    - "qvals": elements of the plot represent statistical significance of the interaction
                    - "fold_change": elements of the plot represent fold change induced in the receiver by the sender
            gene_subset : optional list of str
                Names of genes to subset for plot. If None, will use all genes that were used in the regression.
            significance_threshold : float, default 0.05
                Set non-significant fold changes to zero, where the threshold is given here
            cut_pvals : float, default -5
                Minimum allowable log10(pval)- anything below this will be clipped to this value
            fontsize : optional int
                Size of figure title and axis labels
            figsize : optional tuple of form (int, int)
                Width and height of plotting window
            cmap : str, default "seismic"
                Name of matplotlib colormap specifying colormap to use
            save_show_or_return: Literal["save", "show", "return", "both", "all"], default "show"
                Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs : optional dict
                A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        sender_idx = self.celltype_names.index(sender)

        if plot_mode == 'qvals':
            arr = np.log(self.qvalues[:, sender_idx, :].copy())
            arr[arr < cut_pvals] = cut_pvals
            df = pd.DataFrame(
                arr,
                index=self.celltype_names,
                columns=self.genes
            )
            if gene_subset:
                df = df.drop(index=sender)[gene_subset]

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor='grey',
                linewidths=0.3,
                cbar_kws={'label': "$\log_{10}$ FDR-corrected pvalues"},
                cmap=cmap, vmin=-5, vmax=0.
            )

        elif plot_mode == 'fold_change':
            arr = self.fold_change[:, sender_idx, :].copy()
            arr[np.where(self.qvalues[:, sender_idx, :] > significance_threshold)] = 0
            df = pd.DataFrame(
                arr,
                index=self.celltype_names,
                columns=self.genes
            )
            if gene_subset:
                df = df.drop(index=sender)[gene_subset]
            vmax = np.max(np.abs(df.values))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor='grey',
                linewidths=0.3,
                cbar_kws={'label': "$\ln$ fold change",
                          "location": "top"},
                cmap=cmap, vmin=-vmax, vmax=vmax,
            )
        else:
            logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'fold_change'.")

        plt.xlabel("Receiving cell type")
        plt.title("{} effects on receivers".format(sender))
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="{}_effects_on_receiver".format(sender),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def sender_receiver_effect_volcanoplot(
        self,
        receiver: str,
        sender: str,
        significance_threshold: float = 0.05,
        fold_change_threshold: Union[None, float] = None,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = (4.5, 7.0),
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {}
    ):
        """
        Volcano plot to identify differentially expressed genes of a given receiver cell type in the presence of a
        given sender cell type.

        Args:
            receiver : str
                Receiver cell type label
            sender : str
                Sender cell type label
            significance_threshold : float, default 0.05
                Set non-significant fold changes (given by q-values) to zero, where the threshold is given here
            fold_change_threshold : optional float
                Set absolute value fold-change threshold beyond which observations are marked as interesting. If not
                given, will take the 95th percentile fold-change as
            fontsize : optional int
                Size of figure title and axis labels
            figsize : tuple of form (int, int)
                Width and height of plotting window
            save_show_or_return: Literal["save", "show", "return", "both", "all"], default "show"
                Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs : optional dict
                A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        receiver_idx = self.celltype_names.index(receiver)
        sender_idx = self.celltype_names.index(sender)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.grid(False)

        # All non-significant features:
        qval_filter = np.where(self.qvalues[receiver_idx, sender_idx, :] >= significance_threshold)
        vmax = np.max(np.abs(self.fold_change[receiver_idx, sender_idx, :]))

        sns.scatterplot(
            x=self.fold_change[receiver_idx, sender_idx, :][qval_filter],
            y=-np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter],
            color='white', edgecolor='black', s=100, ax=ax)

        # Identify subset that may be significant, but which doesn't pass the fold-change threshold:
        qval_filter = np.where(self.qvalues[receiver_idx, sender_idx, :] < significance_threshold)
        x = self.fold_change[receiver_idx, sender_idx, :][qval_filter]
        fc_filter = np.where(x < fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter])
        sns.scatterplot(
            x=x[fc_filter],
            y=y[fc_filter],
            color='darkgrey', edgecolor='black', s=100, ax=ax)

        # Identify subset that are significantly downregulated:
        dreg_color = matplotlib.cm.get_cmap("winter")(0)
        x = self.fold_change[receiver_idx, sender_idx, :][qval_filter]
        fc_filter = np.where(x <= -fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter], neginf=-14.5)
        sns.scatterplot(
            x=x[fc_filter],
            y=y[fc_filter],
            color=dreg_color, edgecolor='black', s=100, ax=ax)

        # Identify subset that are significantly upregulated:
        ureg_color = matplotlib.cm.get_map("autumn")(0)
        x = self.fold_change[receiver_idx, sender_idx, :][qval_filter]
        fc_filter = np.where(x >= fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter], neginf=-14.5)
        sns.scatterplot(
            x=x[fc_filter],
            y=y[fc_filter],
            color=ureg_color, edgecolor='black', s=100, ax=ax)

        # Plot configuration:
        ax.set_xlim((-vmax * 1.1, vmax * 1.1))
        ax.set_xlabel('$\ln$ fold change')
        ax.set_ylabel('$-\log_{10}$ FDR-corrected pvalues')
        plt.axvline(-fold_change_threshold, color='darkgrey', linestyle='--')
        plt.axvline(fold_change_threshold, color='darkgrey', linestyle='--')
        plt.axhline(-np.log10(significance_threshold), linestyle='--', color='darkgrey')

        plt.tight_layout()
        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="effect_of_{}_on_{}".format(sender, receiver),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

class Category_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware (but not spatially lagged) regression using categorical
    variables (specifically, the prevalence of categories within spatial neighborhoods) to predict the value of gene
    expression.

    The only keyword argument that is used for this class is 'n_neighbors'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Prepare data:
        self.prepare_data(mod_type="category")


class Connections_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware (but not spatially lagged) regression using the
    connections between categories to predict the value of gene expression.

    The only keyword argument that is used for this class is 'n_neighbors'.

    To fit model, run :func `self.run_GM_lag`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Prepare data:
        self.prepare_data(mod_type="connections")


class Niche_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware regression using both the prevalence of and connections
    between categories within spatial neighborhoods to predict the value of gene expression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        self.prepare_data(mod_type="niche")


class Heterologous_Lagged_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-lagged regression using the spatial lag of selected features
    that are not the regression target feature.
    """


class Ligand_Lagged_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-lagged regression using the spatial lag of ligand genes to
    predict the regression target.
    """

    # Copy over the relevant PySal methods (for ref., TwoSLS, sp_att in utils, TwoSLS_sp/GM_Lag), except modify the call
    # to sp_att in TwoSLS_sp to use expression of a custom ligand of choice rather than self.predy

    # Modified version of the GM lag model defined in the base class:
    def run_GM_lag(self, n_jobs: int = 30):
        "filler"

    # Functionalities to additionally include: gene relationships to genes in their spatial niche


class Ligand_Lagged_Connections_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-lagged regression using categorical variables (specifically,
    the prevalence of categories within spatial neighborhoods) to predict the value of gene expression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Process the cell type array to use it as the input to the regression model:
        self.prepare_data(mod_type="ligand_lag_connections")

        # Compute spatial weights matrix given user inputs:
        self.compute_spatial_weights()

        # Instantiate locations in AnnData object to store results from lag model:
        for i in ["const"] + self.param_labels + ["W_log_exp"]:
            self.adata.var[str(i) + "_GM_lag_coeff"] = None
            self.adata.var[str(i) + "_GM_lag_zstat"] = None
            self.adata.var[str(i) + "_GM_lag_pval"] = None


# -------------------------------------------- Regression Metrics -------------------------------------------- #
def mae(y_true, y_pred):
    """
    Mean absolute error- in this context, actually log1p mean absolute error

    Args:
        y_true : np.ndarray, shape [n_samples, 1]
            Regression model output
        y_pred : np.ndarray, shape [n_samples, 1]
            Observed values for the dependent variable

    Returns:
        mae : float
            Mean absolute error value across all samples
    """
    abs = np.abs(y_true - y_pred)
    mean = np.mean(abs)
    return mean


def mse(y_true, y_pred):
    """
    Mean squared error- in this context, actually log1p mean squared error

    Args:
        y_true : np.ndarray, shape [n_samples, 1]
            Regression model output
        y_pred : np.ndarray, shape [n_samples, 1]
            Observed values for the dependent variable

    Returns:
        mse : float
            Mean squared error value across all samples
    """
    se = np.square(y_true - y_pred)
    se = np.mean(se, axis=-1)
    return se


# NOTE: NLL from here: https://github.com/tensorchiefs/dl_book/blob/master/chapter_06/nb_ch06_02.ipynb
def nll(y_true, y_pred):
    """
    Negative log likelihood

    Args:
        y_true : np.ndarray, shape [n_samples, 1]
            Regression model output
        y_pred : np.ndarray, shape [n_samples, 1]
            Observed values for the dependent variable

    Returns:
        neg_ll : float
            Negative log likelihood across all samples
    """
    n = len(y_true)
    sigma_hat_2 = (n - 1.0) / (n - 2.0) * np.var(y_true - y_pred.flatten(), ddof=1)
    nll = 0.5 * np.log(2 * np.pi * sigma_hat_2) + 0.5 * np.mean((y_true - y_pred.flatten()) ** 2) / sigma_hat_2
    return nll


def r_squared(y_true, y_pred):
    """
    Compute custom r squared- in this context, actually log1p R^2

    Args:
        y_true : np.ndarray, shape [n_samples, 1]
            Regression model output
        y_pred : np.ndarray, shape [n_samples, 1]
            Observed values for the dependent variable

    Returns:
        r2 : float
            Coefficient of determination
    """
    resid = np.sum(np.square(y_true - y_pred))
    total = np.sum(np.square(y_true - np.sum(y_true)))
    r2 = 1.0 - resid / total
    return r2
