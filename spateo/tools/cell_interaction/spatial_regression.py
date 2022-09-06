"""
Perform downstream characterization following spatially-informed regression to characterize niche impact on gene
expression
"""
import os
import time
from random import sample
from typing import List, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from patsy import dmatrix
from pysal.lib import weights
from pysal.model import spreg

# For testing purposes keep the absolute imports:
from regression_utils import compute_wald_test, get_fisher_inverse, ols_fit

from spateo.logging import logger_manager as lm
from spateo.preprocessing.transform import log1p
from spateo.tools.find_neighbors import construct_pairwise
from spateo.tools.utils import update_dict

"""
from .regression_utils import compute_wald_test, get_fisher_inverse, ols_fit
from ...logging import logger_manager as lm
from ...preprocessing.transform import log1p
from ...tools.find_neighbors import construct_pairwise
from ...tools.utils import update_dict"""


# ----------------------------- Master wrappers for performing each type of regression ----------------------------- #


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
            Subset to genes that will be used in the regression analysis
        drop_dummy : optional str
            Name of the category to be dropped (the "dummy variable") in the regression. The dummy category can aid
            in interpretation as all model coefficients can be taken to be in reference to the dummy category. If
            None, will randomly select a few samples to constitute the dummy group.
        layer : optional str
            Entry in .layers to use instead of .X
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
        log_transform: bool = False,
        weights_mode: str = "knn",
        data_id: Union[None, str] = None,
        **kwargs,
    ):
        self.adata = adata
        self.spatial_key = spatial_key
        self.group_key = group_key
        self.genes = genes
        self.drop_dummy = drop_dummy
        self.layer = layer
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

    def prepare_data(self, mod_type: str = "category"):
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
                    select receptor and receptor-downstream genes, and additionally considers neighbor expression of
                    the ligands
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
            else:
                raise ValueError(
                    f"Dummy category ({self.drop_dummy}) provided is not in the " f"adata.obs[{self.group_key}]."
                )
            drop_columns = ["group_others"]

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)
            # Compute adjacency matrix- use the KNN value in self.sp_kwargs (which may have been passed as an
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

            # To index all but the dummy column when fitting model:
            self.variable_names = self.X.columns.difference(drop_columns).to_list()

            # Get the names of all remaining groups:
            self.uniq_g, group_name = (
                set(group_name).difference([self.drop_dummy]),
                group_name.to_list(),
            )

            self.uniq_g = list(np.sort(list(self.uniq_g)))

        elif mod_type == "connections" or mod_type == "niche":
            # Convert groups/categories into one-hot array:
            group_name = self.adata.obs[self.group_key]
            db = pd.DataFrame({"group": group_name})
            categories = np.array(self.adata.obs[self.group_key].unique().tolist())
            db["group"] = pd.Categorical(db["group"], categories=categories)

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)

            # Compute adjacency matrix- use the KNN value in self.sp_kwargs (which may have been passed as an
            # argument when initializing the interpreter):
            construct_pairwise(
                self.adata, spatial_key=self.spatial_key, n_neighbors=self.sp_kwargs["n_neighbors"], exclude_self=True
            )
            adj = self.adata.obsm["adj"]

            # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
            # each category are present within the neighborhood of each sample):
            dmat_neighbors = (adj > 0).astype("int").dot(X.values)

            # Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the niche of
            # each sample by documenting the category-category spatial connections within the niche):
            data = {"categories": X, "dmat_neighbours": dmat_neighbors}
            categories = np.asarray(dmatrix("target-1", data))
            connections = np.asarray(dmatrix("target:dmat_neighbours-1", data))
            """
            # If mod_type is 'connections', use only the connections matrix as independent variables in the regression:
            if mod_type == 'connections':
                self.X = pd.DataFrame(connections, index=, columns=
            # Otherwise if 'niche', combine connections and the one-hot category array as independent variables:
            elif mod_type == 'niche':
                self.X = np.concatenate([categories, connections], axis=1)

        elif mod_type == 'ligand_lag':
            # Adjust self.genes to those which can be found in the NicheNet database:"""

        else:
            self.logger.error("Invalid argument to 'mod_type'.")

        # Save model type as an attribute so it can be accessed by other methods:
        self.mod_type = mod_type

        # Filter gene names if specific gene names are provided. If not, use all genes referenced in .X:
        if self.genes is not None:
            self.genes = self.adata.var.index.intersection(self.genes)
        else:
            self.genes = self.adata.var.index
        self.adata = self.adata[:, self.genes]

        # Transform data matrix if 'log_transform' is True:
        if self.log_transform:
            if self.layer is None:
                self.adata.X = log1p(self.adata)
            else:
                self.adata.layers[self.layer] = log1p(self.adata.layers[self.layer])

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

    def get_sender_receiver_effects(self, coeffs: pd.DataFrame, n_jobs: int = 30, significance_threshold: float = 0.05):
        """
        For each category and each feature, determine if the influence of said category in predicting said feature is
        significant. Additionally, for each feature and each sender-receiver category pair, determines the log
        fold-change that the sender induces in the feature for the receiver.

        Only valid if the model specified uses the connections between categories as variables for the regression.

        Args:
            coeffs : pd.DataFrame
                Contains coefficients from regression for each variable
            n_jobs : int, default 30
                Number of features to model at once
            significance_threshold : float, default 0.5
                p-value needed to call a sender-receiver relationship significant
        """
        # coeffs, pred, resid = self.run_GM_lag(n_jobs)
        # Return only the numerical coefficients:
        coeffs = coeffs[[col for col in coeffs.columns if "coeff" in col]].values
        # Remove the first column (the intercept):
        coeffs = coeffs[:, 1:]

        # Get inverse Fisher information matrix:
        inverse_fisher = get_fisher_inverse()

    def type_coupling_analysis(
        self,
    ):
        """
        Generates heatmap of spatially differentially-expressed features for each pair of sender and receiver
        categories; only valid if the model specified uses the connections between categories as variables for
        regression.
        :return:
        """


class Category_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware (but not spatially lagged) regression using categorical
    variables to predict the value of gene expression.

    The only keyword argument that is used for this class is 'n_neighbors'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Prepare data:
        self.prepare_data(mod_type="category")

    def run_OLS(self, n_jobs: int = 30):
        coeffs = Parallel(n_jobs)(delayed(ols_fit)(self.X, cur_g) for cur_g in self.genes)

        coeffs = pd.DataFrame(coeffs, index=self.genes)
        print(coeffs)
        # coeffs.columns =


class Connections_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware (but not spatially lagged) regression using the
    connections between categories to predict the value of gene expression.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Prepare data:
        self.prepare_data(mod_type="connections")

        # Initialize entries in .var to store connections parameters Beta:

    def run_OLS(self, n_jobs: int = 30):
        coeffs, pred, resid = Parallel(n_jobs)(delayed(ols_fit)(self.X, cur_g) for cur_g in self.genes)

        coeffs = pd.DataFrame(coeffs, index=self.genes)


class Lagged_Category_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-lagged regression using categorical variables to predict the
    value of gene expression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Process the cell type array to use it as the input to the regression model:
        self.prepare_data(mod_type="lag_category")

        # Compute spatial weights matrix given user inputs:
        self.compute_spatial_weights()

        # Instantiate locations in AnnData object to store results from lag model:
        for i in ["const"] + self.uniq_g + ["W_log_exp"]:
            self.adata.var[str(i) + "_GM_lag_coeff"] = None
            self.adata.var[str(i) + "_GM_lag_zstat"] = None
            self.adata.var[str(i) + "_GM_lag_pval"] = None

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
                    Variables used for the regression
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
                cur_g, self.genes, self.X, self.variable_names, self.uniq_g, self.adata, self.w, self.layer
            )
            for cur_g in self.genes
        )

        coeffs = [item[0] for item in results]
        pred = [item[1] for item in results]
        resid = [item[2] for item in results]

        coeffs = pd.DataFrame(coeffs, index=self.genes)
        coeffs.columns = self.adata.var.loc[self.genes, :].columns

        pred = pd.DataFrame(pred, index=self.adata.obs_names, columns=self.genes)
        resid = pd.DataFrame(resid, index=self.adata.obs_names, columns=self.genes)

        # Update AnnData object:
        self.adata.obs.loc[:, "ypred"] = pred
        self.adata.obs.loc[:, "resid"] = resid

        for ind, g in enumerate(["const"] + self.uniq_g + ["W_log_exp"]):
            self.adata.var.loc[:, str(g) + "_GM_lag_coeff"] = coeffs.iloc[:, 0]
            self.adata.var.loc[:, str(g) + "_GM_lag_zstat"] = coeffs.iloc[:, 1]
            self.adata.var.loc[:, str(g) + "_GM_lag_pval"] = coeffs.iloc[:, 2]

        return coeffs, pred, resid


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

    # Parallelized run for all genes:
    # Resource on how to properly run Parallel in this scenario here:
    # http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html

    # Functionalities to additionally include: gene relationships to genes in their spatial niche


# Functionalities to additionally include: gene relationships to genes in their spatial niche

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
    abs = np.absolute(y_true - y_pred)
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