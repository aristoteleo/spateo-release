"""
Perform downstream characterization following spatially-informed regression to characterize niche impact on gene
expression
"""
from typing import Union, List

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from pysal.lib import weights
from pysal.model import spreg
from random import sample

from .lag_utils import get_fisher_inverse
from ...logging import logger_manager as lm
from ...preprocessing.transform import log1p
from ...tools.utils import update_dict


# NOTE TO SELF: output = predicted influence of spatially-lagged expression on expression at particular spot
# X = cell type category, Y = expression of a gene (since it's a spatially-lagged model, the values of gene
# expression @ each neighbor are implicitly considered)

# NOTE ABOUT NCEM: references to "h_0" refer to array encoding cell type information, references to "h_1" refer to
# the gene expression. The shape of the target batch is 8- it's a one-hot encoding for cell type, interaction batch
# is 64- I'm assuming it's a flattened version of a cell type-pairwise interaction matrix? Their x block is thus
# 72-dimensional (made up of targets + interactions) and resulting parameters array is n_features x 72

# ANOTHER NOTE ABOUT NCEM: the "target" array is for the spatial neighborhood of each spot- n_neighbors x
# n_cell_types, the 1 corresponds to the cell type. My interpretation is that the "interactions" array is populated
# by 0 or 1 if, for each spatial neighborhood, both cell types of the pair can be found in the neighborhood?
# TASK: figure out how the "interactions" array is created


# -------------------------------------------- Wrappers for model running -------------------------------------------- #
class SpReg_Category_Interpreter():
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware regression using categorical variables to predict the
    value of gene expression.

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
        kwargs : optional dict
            Provides arguments . Note that for spatial weight-finding arguments, these must specifically match the
            name that the function will look for (case sensitive). For reference:
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
    def __init__(self,
                 adata: AnnData,
                 group_key: str,
                 spatial_key: str = "spatial",
                 genes: Union[None, List] = None,
                 drop_dummy: Union[None, str] = None,
                 layer: Union[None, str] = None,
                 log_transform: bool = False,
                 weights_mode: str = "knn",
                 **kwargs):
        self.adata = adata
        self.genes = genes
        self.layer = layer

        logger = lm.get_main_logger()
        # Prepare data:
        group_num = adata.obs[group_key].value_counts()
        max_group, min_group, min_group_ncells = (
            group_num.index[0],
            group_num.index[-1],
            group_num.values[-1],
        )

        group_name = adata.obs[group_key]
        db = pd.DataFrame({"group": group_name})
        categories = np.array(adata.obs[group_key].unique().tolist() + ["others"])
        db["group"] = pd.Categorical(db["group"], categories=categories)

        # Solve the dummy variable trap by dropping dummy category (deleting rows to avoid issues with
        # multicollinearity):
        if drop_dummy is None:
            # Leave some samples from every group intact
            db.iloc[sample(np.arange(adata.n_obs).tolist(), min_group_ncells), :] = "others"
        elif drop_dummy in group_name:
            group_inds = np.where(db["group"] == drop_dummy)[0]
            db.iloc[group_inds, :] = "others"
        else:
            raise ValueError(f"Dummy category ({drop_dummy}) provided is not in the adata.obs[{group_key}].")
        drop_columns = ["others"]

        self.X = pd.get_dummies(data=db, drop_first=False)
        # To index all but the dummy column when fitting model:
        self.variable_names = self.X.columns.difference(drop_columns).to_list()

        # Get the names of all remaining groups:
        self.uniq_g, group_name = (
            set(group_name).difference([drop_dummy]),
            group_name.to_list(),
        )

        uniq_g = list(np.sort(list(self.uniq_g)))

        # Compute spatial weights matrix given user inputs:
        # Default values for all possible spatial weight parameters:
        sp_kwargs = {
            "n_neighbors": 10,
            "p": 2,
            "distance_metric": "euclidean",
            "bandwidth": None,
            "fixed": True,
            "n_neighbors_bandwidth": 6,
            "kernel_function": "triangular",
            "threshold": 50,
            "alpha": -1.0
        }

        # Update using user input:
        sp_kwargs = update_dict(sp_kwargs, kwargs)

        # Choose how weights are computed:
        if weights_mode == "knn":
            self.w = weights.distance.KNN.from_array(adata.obsm[spatial_key], k=sp_kwargs["n_neighbors"])
        elif weights_mode == "kernel":
            self.w = weights.distance.Kernel.from_array(adata.obsm[spatial_key], bandwidth=sp_kwargs["bandwidth"],
                                                   fixed=sp_kwargs["fixed"], k=sp_kwargs["n_neighbors_bandwidth"],
                                                   function=sp_kwargs["kernel_function"])
        elif weights_mode == "band":
            self.w = weights.distance.DistanceBand.from_array(adata.obsm[spatial_key], threshold=sp_kwargs["threshold"],
                                                         alpha=sp_kwargs["alpha"])
        else:
            logger.error("Invalid input to 'weights_mode'. Options: 'knn', 'kernel', 'band'.")

        # Row standardize spatial weights matrix:
        self.w.transform = "R"

        # Filter gene names if specific gene names are provided. If not, use all genes referenced in .X:
        if genes is not None:
            self.genes = adata.var.index.intersection(genes)
        else:
            self.genes = adata.var.index
        self.adata = adata[:, self.genes]

        # Transform data matrix if 'log_transform' is True:
        if log_transform:
            if layer is None:
                self.adata.X = log1p(self.adata)
            else:
                self.adata.layers[layer] = log1p(self.adata.layers[layer])


        # Initialize entries in .var- intercept, cell type parameters Beta, spatial lag parameter gamma:
        for i in ["const"] + uniq_g + ["W_log_exp"]:
            adata.var[str(i) + "_GM_lag_coeff"] = None
            adata.var[str(i) + "_GM_lag_zstat"] = None
            adata.var[str(i) + "_GM_lag_pval"] = None


        # Define reconstruction metrics:
        self.metrics = [
            mae,
            mse,
            nll,
            r_squared
        ]


    def run_GM_lag(self, n_jobs: int = 30):
        """Runs spatially lagged two-stage least squares model"""

        def _single(cur_g: str,
                    X: pd.DataFrame,
                    X_variable_names: List[str],
                    uniq_g: List[str],
                    adata: AnnData,
                    w: np.ndarray,
                    layer: Union[None, str] = None
                    ):
            """
            Defines model run process for a single feature- not callable by the user, all arguments populated by
            arguments passed on instantiation of Estimator.

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

                adata.obs[cur_g + "_pred"] = y_pred
                adata.obs[cur_g + "_resid"] = resid

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
                adata.obs[cur_g + "_pred"] = np.full((X.shape[0],), np.nan)
                adata.obs[cur_g + "_resid"] = np.full((X.shape[0],), np.nan)

                for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                    adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = np.nan
                    adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = np.nan
                    adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = np.nan

            return adata.var.loc[cur_g, :].values, adata.obs[cur_g + "_pred"].values, adata.obs[cur_g + "_resid"].values

        coeffs, pred, resid = Parallel(n_jobs)(
                                delayed(_single)(
                                    cur_g,
                                    self.genes,
                                    self.X,
                                    self.variable_names,
                                    self.uniq_g,
                                    self.adata,
                                    self.w,
                                    self.layer
                                )
                                for cur_g in self.genes
        )

        coeffs = pd.DataFrame(coeffs, index=self.genes)
        coeffs.columns = self.adata.var.loc[self.genes, :].columns

        pred = pd.DataFrame(pred, index=self.adata.obs_names, columns=self.genes)
        resid = pd.DataFrame(resid, index=self.adata.obs_names, columns=self.genes)

        return coeffs, pred, resid



    def get_sender_receiver_effects(self,
                                    n_jobs: int = 30,
                                    significance_threshold: float = 0.05):
        """
        For each category and each feature, determine if the influence of said category in predicting said feature is
        significant. Additionally, for each feature and each sender-receiver category pair, determines the log
        fold-change that the sender induces in the feature for the receiver.

        Args:
            n_jobs : int, default 30
                Number of features to model at once
            significance_threshold : float, default 0.5
                p-value needed to call a sender-receiver relationship significant
        """
        coeffs, pred, resid = self.run_GM_lag(n_jobs)
        # Return only the numerical coefficients:
        coeffs = coeffs[[col for col in coeffs.columns if 'coeff' in col]].values

        # To compute inverse

        # Get inverse Fisher information matrix:
        inverse_fisher = get_fisher_inverse()



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
    sigma_hat_2 = (n - 1.) / (n - 2.) * np.var(y_true - y_pred.flatten(), ddof=1)
    nll = 0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_true - y_pred.flatten())**2)/sigma_hat_2
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