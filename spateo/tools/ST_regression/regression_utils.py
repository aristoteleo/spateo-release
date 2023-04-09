"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
import sys
from typing import Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
from anndata import AnnData
from numpy import linalg
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.configuration import SKM
from spateo.logging import logger_manager as lm
from spateo.preprocessing.transform import log1p
from spateo.tools.ST_regression.distributions import (
    Gaussian,
    Link,
    NegativeBinomial,
    Poisson,
)

# from ...configuration import SKM
# from ...logging import logger_manager as lm
# from ...preprocessing.transform import log1p
# from .distributions import Gaussian, Link, NegativeBinomial, Poisson


# ---------------------------------------------------------------------------------------------------
# Sparse matrix operations
# ---------------------------------------------------------------------------------------------------
def sparse_dot(
    a: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    b: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    return_array: bool = True,
):
    """
    Matrix multiplication function to deal with sparse and dense objects

    Args:
        a: First of two matrices to be multiplied, can be sparse or dense
        b: Second of two matrices to be multiplied, can be sparse or dense
        return_array: Set True to return dense array

    Returns:
        prod: Matrix product of a and b
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        prod = np.dot(a, b)
    elif (
        type(a).__name__ == "csr_matrix"
        or type(b).__name__ == "csr_matrix"
        or type(a).__name__ == "csc_matrix"
        or type(b).__name__ == "csc_matrix"
    ):
        prod = a * b
        if return_array:
            if type(prod).__name__ == "csc_matrix" or type(prod).__name__ == "csr_matrix":
                prod = prod.toarray()
    else:
        raise Exception("Invalid format for 'spdot' argument: %s and %s" % (type(a).__name__, type(b).__name__))
    return prod


def sparse_element_by_element(
    a: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    b: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    return_array: bool = True,
):
    """Element-by-element multiplication function to deal with either sparse or dense objects.

    Args:
        a: First of two matrices to be multiplied, can be sparse or dense
        b: Second of two matrices to be multiplied, can be sparse or dense
        return_array: Set True to return dense array

    Returns:
        prod: Element-wise multiplied product of a and b
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        prod = a * b
    elif (
        type(a).__name__ == "csr_matrix"
        or type(b).__name__ == "csr_matrix"
        or type(a).__name__ == "csc_matrix"
        or type(b).__name__ == "csc_matrix"
    ):
        prod = a.multiply(b)
        if return_array:
            if type(prod).__name__ == "csc_matrix" or type(prod).__name__ == "csr_matrix":
                prod = prod.toarray()
    else:
        raise Exception("Invalid format for 'spdot' argument: %s and %s" % (type(a).__name__, type(b).__name__))
    return prod


# ---------------------------------------------------------------------------------------------------
# Maximum likelihood estimation procedure
# ---------------------------------------------------------------------------------------------------
def compute_betas(
    y: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    x: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
):
    """Maximum likelihood estimation procedure, to be used in iteratively weighted least squares to compute the
    regression coefficients for a given set of dependent and independent variables. Can be combined with either Lasso
    (L1), Ridge (L2), or Elastic Net (L1 + L2) regularization.

    Source: Iteratively (Re)weighted Least Squares (IWLS), Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        y: Array of shape [n_samples,]; dependent variable
        x: Array of shape [n_samples, n_features]; independent variables

    Returns:
        betas: Array of shape [n_features,]; regression coefficients
    """
    xT = x.T
    xtx = sparse_dot(xT, x)
    xtx_inv = linalg.inv(xtx)
    xtx_inv = scipy.sparse.csr_matrix(xtx_inv)
    xTy = sparse_dot(xT, y, return_array=False)
    betas = sparse_dot(xtx_inv, xTy)
    return betas


def compute_betas_local(y: np.ndarray, x: np.ndarray, w: np.ndarray):
    """Maximum likelihood estimation procedure, to be used in iteratively weighted least squares to compute the
    regression coefficients for a given set of dependent and independent variables while accounting for spatial
    heterogeneity in the dependent variable.

    Source: Iteratively (Re)weighted Least Squares (IWLS), Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        y: Array of shape [n_samples,]; dependent variable
        x: Array of shape [n_samples, n_features]; independent variables
        w: Array of shape [n_samples, n_samples]; spatial weights matrix

    Returns:
        betas: Array of shape [n_features,]; regression coefficients
        pseudoinverse: Array of shape [n_samples, n_samples]; Moore-Penrose pseudoinverse of the X matrix
    """
    xT = (x * w).T
    xtx = np.dot(xT, x)
    try:
        xtx_inv_xt = np.dot(linalg.inv(xtx), xT)
        # xtx_inv_xt = linalg.solve(xtx, xT)
    except:
        xtx_inv_xt = np.dot(linalg.pinv(xtx), xT)
    betas = np.dot(xtx_inv_xt, y)

    pseudoinverse = xtx_inv_xt

    return betas, pseudoinverse


def iwls(
    y: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    x: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    distr: Literal["gaussian", "poisson", "nb"] = "gaussian",
    init_betas: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    spatial_weights: Optional[np.ndarray] = None,
    link: Optional[Link] = None,
    alpha: Optional[float] = None,
    tau: Optional[float] = None,
):
    """Iteratively weighted least squares (IWLS) algorithm to compute the regression coefficients for a given set of
    dependent and independent variables.

    Source: Iteratively (Re)weighted Least Squares (IWLS), Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        y: Array of shape [n_samples,]; dependent variable
        x: Array of shape [n_samples, n_features]; independent variables
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        init_betas: Array of shape [n_features,]; initial regression coefficients
        tol: Convergence tolerance
        max_iter: Maximum number of iterations if convergence is not reached
        spatial_weights: Array of shape [n_samples, 1]; weights to transform observations from location i for a
            geographically-weighted regression
        link: Link function for the distribution family. If None, will default to the default value for the specified
            distribution family.
        variance: Variance function for the distribution family. If None, will default to the default value for the
            specified distribution family.
        alpha: L1 regularization strength; must be between 0 and 1. The L2 regularization strength is 1 - alpha.
        tau: Optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
            provided, Tau will default to the identity matrix.

    Returns:
        betas: Array of shape [n_features,]; regression coefficients
        y_hat: Array of shape [n_samples,]; predicted values of the dependent variable
        wx: Array of shape [n_samples,]; weighted independent variables
        n_iter: Number of iterations completed upon convergence
        w_final: Array of shape [n_samples,]; final spatial weights used for IWLS.
        linear_predictor_final: Array of shape [n_samples,]; final unadjusted linear predictor used for IWLS. Only
            returned if "spatial_weights" is not None.
        adjusted_predictor_final: Array of shape [n_samples,]; final adjusted linear predictor used for IWLS. Only
            returned if "spatial_weights" is not None.
        pseudoinverse: Array of shape [n_samples, n_samples]; optional influence matrix that is only returned if
            "spatial_weights" is not None. The pseudoinverse is the Moore-Penrose pseudoinverse of the X matrix.

    """
    logger = lm.get_main_logger()

    # Initialization:
    n_iter = 0
    difference = 1.0e10

    # Get appropriate distribution family based on specified:
    if distr == "gaussian":
        link = link or Gaussian.__init__.__defaults__[0]
        distr = Gaussian(link)
    elif distr == "poisson":
        link = link or Poisson.__init__.__defaults__[0]
        distr = Poisson(link)
    elif distr == "nb":
        link = link or NegativeBinomial.__init__.__defaults__[0]
        distr = NegativeBinomial(link)

    if init_betas is None:
        betas = np.zeros(x.shape[1], 1)
    else:
        betas = init_betas

    # Initial values:
    y_hat = distr.initial_predictions(y)
    linear_predictor = distr.predict(y_hat)

    while difference > tol and n_iter < max_iter:
        n_iter += 1
        weights = distr.weights(linear_predictor)
        # Compute adjusted predictor from the difference between the predicted mean response variable and observed y:
        adjusted_predictor = linear_predictor + (distr.link.deriv(y_hat) * (y - y_hat))
        weights = np.sqrt(weights)
        if not isinstance(x, np.ndarray):
            weights = scipy.sparse.csr_matrix(weights)
            adjusted_predictor = scipy.sparse.csr_matrix(adjusted_predictor)
        wx = sparse_element_by_element(x, weights, return_array=False)
        w_adjusted_predictor = sparse_element_by_element(adjusted_predictor, weights, return_array=False)

        if spatial_weights is None:
            new_betas = compute_betas(w_adjusted_predictor, wx)
        else:
            new_betas, pseudoinverse = compute_betas_local(w_adjusted_predictor, wx, spatial_weights)

        # Update coefficients with the elastic net penalty, if applicable:
        if alpha is not None:
            if not (alpha is None or (0 <= alpha <= 1)):
                logger.error("alpha must be between 0 and 1.")

            l1_penalty = alpha * L1_penalty(new_betas)
            l2_penalty = (1 - alpha) * L2_penalty(new_betas, tau)
            new_betas = (
                np.sign(new_betas)
                * np.maximum(np.abs(new_betas) - l1_penalty, 0)
                / (1 + l2_penalty / (2 * (1 - alpha)))
            )

        linear_predictor = sparse_dot(x, new_betas)
        y_hat = distr.predict(linear_predictor)

        difference = min(abs(new_betas - betas))
        betas = new_betas

    if spatial_weights is None:
        return betas, y_hat, wx, n_iter
    else:
        w_final = weights
        return betas, y_hat, n_iter, w_final, linear_predictor, adjusted_predictor, pseudoinverse


# ---------------------------------------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------------------------------------
def softplus(z):
    """Numerically stable version of log(1 + exp(z))."""
    nl = z.copy()
    nl[z > 35] = z[z > 35]
    nl[z < -10] = np.exp(z[z < -10])
    nl[(z >= -10) & (z <= 35)] = log1p(np.exp(z[(z >= -10) & (z <= 35)]))
    return nl


# ---------------------------------------------------------------------------------------------------
# Check multicollinearity
# ---------------------------------------------------------------------------------------------------
def multicollinearity_check(X: pd.DataFrame, thresh: float = 5.0):
    """Checks for multicollinearity in dependent variable array, and drops the most multicollinear features until
    all features have VIF less than a given threshold.

    Args:
        X: Dependent variable array, in dataframe format
        thresh: VIF threshold; features with values greater than this value will be removed from the regression

    Returns:
        X: Dependent variable array following filtering
    """
    logger = lm.get_main_logger()

    int_cols = X.select_dtypes(
        include=["int", "int16", "int32", "int64", "float", "float16", "float32", "float64"]
    ).shape[1]
    total_cols = X.shape[1]

    if int_cols != total_cols:
        logger.error("All columns should be integer or float.")
    else:
        variables = list(range(X.shape[1]))
        dropped = True
        logger.info(
            f"Iterating through features and calculating respective variance inflation factors (VIF). Will "
            "iteratively drop the highest VIF features until all features have VIF less than the threshold "
            "value of {thresh}"
        )
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in variables]
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                logger.info("Dropping '" + X.iloc[:, variables].columns[maxloc] + "' at index: " + str(maxloc))
                X.drop(X.columns[variables[maxloc]], 1, inplace=True)
                variables = list(range(X.shape[1]))
                dropped = True

        logger.info(f"\n\nRemaining variables:\n {X.columns[variables]}")
        return X


# ---------------------------------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------------------------------
def L1_penalty(beta: np.ndarray) -> float:
    """
    Implementation of the L1 penalty that penalizes based on absolute value of coefficient magnitude.

    Args:
        beta: Array of shape [n_features,]; learned model coefficients

    Returns:
        L1penalty: float, value for the regularization parameter (typically stylized by lambda)
    """
    # Lasso-like penalty- max(sum(abs(beta), axis=0))
    L1penalty = np.linalg.norm(beta, 1)
    return L1penalty


def L2_penalty(beta: np.ndarray, Tau: Union[None, np.ndarray] = None) -> float:
    """Implementation of the L2 penalty that penalizes based on the square of coefficient magnitudes.

    Args:
        beta: Array of shape [n_features,]; learned model coefficients
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
        provided, Tau will default to the identity matrix.
    """
    if Tau is None:
        # Ridge=like penalty
        L2penalty = np.linalg.norm(beta, 2) ** 2
    else:
        # Tikhonov penalty
        if Tau.shape[0] != beta.shape[0] or Tau.shape[1] != beta.shape[0]:
            raise ValueError("Tau should be (n_features x n_features)")
        else:
            L2penalty = np.linalg.norm(np.dot(Tau, beta), 2) ** 2

    return L2penalty


# ---------------------------------------------------------------------------------------------------
# Significance Testing
# ---------------------------------------------------------------------------------------------------
def get_fisher_inverse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the Fisher matrix that measures the amount of information each feature in x provides about y- that is,
    whether the log-likelihood is sensitive to change in the parameter x.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        x: Independent variable array
        y: Dependent variable array

    Returns:
        inverse_fisher : np.ndarray
    """

    var = np.var(y, axis=0)
    fisher = np.expand_dims(np.matmul(x.T, x), axis=0) / np.expand_dims(var, axis=[1, 2])

    fisher = np.nan_to_num(fisher)

    inverse_fisher = np.array([np.linalg.pinv(fisher[i, :, :]) for i in range(fisher.shape[0])])
    return inverse_fisher


def wald_test(theta_mle: np.ndarray, theta_sd: np.ndarray, theta0: Union[float, np.ndarray] = 0) -> np.ndarray:
    """Perform single-coefficient Wald test, informing whether a given coefficient deviates significantly from the
    supplied reference value (theta0), based on the standard deviation of the posterior of the parameter estimate.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        theta_mle: Maximum likelihood estimation of given parameter by feature
        theta_sd: Standard deviation of the maximum likelihood estimation
        theta0: Value(s) to test theta_mle against. Must be either a single number or an array w/ equal number of
            entries to theta_mle.

    Returns:
        pvals : np.ndarray
    """

    if np.size(theta0) == 1:
        theta0 = np.broadcast_to(theta0, theta_mle.shape)

    if theta_mle.shape[0] != theta_sd.shape[0]:
        raise ValueError("stats.wald_test(): theta_mle and theta_sd have to contain the same number of entries")
    if theta0.shape[0] > 1:
        if theta_mle.shape[0] != theta0.shape[0]:
            raise ValueError("stats.wald_test(): theta_mle and theta0 have to contain the same number of entries")

    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    wald_statistic = np.abs(np.divide(theta_mle - theta0, theta_sd))
    pvals = 2 * (1 - scipy.stats.norm(loc=0, scale=1).cdf(wald_statistic))  # two-tailed test
    return pvals


def multitesting_correction(pvals: np.ndarray, method: str = "fdr_bh", alpha: float = 0.05) -> np.ndarray:
    """In the case of testing multiple hypotheses from the same experiment, perform multiple test correction to adjust
    q-values.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
    pvals: Uncorrected p-values; must be given as a one-dimensional array
    method: Method to use for correction. Available methods can be found in the documentation for
        statsmodels.stats.multitest.multipletests(), and are also listed below (in correct case) for convenience:
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
    alpha: Family-wise error rate (FWER)

    Returns
        qval: p-values post-correction
    """

    qval = np.zeros([pvals.shape[0]]) + np.nan
    qval[np.isnan(pvals) == False] = statsmodels.stats.multitest.multipletests(
        pvals=pvals[np.isnan(pvals) == False], alpha=alpha, method=method, is_sorted=False, returnsorted=False
    )[1]

    return qval


def get_p_value(variables: np.array, fisher_inv: np.array, coef_loc: int) -> np.ndarray:
    """Computes p-value for differential expression for a target feature

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        variables: Array where each column corresponds to a feature
        fisher_inv: Inverse Fisher information matrix
        coef_loc: Numerical column of the array corresponding to the coefficient to test

    Returns:
        pvalues: Array of identical shape to variables, where each element is a p-value for that instance of that
            feature
    """

    theta_mle = variables[coef_loc]
    theta_sd = fisher_inv[:, coef_loc, coef_loc]
    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    theta_sd = np.sqrt(theta_sd)

    pvalues = wald_test(theta_mle, theta_sd, theta0=0.0)
    return pvalues


def compute_wald_test(
    params: np.ndarray, fisher_inv: np.ndarray, significance_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        params: Array of shape [n_features, n_params]
        fisher_inv: Inverse Fisher information matrix
        significance_threshold: Upper threshold to be considered significant

    Returns:
        significance: Array of identical shape to variables, where each element is True or False if it meets the
            threshold for significance
        pvalues: Array of identical shape to variables, where each element is a p-value for that instance of that
            feature
        qvalues: Array of identical shape to variables, where each element is a q-value for that instance of that
            feature
    """

    pvalues = []

    # Compute p-values for each feature, store in temporary list:
    for idx in range(params.T.shape[0]):
        pvals = get_p_value(params.T, fisher_inv, idx)
        pvalues.append(pvals)

    pvalues = np.concatenate(pvalues)
    # Multiple testing correction w/ Benjamini-Hochberg procedure and FWER 0.05
    qvalues = multitesting_correction(pvalues)
    pvalues = np.reshape(pvalues, (-1, params.T.shape[1]))
    qvalues = np.reshape(qvalues, (-1, params.T.shape[1]))
    significance = qvalues < significance_threshold

    return significance, pvalues, qvalues


# ---------------------------------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    """Mean absolute error- in this context, actually log1p mean absolute error

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        mae: Mean absolute error value across all samples
    """
    abs = np.abs(y_true - y_pred)
    mean = np.mean(abs)
    return mean


def mse(y_true, y_pred) -> float:
    """Mean squared error- in this context, actually log1p mean squared error

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        mse: Mean squared error value across all samples
    """
    se = np.square(y_true - y_pred)
    se = np.mean(se, axis=-1)
    return se


# ---------------------------------------------------------------------------------------------------
# Spatial smoothing
# ---------------------------------------------------------------------------------------------------
def smooth(X, W, normalize_W=True, return_discrete=False) -> Tuple[scipy.sparse.csr_matrix, Optional[np.ndarray]]:
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = scipy.sparse.diags(1 / d) @ W if scipy.sparse.issparse(W) else np.diag(1 / d) @ W
        x_new = scipy.sparse.csr_matrix(W @ X)
        if return_discrete:
            x_new = scipy.sparse.csr_matrix.round(x_new).astype(int)
        return x_new, d
    else:
        x_new = W @ X
        if return_discrete:
            x_new = scipy.sparse.csr_matrix.round(x_new).astype(int)
        return x_new


# ---------------------------------------------------------------------------------------------------
# Testing Model Accuracy
# ---------------------------------------------------------------------------------------------------
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def plot_prior_vs_data(
    reconst: pd.DataFrame,
    adata: AnnData,
    kind: str = "barplot",
    target_name: Union[None, str] = None,
    title: Union[None, str] = None,
    figsize: Union[None, Tuple[float, float]] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
    save_kwargs: dict = {},
):
    """Plots distribution of observed vs. predicted counts in the form of a comparative density barplot.

    Args:
        reconst: DataFrame containing values for reconstruction/prediction of targets of a regression model
        adata: AnnData object containing observed counts
        kind: Kind of plot to generate. Options: "barplot", "scatterplot". Case sensitive, defaults to "barplot".
        target_name: Optional, can be:
                - Column name in DataFrame/AnnData object: name of gene to subset to
                - "sum": computes sum over all features present in 'reconst' to compare to the corresponding subset of
                'adata'.
                - "mean": computes mean over all features present in 'reconst' to compare to the corresponding subset of
                'adata'.
            If not given, will subset AnnData to features in 'reconst' and flatten both arrays to compare all values.

            If not given, will compute the sum over all
            features present in 'reconst' and compare to the corresponding subset of 'adata'.
        save_show_or_return: Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
            displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    from ...configuration import config_spateo_rcParams
    from ...plotting.static.utils import save_return_show_fig_utils

    logger = lm.get_main_logger()

    config_spateo_rcParams()
    if figsize is None:
        figsize = rcParams.get("figure.figsize")

    if target_name == "sum":
        predicted = reconst.sum(axis=1).values.reshape(-1, 1)
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = np.sum(observed, axis=1).reshape(-1, 1)
    elif target_name == "mean":
        predicted = reconst.mean(axis=1).values.reshape(-1, 1)
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = np.mean(observed, axis=1).reshape(-1, 1)
    elif target_name is not None:
        observed = adata[:, target_name].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, target_name].X
        observed = observed.reshape(-1, 1)
        predicted = reconst[target_name].values.reshape(-1, 1)
    else:
        # Flatten arrays:
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = observed.flatten().reshape(-1, 1)
        predicted = reconst.values.flatten().reshape(-1, 1)

    obs_pred = np.hstack((observed, predicted))
    # Upper limit along the x-axis (99th percentile to prevent outliers from affecting scale too badly):
    xmax = np.percentile(obs_pred, 99)
    # Lower limit along the x-axis:
    xmin = np.min(observed)
    # Divide x-axis into pieces for purposes of setting x labels:
    xrange, step = np.linspace(xmin, xmax, num=10, retstep=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if target_name is None:
        target_name = "Total Counts"

    if kind == "barplot":
        ax.hist(
            obs_pred,
            xrange,
            alpha=0.7,
            label=[f"Observed {target_name}", f"Predicted {target_name}"],
            density=True,
            color=["#FFA07A", "#20B2AA"],
        )

        plt.legend(loc="upper right", fontsize=9)

        ax.set_xticks(ticks=[i + 0.5 * step for i in xrange[:-1]], labels=[np.round(l, 3) for l in xrange[:-1]])
        plt.xlabel("Counts", size=9)
        plt.ylabel("Normalized Proportion of Cells", size=9)
        if title is not None:
            plt.title(title, size=9)
        plt.tight_layout()

    elif kind == "scatterplot":
        from scipy.stats import spearmanr

        observed = observed.flatten()
        predicted = predicted.flatten()
        slope, intercept = np.polyfit(observed, predicted, 1)

        # Extract residuals:
        predicted_model = np.polyval([slope, intercept], observed)
        observed_mean = np.mean(observed)
        predicted_mean = np.mean(predicted)
        n = observed.size  # number of samples
        m = 2  # number of parameters
        dof = n - m  # degrees of freedom
        # Students statistic of interval confidence:
        t = scipy.stats.t.ppf(0.975, dof)
        residual = observed - predicted_model
        # Standard deviation of the error:
        std_error = (np.sum(residual**2) / dof) ** 0.5

        # Calculate spearman correlation and coefficient of determination:
        s = spearmanr(observed, predicted)[0]
        numerator = np.sum((observed - observed_mean) * (predicted - predicted_mean))
        denominator = (np.sum((observed - observed_mean) ** 2) * np.sum((predicted - predicted_mean) ** 2)) ** 0.5
        correlation_coef = numerator / denominator
        r2 = correlation_coef**2

        # Plot best fit line:
        observed_line = np.linspace(np.min(observed), np.max(observed), 100)
        predicted_line = np.polyval([slope, intercept], observed_line)

        # Confidence interval and prediction interval:
        ci = (
            t
            * std_error
            * (1 / n + (observed_line - observed_mean) ** 2 / np.sum((observed - observed_mean) ** 2)) ** 0.5
        )
        pi = (
            t
            * std_error
            * (1 + 1 / n + (observed_line - observed_mean) ** 2 / np.sum((observed - observed_mean) ** 2)) ** 0.5
        )

        ax.plot(observed, predicted, "o", ms=3, color="royalblue", alpha=0.7)
        ax.plot(observed_line, predicted_line, color="royalblue", alpha=0.7)
        ax.fill_between(
            observed_line, predicted_line + pi, predicted_line - pi, color="lightcyan", label="95% prediction interval"
        )
        ax.fill_between(
            observed_line, predicted_line + ci, predicted_line - ci, color="skyblue", label="95% confidence interval"
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.set_xlabel(f"Observed {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        title = title if title is not None else "Observed and Predicted {}".format(target_name)
        ax.set_title(title)

        # Display r^2, Spearman correlation, mean absolute error on plot as well:
        r2s = str(np.round(r2, 2))
        spearman = str(np.round(s, 2))
        ma_err = mae(observed, predicted)
        mae_s = str(np.round(ma_err, 2))

        # Place text at slightly above the minimum x_line value and maximum y_line value to avoid obscuring the plot:
        ax.text(
            1.01 * np.min(observed),
            1.01 * np.max(predicted),
            "$r^2$ = " + r2s + ", Spearman $r$ = " + spearman + ", MAE = " + mae_s,
            fontsize=8,
        )
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), fontsize=8)

    else:
        logger.info(
            ":func `plot_prior_vs_data` error: Invalid input given to 'kind'. Options: 'barplot', " "'scatterplot'."
        )

    save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=True,
        background="white",
        prefix="parameters",
        save_kwargs=save_kwargs,
        total_panels=1,
        fig=fig,
        axes=ax,
        return_all=False,
        return_all_list=None,
    )
