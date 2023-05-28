"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
import sys
from typing import Callable, Optional, Tuple, Union

from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, recall_score

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
import tensorflow as tf
from anndata import AnnData
from numpy import linalg
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(1, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.configuration import SKM
from spateo.logging import logger_manager as lm
from spateo.preprocessing.transform import log1p
from spateo.tools.ST_modeling.distributions import (
    Binomial,
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
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]

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


def sparse_minmax_scale(a: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
    """Column-wise minmax scaling of a sparse matrix."""
    if type(a).__name__ == "csr_matrix" or type(a).__name__ == "csc_matrix":
        scaler = MinMaxScaler()
        a_scaled = scaler.fit_transform(a)
        a_scaled = scipy.sparse.csr_matrix(a_scaled)
    else:
        raise Exception("Invalid format for 'a' argument: %s" % (type(a).__name__))

    return a_scaled


# ---------------------------------------------------------------------------------------------------
# Maximum likelihood estimation procedure
# ---------------------------------------------------------------------------------------------------
def compute_betas(
    y: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    x: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    ridge_lambda: float = 0.0,
    clip: float = 5.0,
):
    """Maximum likelihood estimation procedure, to be used in iteratively weighted least squares to compute the
    regression coefficients for a given set of dependent and independent variables. Can be combined with either Lasso
    (L1), Ridge (L2), or Elastic Net (L1 + L2) regularization.

    Source: Iteratively (Re)weighted Least Squares (IWLS), Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        y: Array of shape [n_samples,]; dependent variable
        x: Array of shape [n_samples, n_features]; independent variables
        ridge_lambda: Regularization parameter for Ridge regression. Higher values will tend to shrink coefficients
            further towards zero.
        clip: Float; upper and lower bound to constrain betas and prevent numerical overflow

    Returns:
        betas: Array of shape [n_features,]; regression coefficients
    """
    xT = x.T
    xtx = sparse_dot(xT, x)

    # Ridge regularization:
    if ridge_lambda is not None:
        identity = np.eye(xtx.shape[0])
        xtx += ridge_lambda * identity

    try:
        xtx_inv = linalg.inv(xtx)
    except:
        xtx_inv = linalg.pinv(xtx)
    xtx_inv = scipy.sparse.csr_matrix(xtx_inv)
    xTy = sparse_dot(xT, y, return_array=False)
    betas = sparse_dot(xtx_inv, xTy)
    # Upper and lower bound to constrain betas and prevent numerical overflow:
    betas = np.clip(betas, -clip, clip)

    return betas


def compute_betas_local(y: np.ndarray, x: np.ndarray, w: np.ndarray, ridge_lambda: float = 0.0, clip: float = 5.0):
    """Maximum likelihood estimation procedure, to be used in iteratively weighted least squares to compute the
    regression coefficients for a given set of dependent and independent variables while accounting for spatial
    heterogeneity in the dependent variable.

    Source: Iteratively (Re)weighted Least Squares (IWLS), Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        y: Array of shape [n_samples,]; dependent variable
        x: Array of shape [n_samples, n_features]; independent variables
        ridge_lambda: Regularization parameter for Ridge regression. Higher values will tend to shrink coefficients
            further towards zero.
        w: Array of shape [n_samples, 1]; spatial weights matrix
        clip: Float; upper and lower bound to constrain betas and prevent numerical overflow

    Returns:
        betas: Array of shape [n_features,]; regression coefficients
        pseudoinverse: Array of shape [n_samples, n_samples]; Moore-Penrose pseudoinverse of the X matrix
        cov_inverse: Array of shape [n_samples, n_samples]; inverse of the covariance matrix
    """
    xT = (x * w).T
    xtx = np.dot(xT, x)

    # Ridge regularization:
    if ridge_lambda is not None:
        identity = np.eye(xtx.shape[0])
        xtx += ridge_lambda * identity

    try:
        cov_inverse = linalg.inv(xtx)
    except:
        cov_inverse = linalg.pinv(xtx)

    # Diagonals of the Gram matrix- used as additional diagnostic- for each feature, this is the sum of squared
    # values- if this is sufficiently low, the coefficient should be zero- theoretically it can take on nearly any
    # value with little impact on the residuals, but it is most likely to be zero:
    diag = np.diag(xtx)
    below_limit = np.abs(diag) < 1e-3
    # Robustness to outlier points:
    n_nonzeros = np.count_nonzero(xT, axis=1)

    to_zero = np.concatenate((np.where(below_limit)[0], np.where(n_nonzeros <= 2)[0]))

    try:
        xtx_inv_xt = np.dot(linalg.inv(xtx), xT)
        # xtx_inv_xt = linalg.solve(xtx, xT)
    except:
        xtx_inv_xt = np.dot(linalg.pinv(xtx), xT)
    pseudoinverse = xtx_inv_xt

    # Avoid issues with all zero dependent variable values in spatial regions:
    yw = (y * w).reshape(-1, 1)
    all_zeros = np.all(yw == 0)
    if all_zeros:
        betas = np.full((x.shape[1], 1), 1e-5)
        return betas, pseudoinverse

    betas = np.dot(xtx_inv_xt, y)
    # Upper and lower bound to constrain betas and prevent numerical overflow:
    betas = np.clip(betas, -clip, clip)
    # And set to zero with small offset for numerical overflow if the diagonal of the Gram matrix is below a certain
    # threshold:
    betas[to_zero] = 1e-5

    return betas, pseudoinverse, cov_inverse


def iwls(
    y: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    x: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    distr: Literal["gaussian", "poisson", "nb", "binomial"] = "gaussian",
    init_betas: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    clip: float = 5.0,
    max_iter: int = 200,
    spatial_weights: Optional[np.ndarray] = None,
    link: Optional[Link] = None,
    ridge_lambda: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
):
    """Iteratively weighted least squares (IWLS) algorithm to compute the regression coefficients for a given set of
    dependent and independent variables.

    Source: Iteratively (Re)weighted Least Squares (IWLS), Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Args:
        y: Array of shape [n_samples, 1]; dependent variable
        x: Array of shape [n_samples, n_features]; independent variables
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb", "binomial"
        init_betas: Array of shape [n_features,]; initial regression coefficients
        tol: Convergence tolerance
        clip: Sets magnitude of the upper and lower bound to constrain betas and prevent numerical overflow
        max_iter: Maximum number of iterations if convergence is not reached
        spatial_weights: Array of shape [n_samples, 1]; weights to transform observations from location i for a
            geographically-weighted regression
        link: Link function for the distribution family. If None, will default to the default value for the specified
            distribution family.
        variance: Variance function for the distribution family. If None, will default to the default value for the
            specified distribution family.
        ridge_lambda: Ridge regularization parameter.
        mask: Optional array of shape [n_features,]; if provided, final coefficients will be multiplied by mask values

    Returns:
        betas: Array of shape [n_features, 1]; regression coefficients
        y_hat: Array of shape [n_samples, 1]; predicted values of the dependent variable
        wx: Array of shape [n_samples, 1]; weighted independent variables
        n_iter: Number of iterations completed upon convergence
        w_final: Array of shape [n_samples, 1]; final spatial weights used for IWLS.
        linear_predictor_final: Array of shape [n_samples, 1]; final unadjusted linear predictor used for IWLS. Only
            returned if "spatial_weights" is not None.
        adjusted_predictor_final: Array of shape [n_samples, 1]; final adjusted linear predictor used for IWLS. Only
            returned if "spatial_weights" is not None.
        pseudoinverse: Array of shape [n_samples, n_samples]; optional influence matrix that is only returned if
            "spatial_weights" is not None. The pseudoinverse is the Moore-Penrose pseudoinverse of the X matrix.
        inv: Array of shape [n_samples, n_samples]; the inverse covariance matrix (for Gaussian modeling) or the
            inverse Fisher matrix (for GLM models).
    """
    logger = lm.get_main_logger()

    # Initialization:
    n_iter = 0
    difference = 1.0e6

    # Get appropriate distribution family based on specified:
    mod_distr = distr  # string specifying distribution assumption of the model

    if distr == "gaussian":
        link = link or Gaussian.__init__.__defaults__[0]
        distr = Gaussian(link)
    elif distr == "poisson":
        link = link or Poisson.__init__.__defaults__[0]
        distr = Poisson(link)
    elif distr == "nb":
        link = link or NegativeBinomial.__init__.__defaults__[0]
        distr = NegativeBinomial(link)
    elif distr == "binomial":
        link = link or Binomial.__init__.__defaults__[0]
        distr = Binomial(link)

    if init_betas is None:
        betas = np.zeros((x.shape[1], 1))
    else:
        betas = init_betas

    # Initial values:
    y_hat = distr.initial_predictions(y)
    linear_predictor = distr.get_predictors(y_hat)

    while difference > tol and n_iter < max_iter:
        n_iter += 1
        if mod_distr == "binomial":
            weights = distr.weights(y_hat)
        else:
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
            new_betas = compute_betas(w_adjusted_predictor, wx, ridge_lambda=ridge_lambda, clip=clip)
        else:
            new_betas, pseudoinverse, inverse_cov = compute_betas_local(
                w_adjusted_predictor, wx, spatial_weights, ridge_lambda=ridge_lambda, clip=clip
            )

        if mask is not None:
            new_betas = np.multiply(new_betas, mask.reshape(-1, 1)).astype(np.float32)

        linear_predictor = sparse_dot(x, new_betas)
        y_hat = distr.predict(linear_predictor)

        difference = np.min(abs(new_betas - betas))
        betas = new_betas
    # Set zero coefficients to zero:
    betas[betas == 1e-5] = 0.0

    if mod_distr == "gaussian":
        if spatial_weights is not None:
            xT = (x * spatial_weights).T
            xtx = np.dot(xT, x)
            try:
                inv = linalg.inv(xtx)
            except:
                inv = linalg.pinv(xtx)
        else:
            xtx = np.dot(x.T, x)
            try:
                inv = linalg.inv(xtx)
            except:
                inv = linalg.pinv(xtx)

    elif mod_distr == "poisson" or mod_distr == "nb":
        inv = get_fisher_inverse(x, linear_predictor)
    else:
        inv = None

    if spatial_weights is None:
        return betas, y_hat, wx, n_iter
    else:
        w_final = weights
        return betas, y_hat, n_iter, w_final, linear_predictor, adjusted_predictor, pseudoinverse, inv


# ---------------------------------------------------------------------------------------------------
# Objective functions for logistic models
# ---------------------------------------------------------------------------------------------------
def weighted_binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, weight_0: float = 1.0, weight_1: float = 1.0):
    """
    Custom binary cross-entropy loss function with class weights.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        weight_0: Weight for class 0 (negative class)
        weight_1: Weight for class 1 (positive class)

    Returns:
        Weighted binary cross-entropy loss
    """
    # Small constant to avoid division by zero
    epsilon = tf.keras.backend.epsilon()

    # Apply class weights
    weights = y_true * weight_1 + (1 - y_true) * weight_0

    # Clip predicted probabilities to avoid log(0) and log(1)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    # Compute weighted binary cross-entropy loss
    loss = -tf.reduce_mean(weights * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return loss


def logistic_objective(threshold: float, proba: np.ndarray, y_true: np.ndarray):
    """For binomial regression models with IWLS, the objective function is the weighted sum of recall and specificity.

    Args:
        threshold: Threshold for converting predicted probabilities to binary predictions
        proba: Predicted probabilities from logistic model
        y_true: True binary labels

    Returns:
        score: Weighted sum of recall and specificity
    """
    predictions = (proba >= threshold).astype(int)

    # Compute true positive rate
    recall = recall_score(y_true, predictions, pos_label=1)

    # Compute true negative rate
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    specificity = tn / (tn + fp)

    # Define weights for the two metrics
    # Calculate weights based on the ratio of the number of 0s to 1s in y_true to buffer against class imbalances
    zero_ratio = (y_true == 0).sum() / len(y_true)
    one_ratio = (y_true == 1).sum() / len(y_true)
    w1 = 1.0 * zero_ratio
    w2 = 1.0 * one_ratio

    # Return weighted sum of recall and true negative rate
    # Golden search aims to minimize, so negative sign to get the maximum recall + TNR
    score = -(w1 * recall + w2 * specificity)
    return score


def golden_section_search(func: Callable, a: float, b: float, tol: float = 1e-5, min_or_max: str = "min"):
    """Find the extremum of a function within a specified range using Golden Section Search.

    Args:
        func: The function to find the extremum of.
        a: Lower bound of the range.
        b: Upper bound of the range.
        tol: Tolerance for stopping criterion.
        min_or_max: Whether to find the minimum or maximum of the function.

    Returns:
        The x-value of the function's extremum.
    """
    phi = (np.sqrt(5) - 1) / 2  # golden ratio

    c = b - phi * (b - a)
    d = a + phi * (b - a)

    while abs(c - d) > tol:
        if min_or_max == "min":
            if func(c) < func(d):
                b = d
            else:
                a = c
        elif min_or_max == "max":
            if func(c) > func(d):
                b = d
            else:
                a = c

        # Compute new bounds
        c = b - phi * (b - a)
        d = a + phi * (b - a)

    return (b + a) / 2


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
def multicollinearity_check(X: pd.DataFrame, thresh: float = 5.0, logger: Optional = None):
    """Checks for multicollinearity in dependent variable array, and drops the most multicollinear features until
    all features have VIF less than a given threshold.

    Args:
        X: Dependent variable array, in dataframe format
        thresh: VIF threshold; features with values greater than this value will be removed from the regression
        logger: If not provided, will create a new logger

    Returns:
        X: Dependent variable array following filtering
    """
    if logger is None:
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
            print(vif)
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                logger.info("Dropping '" + X.iloc[:, variables].columns[maxloc] + "' at index: " + str(maxloc))
                X.drop(X.columns[variables[maxloc]], 1, inplace=True)
                variables = list(range(X.shape[1]))
                dropped = True

        logger.info(f"\n\nRemaining variables:\n {list(X.columns[variables])}")
        return X


# ---------------------------------------------------------------------------------------------------
# Significance Testing
# ---------------------------------------------------------------------------------------------------
def wald_test(
    theta_mle: Union[float, np.ndarray], theta_sd: Union[float, np.ndarray], theta0: Union[float, np.ndarray] = 0
) -> np.ndarray:
    """Perform Wald test, informing whether a given coefficient deviates significantly from the
    supplied reference value (theta0), based on the standard deviation of the posterior of the parameter estimate.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        theta_mle: Maximum likelihood estimation of given parameter by feature
        theta_sd: Standard deviation of the maximum likelihood estimation
        theta0: Value(s) to test theta_mle against. Must be either a single number or an array w/ equal number of
            entries to theta_mle.

    Returns:
        pvals: p-values for each feature, indicating whether the feature's coefficient deviates significantly from
            the reference value
    """
    if np.size(theta0) == 1:
        theta0 = np.broadcast_to(theta0, np.shape(theta_mle))

    if np.shape(theta_mle) != np.shape(theta_sd):
        raise ValueError("stats.wald_test(): theta_mle and theta_sd have to contain the same number of entries")
    if np.size(theta0) > 1 and np.shape(theta_mle) != np.shape(theta0):
        raise ValueError("stats.wald_test(): theta_mle and theta0 have to contain the same number of entries")

    # If sd is zero, instead set deviation equal to a small floating point
    if isinstance(theta_sd, (float, np.floating)):
        if theta_sd < np.nextafter(0, np.inf):
            theta_sd = np.nextafter(0, np.inf)

    elif isinstance(theta_sd, np.ndarray):
        theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    wald_statistic = np.abs(np.divide(theta_mle - theta0, theta_sd))
    pvals = 2 * (1 - scipy.stats.norm.cdf(np.abs(wald_statistic)))  # two-sided
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


def get_fisher_inverse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the Fisher matrix that measures the amount of information each feature in x provides about y- that is,
    whether the log-likelihood is sensitive to change in the parameter x.

    Function derived from diffxpy: https://github.com/theislab/diffxpy

    Args:
        x: Array of shape [n_samples, n_features]; independent variable array
        fitted: Array of shape [n_samples, 1] or [n_samples, n_variables]; estimated dependent variable

    Returns:
        inverse_fisher : np.ndarray
    """
    if len(y.shape) > 1 and y.shape[1] > 1:
        var = np.var(y, axis=0)
        fisher = np.expand_dims(np.matmul(x.T, x), axis=0) / np.expand_dims(var, axis=[1, 2])
        fisher = np.nan_to_num(fisher)
        try:
            inverse_fisher = np.array([np.linalg.inv(fisher[i, :, :]) for i in range(fisher.shape[0])])
        except:
            inverse_fisher = np.array([np.linalg.pinv(fisher[i, :, :]) for i in range(fisher.shape[0])])
    else:
        var = np.var(y)
        fisher = np.matmul(x.T, x) / var
        fisher = np.nan_to_num(fisher)
        try:
            inverse_fisher = np.linalg.inv(fisher)
        except:
            inverse_fisher = np.linalg.pinv(fisher)

    return inverse_fisher


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
            x_new = x_new.todense()
            x_new = scipy.sparse.csr_matrix(np.round(x_new)).astype(int)
        return x_new, d
    else:
        x_new = W @ X
        if return_discrete:
            x_new = x_new.todense()
            x_new = scipy.sparse.csr_matrix(np.round(x_new)).astype(int)
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
