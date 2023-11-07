"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
import functools
import gc
from typing import Callable, List, Optional, Tuple, Union

import psutil
from joblib import Parallel, delayed

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from multiprocessing import Pool, cpu_count

import anndata
import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
import tensorflow as tf
from numpy import linalg
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ...configuration import SKM
from ...logging import logger_manager as lm
from ...preprocessing.normalize import calcNormFactors
from ...preprocessing.transform import log1p
from .distributions import Binomial, Gaussian, Link, NegativeBinomial, Poisson

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


def sparse_add_pseudocount(a: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix], pseudocount: float = 1.0):
    """Add pseudocount to sparse matrix."""
    if type(a).__name__ == "csr_matrix" or type(a).__name__ == "csc_matrix":
        a.data += pseudocount
    else:
        raise Exception("Invalid format for 'a' argument: %s" % (type(a).__name__))

    return a


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


def compute_betas_local(
    y: np.ndarray, x: np.ndarray, w: np.ndarray, ridge_lambda: float = 0.0, clip: Optional[float] = None
):
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
    yw = (y * w).reshape(-1, 1)
    if np.all(yw == 0):
        betas = np.full((x.shape[1], 1), 1e-20)
        pseudoinverse = np.zeros((x.shape[1], x.shape[0]))
        cov_inverse = np.zeros((x.shape[1], x.shape[1]))
        return betas, pseudoinverse, cov_inverse

    xT = (x * w).T
    if np.all(xT == 0):
        betas = np.full((x.shape[1], 1), 1e-20)
        pseudoinverse = np.zeros((x.shape[1], x.shape[0]))
        cov_inverse = np.zeros((x.shape[1], x.shape[1]))
        return betas, pseudoinverse, cov_inverse

    xtx = np.dot(xT, x)

    # Ridge regularization:
    if ridge_lambda is not None:
        identity = np.eye(xtx.shape[0])
        xtx += ridge_lambda * identity

    try:
        cov_inverse = linalg.inv(xtx)
    except:
        cov_inverse = linalg.pinv(xtx)

    try:
        xtx_inv_xt = np.dot(linalg.inv(xtx), xT)
        # xtx_inv_xt = linalg.solve(xtx, xT)
    except:
        xtx_inv_xt = np.dot(linalg.pinv(xtx), xT)
    pseudoinverse = xtx_inv_xt

    betas = np.dot(xtx_inv_xt, y)
    if clip is not None:
        # Upper and lower bound to constrain betas and prevent numerical overflow:
        betas = np.clip(betas, -clip, clip)

    return betas, pseudoinverse, cov_inverse


def iwls(
    y: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    x: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    distr: Literal["gaussian", "poisson", "nb", "binomial"] = "gaussian",
    init_betas: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    clip: Optional[Union[float, np.ndarray]] = None,
    threshold: float = 1e-4,
    max_iter: int = 200,
    spatial_weights: Optional[np.ndarray] = None,
    i: Optional[int] = None,
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
        offset: Optional array of shape [n_samples,]; if provided, will be added to the linear predictor. This is
            meant to deal with differences in scale that are not caused by the predictor variables,
            e.g. by differences in library size
        tol: Convergence tolerance
        clip: Sets magnitude of the upper and lower bound to constrain betas and prevent numerical overflow. Either
            one floating point value or an array for sample-specific clipping.
        threshold: Coefficients with absolute values below this threshold will be set to zero (as these are
            insignificant)
        max_iter: Maximum number of iterations if convergence is not reached
        spatial_weights: Array of shape [n_samples, 1]; weights to transform observations from location i for a
            geographically-weighted regression
        i: Optional integer; index of the observation to be used as the center of the geographically-weighted
            regression. Required if "clip" is an array.
        link: Link function for the distribution family. If None, will default to the default value for the specified
            distribution family.
        variance: Variance function for the distribution family. If None, will default to the default value for the
            specified distribution family.
        ridge_lambda: Ridge regularization parameter.

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
    # If y or x is all zeros, return all zeros:
    if spatial_weights is None:
        if np.all(y == 0) or np.all(x == 0):
            betas = np.zeros((x.shape[1], 1))
            y_hat = np.zeros((x.shape[0], 1))
            return betas, y_hat, None, None
    else:
        xw = (x * spatial_weights).reshape(-1, 1)
        yw = (y * spatial_weights).reshape(-1, 1)
        if np.all(yw == 0) or np.all(xw == 0):
            betas = np.zeros((x.shape[1], 1))
            y_hat = np.zeros_like(y)
            n_iter = 0
            w_final = np.zeros_like(y)
            linear_predictor = np.zeros_like(y)
            adjusted_predictor = np.zeros_like(y)
            pseudoinverse = np.zeros((x.shape[1], x.shape[0]))
            inv = np.zeros((x.shape[1], x.shape[1]))
            return betas, y_hat, n_iter, w_final, linear_predictor, adjusted_predictor, pseudoinverse, inv

    # Initialization:
    n_iter = 0
    difference = 1.0e6

    if isinstance(clip, np.ndarray):
        assert i is not None, "If clip is an array, i must be specified."
        clip = clip[i]

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
        if offset is None:
            adjusted_predictor = linear_predictor + (distr.link.deriv(y_hat) * (y - y_hat))
        else:
            adjusted_predictor = linear_predictor + (distr.link.deriv(y_hat) * (y - y_hat)) + offset
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

        # Mask operations:
        if mask is not None:
            mask = mask.reshape(-1, 1)
            neg_mask = (new_betas < 0) & (mask == -1.0) | (new_betas > 0)
            new_betas[~neg_mask] = 1e-6
            mask[mask == -1.0] = 1
            new_betas = np.multiply(new_betas, mask).astype(np.float32)

        linear_predictor = sparse_dot(x, new_betas)
        y_hat = distr.predict(linear_predictor)

        difference = np.min(abs(new_betas - betas))
        betas = new_betas

    # Set zero coefficients to zero:
    betas[betas == 1e-6] = 0.0
    # Threshold coefficients where appropriate:
    betas[np.abs(betas) < threshold] = 0.0
    # For robustness, more than one feature is required to

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


def library_scaling_factors(
    offset: Optional[np.ndarray] = None,
    counts: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
    distr: Literal["gaussian", "poisson", "nb"] = "gaussian",
):
    """Get offset values to account for differences in library sizes when comparing expression levels between samples.

    If the offset is not provided, it calculates the offset based on library sizes. The offset is the logarithm of
    the library sizes.

    Args:
        offset: Offset values. If provided, it is returned as is. If None, the offset is calculated based on library
            sizes.
        counts: Gene expression array
        distr: Distribution of the data. Defaults to "gaussian", but can also be "poisson" or "nb".

    Returns:
        offset: Array of shape [n_samples, ] containing offset values.
    """
    logger = lm.get_main_logger()

    if offset is None and counts is None:
        raise ValueError("Either offset or counts must be provided.")

    if isinstance(counts, scipy.sparse.spmatrix):
        counts = counts.toarray()

    if offset is None:
        try:
            nf = calcNormFactors(counts, method="TMMwsp")
        except:
            logger.info("TMMwsp normalization failed. Will use unnormalized library sizes as offset.")
            nf = np.ones(counts.shape[0])

        libsize = np.sum(counts, axis=1) * nf
        if distr != "gaussian":
            offset = np.log(libsize)
        else:
            offset = libsize

        if np.any(libsize == 0):
            logger.info("Some library sizes are zero. Offsetting these to 0.")
            offset[libsize == 0] = 0

    return offset


# ---------------------------------------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------------------------------------
def softplus(z: np.ndarray):
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
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                logger.info("Dropping '" + X.iloc[:, variables].columns[maxloc] + "' at index: " + str(maxloc))
                X.drop(X.columns[variables[maxloc]], axis=1, inplace=True)
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


def multitesting_correction(
    pvals: Union[List[float], np.ndarray], method: str = "fdr_bh", alpha: float = 0.05
) -> np.ndarray:
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
    if isinstance(pvals, list):
        pvals = np.array(pvals)
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


def run_permutation_test(data, thresh, subset_rows=None, subset_cols=None):
    """Permutes the input data array and calculates whether the mean of the permuted array is higher than the
        provided value.

    Args:
        data: Input array or sparse matrix
        thresh: Value to compare against the permuted means
        subset_rows: Optional indices to subset the rows of 'data'
        subset_cols: Optional indices to subset the columns of 'data'

    Returns:
        is_higher: Boolean indicating whether the mean of the permuted data is higher than 'thresh'
    """
    if scipy.sparse.issparse(data):
        permuted = data.copy()
        np.random.shuffle(permuted.data)

        if subset_rows is not None:
            if subset_cols is not None:
                permuted_mean = permuted[subset_rows, subset_cols].mean()
            else:
                permuted_mean = permuted[subset_rows, :].mean()
        elif subset_cols is not None:
            permuted_mean = permuted[:, subset_cols].mean()
        else:
            permuted_mean = permuted.mean()

    else:
        permuted = np.random.permutation(data)
        if subset_rows is not None:
            if subset_cols is not None:
                permuted_mean = np.mean(permuted[subset_rows, subset_cols])
            else:
                permuted_mean = np.mean(permuted[subset_rows, :])
        elif subset_cols is not None:
            permuted_mean = np.mean(permuted[:, subset_cols])
        else:
            permuted_mean = np.mean(permuted)

    is_higher = permuted_mean > thresh
    return is_higher


def permutation_testing(
    data: Union[np.ndarray, scipy.sparse.spmatrix],
    n_permutations: int = 10000,
    n_jobs: int = 1,
    subset_rows: Optional[Union[np.ndarray, List[int]]] = None,
    subset_cols: Optional[Union[np.ndarray, List[int]]] = None,
) -> float:
    """Permutes the input array and calculates the p-value based on the number of times the mean of the permuted
        array is higher than the provided value.

    Args:
        data: Input array or sparse matrix
        n_permutations: Number of permutations
        n_jobs: Number of parallel jobs to use (-1 for all available CPUs)
        subset_rows: Optional indices to subset the rows of 'data' (to take the mean value of only the subset of
            interest)
        subset_cols: Optional indices to subset the columns of 'data' (to take the mean value of only the subset of
            interest)

    Returns:
        pval: The calculated p-value.
    """
    if scipy.sparse.issparse(data):
        data = data.A

    if subset_rows is not None:
        if subset_cols is not None:
            mean_observed = np.mean(data[subset_rows, subset_cols])
        else:
            mean_observed = np.mean(data[subset_rows, :])
    elif subset_cols is not None:
        mean_observed = np.mean(data[:, subset_cols])
    else:
        mean_observed = np.mean(data)

    n_trials_higher = sum(
        Parallel(n_jobs=n_jobs)(
            delayed(run_permutation_test)(data, mean_observed, subset_rows, subset_cols) for _ in range(n_permutations)
        )
    )

    # Add 1 to numerator and denominator for continuity correction
    p_value = (n_trials_higher + 1) / (n_permutations + 1)
    return p_value


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
def smooth(
    X: Union[np.ndarray, scipy.sparse.spmatrix],
    W: Union[np.ndarray, scipy.sparse.spmatrix],
    ct: Optional[np.ndarray] = None,
    gene_expr_subset: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
    min_jaccard: Optional[float] = 0.05,
    manual_mask: Optional[np.ndarray] = None,
    normalize_W: bool = True,
    return_discrete: bool = False,
    n_subsample: Optional[int] = None,
) -> Tuple[scipy.sparse.csr_matrix, Optional[np.ndarray]]:
    """Leverages neighborhood information to smooth gene expression.

    Args:
        X: Gene expression array or sparse matrix (shape n x m, where n is the number of cells and m is the number of
            genes)
        W: Spatial weights matrix (shape n x n)
        ct: Optional, indicates the cell type label for each cell (shape n x 1). If given, will smooth only within each
            cell type.
        gene_expr_subset: Optional, array corresponding to the expression of select genes (shape n x k,
            where k is the number of genes in the subset). If given, will smooth only over cells that largely match
            the expression patterns over these genes (assessed using a Jaccard index threshold that is greater than
            the median score).
        min_jaccard: Optional, and only used if 'gene_expr_subset' is also given. Minimum Jaccard similarity score to
            be considered "nonzero".
        manual_mask: Optional, binary array of shape n x n. For each cell (row), manually indicate which neighbors (
            if any) to use for smoothing.
        normalize_W: Set True to scale the rows of the weights matrix to sum to 1. Use this to smooth by taking an
            average over the entire neighborhood, including zeros. Set False to take the average over only the
            nonzero elements in the neighborhood.
        return_discrete: Set True to return
        n_subsample: Optional, sets the number of random neighbor samples to use in the smoothing. If not given,
            will use all neighbors (nonzero weights) for each cell.

    Returns:
        x_new: Smoothed gene expression array or sparse matrix
        d: Only if normalize_W is True, returns the row sums of the weights matrix
    """
    logger = lm.get_main_logger()
    logger.info(
        "Warning- this can be quite memory intensive when 'gene_expr_subset' is provided, depending on the "
        "size of the AnnData object. If this is the case it is recommended only to use cell types."
    )

    if scipy.sparse.issparse(X):
        logger.info(f"Initial sparsity of array: {X.count_nonzero()}")
    else:
        logger.info(f"Initial sparsity of array: {np.count_nonzero(X)}")

    # Subsample weights array if applicable:
    if n_subsample is not None:
        # Threshold for smoothing (check that a sufficient number of neighbors express a given gene for increased
        # confidence of biological signal- must be greater than or equal to this threshold for smoothing):
        threshold = int(np.ceil(n_subsample / 4))

        if scipy.sparse.issparse(W):
            W = subsample_neighbors_sparse(W, n_subsample)
        else:
            W = subsample_neighbors_dense(W, n_subsample)
    else:
        # Threshold of zero means all neighboring cells get incorporated into the smoothing operation
        threshold = 0
    logger.info(f"Threshold for smoothing: {threshold} neighbors")

    # If mask is manually given, no need to use cell type & expression information:
    if manual_mask is not None:
        logger.info(
            "Manual mask provided. Will use this to smooth, ignoring inputs to 'ct' and 'gene_expr_subset' if "
            "provided."
        )
        W = W.multiply(manual_mask)
    else:
        # Incorporate cell type information
        if ct is not None:
            logger.info(
                "Conditioning smoothing on cell type- only information from cells of the same type will be used."
            )
            ct_masks = ct == ct.transpose()
            W = W.multiply(ct_masks)

        # Incorporate gene expression information
        if gene_expr_subset is not None:
            logger.info(
                "Conditioning smoothing on gene expression- only information from cells with similar gene "
                "expression patterns will be used."
            )
            jaccard_mat = compute_jaccard_similarity_matrix(gene_expr_subset, min_jaccard=min_jaccard)
            logger.info("Computing median Jaccard score from nonzero entries only")
            if scipy.sparse.issparse(jaccard_mat):
                jaccard_threshold = sparse_matrix_median(jaccard_mat, nonzero_only=True)
            else:
                jaccard_threshold = np.median(jaccard_mat[jaccard_mat != 0])
            logger.info(f"Threshold Jaccard score: {jaccard_threshold}")
            # Generate a mask where Jaccard similarities are greater than the threshold, and apply to the weights
            # matrix:
            jaccard_mask = jaccard_mat >= jaccard_threshold
            W = W.multiply(jaccard_mask)

    if scipy.sparse.issparse(W):
        # Calculate the average number of non-zero weights per row for a sparse matrix
        row_nonzeros = W.getnnz(axis=1)  # Number of non-zero elements per row
        average_nonzeros = row_nonzeros.mean()  # Average over all rows
    else:
        # Calculate the average number of non-zero weights per row for a dense matrix
        row_nonzeros = (W != 0).sum(axis=1)  # Boolean matrix where True=1 for non-zeros, then sum per row
        average_nonzeros = row_nonzeros.mean()  # Average over all rows
    logger.info(f"Average number of non-zero weights per cell: {average_nonzeros}")

    # Original nonzero entries (keep these around):
    initial_nz_rows, initial_nz_cols = X.nonzero()

    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = scipy.sparse.diags(1 / d) @ W if scipy.sparse.issparse(W) else np.diag(1 / d) @ W
        # Note that W @ X already returns sparse in this scenario, csr_matrix is just used to convert to common format
        x_new = scipy.sparse.csr_matrix(W @ X) if scipy.sparse.issparse(X) else W @ X

        if return_discrete:
            if scipy.sparse.issparse(x_new):
                data = x_new.data
                data[:] = np.where((0 < data) & (data < 1), 1, np.round(data))
            else:
                x_new = np.where((0 < x_new) & (x_new < 1), 1, np.round(x_new))
        if scipy.sparse.issparse(x_new):
            logger.info(f"Sparsity of smoothed array: {x_new.count_nonzero()}")
        else:
            logger.info(f"Sparsity of smoothed array: {np.count_nonzero(x_new)}")
        return x_new, d
    else:
        # Average of the nonzero elements:
        processor_func = functools.partial(smooth_process_column, X=X, W=W, threshold=threshold)
        pool = Pool(cpu_count())
        mod = pool.map(processor_func, range(X.shape[1]))
        mod = scipy.sparse.hstack(mod)
        mod.data = 1.0 / mod.data

        mod_nz_rows, mod_nz_cols = mod.nonzero()

        # Note that W @ X already returns sparse in this scenario, csr_matrix is just used to convert to common format
        product = scipy.sparse.csr_matrix(W @ X) if scipy.sparse.issparse(X) else W @ X
        product = product.multiply(mod)
        x_new = (
            scipy.sparse.csr_matrix((X.shape[0], X.shape[1]), dtype=np.float32)
            if scipy.sparse.issparse(X)
            else np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        )
        x_new[mod_nz_rows, mod_nz_cols] = product[mod_nz_rows, mod_nz_cols]
        # For any zeros introduced by this process that were initially nonzeros, set back to the original value:
        x_new[initial_nz_rows, initial_nz_cols] = X[initial_nz_rows, initial_nz_cols]
        if scipy.sparse.issparse(x_new):
            logger.info(f"Sparsity of smoothed array: {x_new.count_nonzero()}")
        else:
            logger.info(f"Sparsity of smoothed array: {np.count_nonzero(x_new)}")

        if return_discrete:
            if scipy.sparse.issparse(x_new):
                data = x_new.data
                data[:] = np.round(data)
            else:
                x_new = np.round(x_new)
        return x_new


def compute_jaccard_similarity_matrix(
    data: Union[np.ndarray, scipy.sparse.spmatrix], chunk_size: int = 1000, min_jaccard: float = 0.1
) -> np.ndarray:
    """Compute the Jaccard similarity matrix for input data with rows corresponding to samples and columns
    corresponding to features, processing in chunks for memory efficiency.

    Args:
        data: A dense numpy array or a sparse matrix in CSR format, with rows as features
        chunk_size: The number of rows to process in a single chunk
        min_jaccard: Minimum Jaccard similarity to be considered "nonzero"

    Returns:
        jaccard_matrix: A square matrix of Jaccard similarity coefficients
    """
    n_samples = data.shape[0]
    jaccard_matrix = np.zeros((n_samples, n_samples))

    if scipy.sparse.isspmatrix(data):  # Check if input is a sparse matrix
        # Ensure the matrix is in CSR format for efficient row slicing
        data = scipy.sparse.csr_matrix(data)
        data_bool = data.astype(bool).astype(int)
        row_sums = data_bool.sum(axis=1)
        data_bool_T = data_bool.T
        row_sums_T = row_sums.T
    else:
        data_bool = (data > 0).astype(int)
        row_sums = data_bool.sum(axis=1).reshape(-1, 1)
        data_bool_T = data_bool.T
        row_sums_T = row_sums.T

    # Compute Jaccard similarities in chunks
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        print(f"Pairwise Jaccard similarity computation: processing rows {chunk_start} to {chunk_end}...")

        # Compute similarities for the current chunk
        chunk_intersection = data_bool[chunk_start:chunk_end].dot(data_bool_T)
        chunk_union = row_sums[chunk_start:chunk_end] + row_sums_T - chunk_intersection
        chunk_similarity = chunk_intersection / np.maximum(chunk_union, 1)
        chunk_similarity[chunk_similarity < min_jaccard] = 0.0

        jaccard_matrix[chunk_start:chunk_end] = chunk_similarity

        # Print available memory
        memory_stats = psutil.virtual_memory()
        print(f"Available memory: {memory_stats.available / (1024 * 1024):.2f} MB")

        print(f"Completed rows {chunk_start} to {chunk_end}.")

        # Clean up memory for the next chunk
        del chunk_intersection, chunk_union, chunk_similarity
        gc.collect()

    print("Pairwise Jaccard similarity computation: all chunks processed. Converting to final format...")
    if np.any(np.isnan(jaccard_matrix)) or np.any(np.isinf(jaccard_matrix)):
        raise ValueError("jaccard_matrix contains NaN or Inf values")

    if scipy.sparse.issparse(data):
        jaccard_matrix = scipy.sparse.csr_matrix(jaccard_matrix)
    print("Returning pairwise Jaccard similarity matrix...")

    return jaccard_matrix


def sparse_matrix_median(spmat: scipy.sparse.spmatrix, nonzero_only: bool = False) -> scipy.sparse.spmatrix:
    """Computes the median value of a sparse matrix, used here for determining a threshold value for Jaccard similarity.

    Args:
        spmat: The sparse matrix to compute the median value of
        nonzero_only: If True, only consider nonzero values in the sparse matrix

    Returns:
        median_value: The median value of the sparse matrix
    """
    # Flatten the sparse matrix data and sort it
    spmat_data_sorted = np.sort(spmat.data)

    if nonzero_only:
        # If the number of non-zero elements is even, take the average of the two middle values
        middle_index = spmat.nnz // 2
        if spmat.nnz % 2 == 0:  # Even number of non-zero elements
            median_value = (spmat_data_sorted[middle_index - 1] + spmat_data_sorted[middle_index]) / 2
        else:  # Odd number of non-zero elements, take the middle one
            median_value = spmat_data_sorted[middle_index]
    else:
        # Get the total number of elements in the matrix
        total_elements = spmat.shape[0] * spmat.shape[1]
        # Calculate the number of zeros
        num_zeros = total_elements - spmat.nnz
        # Find the number of sorted elements that need to be sorted through to find the median
        median_idx = total_elements // 2

        # If there are more zeros than the index for the median, then the median is zero
        if num_zeros > median_idx:
            median_value = 0
        else:
            median_idx_non_zero = median_idx - num_zeros
            median_value = spmat_data_sorted[median_idx_non_zero]

    return median_value


def smooth_process_column(
    i: int,
    X: Union[np.ndarray, scipy.sparse.csr_matrix],
    W: Union[np.ndarray, scipy.sparse.csr_matrix],
    threshold: float,
) -> scipy.sparse.csr_matrix:
    """Helper function for parallelization of smoothing.

    Args:
        i: Index of the column to be processed
        X: Dense or sparse array input data matrix
        W: Dense or sparse array pairwise spatial weights matrix
        threshold: Threshold value for smoothing- anything below this will be set to 0.

    Returns:
        count_nnz: The processed column.
    """
    feat = X[:, i].reshape(1, -1)
    if scipy.sparse.issparse(X):
        temp = W.multiply(feat)
        count_nnz = np.diff(temp.indptr)
        count_nnz[count_nnz < threshold] = 0.0
    else:
        temp = np.multiply(W, feat)
        count_nnz = np.count_nonzero(temp, axis=1)
        count_nnz[count_nnz < threshold] = 0.0
    count_nnz = scipy.sparse.csr_matrix(count_nnz.reshape(-1, 1))

    return count_nnz


def subsample_neighbors_dense(W: np.ndarray, n: int) -> np.ndarray:
    """Given dense spatial weights matrix W and number of random neighbors n to take, perform subsampling.

    Parameters:
        W: Spatial weights matrix
        n: Number of neighbors to keep for each row

    Returns:
        W_new: Subsampled spatial weights matrix
    """
    logger = lm.get_main_logger()

    W_new = W.copy()
    for i in range(W_new.shape[0]):
        nonzero_indices = np.nonzero(W_new[i])[0]
        m = len(nonzero_indices)
        if m > n:
            np.random.shuffle(nonzero_indices)
            indices_to_zero = nonzero_indices[: m - n]
            W_new[i, indices_to_zero] = 0
        else:
            logger.warning(f"Cell {i} has fewer than {n} neighbors. Subsampling not performed.")
    return W_new


def subsample_neighbors_sparse(W: scipy.sparse.spmatrix, n: int) -> scipy.sparse.spmatrix:
    """Given sparse spatial weights matrix W and number of random neighbors n to take, perform subsampling.

    Parameters:
        W: Spatial weights matrix
        n: Number of neighbors to keep for each row

    Returns:
        W_new: Subsampled spatial weights matrix
    """
    logger = lm.get_main_logger()

    W_new = W.copy().tocoo()
    rows, cols = W_new.nonzero()
    unique_rows = np.unique(rows)
    for row in unique_rows:
        row_nonzero_cols = cols[rows == row]
        m = len(row_nonzero_cols)
        if m > n:
            np.random.shuffle(row_nonzero_cols)
            cols_to_zero = row_nonzero_cols[: m - n]
            for col in cols_to_zero:
                indices = (W_new.row == row) & (W_new.col == col)
                W_new.data[indices] = 0
        else:
            logger.warning(f"Cell {row} has fewer than {n} neighbors. Subsampling not performed.")
    W_new = W_new.tocsr()
    return W_new
