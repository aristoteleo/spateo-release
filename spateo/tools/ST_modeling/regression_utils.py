"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
import multiprocessing as mp
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

from joblib import Parallel, delayed

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.stats.multitest
import tensorflow as tf
from numpy import linalg
from scipy.linalg import cholesky, solve_triangular
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.gam.smooth_basis import get_knots_bsplines
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(1, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.configuration import SKM
from spateo.logging import logger_manager as lm
from spateo.preprocessing.normalize import calcNormFactors
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
        return betas, pseudoinverse, cov_inverse

    betas = np.dot(xtx_inv_xt, y)
    # Upper and lower bound to constrain betas and prevent numerical overflow:
    betas = np.clip(betas, -clip, clip)
    # And set to zero with small offset for numerical overflow if the diagonal of the Gram matrix is below a certain
    # threshold:
    betas[to_zero] = 1e-6

    return betas, pseudoinverse, cov_inverse


def iwls(
    y: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    x: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix],
    distr: Literal["gaussian", "poisson", "nb", "binomial"] = "gaussian",
    init_betas: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    clip: float = 5.0,
    threshold: float = 1e-3,
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
        offset: Optional array of shape [n_samples,]; if provided, will be added to the linear predictor. This is
            meant to deal with differences in scale that are not caused by the predictor variables,
            e.g. by differences in library size
        tol: Convergence tolerance
        clip: Sets magnitude of the upper and lower bound to constrain betas and prevent numerical overflow
        threshold: Coefficients with absolute values below this threshold will be set to zero (as these are
            insignificant)
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
# Differential testing functions
# ---------------------------------------------------------------------------------------------------
def fit_DE_GAM(
    diff_sending_or_receiving: Literal["sending", "receiving"],
    endog: Union[np.ndarray, scipy.sparse.spmatrix],
    exog: Union[np.ndarray, scipy.sparse.spmatrix],
    genes: List[str],
    cells: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    n_df: int = 8,
    parallel: bool = True,
    family: Literal["gaussian", "poisson", "nb"] = "nb",
    include_offset: bool = False,
    label: Optional[str] = None,
) -> Tuple[anndata.AnnData, Dict[str, BSplines]]:
    """Fit generalized additive models for the purpose of differential expression testing from spatial
    transcriptomics data.

    Args:
        diff_sending_or_receiving: Here, `genes` refers to either the labels for the endogenous variable array (as in
            the case of "receiving", where we identify genes associated with the received effect) or for the exogenous
            variable array (as in the case of "sending", where we identify genes associated with the expression of
            particular ligand(s)). Note that for "sending", the endogenous array must be 1D.
        endog: Matrix of dependent/target variables- for received effect DEGs, this is the gene expression counts,
            with cells as rows and genes as columns (so shape [n_samples, n_genes]). This can also be 1D and
            constitute expression of a particular ligand.
        exog: The continuous predictor variable(s). For received effect DEGs, this is the received effect potential.
            For sent effect DEGs, this is the gene expression counts, cells as rows and genes as columns (so shape [
            n_samples, n_genes]).
        genes: List of genes present in the gene expression array
        cells: Optional list of cell names
        offset: Optional offset term of shape [n_samples, ]
        n_df: The number of degrees of freedom to be used in the smoothing function. Defaults to 8.
        parallel: Whether to use parallel processing. Defaults to True.
        family: The family of the generalized additive model. Defaults to "nb" for negative binomial, with "gaussian"
            and "poisson" as other options.
        include_offset: If True, include offset to account for differences in library size in predictions. If True,
            will compute scaling factor using trimmed mean of M-value with singleton pairing (TMMswp).
        label: Optional label for the AnnData object, will be used for downstream plotting purposes. This should
            somehow describe the signal of interest (e.g. the ligand, downstream target of interest, etc.)

    Returns:
        sc_obj: AnnData object containing information gathered from fitting all GAM models- this includes p-values,
            Wald statistics, etc.
        basis: Dictionary containing BSplines object with information about the fitted splines
    """

    if weights is None:
        weights = np.ones((endog.shape[0], 1))

    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    if endog.ndim == 1:
        endog = endog.reshape(-1, 1)

    if scipy.sparse.issparse(exog):
        exog = exog.toarray()
    if scipy.sparse.issparse(endog):
        endog = endog.toarray()

    if diff_sending_or_receiving == "sending" and endog.shape[1] > 1:
        raise ValueError(
            "For this configuration, ('diff_sending_or_receiving' = 'sending'), dependent variable array must be 1D. "
            "For multiple targets, run this function once for each target."
        )
    elif diff_sending_or_receiving == "receiving" and exog.shape[1] > 1:
        raise ValueError(
            "For this configuration, ('diff_sending_or_receiving' = 'receiving'), predictor variable array must be "
            "1D. For multiple predictors, run this function once for each predictor."
        )

    targets = {}
    predictors = {}
    subsetted_cells = {}
    n_subset = {}
    # Measure converged and/or genes that had enough cells post-filtering to model:
    fitted = {}
    pvals = {}
    wald_stats = {}
    basis = {}
    # For each independent-dependent relationship, subset to cells that have nonzero values for both- gene-level
    # relationships can be multifaceted due to promiscuity and so presence or absence of gene expression can be
    # misleading- we can use cells for which both signals are present to get a sense of the true context-independent
    # effect of one on the other:

    if diff_sending_or_receiving == "receiving":
        # Check non-negativity:
        if (exog < 0).any():
            raise ValueError("All values of the count matrix should be non-negative")

        for idx in range(endog.shape[1]):
            gene = genes[idx]
            to_subset = np.nonzero(np.logical_and(np.any(exog != 0, axis=1), endog[:, idx] != 0))[0]
            # Outliers will obscure the true effect- remove them:
            temp_endog = endog[to_subset, idx]
            temp_exog = exog[to_subset, :]
            z_scores_endog = (temp_endog - np.mean(temp_endog)) / np.std(temp_endog)
            outliers_endog = np.where(np.abs(z_scores_endog) > 3)[0]
            z_scores_exog = (temp_exog - np.mean(temp_exog)) / np.std(temp_exog)
            outliers_exog = np.where(np.abs(z_scores_exog) > 3)[0]
            outliers = np.union1d(outliers_endog, outliers_exog)

            to_subset = np.setdiff1d(to_subset, outliers)

            # If the size of the subset is not large enough, set this gene to insignificant and set p-values and Wald
            # stats to 1 and 0, respectively:
            if len(to_subset) < 20:
                fitted[gene] = False
                pvals[gene] = 1
                wald_stats[gene] = 0
                continue
            else:
                # Placeholders:
                fitted[gene] = True
                pvals[gene] = np.nan
                wald_stats[gene] = np.nan
            targets[gene] = endog[to_subset, idx]

            # The independent variable(s) are shared, but subsetting differences may exist, so store the determined
            # subset for each gene:
            predictors[gene] = exog[to_subset, :]
            # Store indices of subset:
            subsetted_cells[gene] = to_subset
            n_subset[gene] = len(to_subset)

    elif diff_sending_or_receiving == "sending":
        # Check non-negativity:
        if (endog < 0).any():
            raise ValueError("All values of the count matrix should be non-negative")

        for idx in range(exog.shape[1]):
            gene = genes[idx]
            to_subset = np.nonzero(np.logical_and(np.any(endog != 0, axis=1), exog[:, idx] != 0))[0]
            # Outliers will obscure the true effect- remove them:
            temp_endog = endog[to_subset, :]
            temp_exog = exog[to_subset, idx]
            z_scores_endog = (temp_endog - np.mean(temp_endog)) / np.std(temp_endog)
            outliers_endog = np.where(np.abs(z_scores_endog) > 3)[0]
            z_scores_exog = (temp_exog - np.mean(temp_exog)) / np.std(temp_exog)
            outliers_exog = np.where(np.abs(z_scores_exog) > 3)[0]
            outliers = np.union1d(outliers_endog, outliers_exog)

            to_subset = np.setdiff1d(to_subset, outliers)

            # If the size of the subset is not large enough, set this gene to insignificant and set p-values and Wald
            # stats to 1 and 0, respectively:
            if len(to_subset) < 20:
                fitted[gene] = False
                pvals[gene] = 1
                wald_stats[gene] = 0
                continue
            else:
                # Placeholders:
                fitted[gene] = True
                pvals[gene] = np.nan
                wald_stats[gene] = np.nan
            targets[gene] = exog[to_subset, idx]

            # The independent variable(s) are shared, but subsetting differences may exist, so store the determined
            # subset for each gene:
            predictors[gene] = endog[to_subset, :]
            # Store indices of subset:
            subsetted_cells[gene] = to_subset
            n_subset[gene] = len(to_subset)

    else:
        raise ValueError("The independent variable array must be 2D- one-to-one tests currently not supported.")

    if np.isnan(exog).any():
        raise ValueError("Predictors array contains NA values.")
    if np.isnan(endog).any():
        raise ValueError("Targets array contains NA values.")

    counts = endog if diff_sending_or_receiving == "receiving" else exog
    if include_offset:
        # Get library scaling factors
        offset = library_scaling_factors(counts=counts, distr=family)

    # Fit the GAMs:
    # Define family based on provided input argument- first check if the dependent variable is binary- a logistic
    # regression should be applied in this case. This will only happen in the case of a "sending" analysis where the
    # dependent variable is the cell type:
    binary_endog = False
    if all(x in [0, 1] for x in endog):
        binary_endog = True
        family = sm.families.Binomial()

    if not binary_endog:
        if family == "gaussian":
            family = sm.families.Gaussian()
        elif family == "poisson":
            family = sm.families.Poisson()
        elif family == "nb":
            family = sm.families.NegativeBinomial()

    # For each combination of predictor-target, define cubic regression splines:
    for gene in targets.keys():
        basis[gene] = BSplines(predictors[gene], df=[n_df], degree=[3], include_intercept=True)

    # Valid genes:
    genes_to_model = [gene for gene in genes if gene in targets.keys()]
    # Parallel processing:
    if parallel:
        args_list = [
            (targets[gene], gene, weights, offset, predictors[gene], basis[gene], family) for gene in genes_to_model
        ]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            gamList = pool.starmap(counts_to_Gam, args_list)
    else:
        gamList = [
            counts_to_Gam(targets[gene], gene, weights, offset, predictors[gene], basis[gene], family)
            for gene in genes_to_model
        ]

    # Store results in AnnData object:
    sc_obj = anndata.AnnData(X=counts)
    sc_obj.var_names = genes
    if cells is not None:
        sc_obj.obs_names = cells

    # Filter for genes that converged:
    gamList = [gam for gam in gamList if gam.converged]
    geneList = [gam.gene for gam in gamList]
    # Store in format that can be accessed by gene labels:
    gamList = {gam.gene: gam for gam in gamList}

    # Get fitted coefficients to dataframe:
    betaAll = []
    SigmaAll = []

    for i, m in enumerate(gamList):
        beta = pd.DataFrame(m.params.values.reshape(1, -1), index=[geneList[i]], columns=m.params.index)
        betaAll.append(beta)
        # Variance-covariance matrix:
        Sigma = m.cov_params()
        # Convert to dataframe:
        SigmaAll.append(Sigma)
    betaAllDf = pd.DataFrame(np.concatenate(betaAll, axis=0), index=geneList, columns=m.params.index)

    df_list = []
    for sigma in SigmaAll:
        df_matrix = pd.DataFrame(sigma)
        df_list.append(df_matrix)
    # Concatenate:
    df = pd.concat(df_list, keys=geneList, names=["Target gene"])
    sc_obj.uns["Sigma"] = df
    sc_obj.uns["beta"] = betaAllDf

    # To collect knot points:
    knotpoints = {}

    # Store design matrix (linear predictor) for each GAM model:
    for gene in geneList:
        gam_obj = gamList[gene]
        sc_obj.obsm[f"lin_pred_{gene}"] = gam_obj.model.exog
        sc_obj.uns[f"var_{gene}"] = predictors[gene]

        # Get knot points:
        all_points = get_knots_bsplines(
            predictors[gene],
            df=n_df,
            degree=3,
            spacing="quantile",
            lower_bound=np.min(predictors[gene]),
            upper_bound=np.max(predictors[gene]),
        )
        knotpoints[gene] = list(set(all_points))

    sc_obj.var["fitted"] = pd.Series(fitted)
    sc_obj.var["knotpoints"] = pd.Series(knotpoints)
    sc_obj.var["pvals"] = pd.Series(pvals)
    sc_obj.var["wald_stats"] = pd.Series(wald_stats)
    sc_obj.var["subsetted_cells"] = pd.Series(subsetted_cells)
    sc_obj.var["n_subset"] = pd.Series(n_subset)
    if label is not None:
        sc_obj.uns["label"] = label

    return sc_obj, basis


class GAM_DummyModel:
    """Dummy class to store whether model converged or not."""

    def __init__(self, label: str, converged: bool = False):
        self.label = label
        self.converged = converged


def counts_to_Gam(
    y: Union[np.ndarray, scipy.sparse.spmatrix],
    label: str,
    weights: np.ndarray,
    offset: Optional[np.ndarray],
    X: np.ndarray,
    bs: BSplines,
    family: sm.families.family.Family,
):
    """Fitting function- fits a GAM model to a single variable.

    Args:
        y: Continuous values for a single variable
        label: Name of the variable being fit
        converged: Indicator for whether the fitting converged for this gene
        parallel: Whether parallel processing is being used
        n_df: Number of degrees of freedom to use for the cubic regression splines
        weights: Weights for each cell
        offset: Optional offset for each cell
        X: Explanatory variables
        bs: Cubic regression splines corresponding to the independent variable(s)
        family: Family to use for the GLM

    Returns:
        gam_fit: Fitted GAM model
    """
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    data = data.sort_index(axis=1)

    if scipy.sparse.issparse(y):
        y = y.toarray()
    data["y"] = y

    # Construct formula from dataframe columns:
    colnames = data.columns.tolist()
    colnames.remove("y")
    # Fit without intercept
    formula_string = "y ~ -1 + " + " + ".join(colnames)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            gam_model = GLMGam.from_formula(
                formula_string, data=data, smoother=bs, family=family, weights=weights, offset=offset
            )
            gam_fit = gam_model.fit()

            # Keep track of the label and whether the model converged:
            gam_fit.label = label
            gam_fit.converged = True
    except Exception:
        print(f"\nConvergence not achieved for {label}.")
        # Dummy model to store results:
        gam_fit = GAM_DummyModel(label, False)

    return gam_fit


def DE_GAM_test(
    GAM_adata: anndata.AnnData,
    bs_obj: Dict[str, BSplines],
    lfc_thresh: float = 0.0,
    contrast_type: Literal["start", "end", "consecutive"] = "start",
    n_points: Optional[int] = None,
    inverse: Literal["cholesky", "qr", "generalized"] = "cholesky",
    fdr_method: str = "fdr_bh",
) -> Tuple[anndata.AnnData, Dict[str, BSplines]]:
    """Statistical testing for outputs from GAM model(s).

    Args:
        GAM_adata: AnnData object containing results of GAM models
        bs_obj: Dictionary containing BSplines objects with information about the basis functions used for the
            GAM models
        l2fc_thresh: The threshold for natural log fold change (default is 0)
        contrast_type: For use when constructing the contrast matrix, in this context to compare expression levels
            along the continuous predictor. Options:
                - "start": compares the expression levels at each contrast point with the expression level at the
                    starting point of the predictor
                - "end": compares the expression levels at each contrast point with the expression level at the
                    ending point of the predictor
                - "consecutive": compares the expression levels between consecutive contrast points. Computes the
                    difference in expression levels between each point and the immediately preceding point.
        n_points: The number of contrast points to use. If not given, defaults to 2 * the number of knot points.
        inverse: The method for inverting the covariance matrix. Options are "cholesky" (for Cholesky decomposition),
            "qr" (for QR decomposition), or "generalized" (for inverse using the Moore-Penrose pseudoinverse).
        fdr_method: Method to use for correction. Available methods can be found in the documentation for
                :func `statsmodels.stats.multitest.multipletests()`, and are also listed below (in correct case) for
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

    Returns:
        GAM_adata: AnnData object with results of statistical testing added
        bs: BSplines object containing information about the basis splines
    """
    betaAll = GAM_adata.uns["beta"]
    SigmaAll = GAM_adata.uns["Sigma"]

    # Wald test results for each fitted gene and store results in DataFrame:
    for gene in GAM_adata.var_names:
        if GAM_adata.var["fitted"][gene]:
            # BSplines for this gene:
            bs_obj_gene = bs_obj[gene]

            # Design matrix and linear predictor for the fitted gene:
            design_matrix = GAM_adata.uns[f"var_{gene}"]
            linear_predictor = GAM_adata.obsm[f"lin_pred_{gene}"]

            # Get knot points:
            knot_points = GAM_adata.var["knotpoints"][gene]
            if n_points is None:
                n_points = 2 * len(knot_points)

            # Construct the contrast matrix:
            max_val = np.max(design_matrix)

            contrast_matrix = pd.DataFrame(np.zeros((linear_predictor.shape[1], n_points - 1)))
            # Set column names for L1 matrix
            contrast_matrix.columns = ["point" + str(i) for i in range(1, n_points)]
            # Get predictor matrix
            contrastPoints = np.linspace(0, max_val, num=n_points).reshape(-1, 1)
            # Compute linear predictor at the contrast points:
            exog_smooth_interp = bs_obj_gene.transform(contrastPoints)
            exog_smooth_pred = np.hstack((contrastPoints, exog_smooth_interp))

            # Fill in contrast matrix:
            if contrast_type == "start":
                for i in range(1, n_points):
                    contrast_matrix.iloc[:, i - 1] = exog_smooth_pred[i, :] - exog_smooth_pred[0, :]

            elif contrast_type == "end":
                for i in range(n_points - 1):
                    contrast_matrix.iloc[:, i] = exog_smooth_pred[i, :] - exog_smooth_pred[n_points - 1, :]

            elif contrast_type == "consecutive":
                for i in range(n_points - 1):
                    contrast_matrix.iloc[:, i] = exog_smooth_pred[i + 1, :] - exog_smooth_pred[i, :]

            Sigma_gene = SigmaAll.loc[gene].values
            beta_gene = betaAll.loc[gene].values

            wald_stat_gene, _, pval_gene = wald_test_GAM(
                beta_gene, Sigma_gene, contrast_matrix, lfc=lfc_thresh, inverse=inverse
            )

            # Store results in AnnData object:
            GAM_adata.var["pvals"].loc[gene] = pval_gene
            GAM_adata.var["wald_stats"].loc[gene] = wald_stat_gene

    qvals = multitesting_correction(GAM_adata.var["pvals"], method=fdr_method)
    GAM_adata.var["qvals"] = qvals

    return GAM_adata, bs_obj


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


def wald_test_GAM(
    beta: np.ndarray,
    Sigma: np.ndarray,
    contrast_mat: Union[np.ndarray, pd.DataFrame],
    lfc: float = 0.0,
    inverse: str = "QR",
) -> Tuple[float, int, float]:
    """Variant of the Wald test function for generalized additive models, computes Wald statistic and p-value
        considering each independent variable alongside all of its spline bases.

    Args:
        beta: Vector of regression coefficients
        Sigma: Variance-covariance matrix for beta
        contrast_mat: Contrast matrix
        lfc: Natural log fold change threshold. Default is 0.
        inverse: Method to use for inverting Sigma. Options are "cholesky" (for Cholesky decomposition),
            "qr" (for QR decomposition), or "generalized" (for inverse using the Moore-Penrose pseudoinverse).

    Returns:
        wald: The Wald statistic for this feature (i.e. for this beta vector/sigma array)
        df: Degrees of freedom for this feature
        pval: p-value for this feature
    """
    # If contrast matrix is given in another form:
    if isinstance(contrast_mat, pd.DataFrame):
        contrast_mat = contrast_mat.values

    # Contrast matrix for multivariate Wald test:
    # R = upper triangular matrix, P = permutation matrix (mapping between the permuted matrix and the original
    # matrix- needed if pivoting is done for numerical stability)
    _, R, P = scipy.linalg.qr(contrast_mat, pivoting=True)
    rank = np.linalg.matrix_rank(R)
    mod_contrast_matrix = contrast_mat[:, P[:rank]]

    # Invert Sigma
    if inverse == "cholesky":
        try:
            sigmaInv = np.linalg.inv(np.linalg.cholesky(mod_contrast_matrix.T @ Sigma @ mod_contrast_matrix))
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan

    elif inverse == "qr":
        try:
            sigmaInv = np.linalg.lstsq(
                mod_contrast_matrix.T @ Sigma @ mod_contrast_matrix, np.eye(mod_contrast_matrix.shape[1]), rcond=None
            )[0]
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan

    elif inverse == "generalized":
        try:
            sigmaInv = np.linalg.pinv(mod_contrast_matrix.T @ Sigma @ mod_contrast_matrix)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan

    else:
        raise ValueError(f"Invalid value for 'inverse' argument: {inverse}. Options: 'cholesky', 'qr', 'generalized'.")

    # Differential testing with a likelihood ratio test:
    # Estimated log-fold change:
    est_fc = mod_contrast_matrix.T @ beta
    # Do not consider features with absolute log-fold change below threshold (will not do anything if lfc is the
    # default of 0):
    est = np.sign(est_fc) * np.maximum(0, np.abs(est_fc) - lfc)
    est = np.reshape(est, (1, est.shape[0]))

    # Wald statistic and p-value:
    wald = np.matmul(np.matmul(est, sigmaInv), est.T)
    if wald < 0:
        wald = 0
    df = mod_contrast_matrix.shape[1]
    pval = 1 - scipy.stats.chi2.cdf(wald, df)
    # Convert to floating point:
    if not np.isscalar(wald):
        wald = wald.item()
    if not np.isscalar(pval):
        pval = pval.item()

    return wald, df, pval


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
