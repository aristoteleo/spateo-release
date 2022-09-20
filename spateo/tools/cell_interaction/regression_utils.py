"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
from anndata import AnnData
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split

from ...configuration import SKM
from ...logging import logger_manager as lm


# ------------------------------------------- Ordinary Least-Squares ------------------------------------------- #
# Custom implementation of ordinary least-squares regression for an AnnData object:
def ols_fit(
    X: pd.DataFrame, adata: AnnData, x_feats: List[str], y_feat: str, layer: Union[None, str] = None
) -> np.ndarray:
    """
    Ordinary least squares regression for a single variable

    Args:
        X: Contains data to be used as independent variables in the regression
        adata: Object of class `anndata.AnnData` to store results in
        x_feats: Names of the features to use in the regression. Must be present as columns of 'X'.
        y_feat: Name of the feature to regress on. Must be present in adata 'var_names'.
        layer: Can specify layer of adata to use. If not given, will use .X.

    Returns:
        Beta : Array of shape [n_parameters, 1]. Contains weight for each parameter.
    """

    # Beta = (X^T * X)^-1 * X^T * y
    if layer is None:
        X["log_expr"] = adata[:, y_feat].X.A
    else:
        X["log_expr"] = adata[:, y_feat].layers[layer].A
    y = X["log_expr"].values

    # Get values corresponding to the features to be used as regressors:
    x = X[x_feats].values

    res = np.matmul(np.linalg.pinv(np.matmul(x.T, x)), x.T)

    Beta = np.matmul(res, y)
    return Beta


def ols_predict(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Given predictor values and parameter values, reconstruct dependent expression"""
    ypred = np.dot(X, params)
    return ypred


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def ols_fit_predict(
    X: pd.DataFrame, adata: AnnData, x_feats: List[str], y_feat: str, layer: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For single variable, fits ordinary least squares model and then uses the fitted parameters to predict dependent
    feature expression.

    Args: see :func `ols_fit` docstring.

    Returns:
        Beta: Array of shape [n_parameters, 1], contains weight for each parameter
        rex: Array of shape [n_samples, 1]. Reconstructed independent variable values.
    """

    # Beta = (X^T * X)^-1 * X^T * y
    if layer is None:
        X["log_expr"] = adata[:, y_feat].X.A
    else:
        X["log_expr"] = adata[:, y_feat].layers[layer].A
    y = X["log_expr"].values

    # Get values corresponding to the features to be used as regressors:
    x = X[x_feats].values

    res = np.matmul(np.linalg.pinv(np.matmul(x.T, x)), x.T)

    Beta = np.matmul(res, y)

    rex = ols_predict(x, Beta)
    return Beta, rex


# ---------------------------------------- LASSO Ordinary Least-Squares ---------------------------------------- #
def lasso_fit(
    X: pd.DataFrame,
    adata: AnnData,
    x_feats: List[str],
    y_feat: str,
    iterations: int,
    l1_penalty: float = 0.2,
    test_size: float = 0.2,
    num_folds: Union[None, int] = None,
    layer: Union[None, str] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
    """
    For single variable, trains a Lasso least squares regression model. Has the capability of dealing with
    multicollinear data.

    Args:
        X: Contains data to be used as independent variables in the regression
        adata: Object of class `anndata.AnnData` to store results in
        x_feats: Names of the features to use in the regression. Must be present as columns of 'X'.
        y_feat: Name of the feature to regress on. Must be present in adata 'var_names'.
        iterations: Number of weight updates to perform
        l1_penalty: Corresponds to lambda, the strength of the regularization- higher values lead to increasingly
            strict weight shrinkage. This set value will only be used if 'num_folds' is given. Defaults to 0.2.
        test_size: Size of the evaluation set, given as a proportion of the total dataset size. Should be between 0
            and 1, exclusive.
        num_folds: Can be used to specify number of folds for cross-validation. If not given, will not perform
            cross-validation.
        layer: Can specify layer of adata to use. If not given, will use .X.

    Returns:
        W: Array, shape [n_parameters, 1]. Contains weight for each parameter.
        b: Array, shape [n_parameters, 1]. Intercept term.
        alpha: float, only returns if 'num_folds' is given. This is the best penalization value as chosen by
        cross-validation
    """

    if layer is None:
        X["log_expr"] = adata[:, y_feat].X.A
    else:
        X["log_expr"] = adata[:, y_feat].layers[layer].A
    y = X["log_expr"].values

    # Get values corresponding to the features to be used as regressors:
    print(X[x_feats])
    x = X[x_feats].values

    # Model initialization and fitting:
    if num_folds is None:
        # Split data into training and test set (training will be used for fitting):
        x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, random_state=0)

        mod = Lasso(alpha=l1_penalty, max_iter=iterations)
        mod.fit(x_train, y_train)
        W = mod.coef_
        b = mod.intercept_
        return W, b
    else:
        mod = LassoCV(cv=num_folds, random_state=42)
        mod.fit(x, y)
        W = mod.coef_
        b = mod.intercept_
        # The best value for alpha chosen by the cross-validation:
        alpha = mod.alpha_
        return W, b, alpha


def lasso_predict(X: np.ndarray, params: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Given predictor values and parameter values, reconstruct dependent expression

    Args:
        X: independent feature array
        params: Parameter vector
        b: Intercept/independent term in decision function
    """
    return X.dot(params) + b


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def lasso_fit_predict(
    X: pd.DataFrame,
    adata: AnnData,
    x_feats: List[str],
    y_feat: str,
    iterations: int,
    l1_penalty: float = 0.2,
    test_size: float = 0.2,
    num_folds: Union[None, int] = None,
    layer: Union[None, str] = None,
):
    """
    For single variable, fits Lasso least squares model and then uses the fitted parameters to predict dependent
    feature expression.

    Args: see :func `lasso_fit` docstring.

    Returns:
        W: Array of shape [n_parameters, 1], contains weight for each parameter
        rex: Array of shape [n_samples, 1]. Reconstructed independent variable values.
    """

    logger = lm.get_main_logger()

    # Define model, learn weights:
    if num_folds is None:
        W, b = lasso_fit(X, adata, x_feats, y_feat, iterations, l1_penalty, test_size, None, layer)
    else:
        logger.info(f"Initializing cross-validation model with {num_folds} folds, regressing on {y_feat}.")
        W, b, alpha_opt = lasso_fit(X, adata, x_feats, y_feat, iterations, l1_penalty, test_size, num_folds, layer)
        logger.info(f"Optimal L1 penalty term for {y_feat}: {alpha_opt}")

        # Re-fit model based on the optimal l1 penalty discovered by cross-validation on all data:
        if layer is None:
            X["log_expr"] = adata[:, y_feat].X.A
        else:
            X["log_expr"] = adata[:, y_feat].layers[layer].A
        y = X["log_expr"].values

        # Get values corresponding to the features to be used as regressors:
        x = X[x_feats].values

        lasso_best = Lasso(alpha=alpha_opt, max_iter=iterations)
        lasso_best.fit(x, y)
        # Final weights and intercept:
        W = lasso_best.coef_
        b = lasso_best.intercept_

    # Prediction on entire dataset:
    rex = lasso_predict(X[x_feats].values, W, b)

    return W, rex


# ------------------------------------------- Significance Testing ------------------------------------------- #
def get_fisher_inverse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Fisher matrix that measures the amount of information each feature in x provides about y- that is,
    whether the log-likelihood is sensitive to change in the parameter x.

    Args:
        x : np.ndarray
            Independent variable array
        y : np.ndarray
            Dependent variable array

    Returns:
        inverse_fisher : np.ndarray
    """

    var = np.var(y, axis=0)
    fisher = np.expand_dims(np.matmul(x.T, x), axis=0) / np.expand_dims(var, axis=[1, 2])

    fisher = np.nan_to_num(fisher)

    inverse_fisher = np.array([np.linalg.pinv(fisher[i, :, :]) for i in range(fisher.shape[0])])
    return inverse_fisher


def wald_test(theta_mle: np.ndarray, theta_sd: np.ndarray, theta0: Union[float, np.ndarray] = 0) -> np.ndarray:
    """
    Perform single-coefficient Wald test, informing whether a given coefficient deviates significantly from the
    supplied reference value (theta0), based on the standard deviation of the posterior of the parameter estimate.

    Args:
        theta_mle : np.ndarray
            Maximum likelihood estimation of given parameter by feature
        theta_sd : np.ndarray
            Standard deviation of the maximum likelihood estimation
        theta0 : float or np.ndarray
            Value(s) to test theta_mle against. Must be either a single number or an array w/ equal number of entries
            to theta_mle.

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
    """
    In the case of testing multiple hypotheses from the same experiment, perform multiple test correction to adjust
    q-values.

    Args:
    pvals : np.ndarray
        Uncorrected p-values; must be given as a one-dimensional array
    method : str, default 'fdr_bh'
        Method to use for correction. Available methods can be found in the documentation for
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
    alpha : float, default 0.05
        Family-wise error rate (FWER)

    Returns
        qval : np.ndarray
            p-values post-correction
    """

    qval = np.zeros([pvals.shape[0]]) + np.nan
    qval[np.isnan(pvals) == False] = statsmodels.stats.multitest.multipletests(
        pvals=pvals[np.isnan(pvals) == False], alpha=alpha, method=method, is_sorted=False, returnsorted=False
    )[1]

    return qval


def _get_p_value(variables: np.array, fisher_inv: np.array, coef_loc_totest: int) -> np.ndarray:
    """
    Computes p-values for differential expression for each feature

    Args:
        variables : np.ndarray
            Array where each column corresponds to a feature
        fisher_inv : np.ndarray
            Inverse Fisher information matrix
        coef_loc_totest : int
            Numerical column of the array corresponding to the coefficient to test

    Returns:
        pvalues : np.ndarray
            Array of identical shape to variables, where each element is a p-value for that instance of that feature
    """

    theta_mle = variables[coef_loc_totest]
    theta_sd = fisher_inv[:, coef_loc_totest, coef_loc_totest]
    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    theta_sd = np.sqrt(theta_sd)

    pvalues = wald_test(theta_mle, theta_sd, theta0=0.0)
    return pvalues


def compute_wald_test(
    params: np.ndarray, fisher_inv: np.ndarray, significance_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        params : np.ndarray
            Array of shape [n_features, n_params]
        fisher_inv : np.ndarray
            Inverse Fisher information matrix
        significance_threshold : float, default 0.01
            Upper threshold to be considered significant

    Returns
        significance : np.ndarray
            Array of identical shape to variables, where each element is True or False if it meets the threshold for
            significance
        pvalues : np.ndarray
            Array of identical shape to variables, where each element is a p-value for that instance of that feature
        qvalues : np.ndarray
            Array of identical shape to variables, where each element is a q-value for that instance of that feature
    """

    pvalues = []

    # Compute p-values for each feature, store in temporary list:
    for idx in range(params.T.shape[0]):
        pvals = _get_p_value(params.T, fisher_inv, idx)
        pvalues.append(pvals)

    pvalues = np.concatenate(pvalues)
    # Multiple testing correction w/ Benjamini-Hochberg procedure and FWER 0.05
    qvalues = multitesting_correction(pvalues)
    pvalues = np.reshape(pvalues, (-1, params.T.shape[1]))
    qvalues = np.reshape(qvalues, (-1, params.T.shape[1]))
    significance = qvalues < significance_threshold

    return significance, pvalues, qvalues
