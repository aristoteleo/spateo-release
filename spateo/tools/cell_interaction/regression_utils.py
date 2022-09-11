"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
from anndata import AnnData


# ------------------------------------------- Ordinary Least-Squares ------------------------------------------- #
# Custom implementation of ordinary least-squares regression for an AnnData object:
def ols_fit(
    X: pd.DataFrame, adata: AnnData, x_feats: List[str], y_feat: str, layer: Union[None, str] = None
) -> np.ndarray:
    """
    Ordinary least squares regression for a single variable

    Args:
        X : pd.DataFrame
            Contains data to be used as independent variables in the regression
        adata : class `anndata.AnnData`
            AnnData object to store results in
        x_feats : list of str
            Names of the features to use in the regression. Must be present as columns of 'X'.
        y_feat : str
            Name of the feature to regress on. Must be present in adata 'var_names'.
        var_names : list of str
            Names of variables corresponding to regression parameters (e.g. cell types or combinations of cell types)
        layer : optional str
            Can specify layer of adata to use. If not given, will use .X.

    Returns:
        Beta : np.ndarray, shape [n_features, n_parameters]
            Contains parameter for each column in y_feat
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


def ols_fit_predict(
    X: pd.DataFrame, adata: AnnData, x_feats: List[str], y_feat: str, layer: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For single variable, fits ordinary least squares model and then uses the fitted parameters to predict dependent
    feature expression.

    Args: see :func `ols_fit` docstring.

    Returns:
        Beta : np.ndarray, shape [n_features, n_parameters]
            Contains parameter for each column in y_feat
        rex : np.ndarray, shape [n_samples, n_y_feats]
            Reconstructed independent variable values
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
