"""
General linear model regression for spatially-aware regression. Assumes the response variable follows the normal
distribution.
"""
from typing import List, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
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