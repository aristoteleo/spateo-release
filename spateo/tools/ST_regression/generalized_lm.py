"""
Generalized linear model regression for spatially-aware regression of spatial transcriptomic (gene expression) data.
Rather than assuming the response variable necessarily follows the normal distribution, instead allows the
specification of models whose response variable follows different distributions (e.g. Poisson or Gamma),
although allows also for normal (Gaussian) modeling.
Additionally features capability to perform elastic net regularized regression.
"""
import time
from typing import List, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import diags, issparse
from scipy.special import expit, loggamma
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ...configuration import SKM
from ...logging import logger_manager as lm
from ...preprocessing.normalize import normalize_total
from ...preprocessing.transform import log1p
from ...tools.find_neighbors import transcriptomic_connectivity
from .regression_utils import L1_L2_penalty, softplus


# ---------------------------------------------------------------------------------------------------
# Intermediates for generalized linear modeling
# ---------------------------------------------------------------------------------------------------
def _z(beta0: float, beta: np.ndarray, X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    """Computes z, an intermediate comprising the result of a linear regression, just before non-linearity is applied.

    Args:
        beta0: The intercept
        beta: Array of shape [n_features,]; learned model coefficients
        X: Array of shape [n_samples, n_features]; input data
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
            Defaults to True.

    Returns:
        z: Array of shape [n_samples, n_features]; prediction of the target values
    """
    if fit_intercept:
        z = beta0 + np.dot(X, beta)
    else:
        z = np.dot(X, np.r_[beta0, beta])
    return z


def _nl(
    distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"],
    z: np.ndarray,
    eta: float,
    fit_intercept: bool,
) -> np.ndarray:
    """Applies nonlinear operation to linear estimation.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma"
        z: Array of shape [n_samples, n_features]; prediction of the target values
        eta: A threshold parameter that linearizes the exp() function above threshold eta
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
            Defaults to True.

    Returns:
         nl: An array of size [n_samples, n_features]; result following application of the nonlinear layer
    """
    logger = lm.get_main_logger()

    if distr in ["softplus", "gamma", "neg-binomial"]:
        nl = softplus(z)
    elif distr == "poisson":
        nl = z.copy()
        beta0 = (1 - eta) * np.exp(eta) if fit_intercept else 0.0
        nl[z > eta] = z[z > eta] * np.exp(eta) + beta0
        nl[z <= eta] = np.exp(z[z <= eta])
    elif distr == "gaussian":
        nl = z

    return nl


def _grad_nl(distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"], z: np.ndarray, eta: float):
    """Derivative of the non-linearity.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma"
        z: Array of shape [n_samples, n_features]; prediction of the target values
        eta: A threshold parameter that linearizes the exp() function above threshold eta

    Returns:
        grad_nl: Array of size [n_samples, n_features]; first derivative of each parameter estimate
    """
    logger = lm.get_main_logger()

    if distr in ["softplus", "gamma", "neg-binomial"]:
        grad_nl = expit(z)
    elif distr == "poisson":
        grad_nl = z.copy()
        grad_nl[z > eta] = np.ones_like(z)[z > eta] * np.exp(eta)
        grad_nl[z <= eta] = np.exp(z[z <= eta])
    elif distr == "gaussian":
        grad_nl = np.ones_like(z)

    return grad_nl


# ---------------------------------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------------------------------
def batch_grad(
    distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"],
    alpha: float,
    reg_lambda: float,
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    Tau: Union[None, np.ndarray] = None,
    eta: float = 2.0,
    theta: float = 1.0,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Computes the gradient (for parameter updating) via batch gradient descent

    Args:
        distr: Distribution family- can be "gaussian", "softplus", "poisson", "neg-binomial", or "gamma". Case
            sensitive.
        alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.)
            term of the loss function
        reg_lambda: Regularization parameter :math:`\\lambda` of penalty term
        X: Array of shape [n_samples, n_features]; input data
        y: Array of shape [n_samples, 1]; labels or targets for the data
        beta: Array of shape [n_features,]; learned model coefficients
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
            provided, Tau will default to the identity matrix.
        eta: A threshold parameter that linearizes the exp() function above threshold eta
        theta: Shape parameter of the negative binomial distribution (number of successes before the first failure).
            Used only if 'distr' is "neg-binomial"
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
            Defaults to True.

    Returns:
        g: Gradient for each parameter
    """
    n_samples, n_features = X.shape
    n_samples = np.float(n_samples)

    if Tau is None:
        if fit_intercept:
            Tau = np.eye(beta[1:].shape[0])
        else:
            Tau = np.eye(beta.shape[0])
    InvCov = np.dot(Tau.T, Tau)

    # Compute linear intermediate, nonlinearity, first derivative of the nonlinearity
    z = _z(beta[0], beta[1:], X, fit_intercept)
    nl = _nl(distr, z, eta, fit_intercept)
    grad_nl = _grad_nl(distr, z, eta)

    # Initialize gradient:
    grad_beta0 = 0.0

    if distr in ["poisson", "softplus"]:
        if fit_intercept:
            grad_beta0 = np.sum(grad_nl) - np.sum(y * grad_nl / nl)
        grad_beta = (np.dot(grad_nl.T, X) - np.dot((y * grad_nl / nl).T, X)).T

    elif distr == "gamma":
        # Degrees of freedom (one because the parameter array is 1D)
        nu = 1.0
        grad_logl = (y / nl**2 - 1 / nl) * grad_nl
        if fit_intercept:
            grad_beta0 = -nu * np.sum(grad_logl)
        grad_beta = -nu * np.dot(grad_logl.T, X).T

    elif distr == "neg-binomial":
        partial_beta_0 = grad_nl * ((theta + y) / (nl + theta) - y / nl)
        if fit_intercept:
            grad_beta0 = np.sum(partial_beta_0)
        grad_beta = np.dot(partial_beta_0.T, X)

    elif distr == "gaussian":
        if fit_intercept:
            grad_beta0 = np.sum((nl - y) * grad_nl)
        grad_beta = np.dot((nl - y).T, X * grad_nl[:, None]).T

    grad_beta0 *= 1.0 / n_samples
    grad_beta *= 1.0 / n_samples
    if fit_intercept:
        grad_beta += reg_lambda * (1 - alpha) * np.dot(InvCov, beta[1:])  # + reg_lambda * alpha * np.sign(beta[1:])
        g = np.zeros((n_features + 1,))
        g[0] = grad_beta0
        g[1:] = grad_beta
    else:
        grad_beta += reg_lambda * (1 - alpha) * np.dot(InvCov, beta)  # + reg_lambda * alpha * np.sign(beta)
        g = grad_beta

    return g


# ---------------------------------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------------------------------
def log_likelihood(
    distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"],
    y: np.ndarray,
    y_hat: Union[np.ndarray, float],
    theta: float = 1.0,
) -> float:
    """Computes negative log-likelihood of an observation, based on true values and predictions from the regression.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma". Case
            sensitive.
        y: Target values
        y_hat: Predicted values, either array of predictions or scalar value

    Returns:
        logL: Numerical value for the log-likelihood
    """
    if distr in ["poisson", "softplus"]:
        eps = np.spacing(1)
        logL = np.sum(y * np.log(y_hat + eps) - y_hat)

    elif distr == "gamma":
        nu = 1.0  # shape parameter, exponential for now
        logL = np.sum(nu * (-y / y_hat - np.log(y_hat)))

    elif distr == "neg-binomial":
        logL = np.sum(
            loggamma(y + theta)
            - loggamma(theta)
            - loggamma(y + 1)
            + theta * np.log(theta)
            + y * np.log(y_hat)
            - (theta + y) * np.log(y_hat + theta)
        )

    elif distr == "gaussian":
        logL = -0.5 * np.sum((y - y_hat) ** 2)

    return logL


def _loss(
    distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"],
    alpha: float,
    reg_lambda: float,
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    Tau: Union[None, np.ndarray] = None,
    eta: float = 2.0,
    theta: float = 1.0,
    fit_intercept: bool = True,
) -> float:
    """Objective function, comprised of a combination of the log-likelihood and regularization losses.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma". Case
            sensitive.
        alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.) term of the loss function
        reg_lambda: Regularization parameter :math:`\\lambda` of penalty term
        X: Array of shape [n_samples, n_features]; input data
        y: Array of shape [n_samples, 1]; labels or targets for the data
        beta: Array of shape [n_features,]; learned model coefficients
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
            provided, Tau will default to the identity matrix.
        eta: A threshold parameter that linearizes the exp() function above threshold eta
        theta: Shape parameter of the negative binomial distribution (number of successes before the first failure).
            Used only if 'distr' is "neg-binomial"
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
            Defaults to True.

    Returns:
        loss: Numerical value for loss
    """
    n_samples, n_features = X.shape
    z = _z(beta[0], beta[1:], X, fit_intercept)
    y_hat = _nl(distr, z, eta, fit_intercept)
    ll = 1.0 / n_samples * log_likelihood(distr, y, y_hat, z, theta)

    if fit_intercept:
        P = L1_L2_penalty(alpha, beta[1:], Tau)
        # P = 0.5 * (1 - alpha) * L2_penalty(beta[1:], Tau)
    else:
        P = L1_L2_penalty(alpha, beta, Tau)
        # P = 0.5 * (1 - alpha) * L2_penalty(beta, Tau)

    loss = -ll + reg_lambda * P
    return loss


# ---------------------------------------------------------------------------------------------------
# Custom metric
# ---------------------------------------------------------------------------------------------------
def pseudo_r2(
    y: np.ndarray,
    yhat: np.ndarray,
    ynull_: float,
    distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"],
    theta: float,
):
    """Compute r^2 using log-likelihood, taking into account the observed and predicted distributions as well as the
    observed and predicted values.

    Args:
        y: Array of shape [n_samples,]; target values for regression
        yhat: Predicted targets of shape [n_samples,]
        ynull_: Mean of the target labels (null model prediction)
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma". Case
            sensitive.
        theta: Shape parameter of the negative binomial distribution (number of successes before the first
            failure). It is used only if 'distr' is equal to "neg-binomial", otherwise it is ignored.
    """
    if distr in ["poisson", "neg-binomial", "softplus"]:
        LS = log_likelihood(distr, y, y, theta=theta)
    else:
        LS = 0

    L0 = log_likelihood(distr, y, ynull_, theta=theta)
    L1 = log_likelihood(distr, y, yhat, theta=theta)

    if distr in ["poisson", "neg-binomial", "softplus"]:
        score = 1 - (LS - L1) / (LS - L0)
    else:
        score = 1 - L1 / L0
    return score


def deviance(
    y: np.ndarray,
    yhat: np.ndarray,
    distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"],
    theta: float,
):
    """Deviance goodness-of-fit

    Args:
        y: Array of shape [n_samples,]; target values for regression
        yhat: Predicted targets of shape [n_samples,]
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma". Case
            sensitive.
        theta: Shape parameter of the negative binomial distribution (number of successes before the first
            failure). It is used only if 'distr' is equal to "neg-binomial", otherwise it is ignored.

    Returns:
        score: Deviance of the predicted labels
    """
    if distr in ["poisson", "neg-binomial", "softplus"]:
        LS = log_likelihood(distr, y, y, theta=theta)
    else:
        LS = 0

    L1 = log_likelihood(distr, y, yhat, theta=theta)
    score = -2 * (L1 - LS)
    return score


# ---------------------------------------------------------------------------------------------------
# Generalized linear modeling master class
# ---------------------------------------------------------------------------------------------------
class GLM(BaseEstimator):
    """Fitting generalized linear models (Gaussian, Poisson, negative binomial, gamma) for modeling gene expression.

    NOTES: 'Tau' is the Tikhonov matrix (a square factorization of the inverse covariance matrix), used to set the
    degree to which the algorithm tends towards solutions with smaller norms. If not given, defaults to the ridge (
    L2) penalty.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "neg-binomial", or "gamma". Case sensitive.
        alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.) term of the loss function
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
            provided, Tau will default to the identity matrix.
        reg_lambda: Regularization parameter :math:`\\lambda` of penalty term
        learning_rate: Governs the magnitude of parameter updates for the gradient descent algorithm
        max_iter: Maximum number of iterations for the solver
        tol: Convergence threshold or stopping criteria. Optimization loop will stop when relative change in
            parameter norm is below the threshold.
        eta: A threshold parameter that linearizes the exp() function above eta.
        clip_coeffs: Coefficients of lower absolute value than this threshold are set to zero.
        score_metric: Scoring metric. Options:
            - "deviance": Uses the difference between the saturated (perfectly predictive) model and the true model.
            - "pseudo_r2": Uses the coefficient of determination b/w the true and predicted values.
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function
        random_seed: Seed of the random number generator used to initialize the solution. Default: 888
        theta: Shape parameter of the negative binomial distribution (number of successes before the first
            failure). It is used only if 'distr' is equal to "neg-binomial", otherwise it is ignored.
        verbose: If True, will display information about number of iterations until convergence. Defaults to False.

    Attributes:
        beta0_: The intercept
        beta_: Learned parameters
        n_iter: Number of iterations
    """

    def __init__(
        self,
        distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"] = "poisson",
        alpha: float = 0.5,
        Tau: Union[None, np.ndarray] = None,
        reg_lambda: float = 0.1,
        learning_rate: float = 0.2,
        max_iter: int = 1000,
        tol: float = 1e-6,
        eta: float = 2.0,
        clip_coeffs: float = 0.01,
        score_metric: Literal["deviance", "pseudo_r2"] = "deviance",
        fit_intercept: bool = True,
        random_seed: int = 888,
        theta: float = 1.0,
        verbose: bool = True,
    ):

        self.logger = lm.get_main_logger()
        allowable_dists = ["gaussian", "poisson", "softplus", "neg-binomial", "gamma"]
        if distr not in allowable_dists:
            self.logger.error(f"'distr' must be one of {', '.join(allowable_dists)}, got {distr}.")
        if not isinstance(max_iter, int):
            self.logger.error("'max_iter' must be an integer.")
        if not isinstance(fit_intercept, bool):
            self.logger.error(f"'fit_intercept' must be Boolean, got {type(fit_intercept)}")

        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.Tau = Tau
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self.clip_coeffs = clip_coeffs
        self.score_metric = score_metric
        self.fit_intercept = fit_intercept
        # Seed into instance of np.random.RandomState
        self.random_state = np.random.RandomState(random_seed)
        self.theta = theta
        self.verbose = verbose

    def __repr__(self):
        reg_lambda = self.reg_lambda
        s = "<GLM object attributes: "
        s += "\nDistribution | %s" % self.distr
        s += "\nalpha | %0.2f" % self.alpha
        s += "\nmax_iter | %0.2f" % self.max_iter
        s += "\nlambda: %0.2f\n>" % reg_lambda
        return s

    def _prox(self, beta: np.ndarray, thresh: float):
        """Proximal operator to slowly guide convergence during gradient descent."""
        return np.sign(beta) * (np.abs(beta) - thresh) * (np.abs(beta) > thresh)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """The fit function.

        Args:
            X: 2D array of shape [n_samples, n_features]; input data
            y: 1D array of shape [n_samples,]; target data

        Returns:
            self: Fitted instance of class GLM
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        self.beta0_ = None
        self.beta_ = None
        self.ynull_ = None
        self.n_iter_ = 0

        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            self.logger.error("Input must be ndarray. Got {} and {}".format(type(X), type(y)))

        if X.ndim != 2:
            self.logger.error(f"X must be a 2D array, got {X.ndim}")

        if y.ndim != 1:
            self.logger.error(f"y must be 1D, got {y.ndim}")

        n_observations, n_features = X.shape

        if n_observations != len(y):
            self.logger.error("Shape mismatch." + "X has {} observations, y has {}.".format(n_observations, len(y)))

        # Initialize parameters
        beta = np.zeros((n_features + int(self.fit_intercept),))
        if self.fit_intercept:
            if self.beta0_ is None and self.beta_ is None:
                beta[0] = 1 / (n_features + 1) * self.random_state.normal(0.0, 1.0, 1)
                beta[1:] = 1 / (n_features + 1) * self.random_state.normal(0.0, 1.0, (n_features,))
            else:
                beta[0] = self.beta0_
                beta[1:] = self.beta_

        tol = self.tol
        alpha = self.alpha
        reg_lambda = self.reg_lambda

        self._convergence = list()
        train_iterations = range(self.max_iter)

        # Iterative updates
        for t in train_iterations:
            self.n_iter_ += 1
            beta_old = beta.copy()
            grad = batch_grad(
                self.distr, alpha, reg_lambda, X, y, beta, self.Tau, self.eta, self.theta, self.fit_intercept
            )
            beta = beta - self.learning_rate * grad

            # Apply proximal operator
            if self.fit_intercept:
                beta[1:] = self._prox(beta[1:], self.learning_rate * reg_lambda * alpha)
            else:
                beta = self._prox(beta, self.learning_rate * reg_lambda * alpha)

            # Convergence by relative parameter change tolerance
            norm_update = np.linalg.norm(beta - beta_old)
            norm_update /= np.linalg.norm(beta)
            self._convergence.append(norm_update)
            if t > 1 and self._convergence[-1] < tol and self.verbose:
                self.logger.info("\tParameter update tolerance. " + "Converged in {0:d} iterations".format(t))
                break

        if self.n_iter_ == self.max_iter and self.verbose:
            self.logger.warning("Reached max number of iterations without convergence.")

        # Update the estimated variables
        if self.fit_intercept:
            self.beta0_ = beta[0]
            self.beta_ = beta[1:]
        else:
            self.beta0_ = 0
            self.beta_ = beta
        self.ynull_ = np.mean(y)
        self.is_fitted_ = True

        # Clip small nonzero values w/ absolute value below the provided threshold:
        self.beta_[np.abs(self.beta_) < self.clip_coeffs] = 0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given predictor values, reconstruct expression of dependent/target variables.

        Args:
            X: Array of shape [n_samples, n_features]; input data for prediction

        Returns:
            yhat: Predicted targets of shape [n_samples,]
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "is_fitted_")

        if not isinstance(X, np.ndarray):
            self.logger.error(f"Input data should be of type ndarray (got {type(X)}).")

        # Compute intermediate state and then apply nonlinearity:
        z = _z(self.beta0_, self.beta_, X, self.fit_intercept)
        yhat = _nl(self.distr, z, self.eta, self.fit_intercept)
        yhat = np.asarray(yhat)
        return yhat

    def fit_predict(self, X: np.ndarray, y: np.ndarray):
        """Fit the model and predict on the same data.

        Args:
            X: array of shape [n_samples, n_features]; input data to fit and predict
            y: array of shape [n_samples,]; target values for regression

        Returns:
            yhat: Predicted targets of shape [n_samples,]
        """
        yhat = self.fit(X, y).predict(X)
        return yhat

    def score(self, X: np.ndarray, y: np.ndarray):
        """Score model by computing either the deviance or R^2 for predicted values.

        Args:
            X: array of shape [n_samples, n_features]; input data to fit and predict
            y: array of shape [n_samples,]; target values for regression

        Returns:
            score: Value of chosen metric (any pos number for deviance, 0-1 for R^2)
        """
        check_is_fitted(self, "is_fitted_")
        valid_metrics = ["deviance", "pseudo_r2"]
        if self.score_metric not in valid_metrics:
            self.logger.error(f"score_metric has to be one of: {','.join(valid_metrics)}")
        # Model must be fit before scoring:
        if not hasattr(self, "ynull_"):
            self.logger.error("Model must be fit before prediction can be scored.")

        y = np.asarray(y).ravel()
        yhat = self.predict(X)

        if self.score_metric == "deviance":
            score = deviance(y, yhat, self.distr, self.theta)
        elif self.score_metric == "pseudo_r2":
            score = pseudo_r2(y, yhat, self.ynull_, self.distr, self.theta)
        return score


class GLMCV(BaseEstimator):
    """For estimating regularized generalized linear models (GLM) along a regularization path with warm restarts.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "neg-binomial", or "gamma". Case sensitive.
        alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.) term of the loss function
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
            provided, Tau will default to the identity matrix.
        reg_lambda: Regularization parameter :math:`\\lambda` of penalty term
        n_lambdas: Number of lambdas along the regularization path. Defaults to 25.
        cv: Number of cross-validation repeats
        learning_rate: Governs the magnitude of parameter updates for the gradient descent algorithm
        max_iter: Maximum number of iterations for the solver
        tol: Convergence threshold or stopping criteria. Optimization loop will stop when relative change in
            parameter norm is below the threshold.
        eta: A threshold parameter that linearizes the exp() function above eta.
        clip_coeffs: Absolute value below which to set coefficients to zero.
        score_metric: Scoring metric. Options:
            - "deviance": Uses the difference between the saturated (perfectly predictive) model and the true model.
            - "pseudo_r2": Uses the coefficient of determination b/w the true and predicted values.
        fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function
        random_seed: Seed of the random number generator used to initialize the solution. Default: 888
        theta: Shape parameter of the negative binomial distribution (number of successes before the first
            failure). It is used only if 'distr' is equal to "neg-binomial", otherwise it is ignored.
        verbose: If True, returns logging information as program runs. Recommended to set to False for any
            parallelized processes.

    Attributes:
        beta0_: The intercept
        beta_: Learned parameters
        glm_: The GLM object with the best score
        reg_lambda_opt: The value of reg_lambda for the best GLM model
        n_iter: Number of iterations
    """

    def __init__(
        self,
        distr: Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"] = "poisson",
        alpha: float = 0.5,
        Tau: Union[None, np.ndarray] = None,
        reg_lambda: Union[None, List[float]] = None,
        n_lambdas: int = 25,
        cv: int = 5,
        learning_rate: float = 0.2,
        max_iter: int = 1000,
        tol: float = 1e-6,
        eta: float = 2.0,
        clip_coeffs: float = 0.01,
        score_metric: Literal["deviance", "pseudo_r2"] = "deviance",
        fit_intercept: bool = True,
        random_seed: int = 888,
        theta: float = 1.0,
    ):
        if reg_lambda is None:
            reg_lambda = np.logspace(np.log(0.1), np.log(1e-6), n_lambdas, base=np.exp(1))
        if not isinstance(reg_lambda, (list, np.ndarray)):
            reg_lambda = [reg_lambda]

        self.logger = lm.get_main_logger()
        allowable_dists = ["gaussian", "poisson", "softplus", "neg-binomial", "gamma"]
        if distr not in allowable_dists:
            self.logger.error(f"'distr' must be one of {', '.join(allowable_dists)}, got {distr}.")
        if not isinstance(max_iter, int):
            self.logger.error("'max_iter' must be an integer.")
        if not isinstance(fit_intercept, bool):
            self.logger.error(f"'fit_intercept' must be Boolean, got {type(fit_intercept)}")

        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.n_lambdas = n_lambdas
        self.cv = cv
        self.Tau = Tau
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta0_ = None
        self.beta_ = None
        self.reg_lambda_opt_ = None
        self.glm_ = None
        self.scores_ = None
        self.ynull_ = None
        self.tol = tol
        self.eta = eta
        self.clip_coeffs = clip_coeffs
        self.theta = theta
        self.score_metric = score_metric
        self.fit_intercept = fit_intercept
        self.random_seed = random_seed

    def __repr__(self):
        reg_lambda = self.reg_lambda
        s = "<GLMCV object attributes: "
        s += f"\nDistribution | {self.distr}"
        s += "\nalpha | %0.2f" % self.alpha
        s += "\nmax_iter | %0.2f" % self.max_iter
        if len(reg_lambda) > 1:
            s += "\nlambda: %0.2f to %0.2f\n>" % (reg_lambda[0], reg_lambda[-1])
        else:
            s += "\nlambda: %0.2f\n>" % reg_lambda[0]
        return s

    def fit(self, X: np.ndarray, y: np.ndarray):
        """The fit function.

        Args:
            X: 2D array of shape [n_samples, n_features]; input data
            y: 1D array of shape [n_samples,]; target data

        Returns:
            self: Fitted instance of class GLM
        """
        glms, scores = list(), list()
        self.ynull_ = np.mean(y)

        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)
        # Ensure dataset is large enough for cross-validation; if not, adjust number of folds to number of data
        # points - 1 for leave-one-out cross validation:
        if idxs.shape[0] < self.cv:
            self.logger.info(
                f"Too few samples for {self.cv}-fold cross-validation- performing leave-one-out cross "
                f"validation instead."
            )
            n_folds = idxs.shape[0] - 1
        else:
            n_folds = self.cv
        cv_splits = np.array_split(idxs, n_folds)

        cv_training_iterations = self.reg_lambda

        for idx, rl in enumerate(cv_training_iterations):
            glm = GLM(
                distr=self.distr,
                alpha=self.alpha,
                Tau=self.Tau,
                reg_lambda=rl,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                eta=self.eta,
                clip_coeffs=self.clip_coeffs,
                theta=self.theta,
                score_metric=self.score_metric,
                fit_intercept=self.fit_intercept,
                random_seed=self.random_seed,
                verbose=False,
            )

            scores_fold = list()
            for fold in range(n_folds):
                val = cv_splits[fold]
                train = np.setdiff1d(idxs, val)
                # Initialize parameters:
                if idx == 0:
                    glm.beta0_, glm.beta_ = self.beta0_, self.beta_
                else:
                    glm.beta0_, glm.beta_ = glms[-1].beta0_, glms[-1].beta_

                glm.n_iter_ = 0
                glm.fit(X[train], y[train])
                scores_fold.append(glm.score(X[val], y[val]))
            avg_score = np.mean(scores_fold)
            scores.append(avg_score)

            # Extract final parameters for this value of lambda:
            if idx == 0:
                glm.beta0_, glm.beta_ = self.beta0_, self.beta_
            else:
                glm.beta0_, glm.beta_ = glms[-1].beta0_, glms[-1].beta_

            glm.n_iter_ = 0
            glm.fit(X, y)
            glms.append(glm)

        # Find the lambda that maximizes (for r-squared) or minimizes (for deviance) the scoring metric:
        if self.score_metric == "deviance":
            opt = np.array(scores).argmin()
            opt_score = np.array(scores).min()
        elif self.score_metric == "pseudo_r2":
            opt = np.array(scores).argmax()
            opt_score = np.array(scores).max()
        else:
            self.logger.error(f"Unknown score_metric: {self.score_metric}")

        self.beta0_, self.beta_ = glms[opt].beta0_, glms[opt].beta_
        self.reg_lambda_opt_ = self.reg_lambda[opt]
        self.glm_ = glms[opt]
        self.scores_ = scores
        # Optimal score:
        self.opt_score = opt_score
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Using the best scoring model, predict target values.

        Args:
            X: Array of shape [n_samples, n_features]; input data for prediction

        Returns:
            yhat: Predicted targets based on the model with optimal reg_lambda, of shape [n_samples,]
        """
        self.logger = lm.get_main_logger()

        if not hasattr(self, "beta_"):
            self.logger.error("Error: model of :class `GLMCV` not yet fitted. Call :func `fit()` method.")
        X = check_array(X)

        yhat = self.glm_.predict(X)
        return yhat

    def fit_predict(self, X: np.ndarray, y: np.ndarray):
        """Fit the model and, after finding the best model, predict on the same data using that model.

        Args:
            X: array of shape [n_samples, n_features]; input data to fit and predict
            y: array of shape [n_samples,]; target values for regression

        Returns:
            yhat: Predicted targets based on the model with optimal reg_lambda, of shape [n_samples,]
        """
        self.fit(X, y)
        yhat = self.predict(X)
        return yhat

    def score(self, X: np.ndarray, y: np.ndarray):
        """Score model by computing either the deviance or R^2 for predicted values.

        Args:
            X: array of shape [n_samples, n_features]; input data to fit and predict
            y: array of shape [n_samples,]; target values for regression

        Returns:
            score: Value of chosen metric (any pos number for deviance, 0-1 for R^2) for the optimal reg_lambda
        """
        score = self.glm_.score(X, y)
        return score


# ---------------------------------------------------------------------------------------------------
# Wrapper for GLM CV, with parameter optimization
# ---------------------------------------------------------------------------------------------------
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def fit_glm(
    X: Union[np.ndarray, pd.DataFrame],
    adata: AnnData,
    y_feat,
    calc_first_moment: bool = True,
    log_transform: bool = True,
    gs_params: Union[None, dict] = None,
    n_gs_cv: Union[None, int] = None,
    return_model: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, Union[None, GLMCV]]:
    """Wrapper for fitting a generalized elastic net linear model to large biological data, with automated finding of
    optimum lambda regularization parameter and optional further grid search for parameter optimization.

    Args:
        X: Array or DataFrame containing data for fitting- all columns in this array will be used as independent
            variables
        adata: AnnData object from which dependent variable gene expression values will be taken from
        y_feat: Name of the feature in 'adata' corresponding to the dependent variable
        log_transform: If True, will log transform expression. Defaults to True.
        calc_first_moment: If True, will alleviate dropout effects by computing the first moment of each gene across
            cells, consistent with the method used by the original RNA velocity method (La Manno et al.,
            2018). Defaults to True.
        gs_params: Optional dictionary where keys are variable names for either the classifier or the regressor and
            values are lists of potential values for which to find the best combination using grid search.
            Classifier parameters should be given in the following form: 'classifier__{parameter name}'.
        n_gs_cv: Number of folds for cross-validation, will only be used if gs_params is not None. If None,
            will default to a 5-fold cross-validation.
        return_model: If True, returns fitted model. Defaults to True.
        kwargs: Additional named arguments that will be provided to :class `GLMCV`. Valid options are:
            - distr: Distribution family- can be "gaussian", "poisson", "neg-binomial", or "gamma". Case sensitive.
            - alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.) term of the loss function
            - Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
                    provided, Tau will default to the identity matrix.
            - reg_lambda: Regularization parameter :math:`\\lambda` of penalty term
            - n_lambdas: Number of lambdas along the regularization path. Only used if 'reg_lambda' is not given.
            - cv: Number of cross-validation repeats
            - learning_rate: Governs the magnitude of parameter updates for the gradient descent algorithm
            - max_iter: Maximum number of iterations for the solver
            - tol: Convergence threshold or stopping criteria. Optimization loop will stop when relative change in
                    parameter norm is below the threshold.
            - eta: A threshold parameter that linearizes the exp() function above eta.
            - score_metric: Scoring metric. Options:
                - "deviance": Uses the difference between the saturated (perfectly predictive) model and the true model.
                - "pseudo_r2": Uses the coefficient of determination b/w the true and predicted values.
            - fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function
            - random_seed: Seed of the random number generator used to initialize the solution. Default: 888
            - theta: Shape parameter of the negative binomial distribution (number of successes before the first
                    failure). It is used only if 'distr' is equal to "neg-binomial", otherwise it is ignored.

    Returns:
        Beta: Array of shape [n_parameters, 1], contains weight for each parameter
        rex: Array of shape [n_samples, 1]. Reconstructed independent variable values.
        reg: Instance of regression model. Returned only if 'return_model' is True.
    """
    logger = lm.get_main_logger()
    if not "distr" in kwargs:
        kwargs["distr"] = "poisson"
    if not "score_metric" in kwargs:
        kwargs["score_metric"] = "pseudo_r2"

    if kwargs["distr"] in ["poisson", "softplus", "neg-binomial"]:
        if calc_first_moment or log_transform:
            logger.info(
                f"With a {kwargs['distr']} assumption, it is recommended to fit to raw counts. Setting all "
                f"preprocessing settings to False."
            )
            calc_first_moment = False
            log_transform = False

    if calc_first_moment:
        normalize_total(adata)
        _, adata = transcriptomic_connectivity(adata, n_neighbors_method="ball_tree")
        conn = adata.obsp["expression_connectivities"]
        adata_smooth_norm, _ = calc_1nd_moment(adata.X, conn, normalize_W=True)
        adata.layers["M_s"] = adata_smooth_norm

        adata.layers["raw"] = adata.X
        adata.X = adata.layers["M_s"]

    if log_transform:
        log1p(adata)

    y = adata[:, y_feat].X.toarray()
    if isinstance(X, pd.DataFrame):
        X = X.values

    # logger.info("<Grid search CV model fitting for parameters : ")
    # for param in gs_params.keys():
    #    logger.info(f"{param} to test | {gs_params[param]}")

    start_gs_time = time.time()
    if gs_params is not None:
        reg_lambda_given = kwargs.get("reg_lambda", None)
        if reg_lambda_given is None or len(reg_lambda_given) > 3:
            logger.info(
                "Beginning grid search procedure. Temporarily running on reduced range of lambda values for "
                "conciseness."
            )
            kwargs["reg_lambda"] = [0.1, 1e-4]
        else:
            logger.info("Beginning grid search procedure.")

        reg = GLMCV(**kwargs)
        grid = GridSearchCV(estimator=reg, param_grid=gs_params, cv=n_gs_cv)
        grid.fit(X, y)
        logger.info(f"Grid search finished for {y_feat}. Elapsed time: {time.time()-start_gs_time}s.")
        best_params = grid.best_params_
        msg = f"Grid search best parameters for {y_feat}: "
        for k, v in best_params.items():
            msg += f"\n{k}: {v}"
        logger.info(msg)

        # Select parameters in the classifier signature to update classifier keyword arguments:
        for param, value in best_params.items():
            kwargs[param] = value
        # Restore lambda to its original configuration:
        kwargs["reg_lambda"] = reg_lambda_given

    reg = GLMCV(**kwargs)

    rex = reg.fit_predict(X, y)
    logger.info(f"Optimal lambda regularization value for {y_feat}: {reg.reg_lambda_opt_}.")
    intercept = reg.beta0_
    Beta = reg.beta_
    opt_score = reg.opt_score
    # Returns: intercept, coefficients, metric for the optimum lambda, reconstruction, optionally model object
    if return_model:
        return intercept, Beta, opt_score, rex, reg
    else:
        return intercept, Beta, opt_score, rex


def calc_1nd_moment(X, W, normalize_W=True):
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W
        return W @ X, W
    else:
        return W @ X
