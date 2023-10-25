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
    geographically_weighted: bool = False,
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
        geographically_weighted: Set True to perform spatially-weighted gradient update

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
    z = calc_z(beta[0], beta[1:], X, fit_intercept)
    nl = apply_nonlinear(distr, z, eta, fit_intercept)
    nl_grad = nonlinear_gradient(distr, z, eta)

    # Initialize gradient:
    grad_beta0 = 0.0

    if distr in ["poisson", "softplus"]:
        if fit_intercept:
            grad_beta0 = np.sum(nl_grad) - np.sum(y * nl_grad / nl)
        grad_beta = (np.dot(nl_grad.T, X) - np.dot((y * nl_grad / nl).T, X)).T

    elif distr == "gamma":
        # Degrees of freedom (one because the parameter array is 1D)
        nu = 1.0
        grad_logl = (y / nl**2 - 1 / nl) * nl_grad
        if fit_intercept:
            grad_beta0 = -nu * np.sum(grad_logl)
        grad_beta = -nu * np.dot(grad_logl.T, X).T

    elif distr == "neg-binomial":
        partial_beta_0 = nl_grad * ((theta + y) / (nl + theta) - y / nl)
        if fit_intercept:
            grad_beta0 = np.sum(partial_beta_0)
        grad_beta = np.dot(partial_beta_0.T, X)

    elif distr == "gaussian":
        if fit_intercept:
            grad_beta0 = np.sum((nl - y) * nl_grad)
        grad_beta = np.dot((nl - y).T, X * nl_grad[:, None]).T

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
    w: Optional[np.ndarray, scipy.sparse.spmatrix] = None,
) -> float:
    """Computes negative log-likelihood of an observation, based on true values and predictions from the regression.

    Args:
        distr: Distribution family- can be "gaussian", "poisson", "softplus", "neg-binomial", or "gamma". Case
            sensitive.
        y: Target values
        y_hat: Predicted values, either array of predictions or scalar value
        theta: Shape parameter of the negative binomial distribution (number of successes before the first failure).
            Used only if 'distr' is "neg-binomial"
        w: Optional weights vector for spatially-weighted regression. If given, will perform calculations for
            spatially-weighted regression rather than generalized linear regression.

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
    w: Optional[np.ndarray, scipy.sparse.spmatrix] = None,
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
        w: Optional weights vector for spatially-weighted regression. If given, will perform calculations for
            spatially-weighted regression rather than generalized linear regression.

    Returns:
        loss: Numerical value for loss
    """
    n_samples, n_features = X.shape
    z = calc_z(beta[0], beta[1:], X, fit_intercept)
    y_hat = apply_nonlinear(distr, z, eta, fit_intercept)
    ll = 1.0 / n_samples * log_likelihood(distr, y, y_hat, theta, w=w)

    if fit_intercept:
        P = L1_L2_penalty(alpha, beta[1:], Tau)
        # P = 0.5 * (1 - alpha) * L2_penalty(beta[1:], Tau)
    else:
        P = L1_L2_penalty(alpha, beta, Tau)
        # P = 0.5 * (1 - alpha) * L2_penalty(beta, Tau)

    loss = -ll + reg_lambda * P
    return loss


# ---------------------------------------------------------------------------------------------------
# GLM implementation
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
        z = calc_z(self.beta0_, self.beta_, X, self.fit_intercept)
        yhat = apply_nonlinear(self.distr, z, self.eta, self.fit_intercept)
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
