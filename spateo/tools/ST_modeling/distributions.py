"""
Defining the types of distributions that the dependent variable can be assumed to take.
"""
from typing import Optional

import numpy as np
from scipy import special

from ...configuration import EPS, MAX
from ...logging import logger_manager as lm


# ---------------------------------------------------------------------------------------------------
# Link functions
# ---------------------------------------------------------------------------------------------------
class Link(object):
    """
    Parent class for transformations of the dependent variable. The link function is used to transform the mean of
    the response variable, which may have a different distributional form than the linear predictor, to the scale of
    the linear predictor.

    For example, the probability returned by a logistic regressor and the mean parameters are related through this
    link function.

    Does nothing, but includes the expected methods for transformations.
    """

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Inverse of the transformation.

        Args:
            z: The prediction of the transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            g^(-1)(z): The value of the inverse transform function (converting from values to probabilities)
        """
        return NotImplementedError

    def deriv(self, fitted: np.ndarray) -> np.ndarray:
        """Derivative of the logit transformation evaluated at the fitted mean response variable and with respect to the
        linear predictor.

        Args:
            fitted: The fitted mean response variable

        Returns:
            deriv: The value of the derivative of the logit function evaluated at the fitted mean response variable
        """
        return NotImplementedError

    def second_deriv(self, p: np.ndarray) -> np.ndarray:
        """Second derivative of the transformation.

        Args:
            p: Logits to model the probability of an event, given predictor variables with specified values

        Returns:
            g''(p): The value of the second derivative of the link function
        """
        from statsmodels.tools.numdiff import approx_fprime_cs

        second_deriv = approx_fprime_cs(p, self.deriv)
        return second_deriv

    def inverse_deriv(self, z: np.ndarray):
        """Derivative of the inverse transformation g^(-1)(z).

        Args:
            z: The prediction of the transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            g^(-1)'(p): The value of the derivative of the fitted mean response variable (link function
        """
        inv_deriv = 1 / self.deriv(self.inverse(z))
        return inv_deriv


class Logit(Link):
    """
    The logit link function to transform the probability of a binary response variable to the scale of a linear
    predictor.
    """

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, 1 - EPS)
        return vals

    def __call__(self, p: np.ndarray):
        """Transforms the probabilities to logits.

        Args:
            p: Probabilities of an event, given predictor variables with specified values

        Returns:
            z: The transformed logits
        """
        p = self.clip(p)
        z = np.log(p / (1 - p))
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Inverse of the transformation, transforms the linear predictor to the scale of the response variable.

        Args:
            z: The prediction of the logit-transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            inv: Transformed linear predictor
        """
        z = np.asarray(z)
        inv = 1 / (1 + np.exp(-z))
        return inv

    def inverse_deriv(self, z: np.ndarray):
        """Derivative of the inverse transformation g^(-1)(z).

        Args:
            z: The prediction of the transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            inv_deriv: The value of the derivative of the inverse of the logit function
        """
        z = np.exp(z)
        inv_deriv = z / (1 + z) ** 2
        return inv_deriv

    def deriv(self, fitted: np.ndarray) -> np.ndarray:
        """Derivative of the logit transformation evaluated at the fitted mean response variable and with respect to the
        linear predictor.

        Args:
            fitted: The fitted mean response variable

        Returns:
            deriv: The value of the derivative of the logit function evaluated at the fitted mean response variable
        """
        fitted = self.clip(fitted)
        deriv = 1.0 / (fitted * (1 - fitted))
        return deriv

    def second_deriv(self, p: np.ndarray) -> np.ndarray:
        """Second derivative of the logit transformation.

        Args:
            p: Logits to model the probability of an event, given predictor variables with specified values

        Returns:
            second_deriv: The value of the second derivative of the logit function at "p"
        """
        p = self.clip(p)
        v = p * (1 - p)
        second_deriv = (2 * p - 1) / v**2
        return second_deriv


class Power(Link):
    """
    Transform by raising to a power to transform the mean parameter to the scale of the linear predictor.

    Aliases of Power:
        identity = Power(power=1)
        squared = Power(power=2)
        sqrt = Power(power=0.5)
        inverse = Power(power=-1)
        inverse_squared = Power(power=-2)

    Args:
        power: The exponent of the power transform
    """

    def __init__(self, power: float):
        self.power = power

    def __call__(self, fitted: np.ndarray):
        """Raises parameters to power.

        Args:
            fitted: Mean parameter values

        Returns:
            z: The transformed logits
        """
        z = fitted**self.power
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Inverse of the transformation, transforms the linear predictor to the scale of the response variable.

        Args:
            z: The prediction of the power-transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            inv: Transformed linear predictor
        """
        z = np.asarray(z)
        inv = z ** (1 / self.power)
        return inv

    def inverse_deriv(self, z: np.ndarray):
        """Derivative of the inverse transformation g^(-1)(z).

        Args:
            z: The prediction of the transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            inv_deriv: The value of the derivative of the inverse of the power transform
        """
        inv_deriv = z ** ((1 - self.power) / self.power) / self.power
        return inv_deriv

    def deriv(self, fitted: np.ndarray) -> np.ndarray:
        """Derivative of the logit transformation evaluated at the fitted mean response variable and with respect to the
        linear predictor.

        Args:
            fitted: The fitted mean response variable

        Returns:
            deriv: The value of the derivative of the logit function evaluated at the fitted mean response variable
        """
        deriv = self.power * fitted ** (self.power - 1)
        return deriv

    def second_deriv(self, p: np.ndarray) -> np.ndarray:
        """Second derivative of the power transformation.

        Args:
            p: (non-transformed) logits to model the probability of an event, given predictor variables with specified
                values

        Returns:
            second_deriv: The value of the second derivative of the logit function at "p"
        """
        second_deriv = self.power * (self.power - 1) * p ** (self.power - 2)
        return second_deriv


class identity(Power):
    """Identity transform.

    g(p) = p

    Alias of Power(power=1.)
    """

    def __init__(self):
        super().__init__(power=1.0)


class inverse_power(Power):
    """Inverse power transform.

    g(p) = 1 / p

    Alias of Power(power=-1.)
    """

    def __init__(self):
        super().__init__(power=-1.0)


class sqrt(Power):
    """Square root transform.

    g(p) = sqrt(p)

    Alias of Power(power=0.5)
    """

    def __init__(self):
        super().__init__(power=0.5)


class Log(Link):
    """Transform by taking the logarithm.

    g(p) = log(p)
    """

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, MAX)
        return vals

    def __call__(self, p: np.ndarray):
        """Transforms the probabilities to logits.

        Args:
            p: Probabilities of an event, given predictor variables with specified values

        Returns:
            z: The transformed logits
        """
        p = self.clip(p)
        z = np.log(p)
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Inverse of the transformation, transforms the linear predictor to the scale of the response variable.

        Args:
            z: The prediction of the log-transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            inv: Transformed linear predictor
        """
        z = np.asarray(z)
        inv = np.exp(z)
        return inv

    def inverse_deriv(self, z: np.ndarray):
        """Derivative of the inverse transformation g^(-1)(z).

        Args:
            z: The prediction of the transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            inv_deriv: The value of the derivative of the inverse of the power transform
        """
        inv_deriv = np.exp(z)
        return inv_deriv

    def deriv(self, fitted: np.ndarray) -> np.ndarray:
        """Derivative of the log transformation evaluated at the fitted mean response variable and with respect to the
        linear predictor.

        Args:
            fitted: The fitted mean response variable

        Returns:
            deriv: The value of the derivative of the logit function evaluated at the fitted mean response variable
        """
        fitted = self.clip(fitted)
        deriv = 1.0 / fitted
        return deriv

    def second_deriv(self, y: np.ndarray) -> np.ndarray:
        """Second derivative of the logit transformation evaluated at the mean response variable and with respect to
        the linear predictor.

        Args:
            y: The mean response variable

        Returns:
            second_deriv: The value of the second derivative of the logit function at "p"
        """
        y = self.clip(y)
        second_deriv = -1.0 / y**2
        return second_deriv


# ---------------------------------------------------------------------------------------------------
# Variance functions
# ---------------------------------------------------------------------------------------------------
class VarianceFunction(object):
    """
    Relates the variance of a random variable to its mean.
    """

    def __call__(self, fitted):
        """Default variance function, assumes variance of each parameter is 1.

        Args:
            fitted: Mean parameter values

        Returns:
            var: Variance
        """
        var = np.ones_like(fitted, dtype=np.float64)
        return var

    def deriv(self, fitted):
        """Returns the derivative of the variance function.

        Args:
            fitted: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        deriv = np.zeros_like(fitted)
        return deriv


constant_var = VarianceFunction()
constant_var.__doc__ = """Constant variance function, assumes variance of each parameter is 1. This is an alias of 
VarianceFunction()."""


class Power_Variance(object):
    """
    Variance function that is a power of the mean.

    Alias for Power_Variance:
        fitted = Power_Variance()
        fitted_squared = Power_Variance(power=2)
        fitted_cubed = Power_Variance(power=3)

    Args:
        power: Exponent used in the variance function
    """

    def __init__(self, power=1.0):
        self.power = power

    def __call__(self, fitted):
        """Computes variance by raising mean parameters to a power.

        Args:
            fitted: Mean parameter values

        Returns:
            var: Variance
        """
        # Variance fitted must be positive:
        fitted_abs = np.fabs(fitted)
        var = fitted_abs**self.power
        return var

    def deriv(self, fitted):
        """Returns the derivative of the variance function.

        Args:
            fitted: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        from statsmodels.tools.numdiff import approx_fprime_cs

        deriv = np.diag(approx_fprime_cs(fitted, self))
        return deriv


fitted = Power_Variance()
fitted.__doc__ = """
Computes variance using np.fabs(fitted).

This is an alias of Power_Variance() for which the variance is equal in magnitude to the mean.
"""

fitted_squared = Power_Variance(power=2)
fitted_squared.__doc__ = """
Computes variance using np.fabs(fitted) ** 2.

This is an alias of Power_Variance() for which the variance is equal in magnitude to the square of the mean.
"""

fitted_cubed = Power_Variance(power=3)
fitted_cubed.__doc__ = """
Computes variance using np.fabs(fitted) ** 3.

This is an alias of Power_Variance() for which the variance is equal in magnitude to the cube of the mean.
"""


class Binomial_Variance(object):
    """
    Variance function for binomial distribution.

    Equations:
        V(fitted) = p * (1 - p) * n, where p = mu / n

    Args:
        n: The number of trials. The default is 1, under which the assumption is that each observation is an
            independent trial with a binary outcome.
    """

    def __init__(self, n=1):
        self.n = n

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, 1 - EPS)
        return vals

    def __call__(self, fitted: np.ndarray):
        """Computes variance for the mean parameters by modeling the output probabilities as a binomial distribution.

        Args:
            fitted: Mean parameter values

        Returns:
            var: Variance
        """
        fitted = self.clip(fitted / self.n)
        var = fitted * (1 - fitted) * self.n
        return var

    def deriv(self, fitted: np.ndarray) -> np.ndarray:
        """Returns the derivative of the variance function.

        Args:
            fitted: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs

        deriv = np.diag(approx_fprime_cs(fitted, self))
        return deriv


binom_variance = Binomial_Variance()
binom_variance.__doc__ = """
Binomial distribution variance function.

This is an alias of Binomial(n=1)
"""


class Negative_Binomial_Variance(object):
    """
    Variance function for the negative binomial distribution.

    Equations:
        V(fitted) = fitted + disp * fitted ** 2

    Args:
        disp: The dispersion parameter for the negative binomial. Assumed to be nonstochastic, defaults to 0.5.
    """

    def __init__(self, disp: float = 0.5):
        self.disp = disp

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, MAX)
        return vals

    def __call__(self, fitted: np.ndarray):
        """Computes variance for the mean parameters by modeling the output probabilities as a negative binomial
        distribution.

        Args:
            fitted: Mean parameter values

        Returns:
            var: Variance, given by fitted + disp * fitted ** 2
        """
        fitted = self.clip(fitted)
        var = fitted + self.disp * fitted**2
        return var

    def deriv(self, fitted):
        """Returns the derivative of the variance function.

        Args:
            fitted: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        deriv = self.clip(fitted)
        deriv = 1 + self.disp * 2 * deriv
        return deriv


nbinom_variance = Negative_Binomial_Variance()
nbinom_variance.__doc__ = """
Negative Binomial variance function.

This is an alias of NegativeBinomial(disp=0.5)
"""


# ---------------------------------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------------------------------
class Distribution(object):
    """
    Parent class for one-parameter exponential distributions that can be used with Spateo's modeling core. Some of
    the methods do nothing, but provide skeletons for the expected methods for the distributions themselves .

    Args:
        link: The link function to use for the distribution, for performing transformation of the linear outputs. See
            the individual distributions for the default link.
        variance: Measures the variance as a function of the mean probabilities. See the individual families that
            inherit from this class for the default variance function.
    """

    def _setlink(self, link):
        """Sets the link function for the distribution.

        If the chosen link function is not valid for a particular distribution family, a ValueError exception will be
        raised.
        """

        self._link = link
        if not isinstance(link, Link):
            raise TypeError("The input should be a valid Link object.")
        if hasattr(self, "links"):
            validlink = link in self.links
            validlink = max([isinstance(link, _) for _ in self.links])
            if not validlink:
                errmsg = "Invalid link for family, should be in %s. (got %s)"
                raise ValueError(errmsg % (repr(self.links), link))

    def _getlink(self):
        return self._link

    link = property(_getlink, _setlink, doc="The link function for the distribution.")

    def __init__(self, link, variance):
        self._link = link
        self.variance = variance

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, MAX)
        return vals

    def initial_predictions(self, y: np.ndarray) -> np.ndarray:
        """Starting value for linear predictions in the IRLS algorithm.

        Args:
            y: The untransformed dependent variable

        Returns:
            y_hat_0 : Array of shape [n_samples,]; the initial linear predictors.
        """
        y_hat_0 = (y + y.mean()) / 2
        return y_hat_0

    def weights(self, fitted: np.ndarray) -> np.ndarray:
        """Weights for the IRLS algorithm.

        Args:
            fitted: Array of shape [n_samples,]; transformed mean response variable

        Returns:
            w: Weights for the IRLS steps
        """
        w = 1.0 / (self.link.deriv(fitted) ** 2 * self.variance(fitted))
        return w

    def predict(self, fitted: np.ndarray) -> np.ndarray:
        """Given the linear predictors, map back to the scale of the dependent variable.

        Args:
            fitted: Linear predictors

        Returns:
            y_hat: The predicted dependent variable values
        """
        y_hat = self.link.inverse(fitted)
        return y_hat

    def get_predictors(self, outputs: np.ndarray) -> np.ndarray:
        """Given model fit (outputs obtained from applying the link function), map back to the scale of the linear
        predictors.

        Args:
            outputs: The predicted dependent variable values

        Returns:
            predictor: The linear predictors
        """
        predictors = self.link(outputs)
        return predictors

    def deviance(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> float:
        """Deviance function to measure goodness-of-fit of model fitting. Defined as twice the log-likelihood ratio.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: The value of the deviance function
        """
        raise NotImplementedError

    def deviance_residuals(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Deviance residuals for the model, representing the difference between the observed and expected values of
        the dependent variable.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_res: The deviance residuals
        """
        raise NotImplementedError

    def log_likelihood(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Log-likelihood function for the model.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        raise NotImplementedError


class Poisson(Distribution):
    """
    Poisson distribution for modeling count data.

    Args:
        link: The link function to use for the distribution, for performing transformation of the linear outputs. The
            default link is the log link, but available links are "log", "identity", and "sqrt".
    """

    valid_links = [Log, identity, sqrt]
    variance = fitted
    valid = [0, np.inf]
    suggested_link = Log

    def __init__(self, link=Log):
        self.logger = lm.get_main_logger()

        if link not in Poisson.valid_links:
            raise ValueError("Invalid link for Poisson distribution. Valid links are: %s" % Poisson.valid_links)
        if link != Poisson.suggested_link:
            self.logger.warning(
                "The suggested link function for Poisson is the log link, but %s is currently " "chosen." % link
            )

        self.variance = Poisson.variance
        self.link = link()

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, MAX)
        return vals

    def deviance(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> float:
        """Poisson deviance function.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: Array of shape [n_samples, ]; the value of the deviance function evaluated for each sample.
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        endog_fitted = self.clip(endog / fitted)

        dev = 2 * np.sum(freq_weights * endog * np.log(endog_fitted)) / scale
        return dev

    def deviance_residuals(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Poisson deviance residuals.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_resid: The deviance residuals
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        endog_fitted = self.clip(endog / fitted)

        dev_resid = (
            np.sign(endog - fitted)
            * np.sqrt(2 * freq_weights * (endog * np.log(endog_fitted) - np.subtract(endog, fitted)))
            / scale
        )
        return dev_resid

    def log_likelihood(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Poisson log likelihood of the fitted mean response.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        is_na = np.isnan(fitted).any()

        ll = np.sum(freq_weights * (endog * np.log(fitted) - fitted - special.gammaln(endog + 1)))
        ll = scale * ll
        return ll


class Gaussian(Distribution):
    """
    Gaussian distribution for modeling continuous data.

    Args:
        link: The link function to use for the distribution, for performing transformation of the linear outputs. The
            default link is the identity link, but available links are "log", "identity", and "inverse".
    """

    valid_links = [Log, identity, inverse_power]
    variance = constant_var
    suggested_link = identity

    def __init__(self, link=identity):
        self.logger = lm.get_main_logger()

        if link not in Gaussian.valid_links:
            raise ValueError("Invalid link for Gaussian distribution. Valid links are: %s" % Gaussian.valid_links)
        if link != Gaussian.suggested_link:
            self.logger.warning(
                "The suggested link function for Gaussian is the identity link, but %s is currently " "chosen." % link
            )

        self.variance = Gaussian.variance
        self.link = link()

    def deviance(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> float:
        """Gaussian deviance function.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: The value of the deviance function
        """
        if freq_weights is None:
            freq_weights = 1.0

        if freq_weights is None:
            freq_weights = 1.0

        dev = np.sum((freq_weights * (endog - fitted) ** 2)) / scale
        return dev

    def deviance_residuals(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Gaussian deviance residuals.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response- residuals will be divided by the scale

        Returns:
            dev_resid: The deviance residuals
        """
        if freq_weights is None:
            freq_weights = 1.0

        dev_resid = (freq_weights * (endog - fitted) / np.sqrt(self.variance(fitted))) / scale
        return dev_resid

    def log_likelihood(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ):
        """Gaussian log likelihood of the fitted mean response.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        if freq_weights is None:
            freq_weights = 1.0

        ll = np.sum(
            freq_weights
            * ((endog * fitted - fitted**2 / 2) / scale - endog**2 / (2 * scale) - 0.5 * np.log(2 * np.pi * scale))
        )
        return ll


class Gamma(Distribution):
    """
    Gamma distribution for modeling continuous data.

    Args:
        link: The link function to use for the distribution, for performing transformation of the linear outputs. The
            default link is the inverse link, but available links are "log", "identity", and "inverse".
    """

    valid_links = [Log, identity, inverse_power]
    variance = fitted_squared
    suggested_link = inverse_power

    def __init__(self, link=Log):
        self.logger = lm.get_main_logger()

        if link not in Gamma.valid_links:
            raise ValueError("Invalid link for Gamma distribution. Valid links are: %s" % Gamma.valid_links)
        if link != Gamma.suggested_link:
            self.logger.warning(
                "The suggested link function for Gamma is the log link, but %s is currently " "chosen." % link
            )

        self.variance = Gamma.variance
        self.link = link()

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, MAX)
        return vals

    def deviance(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> float:
        """Gamma deviance function.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for
                unequal sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: The value of the deviance function
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        endog_fitted = self.clip(endog / fitted)

        dev = 2 * np.sum(freq_weights * ((endog - fitted) / fitted - np.log(endog_fitted))) / scale
        return dev

    def deviance_residuals(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Gamma deviance residuals.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_resid: The deviance residuals
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        endog_fitted = self.clip(endog / fitted)

        dev_resid = (
            np.sign(endog - fitted)
            * np.sqrt(-2 * (-(endog - fitted) / fitted + np.log(endog_fitted + EPS)))
            * np.sqrt(freq_weights)
            / scale
        )
        return dev_resid

    def log_likelihood(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Gamma log likelihood of the fitted mean response.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        if freq_weights is None:
            freq_weights = 1.0

        ll = (
            -1.0
            / scale
            * np.sum(
                (
                    endog / fitted
                    + np.log(fitted)
                    + (scale - 1) * np.log(endog)
                    + np.log(scale)
                    + scale * special.gammaln(1.0 / scale)
                )
                * freq_weights
            )
        )
        return ll


class Binomial(Distribution):
    """
    Binomial distribution for modeling binary data.

    Args:
        link: The link function to use for the distribution, for performing transformation of the linear outputs. The
            default link is the logit link, but available links are "logit" and "log".
    """

    valid_links = [Logit, Log]
    variance = binom_variance
    suggested_link = Logit

    def __init__(self, link=Logit):
        self.logger = lm.get_main_logger()

        if link not in Binomial.valid_links:
            raise ValueError("Invalid link for Binomial distribution. Valid links are: %s" % Binomial.valid_links)

        if link != Binomial.suggested_link:
            self.logger.warning(
                "The suggested link function for Binomial is the logit link, but %s is currently chosen." % link
            )

        self.n = 1
        self.variance = Binomial.variance
        self.link = link()

    def initial_predictions(self, y: np.ndarray) -> np.ndarray:
        """Initial predictions for the IRLS algorithm.

        Args:
            y: Array of shape [n_samples, ]; untransformed dependent variable

        Returns:
            y_hat_0 : Array of shape [n_samples,]; the initial linear predictors.
        """
        y_hat_0 = (y + 0.5) / 2
        return y_hat_0

    def deviance(
        self,
        endog: np.ndarray,
        fitted: np.ndarray,
        freq_weights: Optional[np.ndarray] = None,
        scale: np.float = 1.0,
        axis: Optional[int] = None,
    ) -> float:
        """Binomial deviance function.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable
            axis: Axis along which the deviance is calculated

        Returns:
            dev: The value of the deviance function
        """
        if np.shape(self.n) == () and self.n == 1:
            one = np.equal(endog, 1)
            return -2 * np.sum(
                (one * np.log(fitted + 1e-88) + (1 - one) * np.log(1 - fitted + 1e-88)) * freq_weights, axis=axis
            )
        else:
            return 2 * np.sum(
                self.n
                * freq_weights
                * (endog * np.log(endog / fitted + 1e-88) + (1 - endog) * np.log((1 - endog) / (1 - fitted) + 1e-88)),
                axis=axis,
            )

    def deviance_residuals(self, endog: np.ndarray, fitted: np.ndarray, scale: np.float = 1.0) -> np.ndarray:
        """
        Binomial deviance residuals.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_resid: The deviance residuals
        """
        fitted = self.clip(fitted)
        if np.shape(self.n) == () and self.n == 1:
            one = np.equal(endog, 1)
            dev_resid = np.sign(endog - fitted) * np.sqrt(-2 * np.log(one * fitted + (1 - one) * (1 - fitted))) / scale
            return dev_resid
        else:
            dev_resid = (
                np.sign(endog - fitted)
                * np.sqrt(
                    2
                    * self.n
                    * (
                        endog * np.log(endog / fitted + 1e-88)
                        + (1 - endog) * np.log((1 - endog) / (1 - fitted) + 1e-88)
                    )
                )
                / scale
            )
            return dev_resid

    def log_likelihood(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Binomial log likelihood of the fitted mean response.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
            at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; optional 1D array of frequency weights, used to e.g. adjust for
                unequal sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        if np.shape(self.n) == () and self.n == 1:
            ll = scale * np.sum((endog * np.log(fitted / (1 - fitted) + 1e-88) + np.log(1 - fitted)) * freq_weights)
            return ll
        else:
            y = endog * self.n
            ll = scale * np.sum(
                (
                    special.gammaln(self.n + 1)
                    - special.gammaln(y + 1)
                    - special.gammaln(self.n - y + 1)
                    + y * np.log(fitted / (1 - fitted))
                    + self.n * np.log(1 - fitted)
                )
                * freq_weights
            )
            return ll


class NegativeBinomial(Distribution):
    """
    Negative binomial distribution for modeling count data.

    Args:
        link: The link function to use for the distribution, for performing transformation of the linear outputs. The
            default link is the inverse link, but available links are "log", "identity", and "inverse".
    """

    valid_links = [Log, identity, sqrt]
    variance = nbinom_variance
    suggested_link = Log

    def __init__(self, link=Log, disp: Optional[float] = None):
        self.logger = lm.get_main_logger()

        if link not in NegativeBinomial.valid_links:
            raise ValueError(
                "Invalid link for NegativeBinomial distribution. Valid links are: %s" % NegativeBinomial.valid_links
            )
        if link != NegativeBinomial.suggested_link:
            self.logger.warning(
                "The suggested link function for NegativeBinomial is the log link, but %s is currently "
                "chosen." % link
            )

        self.variance = NegativeBinomial.variance
        # Modify the variance function to update the dispersion parameter, if applicable:
        if disp is not None:
            self.variance.disp = disp

        self.link = link()

    def clip(self, vals: np.ndarray) -> np.ndarray:
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, MAX)
        return vals

    def deviance(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> float:
        """Negative binomial deviance function.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
                at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; 1D array of frequency weights, used to e.g. adjust for unequal
                sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: The value of the deviance function
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        endog_fitted = self.clip(endog / fitted)

        dispersion = self.variance.disp

        dev = (
            2
            * np.sum(
                freq_weights
                * (
                    endog * np.log(endog_fitted + dispersion)
                    - endog * np.log(dispersion)
                    - np.log(1 + fitted / dispersion)
                )
            )
            / scale
        )

        return dev

    def deviance_residuals(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Negative binomial deviance residuals.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
            at the linear predicted values)
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_resid: The deviance residuals
        """
        if freq_weights is None:
            freq_weights = 1.0

        fitted = self.clip(fitted)
        endog_fitted = self.clip(endog / fitted)

        dev_resid = (
            np.sign(endog - fitted)
            * np.sqrt(2 * freq_weights * (endog * np.log(endog_fitted) - np.subtract(endog, fitted)))
            / scale
        )

        return dev_resid

    def log_likelihood(
        self, endog: np.ndarray, fitted: np.ndarray, freq_weights: Optional[np.ndarray] = None, scale: np.float = 1.0
    ) -> np.ndarray:
        """Negative binomial log likelihood of the fitted mean response.

        Args:
            endog: Array of shape [n_samples, ]; untransformed dependent variable
            fitted: Array of shape [n_samples, ]; fitted mean response variable (link function evaluated
            at the linear predicted values)
            freq_weights: Array of shape [n_samples, ]; optional 1D array of frequency weights, used to e.g. adjust for
                unequal sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        if freq_weights is None:
            freq_weights = 1.0

        dispersion = self.variance.disp
        endog = self.clip(endog)
        fitted = self.clip(fitted)

        ll = np.sum(
            freq_weights
            * (
                special.gammaln(dispersion + endog)
                - special.gammaln(dispersion)
                - special.gammaln(endog + 1)
                + dispersion * np.log(dispersion / (dispersion + fitted * scale))
                + endog * np.log(fitted * scale / (dispersion + fitted * scale))
            )
        )

        return ll
