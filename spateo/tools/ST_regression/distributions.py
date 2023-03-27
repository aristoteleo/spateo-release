"""
Defining the types of distributions that the dependent variable can be assumed to take.
"""

import numpy as np
from scipy import special

from ...configuration import EPS


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

    def inverse(self, z: np.ndarray):
        """Inverse of the transformation.

        Args:
            z: The prediction of the transformed dependent variable from the IRLS algorithm as applied with a
                generalized linear model

        Returns:
            g^(-1)(z): The value of the inverse transform function (converting from values to probabilities)
        """
        return NotImplementedError

    def deriv(self, p: np.ndarray):
        """Derivative of the transformation.

        Args:
            p: Logits to model the probability of an event, given predictor variables with specified values

        Returns:
            g'(p): The value of the derivative of the link function
        """
        return NotImplementedError

    def second_deriv(self, p: np.ndarray):
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
            g^(-1)'(p): The value of the derivative of the inverse of the link function
        """
        inv_deriv = 1 / self.deriv(self.inverse(z))
        return inv_deriv


class Logit(Link):
    """
    The logit link function to transform the probability of a binary response variable to the scale of a linear
    predictor.
    """

    def clip(self, vals: np.ndarray):
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

    def inverse(self, z: np.ndarray):
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

    def deriv(self, p: np.ndarray):
        """Derivative of the logit transformation.

        Args:
            p: Logits to model the probability of an event, given predictor variables with specified values

        Returns:
            deriv: The value of the derivative of the logit function at "p"
        """
        p = self.clip(p)
        deriv = 1.0 / (p * (1 - p))
        return deriv

    def second_deriv(self, p: np.ndarray):
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

    def __call__(self, mu: np.ndarray):
        """Raises parameters to power.

        Args:
            mu: Mean parameter values

        Returns:
            z: The transformed logits
        """
        z = mu**self.power
        return z

    def inverse(self, z: np.ndarray):
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

    def deriv(self, p: np.ndarray):
        """Derivative of the power transformation.

        Args:
            p: (non-transformed) logits to model the probability of an event, given predictor variables with specified
                values

        Returns:
            deriv: The value of the derivative of the logit function at "p"
        """
        deriv = self.power * p ** (self.power - 1)
        return deriv

    def second_deriv(self, p: np.ndarray):
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

    def clip(self, vals: np.ndarray):
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, np.inf)
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

    def inverse(self, z: np.ndarray):
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

    def deriv(self, p: np.ndarray):
        """Derivative of the logit transformation.

        Args:
            p: (non-transformed) logits to model the probability of an event, given predictor variables with specified
                values

        Returns:
            deriv: The value of the derivative of the logit function at "p"
        """
        p = self.clip(p)
        deriv = 1.0 / p
        return deriv

    def second_deriv(self, p: np.ndarray):
        """Second derivative of the logit transformation.

        Args:
            p: (non-transformed) logits to model the probability of an event, given predictor variables with specified
                values

        Returns:
            second_deriv: The value of the second derivative of the logit function at "p"
        """
        p = self.clip(p)
        second_deriv = -1.0 / p**2
        return second_deriv


# ---------------------------------------------------------------------------------------------------
# Variance functions
# ---------------------------------------------------------------------------------------------------
class VarianceFunction(object):
    """
    Relates the variance of a random variable to its mean.
    """

    def __call__(self, mu):
        """Default variance function, assumes variance of each parameter is 1.

        Args:
            mu: Mean parameter values

        Returns:
            var: Variance
        """
        var = np.ones_like(mu, dtype=np.float64)
        return var

    def deriv(self, mu):
        """Returns the derivative of the variance function.

        Args:
            mu: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        deriv = np.zeros_like(mu)
        return deriv


constant_var = VarianceFunction()
constant_var.__doc__ = """Constant variance function, assumes variance of each parameter is 1. This is an alias of 
VarianceFunction()."""


class Power_Variance(object):
    """
    Variance function that is a power of the mean.

    Alias for Power_Variance:
        mu = Power_Variance()
        mu_squared = Power_Variance(power=2)
        mu_cubed = Power_Variance(power=3)

    Args:
        power: Exponent used in the variance function
    """

    def __init__(self, power=1.0):
        self.power = power

    def __call__(self, mu):
        """Computes variance by raising mean parameters to a power.

        Args:
            mu: Mean parameter values

        Returns:
            var: Variance
        """
        # Variance must be positive:
        mu_abs = np.fabs(mu)
        var = mu_abs**self.power
        return var

    def deriv(self, mu):
        """Returns the derivative of the variance function.

        Args:
            mu: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs

        deriv = np.diag(approx_fprime_cs(mu, self))
        return deriv


mu = Power_Variance()
mu.__doc__ = """
Computes variance using np.fabs(mu).

This is an alias of Power_Variance() for which the variance is equal in magnitude to the mean.
"""

mu_squared = Power_Variance(power=2)
mu_squared.__doc__ = """
Computes variance using np.fabs(mu) ** 2.

This is an alias of Power_Variance() for which the variance is equal in magnitude to the square of the mean.
"""

mu_cubed = Power_Variance(power=3)
mu_cubed.__doc__ = """
Computes variance using np.fabs(mu) ** 3.

This is an alias of Power_Variance() for which the variance is equal in magnitude to the cube of the mean.
"""


class Binomial_Variance(object):
    """
    Variance function for the binomial distribution.

    Equations:
        V(mu) = p * (1 - p) * n

    Args:
        n: Number of trials. Default is 1.
    """

    def __init__(self, n: int = 1):
        self.n = n

    def clip(self, vals: np.ndarray):
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, 1 - EPS)
        return vals

    def __call__(self, mu: np.ndarray):
        """Computes variance for the mean parameters by modeling the output probabilities as a binomial distribution.

        Args:
            mu: Mean parameter values

        Returns:
            var: Variance, given by mu/n * (1 - mu/n) * self.n
        """
        p = self.clip(mu / self.n)
        var = p * (1 - p) * self.n
        return var

    def deriv(self, mu):
        """Returns the derivative of the variance function.

        Args:
            mu: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        from statsmodels.tools.numdiff import approx_fprime_cs

        deriv = np.diag(approx_fprime_cs(mu, self))
        return deriv


binary = Binomial_Variance()
binary.__doc__ = """
The binomial variance function for n = 1

This is an alias of Binomial_Variance(n=1)
"""


class Negative_Binomial_Variance(object):
    """
    Variance function for the negative binomial distribution.

    Equations:
        V(mu) = mu + disp * mu ** 2

    Args:
        disp: The dispersion parameter for the negative binomial. Assumed to be nonstochastic, defaults to 1.
    """

    def __init__(self, disp: float = 1.0):
        self.disp = disp

    def clip(self, vals: np.ndarray):
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, np.inf)
        return vals

    def __call__(self, mu: np.ndarray):
        """Computes variance for the mean parameters by modeling the output probabilities as a negative binomial
        distribution.

        Args:
            mu: Mean parameter values

        Returns:
            var: Variance, given by mu + disp * mu ** 2
        """
        p = self.clip(mu)
        var = p + self.disp * p**2
        return var

    def deriv(self, mu):
        """Returns the derivative of the variance function.

        Args:
            mu: Mean parameter values

        Returns:
            deriv: Derivative of the variance function
        """
        deriv = self.clip(mu)
        deriv = 1 + self.disp * 2 * deriv
        return deriv


nbinom = Negative_Binomial_Variance()
nbinom.__doc__ = """
Negative Binomial variance function.

This is an alias of NegativeBinomial(disp=1.)
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

    def initial_mu(self, y: np.ndarray):
        """Starting value for mu in the IRLS algorithm.

        Args:
            y: The untransformed dependent variable

        Returns:
            mu_0 : Array of shape [n_samples,]; the initial estimate of the transformed dependent variable.
        """
        mu_0 = (y + y.mean()) / 2
        return mu_0

    def weights(self, mu: np.ndarray):
        """Weights for the IRLS algorithm.

        Args:
            mu: The transformed mean of the dependent variable

        Returns:
            w: Weights for the IRLS steps
        """
        w = 1.0 / (self.link.deriv(mu) ** 2 * self.variance(mu))
        return w

    def deviance(self, endog: np.ndarray, mu: np.ndarray, freq_weights: np.ndarray, scale: np.float = 1.0):
        """Deviance function to measure goodness-of-fit of model fitting. Defined as twice the log-likelihood ratio.

        Args:
            endog: The untransformed dependent variable
            mu: Inverse of the link function evaluated at the linear predicted values
            freq_weights: 1D array of frequency weights, used to e.g. adjust for unequal sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: The value of the deviance function
        """
        raise NotImplementedError

    def deviance_residuals(self, endog: np.ndarray, mu: np.ndarray, freq_weights: np.ndarray, scale: np.float = 1.0):
        """Deviance residuals for the model.

        Args:
            endog: The untransformed dependent variable
            mu: Inverse of the link function evaluated at the linear predicted values
            freq_weights: 1D array of frequency weights, used to e.g. adjust for unequal sampling frequencies
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_res: The deviance residuals
        """
        raise NotImplementedError

    def log_likelihood(self, endog: np.ndarray, mu: np.ndarray, freq_weights: np.ndarray, scale: np.float = 1.0):
        """Log-likelihood function for the model.

        Args:
            endog: The untransformed dependent variable
            mu: Inverse of the link function evaluated at the linear predicted values
            freq_weights: 1D array of frequency weights, used to e.g. adjust for unequal sampling frequencies
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
    variance = mu
    valid = [0, np.inf]
    suggested_link = [Log]

    def __init__(self, link=Log):
        self.variance = Poisson.variance
        self.link = link()

    def clip(self, vals: np.ndarray):
        """Clips values to avoid numerical issues.

        Args:
            vals: Values to clip

        Returns:
            vals: The clipped values
        """
        vals = np.clip(vals, EPS, np.inf)
        return vals

    def deviance(self, endog: np.ndarray, mu: np.ndarray, freq_weights: np.ndarray, scale: np.float = 1.0):
        """Poisson deviance function.

        Args:
            endog: The untransformed dependent variable
            mu: Inverse of the link function evaluated at the linear predicted values
            freq_weights: 1D array of frequency weights, used to e.g. adjust for unequal sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            dev: The value of the deviance function
        """
        endog_mu = self.clip(endog / mu)
        dev = 2 * np.sum(freq_weights * endog * np.log(endog_mu)) / scale
        return dev

    def deviance_residuals(self, endog: np.ndarray, mu: np.ndarray, scale: np.float = 1.0):
        """Poisson deviance residuals.

        Args:
            endog: The untransformed dependent variable
            mu: Inverse of the link function evaluated at the linear predicted values
            scale: Optional scale of the response variable- residuals will be divided by the scale

        Returns:
            dev_res: The deviance residuals
        """
        endog_mu = self.clip(endog / mu)
        dev_res = np.sign(endog - mu) * np.sqrt(2 * (endog * np.log(endog_mu) - np.subtract(endog, mu))) / scale
        return dev_res

    def log_likelihood(self, endog: np.ndarray, mu: np.ndarray, freq_weights: np.ndarray = 1.0, scale: np.float = 1.0):
        """Poisson log likelihood of the fitted mean response.

        Args:
            endog: The untransformed dependent variable
            mu: Inverse of the link function evaluated at the linear predicted values
            freq_weights: 1D array of frequency weights, used to e.g. adjust for unequal sampling frequencies
            scale: Optional scale of the response variable

        Returns:
            ll: The value of the log-likelihood function
        """
        ll = np.sum(freq_weights * (endog * np.log(mu) - mu - special.gammaln(endog + 1)))
