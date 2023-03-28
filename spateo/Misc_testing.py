import numpy as np
from scipy import special

from .tools.ST_regression.distributions import Poisson

"""
a = np.random.randint(1, 5, (10,))
b = np.random.randint(1, 5, (10,))

print(a)
print(b)

test_operation = (3 / 2.0) * (a ** (2 / 3.0) - b ** (2 / 3.0)) / b ** (1 / 6.0)
print(test_operation)

test_operation_b = 1.0 * (a * np.log(b) - b - special.gammaln(a + 1))
print(test_operation_b)

test_operation_c = a / b
print(test_operation_c)

test_operation_d = 2 * np.sum(1.0 * ((a - b) / b - np.log(a / b)))
print(test_operation_d)

test_operation_e = np.sign(a - b) * np.sqrt(-2 * (-(a - b) / b + np.log(a / b)))
print(test_operation_e)

test_operation_f = (
    special.gammaln(0.5 * a / b + a / 0.5)
    - special.gammaln(a / 0.5 + 1)
    - special.gammaln(0.5 * a / b + 1)
    + 0.5 * a / b * np.log(0.5 * a / b / (1 - 0.5))
    + a / 0.5 * np.log(1 - 0.5)
    - np.log(a)
)
print(test_operation_f)"""


# Simulate some Poisson data
np.random.seed(123)
n_obs = 1000
mu = np.exp(np.random.normal(0, 1, n_obs))
y = np.random.poisson(mu)

# Instantiate Poisson family with default log link function
poisson_family = Poisson()

# Test deviance function
mu_null = np.mean(y)
deviance_null = poisson_family.deviance(y, mu_null)
