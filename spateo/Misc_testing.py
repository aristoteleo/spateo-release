import numpy as np
from scipy import special

a = np.random.randint(1, 5, (10,))
b = np.random.randint(1, 5, (10,))

print(a)
print(b)


test_operation = (3 / 2.) * (a**(2/3.) - b**(2 / 3.)) / b**(1 / 6.)
print(test_operation)

test_operation_b = 1.0 * (a * np.log(b) - b - special.gammaln(a + 1))
print(test_operation_b)

test_operation_c = a / b
print(test_operation_c)

test_operation_d = 2 * np.sum(1.0 * ((a - b) / b - np.log(a / b)))
print(test_operation_d)

test_operation_e = np.sign(a - b) * np.sqrt(-2 * (-(a - b) / b + np.log(a / b)))
print(test_operation_e)
