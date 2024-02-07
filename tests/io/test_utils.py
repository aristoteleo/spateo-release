from unittest import TestCase, mock

import numpy as np
from scipy import sparse

import spateo.io.utils as utils

from ..mixins import TestMixin


class TestIOUtils(TestMixin, TestCase):
    def test_bin_matrix(self):
        X = np.random.random((3, 3))
        expected = np.zeros((2, 2))
        expected[0, 0] = X[:2, :2].sum()
        expected[0, 1] = X[:2, 2].sum()
        expected[1, 0] = X[2, :2].sum()
        expected[1, 1] = X[2, 2]
        np.testing.assert_array_equal(expected, utils.bin_matrix(X, 2))
        np.testing.assert_array_equal(expected, utils.bin_matrix(sparse.csr_matrix(X), 2).A)
