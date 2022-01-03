from unittest import TestCase

import numpy as np
from scipy import stats

import spateo.preprocessing.segmentation.em as em
from ...mixins import TestMixin


class TestEM(TestMixin, TestCase):
    def test_nb_pmf(self):
        np.testing.assert_almost_equal(
            stats.nbinom(n=10, p=0.5).pmf(10), em.nb_pmf(10, 10, 0.5)
        )
        # Test underflow
        np.testing.assert_almost_equal(
            stats.nbinom(n=10, p=0.2).pmf(1000), em.nb_pmf(1000, 10, 0.5)
        )

    def test_nbn_em(self):
        pass

    def test_confidence(self):
        pass
