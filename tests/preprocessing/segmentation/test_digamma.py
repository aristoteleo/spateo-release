from unittest import TestCase

import numpy as np
from scipy import special

import spateo.preprocessing.segmentation._digamma as _digamma
from ...mixins import TestMixin


class TestDigamma(TestMixin, TestCase):
    def test_digamma(self):
        np.testing.assert_almost_equal(special.digamma(10), _digamma.digamma(10.0))
        np.testing.assert_almost_equal(special.digamma(1), _digamma.digamma(1.0))
        np.testing.assert_almost_equal(special.digamma(100), _digamma.digamma(100.0))
