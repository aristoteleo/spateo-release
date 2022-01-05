from unittest import TestCase

import numpy as np

import spateo.preprocessing.segmentation.em as em
from ...mixins import TestMixin


class TestEM(TestMixin, TestCase):
    def test_nbn_em(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, 100) + rng.negative_binomial(100, 0.5, 100)
        w, r, p = em.nbn_em(X)
        np.testing.assert_allclose([0.41884403, 0.58115597], w)
        np.testing.assert_allclose([53.75074877, 286.70262741], r)
        np.testing.assert_allclose([0.33038823, 0.72543857], p)

    def test_confidence(self):
        np.testing.assert_allclose(
            [
                8.07793567e-27,
                7.41701366e-26,
                6.30446161e-25,
                4.99507343e-24,
                3.71062598e-23,
            ],
            em.confidence(np.array([1, 2, 3, 4, 5]), (0.5, 0.5), (10, 100), (0.5, 0.5)),
        )
