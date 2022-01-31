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

    def test_run_em(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (10, 10)) + rng.negative_binomial(100, 0.5, (10, 10))

        w, r, p = em.run_em(X)
        np.testing.assert_allclose([0.41884403, 0.58115597], w)
        np.testing.assert_allclose([53.75074877, 286.70262741], r)
        np.testing.assert_allclose([0.33038823, 0.72543857], p)

    def test_run_em_downsample(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20)) + rng.negative_binomial(100, 0.5, (20, 20))

        w, r, p = em.run_em(X, downsample=100, seed=2021)
        np.testing.assert_allclose([0.3591019654794112, 0.6408980345205889], w)
        np.testing.assert_allclose([174.5511759619999, 701.1016921390545], r)
        np.testing.assert_allclose([0.5981170046608679, 0.8692063871420153], p)

    def test_run_em_peaks(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(500, 0.5, (100, 100))
        w, r, p = em.run_em(X, use_peaks=True, min_distance=1, seed=2021)
        np.testing.assert_allclose([3.06e-322, 1.0], w)
        np.testing.assert_allclose([145.7961865893234, 1552.4287058491911], r)
        np.testing.assert_allclose([0.19757881931910512, 0.7212333080184193], p)

    def test_run_em_bins(self):
        pass
