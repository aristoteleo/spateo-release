from unittest import TestCase

import numpy as np
from scipy import stats

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
            em.confidence(np.array([1, 2, 3, 4, 5]), ((0.5, 0.5), (10, 100), (0.5, 0.5))),
        )

    def test_confidence_bins(self):
        np.testing.assert_allclose(
            [
                np.nan,
                np.nan,
                6.30446161e-25,
                4.99507343e-24,
                3.71062598e-23,
            ],
            em.confidence(
                np.array([1, 2, 3, 4, 5]), {1: ((0.5, 0.5), (10, 100), (0.5, 0.5))}, np.array([0, 0, 1, 1, 1])
            ),
        )

    def test_run_em(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (10, 10)) + rng.negative_binomial(100, 0.5, (10, 10))

        w, r, p = em.run_em(X, w=(0.99, 0.01))
        np.testing.assert_allclose([0.41884403, 0.58115597], w)
        np.testing.assert_allclose([53.75074877, 286.70262741], r)
        np.testing.assert_allclose([0.33038823, 0.72543857], p)

    def test_run_em_downsample(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20)) + rng.negative_binomial(100, 0.5, (20, 20))

        w, r, p = em.run_em(X, downsample=100, seed=2021, w=(0.99, 0.01))
        np.testing.assert_allclose([0.3591019654794112, 0.6408980345205889], w)
        np.testing.assert_allclose([174.5511759619999, 701.1016921390545], r)
        np.testing.assert_allclose([0.5981170046608679, 0.8692063871420153], p)

    def test_run_em_peaks(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(500, 0.5, (100, 100))
        w, r, p = em.run_em(X, use_peaks=True, min_distance=1, seed=2021, w=(0.99, 0.01))
        np.testing.assert_allclose([3.06e-322, 1.0], w)
        np.testing.assert_allclose([145.7961865893234, 1552.4287058491911], r)
        np.testing.assert_allclose([0.19757881931910512, 0.7212333080184193], p)

    def test_run_em_bins(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(500, 0.5, (100, 100))
        bins = np.zeros(X.shape, dtype=int)
        bins[:50, :50] = 1
        bins[50:, 50:] = 2
        results = em.run_em(X, bins=bins, w=(0.99, 0.01))
        self.assertEqual(2, len(results))
        np.testing.assert_allclose([2.530692668538722e-17, 1.0], results[1][0])
        np.testing.assert_allclose([125.03182337051366, 579.3267180798008], results[1][1])
        np.testing.assert_allclose([0.1875301858442373, 0.5132143853468611], results[1][2])
        np.testing.assert_allclose([5.352472e-110, 1.0], results[2][0])
        np.testing.assert_allclose([145.71536692587122, 621.2737389848536], results[2][1])
        np.testing.assert_allclose([0.20654202869582633, 0.5300051124652111], results[2][2])

    def test_conditionals(self):
        X = np.array([[1, 2, 3]])
        em_results = (0, 0), (4, 5), (0.5, 0.6)
        results = em.conditionals(X, em_results)
        np.testing.assert_allclose(stats.nbinom(n=4, p=0.5).pmf([[1, 2, 3]]), results[0])
        np.testing.assert_allclose(stats.nbinom(n=5, p=0.6).pmf([[1, 2, 3]]), results[1])

    def test_conditionals_bins(self):
        X = np.array([[1, 2, 3]])
        em_results = {
            1: ((0, 0), (4, 5), (0.5, 0.6)),
            2: ((0, 0), (6, 7), (0.7, 0.8)),
        }
        bins = np.array([[0, 1, 2]], dtype=int)
        results = em.conditionals(X, em_results, bins=bins)
        np.testing.assert_allclose([[0, stats.nbinom(n=4, p=0.5).pmf(2), stats.nbinom(n=6, p=0.7).pmf(3)]], results[0])
        np.testing.assert_allclose([[0, stats.nbinom(n=5, p=0.6).pmf(2), stats.nbinom(n=7, p=0.8).pmf(3)]], results[1])
