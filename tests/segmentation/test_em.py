from unittest import TestCase

import numpy as np
from scipy import stats

import spateo.segmentation.em as em

from ..mixins import TestMixin


class TestEM(TestMixin, TestCase):
    def test_nbn_em(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, 100) + rng.negative_binomial(100, 0.5, 100)
        w, r, p = em.nbn_em(X)
        # np.testing.assert_allclose([0.41884403, 0.58115597], w)
        # np.testing.assert_allclose([53.75074877, 286.70262741], r)
        # np.testing.assert_allclose([0.33038823, 0.72543857], p)

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

        w, r, p = em.run_em(X, downsample=1e6, max_iter=100)
        # np.testing.assert_allclose([0.22986291449828194, 0.770137085501718], w)
        # np.testing.assert_allclose([22.612072509754206, 665.8608166342448], r)
        # np.testing.assert_allclose([0.1730657448451262, 0.8594447707151015], p)

    def test_run_em_downsample(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20)) + rng.negative_binomial(100, 0.5, (20, 20))

        w, r, p = em.run_em(X, downsample=100, seed=2021, max_iter=100)
        # np.testing.assert_allclose([0.07933234411996691, 0.9206676558800331], w)
        # np.testing.assert_allclose([23.990499915230924, 740.2223695491173], r)
        # np.testing.assert_allclose([0.1631365477471769, 0.8734818488242497], p)

    def test_run_em_bins(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(500, 0.5, (100, 100))
        bins = np.zeros(X.shape, dtype=int)
        bins[:50, :50] = 1
        bins[50:, 50:] = 2
        results = em.run_em(X, downsample=1e6, bins=bins, max_iter=100)
        self.assertEqual(2, len(results))
        # np.testing.assert_allclose([1.6120561765343201e-108, 1.0], results[1][0])
        # np.testing.assert_allclose([23.100084326864582, 878.7901287669895], results[1][1])
        # np.testing.assert_allclose([0.04332046694964168, 0.6152784261563431], results[1][2])
        # np.testing.assert_allclose([9.432967710450355e-126, 1.0], results[2][0])
        # np.testing.assert_allclose([24.349175570483077, 886.542518924474], results[2][1])
        # np.testing.assert_allclose([0.0420157787402099, 0.6167383056305975], results[2][2])

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
        np.testing.assert_allclose([[1, stats.nbinom(n=4, p=0.5).pmf(2), stats.nbinom(n=6, p=0.7).pmf(3)]], results[0])
        np.testing.assert_allclose([[0, stats.nbinom(n=5, p=0.6).pmf(2), stats.nbinom(n=7, p=0.8).pmf(3)]], results[1])
