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

        w, r, p = em.run_em(X, w=(0.99, 0.01), max_iter=100)
        np.testing.assert_allclose([0.2383495042266562, 0.7616504957733438], w)
        np.testing.assert_allclose([33.82400850536949, 600.8543296460896], r)
        np.testing.assert_allclose([0.24796212737976334, 0.8454343821517696], p)

    def test_run_em_downsample(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20)) + rng.negative_binomial(100, 0.5, (20, 20))

        w, r, p = em.run_em(X, downsample=100, seed=2021, w=(0.99, 0.01), max_iter=100)
        np.testing.assert_allclose([0.13265334244385113, 0.8673466575561488], w)
        np.testing.assert_allclose([26.374527039240963, 559.0179953067875], r)
        np.testing.assert_allclose([0.1909250367173723, 0.8367866538310141], p)

    def test_run_em_peaks(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(500, 0.5, (100, 100))
        w, r, p = em.run_em(X, use_peaks=True, min_distance=1, seed=2021, w=(0.99, 0.01), max_iter=100)
        np.testing.assert_allclose([3.847497690531427e-204, 1.0], w)
        np.testing.assert_allclose([30.091059744937105, 1031.3178918176766], r)
        np.testing.assert_allclose([0.04729785407753589, 0.6321837694765489], p)

    def test_run_em_bins(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(500, 0.5, (100, 100))
        bins = np.zeros(X.shape, dtype=int)
        bins[:50, :50] = 1
        bins[50:, 50:] = 2
        results = em.run_em(X, bins=bins, w=(0.99, 0.01), max_iter=100)
        self.assertEqual(2, len(results))
        np.testing.assert_allclose([1.6359410595079348e-111, 1.0], results[1][0])
        np.testing.assert_allclose([25.45447008445437, 881.3329952146362], results[1][1])
        np.testing.assert_allclose([0.049578513664564734, 0.6160205873806845], results[1][2])
        np.testing.assert_allclose([3.08767945236589e-131, 1.0], results[2][0])
        np.testing.assert_allclose([25.748715118551672, 888.9653114044754], results[2][1])
        np.testing.assert_allclose([0.04636794899451994, 0.6174397965275665], results[2][2])

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
