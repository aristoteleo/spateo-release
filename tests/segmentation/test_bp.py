from unittest import TestCase

import numpy as np
from scipy import stats

import spateo.segmentation.bp as bp

from ..mixins import TestMixin


class TestBP(TestMixin, TestCase):
    def test_create_neighbor_offsets(self):
        neighborhood = np.zeros((3, 3), dtype=bool)
        neighborhood[0, 1] = True
        neighborhood[1, 0] = True
        neighborhood[1, 1] = True
        neighborhood[1, 2] = True
        neighborhood[2, 1] = True
        offsets = bp.create_neighbor_offsets(neighborhood)

        np.testing.assert_equal([[-1, 0], [0, -1], [0, 1], [1, 0]], offsets)

    def test_cell_marginals(self):
        background_probs = np.full((10, 10), 0.1)
        cell_probs = np.full((10, 10), 0.9)
        marginals = bp.cell_marginals(background_probs, cell_probs, p=0.7, q=0.3)
        np.testing.assert_allclose(np.ones((10, 10)), marginals, atol=0.05)

    def test_run_bp(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20))
        X[5:15, 5:15] = rng.negative_binomial(100, 0.5, (10, 10))

        expected = np.zeros((20, 20))
        expected[5:15, 5:15] = 1
        np.testing.assert_allclose(
            expected,
            bp.run_bp(stats.nbinom(n=10, p=0.5).pmf(X), stats.nbinom(n=100, p=0.5).pmf(X), square=True, p=0.7, q=0.3),
            atol=1e-3,
        )
