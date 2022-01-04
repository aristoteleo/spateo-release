from unittest import TestCase

import numpy as np
from scipy import stats

import spateo.preprocessing.segmentation.icell as icell
from ...mixins import TestMixin


class TestICell(TestMixin, TestCase):
    def test_mclose_mopen(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        mask[5, 5] = False
        mask[0, 0] = True
        expected = mask.copy()
        expected[0, 0] = False
        expected[3, 3] = False
        expected[3, 6] = False
        expected[6, 3] = False
        expected[6, 6] = False
        expected[5, 5] = True
        np.testing.assert_array_equal(expected, icell.mclose_mopen(mask, 3))

    def test_run_em(self):
        pass

    def test_run_bp(self):
        pass

    def test_score_pixels(self):
        pass

    def test_apply_cutoff(self):
        pass
