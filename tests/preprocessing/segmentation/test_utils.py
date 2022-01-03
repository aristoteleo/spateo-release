from unittest import TestCase

import numpy as np
from scipy import stats

import spateo.preprocessing.segmentation.utils as utils
from ...mixins import TestMixin


class TestSegmentationUtils(TestMixin, TestCase):
    def test_circle(self):
        circle = np.ones((3, 3), dtype=np.uint8)
        circle[0, 0] = 0
        circle[0, 2] = 0
        circle[2, 0] = 0
        circle[2, 2] = 0
        np.testing.assert_array_equal(circle, utils.circle(3))

    def test_knee(self):
        X = np.array([0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5])
        self.assertEqual(1, utils.knee(X))

    def test_gaussian_blur(self):
        pass

    def test_scale_to_01(self):
        pass

    def test_scale_to_255(self):
        pass
