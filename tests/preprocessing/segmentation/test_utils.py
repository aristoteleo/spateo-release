from unittest import TestCase

import numpy as np

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
        X = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        np.testing.assert_array_equal(np.full((4, 4), 0.5), utils.gaussian_blur(X, 3))

    def test_conv2d(self):
        X = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        np.testing.assert_array_equal(np.full((4, 4), 0.5), utils.conv2d(X, 3, "gauss"))
        np.testing.assert_array_equal(
            [[2, 2, 3, 3], [2, 4, 1, 3], [3, 1, 4, 2], [3, 3, 2, 2]],
            utils.conv2d(X, 3, "circle"),
        )
        np.testing.assert_array_equal(
            [[4, 4, 5, 5], [4, 4, 5, 5], [5, 5, 4, 4], [5, 5, 4, 4]],
            utils.conv2d(X, 3, "square"),
        )

    def test_scale_to_01(self):
        X = np.array([0, 1, 2, 3, 4])
        np.testing.assert_allclose([0, 0.25, 0.5, 0.75, 1], utils.scale_to_01(X))

    def test_scale_to_255(self):
        X = np.array([0, 1, 2, 3, 4])
        np.testing.assert_allclose(
            np.array([0, 0.25, 0.5, 0.75, 1]) * 255, utils.scale_to_255(X)
        )

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
        np.testing.assert_array_equal(expected, utils.mclose_mopen(mask, 3))
