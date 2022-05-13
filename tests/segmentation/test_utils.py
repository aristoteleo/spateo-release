from unittest import TestCase, mock

import numpy as np

import spateo.segmentation.utils as utils

from ..mixins import TestMixin


class TestSegmentationUtils(TestMixin, TestCase):
    def test_circle(self):
        circle = np.ones((3, 3), dtype=np.uint8)
        circle[0, 0] = 0
        circle[0, 2] = 0
        circle[2, 0] = 0
        circle[2, 2] = 0
        np.testing.assert_array_equal(circle, utils.circle(3))

    def test_knee_threshold(self):
        with mock.patch("spateo.segmentation.utils.KneeLocator") as KneeLocator:
            X = np.array([0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5])
            self.assertEqual(KneeLocator.return_value.knee, utils.knee_threshold(X))
            np.testing.assert_array_equal([0, 1, 2, 3, 4, 5], KneeLocator.call_args[0][0])
            np.testing.assert_allclose(
                [4 / 16, 7 / 16, 8 / 16, 9 / 16, 12 / 16, 1],
                KneeLocator.call_args[0][1],
            )
            KneeLocator.assert_called_once_with(mock.ANY, mock.ANY, curve="concave")

    def test_knee_threshold_float(self):
        with mock.patch("spateo.segmentation.utils.KneeLocator") as KneeLocator:
            X = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9])
            self.assertEqual(KneeLocator.return_value.knee, utils.knee_threshold(X, n_bins=3))
            np.testing.assert_array_equal([0.1, 0.5, 0.9], KneeLocator.call_args[0][0])
            np.testing.assert_allclose([3 / 13, 7 / 13, 1], KneeLocator.call_args[0][1])
            KneeLocator.assert_called_once_with(mock.ANY, mock.ANY, curve="concave")

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

    def test_conv2d_bins(self):
        X = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        bins = np.zeros(X.shape, dtype=int)
        bins[:2, :2] = 1
        bins[2:, 2:] = 2
        np.testing.assert_array_equal(
            [[0.5, 0.375, 0, 0], [0.375, 0.25, 0, 0], [0, 0, 0.25, 0.375], [0, 0, 0.375, 0.5]],
            utils.conv2d(X, 3, "gauss", bins=bins),
        )
        np.testing.assert_array_equal(2 * (bins > 0), utils.conv2d(X, 3, "circle", bins=bins))
        np.testing.assert_array_equal(
            [[4, 3, 0, 0], [3, 2, 0, 0], [0, 0, 2, 3], [0, 0, 3, 4]], utils.conv2d(X, 3, "square", bins=bins)
        )

    def test_scale_to_01(self):
        X = np.array([0, 1, 2, 3, 4])
        np.testing.assert_allclose([0, 0.25, 0.5, 0.75, 1], utils.scale_to_01(X))

    def test_scale_to_255(self):
        X = np.array([0, 1, 2, 3, 4])
        np.testing.assert_allclose(np.array([0, 0.25, 0.5, 0.75, 1]) * 255, utils.scale_to_255(X))

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

    def test_apply_threshold(self):
        with mock.patch("spateo.segmentation.utils.mclose_mopen") as mclose_mopen, mock.patch(
            "spateo.segmentation.utils.knee_threshold"
        ) as knee:
            X = np.array([1, 2, 3, 4, 5])
            self.assertEqual(mclose_mopen.return_value, utils.apply_threshold(X, 3, 4))
            np.testing.assert_array_equal(X >= 4, mclose_mopen.call_args[0][0])
            mclose_mopen.assert_called_once_with(mock.ANY, 3)
            knee.assert_not_called()

    def test_apply_threshold_knee(self):
        with mock.patch("spateo.segmentation.utils.mclose_mopen") as mclose_mopen, mock.patch(
            "spateo.segmentation.utils.knee_threshold"
        ) as knee:
            X = np.array([1, 2, 3, 4, 5])
            knee.return_value = 4
            self.assertEqual(mclose_mopen.return_value, utils.apply_threshold(X, 3))
            np.testing.assert_array_equal(X >= 4, mclose_mopen.call_args[0][0])
            mclose_mopen.assert_called_once_with(mock.ANY, 3)
            knee.assert_called_once_with(X)

    def test_safe_erode(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        expected = np.zeros((10, 10), dtype=bool)
        expected[4:6, 4:6] = True
        np.testing.assert_array_equal(expected, utils.safe_erode(mask, 3, min_area=4, n_iter=10))
