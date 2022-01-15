from unittest import TestCase

import numpy as np

import spateo.preprocessing.segmentation.label as label
from ...mixins import TestMixin


class TestLabel(TestMixin, TestCase):
    def test_watershed(self):
        X = np.zeros((10, 10))
        X[:3, :3] = 1
        X[7:, 7:] = 1
        mask = X > 0
        marker_mask = np.zeros((10, 10), dtype=bool)
        marker_mask[1, 1] = True
        marker_mask[8, 8] = True

        expected = np.zeros((10, 10), dtype=int)
        expected[:3, :3] = 1
        expected[7:, 7:] = 2
        np.testing.assert_array_equal(expected, label.watershed(X, mask, marker_mask, 3))

    def test_expand_labels(self):
        X = np.zeros((10, 10), dtype=int)
        X[:2, :2] = 1
        X[7:, 7:] = 2
        expected = X.copy()
        expected[:3, :3] = 1
        expected[3, :2] = 1
        expected[:2, 3] = 1
        np.testing.assert_array_equal(expected, label.expand_labels(X, 3, 9))
