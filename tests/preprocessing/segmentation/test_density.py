from unittest import mock, TestCase

import networkx as nx
import numpy as np
from scipy import sparse

import spateo.preprocessing.segmentation.density as density
from ...mixins import TestMixin


class TestDensity(TestMixin, TestCase):
    def test_create_spatial_adjacency(self):
        adjacency = density.create_spatial_adjacency((7, 8))
        # networkx dimensions are flipped
        np.testing.assert_array_equal(nx.adjacency_matrix(nx.grid_graph((8, 7))).A, adjacency.A)

    def test_schc(self):
        X = np.zeros((10, 10))
        X[:3, :3] = 1 / 3
        X[4:6, 4:6] = 2 / 3
        X[7:, 7:] = 1
        expected = np.zeros_like(X)
        expected[:3, :3] = 3
        expected[4:6, 4:6] = 2
        expected[7:, 7:] = 1
        np.testing.assert_array_equal(expected, density.schc(X))

    def test_segment_densities(self):
        with mock.patch("spateo.preprocessing.segmentation.density.utils.conv2d") as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.density.schc"
        ) as schc:
            X = sparse.csr_matrix(np.random.random((3, 3)))
            self.assertEqual(schc.return_value, density.segment_densities(X, 5))
            np.testing.assert_array_equal(conv2d.call_args[0][0], X.A / X.max())
            conv2d.assert_called_once_with(mock.ANY, 5, mode="gauss")
            schc.assert_called_once_with(conv2d.return_value, distance_threshold=None)
