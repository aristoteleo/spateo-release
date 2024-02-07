from unittest import TestCase, mock

import networkx as nx
import numpy as np
from scipy import sparse

import spateo.segmentation.density as density

from ..mixins import TestMixin, create_random_adata


class TestDensity(TestMixin, TestCase):
    def test_create_spatial_adjacency(self):
        adjacency = density._create_spatial_adjacency((7, 8))
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
        np.testing.assert_array_equal(expected, density._schc(X))

    def test_segment_densities(self):
        with mock.patch("spateo.segmentation.density.utils.conv2d") as conv2d, mock.patch(
            "spateo.segmentation.density._schc"
        ) as schc:
            schc.return_value = np.zeros((3, 3), dtype=int)
            X = sparse.csr_matrix(np.random.random((3, 3)))
            np.testing.assert_array_equal(np.ones((3, 3), dtype=int), density._segment_densities(X, 5, 7))
            np.testing.assert_array_equal(conv2d.call_args[0][0], X.A / X.max())
            conv2d.assert_called_once_with(mock.ANY, 5, mode="gauss")
            schc.assert_called_once_with(conv2d.return_value, distance_threshold=None)

    def test_segment_densities_adata(self):
        with mock.patch("spateo.segmentation.density._segment_densities") as _segment_densities, mock.patch(
            "spateo.segmentation.density.bin_matrix"
        ) as bin_matrix:
            adata = create_random_adata(shape=(3, 3))
            _segment_densities.return_value = np.random.random((3, 3))
            bin_matrix.return_value = np.random.random((3, 3))
            k = mock.MagicMock()
            distance_threshold = mock.MagicMock()
            dk = mock.MagicMock()
            density.segment_densities(adata, "X", 1, k, dk, distance_threshold)
            np.testing.assert_array_equal(adata.layers["X_bins"], _segment_densities.return_value)
            _segment_densities.assert_called_once_with(mock.ANY, k, dk, distance_threshold)
            np.testing.assert_array_equal(adata.X, _segment_densities.call_args[0][0])
            bin_matrix.assert_not_called()
