from unittest import TestCase, mock

import numpy as np

import spateo.segmentation.label as label

from ..mixins import TestMixin, create_random_adata


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
        np.testing.assert_array_equal(expected, label._watershed(X, mask, marker_mask, 3))

    def test_expand_labels(self):
        X = np.zeros((10, 10), dtype=int)
        X[:2, :2] = 1
        X[7:, 7:] = 2
        expected = X.copy()
        expected[:3, :3] = 1
        expected[3, :2] = 1
        expected[:2, 3] = 1
        np.testing.assert_array_equal(expected, label._expand_labels(X, 3, 9))

    def test_find_peaks_with_erosion_with_scores(self):
        with mock.patch("spateo.segmentation.label.utils.safe_erode") as safe_erode:
            safe_erode.return_value = np.random.random((3, 3))
            adata = create_random_adata(["unspliced_scores", "unspliced_mask"], (3, 3))
            k = mock.MagicMock()
            square = mock.MagicMock()
            min_area = mock.MagicMock()
            n_iter = mock.MagicMock()
            float_k = mock.MagicMock()
            float_threshold = mock.MagicMock()
            label.find_peaks_with_erosion(adata, "unspliced", k, square, min_area, n_iter, float_k, float_threshold)
            np.testing.assert_array_equal(adata.layers["unspliced_markers"], safe_erode.return_value)

            safe_erode.assert_called_once_with(mock.ANY, k, square, min_area, n_iter, float_k, float_threshold)
            np.testing.assert_array_equal(adata.layers["unspliced_scores"], safe_erode.call_args[0][0])

    def test_find_peaks_with_erosion_with_mask(self):
        with mock.patch("spateo.segmentation.label.utils.safe_erode") as safe_erode:
            safe_erode.return_value = np.random.random((3, 3))
            adata = create_random_adata(["unspliced_mask"], (3, 3))
            k = mock.MagicMock()
            square = mock.MagicMock()
            min_area = mock.MagicMock()
            n_iter = mock.MagicMock()
            float_k = mock.MagicMock()
            float_threshold = mock.MagicMock()
            label.find_peaks_with_erosion(adata, "unspliced", k, square, min_area, n_iter, float_k, float_threshold)
            np.testing.assert_array_equal(adata.layers["unspliced_markers"], safe_erode.return_value)

            safe_erode.assert_called_once_with(mock.ANY, k, square, min_area, n_iter, float_k, float_threshold)
            np.testing.assert_array_equal(adata.layers["unspliced_mask"], safe_erode.call_args[0][0])

    def test_watershed_adata(self):
        with mock.patch("spateo.segmentation.label._watershed") as _watershed:
            _watershed.return_value = (np.random.random((3, 3)) * 10).astype(int)
            adata = create_random_adata(["nuclei", "nuclei_mask", "nuclei_markers"], (3, 3))
            adata.layers["nuclei_mask"] = adata.layers["nuclei_mask"] > 0.5
            k = mock.MagicMock()
            label.watershed(adata, "nuclei", k)
            np.testing.assert_array_equal(adata.layers["nuclei_labels"], _watershed.return_value)
            _watershed.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY, k)
            np.testing.assert_array_equal(adata.layers["nuclei"], _watershed.call_args[0][0])
            np.testing.assert_array_equal(
                adata.layers["nuclei_mask"] | (adata.layers["nuclei_markers"] > 0), _watershed.call_args[0][1]
            )
            np.testing.assert_array_equal(adata.layers["nuclei_markers"], _watershed.call_args[0][2])

    def test_expand_labels_adata(self):
        with mock.patch("spateo.segmentation.label._expand_labels") as _expand_labels:
            _expand_labels.return_value = np.random.random((3, 3))
            adata = create_random_adata(["nuclei_labels", "mask"], (3, 3))
            distance = mock.MagicMock()
            max_area = mock.MagicMock()
            label.expand_labels(adata, "nuclei", distance, max_area, mask_layer="mask")
            np.testing.assert_array_equal(adata.layers["nuclei_labels_expanded"], _expand_labels.return_value)
            _expand_labels.assert_called_once_with(mock.ANY, distance, max_area, mask=mock.ANY)
            np.testing.assert_array_equal(adata.layers["nuclei_labels"], _expand_labels.call_args[0][0])
            np.testing.assert_array_equal(adata.layers["mask"], _expand_labels.call_args.kwargs["mask"])

    def test_label_connected_components(self):
        pass

    def test_label_connected_components_adata(self):
        pass
