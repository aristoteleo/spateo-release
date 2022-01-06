from unittest import mock, TestCase

import numpy as np

import spateo.preprocessing.segmentation.icell as icell
from ...mixins import TestMixin


class TestICell(TestMixin, TestCase):
    def setUp(self):
        self.utils_patch = mock.patch("spateo.preprocessing.segmentation.icell.utils")
        self.em_patch = mock.patch("spateo.preprocessing.segmentation.icell.em")
        self.bp_patch = mock.patch("spateo.preprocessing.segmentation.icell.bp")
        self.utils = self.utils_patch.start()
        self.em = self.em_patch.start()
        self.bp = self.bp_patch.start()

    def tearDown(self):
        self.utils_patch.stop()
        self.em_patch.stop()
        self.bp_patch.stop()

    def test_score_pixels_gauss(self):
        X = mock.MagicMock()
        result = icell.score_pixels(X, k=3, method="gauss")
        self.utils.conv2d.assert_called_once_with(X, 3, mode="gauss")
        self.em.run_em.assert_not_called()
        self.bp.run_bp.assert_not_called()
        self.em.confidence.assert_not_called()
        self.utils.scale_to_01.assert_called_once_with(self.utils.conv2d.return_value)
        self.assertEqual(result, self.utils.scale_to_01.return_value)

    def test_score_pixels_em(self):
        X = mock.MagicMock()
        em_kwargs = mock.MagicMock()
        w = mock.MagicMock()
        r = mock.MagicMock()
        p = mock.MagicMock()
        self.em.run_em.return_value = w, r, p
        result = icell.score_pixels(X, k=3, method="EM", em_kwargs=em_kwargs)
        self.utils.conv2d.assert_called_once_with(X, 3, mode="circle")
        self.em.run_em.assert_called_once_with(
            self.utils.conv2d.return_value, **em_kwargs
        )
        self.bp.run_bp.assert_not_called()
        self.em.confidence.assert_called_once_with(
            self.utils.conv2d.return_value, w, r, p
        )
        self.utils.scale_to_01.assert_not_called()
        self.assertEqual(result, self.em.confidence.return_value)

    def test_score_pixels_em_gauss(self):
        X = mock.MagicMock()
        em_kwargs = mock.MagicMock()
        w = mock.MagicMock()
        r = mock.MagicMock()
        p = mock.MagicMock()
        self.em.run_em.return_value = w, r, p
        result = icell.score_pixels(X, k=3, method="EM+gauss", em_kwargs=em_kwargs)
        self.utils.conv2d.assert_has_calls(
            [
                mock.call(X, 3, mode="circle"),
                mock.call(self.em.confidence.return_value, 3, mode="gauss"),
            ]
        )
        self.em.run_em.assert_called_once_with(
            self.utils.conv2d.return_value, **em_kwargs
        )
        self.bp.run_bp.assert_not_called()
        self.em.confidence.assert_called_once_with(
            self.utils.conv2d.return_value, w, r, p
        )
        self.utils.scale_to_01.assert_not_called()
        self.assertEqual(result, self.utils.conv2d.return_value)

    def test_score_pixels_em_bp(self):
        X = mock.MagicMock()
        em_kwargs = mock.MagicMock()
        bp_kwargs = mock.MagicMock()
        w = mock.MagicMock()
        r = mock.MagicMock()
        p = mock.MagicMock()
        self.em.run_em.return_value = w, r, p
        result = icell.score_pixels(
            X, k=3, method="EM+BP", em_kwargs=em_kwargs, bp_kwargs=bp_kwargs
        )
        self.utils.conv2d.assert_called_once_with(X, 3, mode="circle")
        self.em.run_em.assert_called_once_with(
            self.utils.conv2d.return_value, **em_kwargs
        )
        self.bp.run_bp.assert_called_once_with(
            self.utils.conv2d.return_value,
            (r[0], p[0]),
            (r[1], p[1]),
            certain_mask=None,
            **bp_kwargs
        )
        self.em.confidence.assert_not_called()
        self.utils.scale_to_01.assert_not_called()
        self.assertEqual(result, self.bp.run_bp.return_value)

    def test_apply_threshold(self):
        X = np.array([1, 2, 3, 4, 5])
        self.assertEqual(
            self.utils.mclose_mopen.return_value, icell.apply_threshold(X, 3, 4)
        )
        np.testing.assert_array_equal(X >= 4, self.utils.mclose_mopen.call_args[0][0])
        self.utils.mclose_mopen.assert_called_once_with(mock.ANY, 3)
        self.utils.knee.assert_not_called()

    def test_apply_threshold_knee(self):
        X = np.array([1, 2, 3, 4, 5])
        self.utils.knee.return_value = 4
        self.assertEqual(
            self.utils.mclose_mopen.return_value, icell.apply_threshold(X, 3)
        )
        np.testing.assert_array_equal(X >= 4, self.utils.mclose_mopen.call_args[0][0])
        self.utils.mclose_mopen.assert_called_once_with(mock.ANY, 3)
        self.utils.knee.assert_called_once_with(X)
