from unittest import mock, TestCase

import numpy as np

import spateo.preprocessing.segmentation.icell as icell
from ...mixins import TestMixin


class TestICell(TestMixin, TestCase):
    def test_score_pixels_gauss(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.em.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.bp.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.utils.scale_to_01"
        ) as scale_to_01:
            X = mock.MagicMock()
            result = icell.score_pixels(X, k=3, method="gauss")
            conv2d.assert_called_once_with(X, 3, mode="gauss")
            run_em.assert_not_called()
            run_bp.assert_not_called()
            confidence.assert_not_called()
            scale_to_01.assert_called_once_with(conv2d.return_value)
            self.assertEqual(result, scale_to_01.return_value)

    def test_score_pixels_em(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.em.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.bp.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.utils.scale_to_01"
        ) as scale_to_01:
            X = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            w = mock.MagicMock()
            r = mock.MagicMock()
            p = mock.MagicMock()
            run_em.return_value = w, r, p
            result = icell.score_pixels(X, k=3, method="EM", em_kwargs=em_kwargs)
            conv2d.assert_called_once_with(X, 3, mode="circle")
            run_em.assert_called_once_with(conv2d.return_value, **em_kwargs)
            run_bp.assert_not_called()
            confidence.assert_called_once_with(conv2d.return_value, w, r, p)
            scale_to_01.assert_not_called()
            self.assertEqual(result, confidence.return_value)

    def test_score_pixels_em_gauss(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.em.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.bp.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.utils.scale_to_01"
        ) as scale_to_01:
            X = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            w = mock.MagicMock()
            r = mock.MagicMock()
            p = mock.MagicMock()
            run_em.return_value = w, r, p
            result = icell.score_pixels(X, k=3, method="EM+gauss", em_kwargs=em_kwargs)
            conv2d.assert_has_calls(
                [
                    mock.call(X, 3, mode="circle"),
                    mock.call(confidence.return_value, 3, mode="gauss"),
                ]
            )
            run_em.assert_called_once_with(conv2d.return_value, **em_kwargs)
            run_bp.assert_not_called()
            confidence.assert_called_once_with(conv2d.return_value, w, r, p)
            scale_to_01.assert_not_called()
            self.assertEqual(result, conv2d.return_value)

    def test_score_pixels_em_bp(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.em.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.bp.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.utils.scale_to_01"
        ) as scale_to_01:
            X = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            bp_kwargs = mock.MagicMock()
            w = mock.MagicMock()
            r = mock.MagicMock()
            p = mock.MagicMock()
            run_em.return_value = w, r, p
            result = icell.score_pixels(
                X, k=3, method="EM+BP", em_kwargs=em_kwargs, bp_kwargs=bp_kwargs
            )
            conv2d.assert_called_once_with(X, 3, mode="circle")
            run_em.assert_called_once_with(conv2d.return_value, **em_kwargs)
            run_bp.assert_called_once_with(
                conv2d.return_value,
                (r[0], p[0]),
                (r[1], p[1]),
                certain_mask=None,
                **bp_kwargs
            )
            confidence.assert_not_called()
            scale_to_01.assert_not_called()
            self.assertEqual(result, run_bp.return_value)

    def test_apply_threshold(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.utils.knee"
        ) as knee, mock.patch(
            "spateo.preprocessing.segmentation.utils.mclose_mopen"
        ) as mclose_mopen:
            X = np.array([1, 2, 3, 4, 5])
            self.assertEqual(mclose_mopen.return_value, icell.apply_threshold(X, 3, 4))
            np.testing.assert_array_equal(X >= 4, mclose_mopen.call_args[0][0])
            mclose_mopen.assert_called_once_with(mock.ANY, 3)
            knee.assert_not_called()

    def test_apply_threshold_knee(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.utils.knee", return_value=4
        ) as knee, mock.patch(
            "spateo.preprocessing.segmentation.utils.mclose_mopen"
        ) as mclose_mopen:
            X = np.array([1, 2, 3, 4, 5])
            self.assertEqual(mclose_mopen.return_value, icell.apply_threshold(X, 3))
            np.testing.assert_array_equal(X >= 4, mclose_mopen.call_args[0][0])
            mclose_mopen.assert_called_once_with(mock.ANY, 3)
            knee.assert_called_once_with(X)
