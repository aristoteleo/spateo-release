from unittest import mock, TestCase

import numpy as np

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
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (10, 10)) + rng.negative_binomial(
            100, 0.5, (10, 10)
        )

        w, r, p = icell.run_em(X)
        np.testing.assert_allclose([0.41884403, 0.58115597], w)
        np.testing.assert_allclose([53.75074877, 286.70262741], r)
        np.testing.assert_allclose([0.33038823, 0.72543857], p)

    def test_run_em_downsample(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20)) + rng.negative_binomial(
            100, 0.5, (20, 20)
        )

        w, r, p = icell.run_em(X, downsample=100, seed=2021)
        np.testing.assert_allclose([0.3591019654794112, 0.6408980345205889], w)
        np.testing.assert_allclose([174.5511759619999, 701.1016921390545], r)
        np.testing.assert_allclose([0.5981170046608679, 0.8692063871420153], p)

    def test_run_em_peaks(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(50, 0.5, (100, 100)) + rng.negative_binomial(
            500, 0.5, (100, 100)
        )
        w, r, p = icell.run_em(X, use_peaks=True, min_distance=1, seed=2021)
        print(w, r, p)
        np.testing.assert_allclose([3.06e-322, 1.0], w)
        np.testing.assert_allclose([145.7961865893234, 1552.4287058491911], r)
        np.testing.assert_allclose([0.19757881931910512, 0.7212333080184193], p)

    def test_run_bp(self):
        rng = np.random.default_rng(2021)
        X = rng.negative_binomial(10, 0.5, (20, 20))
        X[5:15, 5:15] = rng.negative_binomial(100, 0.5, (10, 10))
        expected = np.zeros((20, 20))
        expected[5:15, 5:15] = 1
        np.testing.assert_allclose(
            expected, icell.run_bp(X, (10, 0.5), (100, 0.5), square=True), atol=1e-3
        )

    def test_score_pixels_gauss(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.icell.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.scale_to_01"
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
            "spateo.preprocessing.segmentation.icell.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.icell.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.scale_to_01"
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
            "spateo.preprocessing.segmentation.icell.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.icell.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.scale_to_01"
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
            "spateo.preprocessing.segmentation.icell.utils.conv2d"
        ) as conv2d, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_em"
        ) as run_em, mock.patch(
            "spateo.preprocessing.segmentation.icell.run_bp"
        ) as run_bp, mock.patch(
            "spateo.preprocessing.segmentation.icell.em.confidence"
        ) as confidence, mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.scale_to_01"
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
                conv2d.return_value, (r[0], p[0]), (r[1], p[1]), **bp_kwargs
            )
            confidence.assert_not_called()
            scale_to_01.assert_not_called()
            self.assertEqual(result, run_bp.return_value)

    def test_apply_cutoff(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.knee"
        ) as knee, mock.patch(
            "spateo.preprocessing.segmentation.icell.mclose_mopen"
        ) as mclose_mopen:
            X = np.array([1, 2, 3, 4, 5])
            self.assertEqual(mclose_mopen.return_value, icell.apply_cutoff(X, 3, 4))
            np.testing.assert_array_equal(X >= 4, mclose_mopen.call_args[0][0])
            mclose_mopen.assert_called_once_with(mock.ANY, 3)
            knee.assert_not_called()

    def test_apply_cutoff_knee(self):
        with mock.patch(
            "spateo.preprocessing.segmentation.icell.utils.knee", return_value=4
        ) as knee, mock.patch(
            "spateo.preprocessing.segmentation.icell.mclose_mopen"
        ) as mclose_mopen:
            X = np.array([1, 2, 3, 4, 5])
            self.assertEqual(mclose_mopen.return_value, icell.apply_cutoff(X, 3))
            np.testing.assert_array_equal(X >= 4, mclose_mopen.call_args[0][0])
            mclose_mopen.assert_called_once_with(mock.ANY, 3)
            knee.assert_called_once_with(X)
