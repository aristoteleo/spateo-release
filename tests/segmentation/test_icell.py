from unittest import TestCase, mock

import numpy as np

import spateo.segmentation.icell as icell

from ..mixins import TestMixin, create_random_adata


class TestICell(TestMixin, TestCase):
    def setUp(self):
        self.utils_patch = mock.patch("spateo.segmentation.icell.utils")
        self.em_patch = mock.patch("spateo.segmentation.icell.em")
        self.bp_patch = mock.patch("spateo.segmentation.icell.bp")
        self.utils = self.utils_patch.start()
        self.em = self.em_patch.start()
        self.bp = self.bp_patch.start()

    def tearDown(self):
        self.utils_patch.stop()
        self.em_patch.stop()
        self.bp_patch.stop()

    def test_mask_cells_from_stain(self):
        pass

    def test_mask_cells_from_stain_adata(self):
        pass

    def test_mask_nuclei_from_stain(self):
        pass
        # with mock.patch("spateo.segmentation.icell.filters") as filters:
        #     X = np.zeros((2, 2), dtype=np.uint8)
        #     X[1, 1] = 1
        #     X[1, 0] = 2
        #     filters.threshold_multiotsu.return_value = [1, 2]
        #     filters.threshold_local.return_value = 0
        #     self.assertEqual(self.utils.mclose_mopen.return_value, icell._mask_nuclei_from_stain(X, offset=0))
        #     np.testing.assert_array_equal([[False, False], [True, True]], self.utils.mclose_mopen.call_args[0][0])
        #     self.utils.mclose_mopen.assert_called_once_with(mock.ANY, 5)

    def test_mask_nuclei_from_stain_adata(self):
        pass
        # with mock.patch("spateo.segmentation.icell._mask_nuclei_from_stain") as _mask_nuclei_from_stain:
        #     _mask_nuclei_from_stain.return_value = np.random.random((3, 3))
        #     adata = create_random_adata(["stain"], (3, 3))
        #     otsu_classes = mock.MagicMock()
        #     otsu_index = mock.MagicMock()
        #     local_k = mock.MagicMock()
        #     offset = mock.MagicMock()
        #     mk = mock.MagicMock()
        #     icell.mask_nuclei_from_stain(adata, otsu_classes, otsu_index, local_k, offset, mk)
        #     np.testing.assert_array_equal(adata.layers["stain_mask"], _mask_nuclei_from_stain.return_value)
        #     _mask_nuclei_from_stain.assert_called_once_with(
        #         mock.ANY, otsu_classes, otsu_index, local_k, offset, mk
        #     )
        #     np.testing.assert_array_equal(adata.layers["stain"], _mask_nuclei_from_stain.call_args[0][0])

    def test_score_pixels_gauss(self):
        X = mock.MagicMock()
        result = icell._score_pixels(X, k=3, method="gauss")
        self.utils.conv2d.assert_called_once_with(X, 3, mode="gauss", bins=None)
        self.em.run_em.assert_not_called()
        self.em.conditional.assert_not_called()
        self.bp.run_bp.assert_not_called()
        self.em.confidence.assert_not_called()
        self.utils.scale_to_01.assert_called_once_with(self.utils.conv2d.return_value)
        self.assertEqual(result, self.utils.scale_to_01.return_value)

    def test_score_pixels_em(self):
        with mock.patch("spateo.segmentation.icell._initial_nb_params") as _initial_nb_params:
            X = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            background_cond = mock.MagicMock()
            cell_cond = mock.MagicMock()
            self.em.conditionals.return_value = background_cond, cell_cond
            result = icell._score_pixels(X, k=3, method="EM", em_kwargs=em_kwargs)
            self.utils.conv2d.assert_called_once_with(X, 3, mode="circle", bins=None)
            self.em.run_em.assert_called_once_with(
                self.utils.conv2d.return_value, bins=None, params=_initial_nb_params.return_value, **em_kwargs
            )
            self.em.conditional.assert_not_called()
            self.bp.run_bp.assert_not_called()
            self.em.confidence.assert_called_once_with(
                self.utils.conv2d.return_value, em_results=self.em.run_em.return_value, bins=None
            )
            self.utils.scale_to_01.assert_not_called()
            self.assertEqual(result, self.em.confidence.return_value)

    def test_score_pixels_em_gauss(self):
        with mock.patch("spateo.segmentation.icell._initial_nb_params") as _initial_nb_params:
            X = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            background_cond = mock.MagicMock()
            cell_cond = mock.MagicMock()
            self.em.conditionals.return_value = background_cond, cell_cond
            result = icell._score_pixels(X, k=3, method="EM+gauss", em_kwargs=em_kwargs)
            self.utils.conv2d.assert_has_calls(
                [
                    mock.call(X, 3, mode="circle", bins=None),
                    mock.call(self.em.confidence.return_value, 3, mode="gauss", bins=None),
                ]
            )
            self.em.run_em.assert_called_once_with(
                self.utils.conv2d.return_value, bins=None, params=_initial_nb_params.return_value, **em_kwargs
            )
            self.em.conditional.assert_not_called()
            self.bp.run_bp.assert_not_called()
            self.em.confidence.assert_called_once_with(
                self.utils.conv2d.return_value, em_results=self.em.run_em.return_value, bins=None
            )
            self.utils.scale_to_01.assert_not_called()
            self.assertEqual(result, self.utils.conv2d.return_value)

    def test_score_pixels_em_bp(self):
        with mock.patch("spateo.segmentation.icell._initial_nb_params") as _initial_nb_params:
            X = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            bp_kwargs = mock.MagicMock()
            background_cond = mock.MagicMock()
            cell_cond = mock.MagicMock()
            self.em.conditionals.return_value = background_cond, cell_cond
            result = icell._score_pixels(X, k=3, method="EM+BP", em_kwargs=em_kwargs, bp_kwargs=bp_kwargs)
            self.utils.conv2d.assert_called_once_with(X, 3, mode="circle", bins=None)
            self.em.run_em.assert_called_once_with(
                self.utils.conv2d.return_value, bins=None, params=_initial_nb_params.return_value, **em_kwargs
            )
            self.em.conditionals.assert_called_once_with(
                self.utils.conv2d.return_value, em_results=self.em.run_em.return_value, bins=None
            )
            self.bp.run_bp.assert_called_once_with(background_cond, cell_cond, **bp_kwargs)
            self.em.confidence.assert_not_called()
            self.utils.scale_to_01.assert_not_called()
            self.assertEqual(result, self.bp.run_bp.return_value)

    def test_score_pixels_adata(self):
        with mock.patch("spateo.segmentation.icell._score_pixels") as _score_pixels, mock.patch(
            "spateo.segmentation.icell.utils.apply_threshold"
        ) as apply_threshold:
            _score_pixels.return_value = np.random.random((3, 3))
            apply_threshold.return_value = np.random.random((3, 3))
            adata = create_random_adata(["unspliced", "unspliced_bins", "certain"], (3, 3))
            k = mock.MagicMock()
            method = mock.MagicMock()
            moran_kwargs = mock.MagicMock()
            em_kwargs = mock.MagicMock()
            vi_kwargs = mock.MagicMock()
            bp_kwargs = mock.MagicMock()
            threshold = mock.MagicMock()
            use_knee = mock.MagicMock()
            mk = mock.MagicMock()
            icell.score_and_mask_pixels(
                adata,
                "unspliced",
                k,
                method,
                moran_kwargs,
                em_kwargs,
                vi_kwargs,
                bp_kwargs,
                threshold,
                use_knee,
                mk,
                certain_layer="certain",
            )
            np.testing.assert_array_equal(adata.layers["unspliced_scores"], _score_pixels.return_value)
            np.testing.assert_array_equal(adata.layers["unspliced_mask"], apply_threshold.return_value)
            _score_pixels.assert_called_once_with(
                mock.ANY, k, method.lower(), moran_kwargs, em_kwargs, vi_kwargs, bp_kwargs, mock.ANY, mock.ANY
            )
            np.testing.assert_array_equal(adata.layers["unspliced"], _score_pixels.call_args[0][0])
            np.testing.assert_array_equal(adata.layers["certain"] > 0, _score_pixels.call_args[0][7])
            np.testing.assert_array_equal(adata.layers["unspliced_bins"], _score_pixels.call_args[0][8])

            # apply_threshold.assert_called_once_with(mock.ANY, mk, threshold)
            np.testing.assert_array_equal(_score_pixels.return_value, apply_threshold.call_args[0][0])
            np.testing.assert_array_equal(adata.layers["unspliced_mask"], apply_threshold.return_value)
