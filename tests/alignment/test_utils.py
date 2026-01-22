import unittest
from typing import List, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from spateo.alignment.methods.utils import check_rep_layer
from spateo.alignment.utils import assign_z_coordinates


class TestCheckRepLayer(unittest.TestCase):
    def setUp(self):
        # Create some example AnnData objects
        self.NA = 100
        self.NB = 200
        self.G, self.D = 10, 3

        self.sampleA = AnnData(
            X=np.random.randn(self.NA, self.G),
            layers={"layer": np.random.randn(self.NA, self.G)},
            obsm={"rep": np.random.randn(self.NA, self.D)},
            obs=pd.DataFrame(
                {
                    "label": pd.Categorical(np.random.choice(["A", "B", "C"], self.NA)),
                    "scalar": np.random.randn(self.NA),  # Adding non-categorical obs
                }
            ),
        )
        self.sampleB = AnnData(
            X=np.random.randn(self.NB, self.G),
            layers={"layer": np.random.randn(self.NB, self.G)},
            obsm={"rep": np.random.randn(self.NB, self.D)},
            obs=pd.DataFrame(
                {
                    "label": pd.Categorical(np.random.choice(["A", "B", "C"], self.NB)),
                    "scalar": np.random.randn(self.NB),  # Adding non-categorical obs
                }
            ),
        )
        self.samples = [self.sampleA, self.sampleB]

    def test_valid_layer(self):
        # Test with valid 'layer'
        self.assertTrue(check_rep_layer(self.samples, rep_layer=["layer"], rep_field=["layer"]))
        self.assertTrue(check_rep_layer(self.samples, rep_layer=["X"], rep_field=["layer"]))

    def test_valid_obsm(self):
        # Test with valid 'obsm'
        self.assertTrue(check_rep_layer(self.samples, rep_layer=["rep"], rep_field=["obsm"]))

    def test_valid_obs(self):
        # Test with valid 'obs'
        self.assertTrue(check_rep_layer(self.samples, rep_layer=["label"], rep_field=["obs"]))

    def test_valid_mix(self):
        # Test with valid 'obs'
        self.assertTrue(
            check_rep_layer(
                self.samples, rep_layer=["X", "rep", "layer", "label"], rep_field=["layer", "obsm", "layer", "obs"]
            )
        )

    def test_invalid_layer(self):
        # Test with invalid 'layer'
        with self.assertRaises(ValueError):
            check_rep_layer(self.samples, rep_layer=["invalid_layer"], rep_field=["layer"])

    def test_invalid_obsm(self):
        # Test with invalid 'obsm'
        with self.assertRaises(ValueError):
            check_rep_layer(self.samples, rep_layer=["invalid_obsm"], rep_field=["obsm"])

    def test_invalid_obs(self):
        # Test with invalid 'obs'
        with self.assertRaises(ValueError):
            check_rep_layer(self.samples, rep_layer=["invalid_obs"], rep_field=["obs"])

    def test_invalid_obs_type(self):
        # Test with invalid 'obs' type (not categorical)
        with self.assertRaises(ValueError):
            check_rep_layer(self.samples, rep_layer=["obs3"], rep_field=["obs"])

    def test_invalid_rep_field(self):
        # Test with invalid 'rep_field'
        with self.assertRaises(ValueError):
            check_rep_layer(self.samples, rep_layer=["layer1"], rep_field=["invalid"])


class TestAssignZCoordinates(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with 2D spatial data"""
        np.random.seed(42)
        self.n_cells = 100
        self.n_genes = 50
        
        # Create test slices with 2D spatial coordinates
        self.slices_2d = []
        for i in range(4):
            spatial_coords = np.random.rand(self.n_cells, 2) * 100
            adata = AnnData(
                X=np.random.randn(self.n_cells, self.n_genes),
                obsm={"spatial": spatial_coords}
            )
            adata.obs["slice_id"] = i
            self.slices_2d.append(adata)
    
    def test_default_spacing(self):
        """Test default uniform spacing of 1.0"""
        result = assign_z_coordinates(self.slices_2d, spatial_key="spatial", inplace=False)
        
        # Check that all slices now have 3D coordinates
        for i, adata in enumerate(result):
            self.assertEqual(adata.obsm["spatial"].shape[1], 3)
            # Check z-coordinate is correct
            expected_z = float(i)
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], expected_z)
    
    def test_custom_uniform_spacing(self):
        """Test custom uniform spacing"""
        spacing = 10.0
        result = assign_z_coordinates(
            self.slices_2d, 
            spatial_key="spatial", 
            z_spacing=spacing,
            inplace=False
        )
        
        # Check z-coordinates
        for i, adata in enumerate(result):
            expected_z = i * spacing
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], expected_z)
    
    def test_tissue_thickness_spacing(self):
        """Test tissue thickness-based spacing"""
        thickness = 15.0
        result = assign_z_coordinates(
            self.slices_2d,
            spatial_key="spatial",
            tissue_thickness=thickness,
            inplace=False
        )
        
        # Check z-coordinates
        for i, adata in enumerate(result):
            expected_z = i * thickness
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], expected_z)
    
    def test_variable_spacing(self):
        """Test variable spacing between slices"""
        spacings = [10.0, 15.0, 12.0]  # For 4 slices, need 3 spacing values
        result = assign_z_coordinates(
            self.slices_2d,
            spatial_key="spatial",
            z_spacing=spacings,
            inplace=False
        )
        
        # Check z-coordinates
        expected_z_values = [0.0, 10.0, 25.0, 37.0]  # Cumulative sum
        for i, adata in enumerate(result):
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], expected_z_values[i])
    
    def test_z_offset(self):
        """Test z-offset parameter"""
        offset = 100.0
        spacing = 5.0
        result = assign_z_coordinates(
            self.slices_2d,
            spatial_key="spatial",
            z_spacing=spacing,
            z_offset=offset,
            inplace=False
        )
        
        # Check z-coordinates start from offset
        for i, adata in enumerate(result):
            expected_z = offset + (i * spacing)
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], expected_z)
    
    def test_inplace_modification(self):
        """Test that inplace=True modifies original objects"""
        slices_copy = [adata.copy() for adata in self.slices_2d]
        result = assign_z_coordinates(slices_copy, spatial_key="spatial", inplace=True)
        
        # Should return None when inplace=True
        self.assertIsNone(result)
        
        # Original slices should be modified
        for i, adata in enumerate(slices_copy):
            self.assertEqual(adata.obsm["spatial"].shape[1], 3)
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], float(i))
    
    def test_single_slice(self):
        """Test with single AnnData object"""
        single_slice = self.slices_2d[0].copy()
        result = assign_z_coordinates(single_slice, spatial_key="spatial", inplace=False)
        
        # Should return single AnnData
        self.assertIsInstance(result, AnnData)
        self.assertEqual(result.obsm["spatial"].shape[1], 3)
        np.testing.assert_allclose(result.obsm["spatial"][:, 2], 0.0)
    
    def test_invalid_spacing_length(self):
        """Test error when spacing list length is incorrect"""
        invalid_spacings = [10.0, 15.0]  # Wrong length for 4 slices
        with self.assertRaises(ValueError):
            assign_z_coordinates(
                self.slices_2d,
                spatial_key="spatial",
                z_spacing=invalid_spacings,
                inplace=False
            )
    
    def test_missing_spatial_key(self):
        """Test error when spatial_key doesn't exist"""
        with self.assertRaises(ValueError):
            assign_z_coordinates(
                self.slices_2d,
                spatial_key="nonexistent_key",
                inplace=False
            )
    
    def test_preserves_xy_coordinates(self):
        """Test that original XY coordinates are preserved"""
        original_xy = [adata.obsm["spatial"].copy() for adata in self.slices_2d]
        result = assign_z_coordinates(self.slices_2d, spatial_key="spatial", inplace=False)
        
        # Check XY coordinates are unchanged
        for i, adata in enumerate(result):
            np.testing.assert_allclose(adata.obsm["spatial"][:, :2], original_xy[i])
    
    def test_overwrite_existing_z(self):
        """Test that existing z-coordinates are overwritten"""
        # Create slices with existing 3D coordinates
        slices_3d = []
        for i in range(3):
            spatial_coords = np.random.rand(self.n_cells, 3) * 100
            adata = AnnData(
                X=np.random.randn(self.n_cells, self.n_genes),
                obsm={"spatial": spatial_coords}
            )
            slices_3d.append(adata)
        
        result = assign_z_coordinates(slices_3d, spatial_key="spatial", z_spacing=20.0, inplace=False)
        
        # Check that z-coordinates were overwritten
        for i, adata in enumerate(result):
            expected_z = i * 20.0
            np.testing.assert_allclose(adata.obsm["spatial"][:, 2], expected_z)


if __name__ == "__main__":
    unittest.main()
