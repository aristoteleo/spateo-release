import unittest
from typing import List, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from spateo.alignment.methods.utils import check_rep_layer


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


if __name__ == "__main__":
    unittest.main()
