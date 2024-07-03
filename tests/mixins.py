import os
import shutil
import tempfile
from unittest import TestCase

import numpy as np
from anndata import AnnData
from spateo.configuration import SKM


def create_random_adata(layers=None, shape=(3, 3), t=SKM.ADATA_AGG_TYPE):
    adata = AnnData(X=np.random.random(shape))
    if layers:
        for layer in layers:
            adata.layers[layer] = np.random.random(shape)
    SKM.init_adata_type(adata, t)
    return adata


class TestMixin(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.fixtures_dir = os.path.join(cls.base_dir, "fixtures")
        cls.bgi_dir = os.path.join(cls.fixtures_dir, "bgi")
        cls.bgi_counts_path = os.path.join(cls.bgi_dir, "SS200000135TL_D1_bin1_small.gem.gz")
