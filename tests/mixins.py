import os
import shutil
import tempfile
from unittest import TestCase


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
