from unittest import TestCase

import spateo.io.bgi as bgi
from ..mixins import TestMixin


class TestIOBGI(TestMixin, TestCase):
    def test_read_bgi_as_dataframe(self):
        df = bgi.read_bgi_as_dataframe(self.bgi_counts_path)
        self.assertEqual(
            {"geneID": "0610009B22Rik", "x": 9776, "y": 12669, "MIDCounts": 1},
            df.iloc[0].to_dict(),
        )
        self.assertEqual(77634, df.shape[0])

    def test_read_bgi_agg(self):
        total, spliced, unspliced, x_min, y_min = bgi.read_bgi_agg(self.bgi_counts_path)
        self.assertIsNone(spliced)
        self.assertIsNone(unspliced)
        self.assertEqual((299, 300), total.shape)
        self.assertEqual([9600, 12600], [x_min, y_min])
