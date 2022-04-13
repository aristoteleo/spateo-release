"""IO functions for 10x Visium technology.
"""
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.io
from anndata import AnnData

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import get_points_props


def read_10x_as_anndata(matrix_dir: str) -> AnnData:
    """Read 10x Visium matrix directory as AnnData.

    Args:
        matrix_dir: Path to directory containing matrix files.

    Returns:
        AnnData of barcodes x genes.
    """
    obs = pd.read_csv(os.path.join(matrix_dir, "barcodes.tsv.gz"), names=["barcode"]).set_index("barcode")
    var = pd.read_csv(os.path.join(matrix_dir, "features.tsv.gz"), names=["gene_name", "gene_id", "library"]).set_index(
        "gene_id"
    )
    X = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz")).tocsr()
    return AnnData(X=X, obs=obs, var=var)


def read_10x_positions_as_dataframe(path: str) -> pd.DataFrame:
    """Read 10x tissue positions CSV as dataframe.
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images

    Args:
        path: Path to file

    Returns:
        DataFrame containing the following standardized columns.
            * `barcode`: Barcode
            * `x`, `y`: X, Y coordinates
    """
    df = pd.read_csv(
        path, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    )
    return df.rename(
        columns={
            "pxl_row_in_fullres": "x",
            "pxl_col_in_fullres": "y",
        }
    )
