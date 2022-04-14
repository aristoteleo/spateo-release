"""IO functions for 10x Visium technology.
"""
import os
from typing import List, Optional, Union

import ngs_tools as ngs
import numpy as np
import pandas as pd
import scipy.io
from anndata import AnnData
from typing_extensions import Literal

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import get_points_props

VERSIONS = {
    "visium": ngs.chemistry.get_chemistry("Visium"),
}


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
        DataFrame containing barcode positions.
    """
    df = pd.read_csv(
        path, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    )
    return df


def read_10x(matrix_dir: str, positions_path: str, version: Literal["visium"] = "visium") -> AnnData:
    """Read 10x Visium data as AnnData.

    Args:
        matrix_dir: Directory containing matrix files
            (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
        positions_path: Path to CSV containing spatial coordinates
        version: 10x technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.
    """
    adata = read_10x_as_anndata(matrix_dir)
    positions = read_10x_positions_as_dataframe(positions_path)
    adata.obs = positions.set_index("barcode").loc[adata.obs_names]
    adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values

    scale, scale_unit = 1.0, None
    if version in VERSIONS:
        resolution = VERSIONS[version].resolution
        scale, scale_unit = resolution.scale, resolution.unit

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
