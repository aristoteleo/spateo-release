"""IO functions for MERFISH technology.
"""
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from ..logging import logger_manager as lm


def read_merfish_as_anndata(path: str) -> AnnData:
    """Read MERFISH matrix as AnnData.

    Args:
        matrix_dir: Path to matrix file.

    Returns:
        AnnData of cell x genes.
    """
    lm.main_info("Constructing count matrix.")
    X = pd.read_csv(path, index_col=0).transpose()
    obs = pd.DataFrame(index=X.index)
    var = pd.DataFrame(index=X.columns)

    return AnnData(X=csr_matrix(X, dtype=np.uint16), obs=obs, var=var)


def read_merfish_positions_as_dataframe(path: str) -> pd.DataFrame:
    """Read MERFISH cell positions CSV as dataframe.

    Args:
        path: Path to file

    Returns:
        DataFrame containing cell positions.
    """
    df_loc = pd.read_excel(path, names=["x", "y"], index_col=0, dtype=np.float32)

    df_loc = df_loc - min(df_loc["x"].min(), df_loc["y"].min())
    return df_loc


def read_merfish(
    path: str,
    positions_path: str,
) -> AnnData:
    """Read MERFISH data as AnnData.

    Args:
        path: Path to matrix files
        positions_path: Path to xlsx containing spatial coordinates
    """

    adata = read_merfish_as_anndata(path)

    df_loc = read_merfish_positions_as_dataframe(positions_path)
    adata = adata[np.intersect1d(df_loc.index, adata.obs_names), :]
    adata.obsm["spatial"] = np.array(df_loc)

    scale, scale_unit = 1.0, None

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
