"""IO functions for seqFISH-PLUS technology.
"""
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from ..logging import logger_manager as lm


def read_seqfish_meta_as_dataframe(
    path: str, fov_offset: pd.DataFrame = None, accumulate_x: bool = False, accumulate_y: bool = False
) -> pd.DataFrame:
    """Read a seqFISH cell centroid locations file.

    Args:
        path: Path to file
        fov_offset: a dataframe contains the x/y offset of each fov (field of view), for example,
            {'fov':[fov_1, ..], 'x_offset':[x_offset_1, ..], 'y_offset':[y_offset_1, ..]}
        accumulate_x: whether to accumulate x_offset
        accumulate_y: whether to accumulate y_offset

    Return:
        Pandas DataFrame with the following columns.
            * `fov`: ID of field of view
            * `cell_id`: ID of cell in each fov
            * `x`, `y`: X, Y coordinates of the cell centroids
            * `region`: sample region(tissue)
    """
    dtype = {
        "Field of View": np.uint8,
        "Cell ID": np.uint16,
        "X": np.float32,
        "Y": np.float32,
        "Region": "category",
    }
    df_loc = pd.read_csv(
        path,
        dtype=dtype,
    )

    rename = {
        "Field of View": "fov",
        "Cell ID": "cell_id",
        "X": "x",
        "Y": "y",
        "Region": "region",
    }
    df_loc = df_loc.rename(columns=rename)

    if fov_offset is not None:
        if accumulate_x:
            for i in range(1, fov_offset.shape[0]):
                fov_offset["x_offset"][i] = fov_offset["x_offset"][i] + fov_offset["x_offset"][i - 1]
        if accumulate_y:
            for i in range(1, fov_offset.shape[0]):
                fov_offset["y_offset"][i] = fov_offset["y_offset"][i] + fov_offset["y_offset"][i - 1]

        for i in range(fov_offset.shape[0]):
            df_loc["x"][df_loc["fov"] == fov_offset["fov"][i]] = (
                df_loc["x"][df_loc["fov"] == fov_offset["fov"][i]] + fov_offset["x_offset"][i]
            )
            df_loc["y"][df_loc["fov"] == fov_offset["fov"][i]] = (
                df_loc["y"][df_loc["fov"] == fov_offset["fov"][i]] + fov_offset["y_offset"][i]
            )

    df_loc["spatial"] = [[int(df_loc["x"][i]), int(df_loc["y"][i])] for i in range(df_loc.shape[0])]
    return df_loc


def read_seqfish(
    path: str,
    meta_path: str,
    fov_offset: pd.DataFrame = None,
    accumulate_x: bool = False,
    accumulate_y: bool = False,
) -> AnnData:
    """Read seqFISH data as AnnData.

    Args:
        path: Path to seqFISH digital expression matrix CSV.
        meta_path: Path to CSV file containing cell centroid locations.
        fov_offset: a dataframe contain offset of each fov, for example,
            {'fov':[fov_1, ..], 'x_offset':[x_offset_1, ..], 'y_offset':[y_offset_1, ..]}
        accumulate_x: whether to accumulate x_offset
        accumulate_y: whether to accumulate y_offset
    """
    df = pd.read_csv(path, dtype=np.uint16)

    X = csr_matrix(df)
    obs = pd.DataFrame(index=df.index.to_list())
    var = pd.DataFrame(index=df.columns.to_list())

    df_loc = read_seqfish_meta_as_dataframe(meta_path, fov_offset, accumulate_x, accumulate_y)

    lm.main_info("Constructing count matrix.")
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs["fov"] = df_loc["fov"].to_list()
    adata.obs["cell_id"] = df_loc["cell_id"].to_list()
    adata.obs["region"] = df_loc["region"].to_list()

    adata.obsm = pd.DataFrame(index=df_loc.index.to_list())
    adata.obsm["spatial"] = np.array(df_loc["spatial"].to_list())

    scale, scale_unit = 1.0, None

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
