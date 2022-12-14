"""IO functions for SeqScope technology.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io
from anndata import AnnData
from scipy.sparse import coo_matrix

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import bin_indices, get_bin_props


def read_seqscope_as_anndata(matrix_dir: str) -> AnnData:
    """Read SeqScope matrix directory as AnnData.

    Args:
        matrix_dir: Path to directory containing matrix files.

    Returns:
        AnnData of barcode x genes.
    """
    lm.main_info("Constructing count matrix.")
    obs = pd.read_csv(os.path.join(matrix_dir, "barcodes.tsv"), names=["barcode"]).set_index("barcode")
    var = pd.read_csv(os.path.join(matrix_dir, "features.tsv"), names=["gene_name", "gene_id", "library"]).set_index(
        "gene_id"
    )
    X = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx")).transpose().tocsr()
    return AnnData(X=X, obs=obs, var=var)


def read_seqscope_positions_as_dataframe(path: str) -> pd.DataFrame:
    """Read SeqScope barcode positions CSV as dataframe.

    Args:
        path: Path to file

    Returns:
        DataFrame containing barcode positions.
    """
    dtype = {
        "barcode": "category",
        "lane": np.uint16,
        "tile": np.uint16,
        "x": np.uint32,
        "y": np.uint32,
    }

    df = pd.read_table(path, names=["barcode", "lane", "tile", "x", "y"], sep="\s+", dtype=dtype)
    return df


def read_seqscope(
    matrix_dir: str,
    positions_path: str,
    binsize: Optional[int] = 1,
    add_props: bool = True,
    version: Literal["seqscope"] = "seqscope",
) -> AnnData:
    """Read SeqScope data as AnnData.

    Args:
        matrix_dir: Directory containing matrix files
            (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
        positions_path: Path to CSV containing spatial coordinates
        binsize: Size of pixel bins
        add_props: Whether or not to compute label properties, such as area,
            bounding box, centroid, etc.
        version: SeqScope technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.
    """
    if binsize is not None and abs(int(binsize)) != binsize:
        raise IOError("Positive integer `binsize` must be provided when `segmentation_adata` is not provided.")

    adata = read_seqscope_as_anndata(matrix_dir)
    positions = read_seqscope_positions_as_dataframe(positions_path)
    adata.obs = positions.set_index("barcode").loc[adata.obs_names]

    props = None
    if binsize is not None:
        lm.main_info(f"Using binsize={binsize}")
        if binsize < 2:
            lm.main_warning("Please consider using a larger bin size.")
        if binsize > 1:
            x_bin = bin_indices(adata.obs["x"].values, 0, binsize)
            y_bin = bin_indices(adata.obs["y"].values, 0, binsize)
            adata.obs["x"], adata.obs["y"] = x_bin, y_bin

        adata.obs["label"] = adata.obs["x"].astype(str) + "-" + adata.obs["y"].astype(str)
        if add_props:
            props = get_bin_props(adata.obs[["x", "y", "label"]].drop_duplicates(), binsize)

    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs["label"])

    cat = adata.obs["label"].values
    indicator = coo_matrix(
        (np.broadcast_to(True, adata.n_obs), (cat.codes, np.arange(adata.n_obs))),
        shape=(len(cat.categories), adata.n_obs),
    )

    adata = AnnData(
        indicator @ adata.X, var=adata.var, obs=adata.obs.set_index("label").drop_duplicates().loc[cat.categories]
    )

    if props is not None:
        ordered_props = props.loc[adata.obs_names]
        adata.obs["area"] = ordered_props["area"].values
        adata.obsm["spatial"] = ordered_props.filter(regex="centroid-").values
        adata.obsm["contour"] = ordered_props["contour"].values
        adata.obsm["bbox"] = ordered_props.filter(regex="bbox-").values
    else:
        adata.obsm["spatial"] = adata.obs[["x", "y"]].values

    scale, scale_unit = 1.0, None
    # if version in VERSIONS:
    #    resolution = VERSIONS[version].resolution
    #    scale, scale_unit = resolution.scale, resolution.unit

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
