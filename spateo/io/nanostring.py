"""IO functions for NanoString CosMx technology.
"""
from typing import List, Optional, Union

import ngs_tools as ngs
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from typing_extensions import Literal

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import get_points_props

VERSIONS = {
    "cosmx": ngs.chemistry.get_chemistry("CosMx"),
}


def read_nanostring_as_dataframe(path: str, label_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a NanoString CSV tx or metadata file as pandas DataFrame.

    Args:
        path: Path to file.
        label_columns: Column names, the combination of which indicates unique cells.

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `gene`: Gene name/ID (whatever was used in the original file)
            * `x`, `y`: X and Y coordinates
    """
    dtype = {
        "target": "category",
        "CellComp": "category",
        "fov": np.uint8,
        "cell_ID": np.uint16,
        "Area": np.uint32,
        "Width": np.uint16,
        "Height": np.uint16,
    }
    # These are actually stored as floats in the CSV, but we cast them to
    # integers because they are in terms of the TIFF coordinates,
    # which are integers.
    convert = {
        "x_global_px": np.uint32,
        "y_global_px": np.uint32,
        "x_local_px": np.uint16,
        "y_local_px": np.uint16,
        "CenterX_local_px": np.uint16,
        "CenterY_local_px": np.uint16,
        "CenterX_global_px": np.uint32,
        "CenterY_global_px": np.uint32,
    }
    rename = {
        "target": "gene",
        "x_global_px": "x",
        "y_global_px": "y",
    }
    # Use first 10 rows for validation.
    df = pd.read_csv(path, dtype=dtype, nrows=10)
    if label_columns:
        for column in label_columns:
            if column not in df.columns:
                raise IOError(f"Column `{column}` is not present.")

    df = pd.read_csv(path, dtype=dtype)
    for column, t in convert.items():
        if column in df.columns:
            df[column] = df[column].astype(t)
    if label_columns:
        labels = df[label_columns[0]].astype(str)
        for label in label_columns[1:]:
            labels += "-" + df[label].astype(str)
    df["label"] = labels
    return df.rename(columns=rename)


def read_nanostring(
    path: str,
    meta_path: Optional[str] = None,
    binsize: Optional[int] = None,
    label_columns: Optional[Union[str, List[str]]] = None,
    add_props: bool = True,
    version: Literal["cosmx"] = "cosmx",
) -> AnnData:
    """Read NanoString CosMx data as AnnData.

    Args:
        path: Path to transcript detection CSV file.
        meta_path: Path to cell metadata CSV file.
        scale: Physical length per coordinate. For visualization only.
        scale_unit: Scale unit.
        binsize: Size of pixel bins
        label_columns: Columns that contain already-segmented cell labels. Each
            unique combination is considered a unique cell.
        add_props: Whether or not to compute label properties, such as area,
            bounding box, centroid, etc.
        version: NanoString technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.

    Returns:
        Bins x genes or labels x genes AnnData.
    """
    if sum([binsize is not None, label_columns is not None]) != 1:
        raise IOError("Exactly one of `binsize`, `label_columns` must be provided.")
    if binsize is not None and abs(int(binsize)) != binsize:
        raise IOError("Positive integer `binsize` must be provided when `segmentation_adata` is not provided.")

    label_columns = [label_columns] if isinstance(label_columns, str) else label_columns
    data = read_nanostring_as_dataframe(path, label_columns)
    metadata = None

    uniq_gene = sorted(data["gene"].unique())

    props = None
    if label_columns:
        if meta_path:
            metadata = read_nanostring_as_dataframe(meta_path, label_columns)

        lm.main_info(f"Using cell labels from `{label_columns}` columns.")
        binsize = 1
        # cell_ID == 0 indicates not assigned
        data = data[data["cell_ID"] > 0]
        if add_props:
            props = get_points_props(data[["x", "y", "label"]])
    elif binsize is not None:
        lm.main_info(f"Using binsize={binsize}")
        if binsize < 2:
            lm.main_warning("Please consider using a larger bin size.")
        if binsize > 1:
            x_bin = bin_indices(data["x"].values, 0, binsize)
            y_bin = bin_indices(data["y"].values, 0, binsize)
            data["x"], data["y"] = x_bin, y_bin

        data["label"] = data["x"].astype(str) + "-" + data["y"].astype(str)
        if add_props:
            props = get_bin_props(data[["x", "y", "label"]].drop_duplicates(), binsize)
    else:
        raise NotImplementedError()

    uniq_cell = sorted(data["label"].unique())
    shape = (len(uniq_cell), len(uniq_gene))
    cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))

    label_gene_counts = data.groupby(["label", "gene"], observed=True, sort=False).size()
    label_gene_counts.name = "count"
    label_gene_counts = label_gene_counts.reset_index()
    x_ind = label_gene_counts["label"].map(cell_dict).astype(int).values
    y_ind = label_gene_counts["gene"].map(gene_dict).astype(int).values

    lm.main_info("Constructing count matrix.")
    X = csr_matrix((label_gene_counts["count"].values, (x_ind, y_ind)), shape=shape)
    obs = pd.DataFrame(index=uniq_cell)
    var = pd.DataFrame(index=uniq_gene)
    adata = AnnData(X=X, obs=obs, var=var)
    if metadata is not None:
        adata.obs = metadata.set_index("label").loc[adata.obs_names]
    if props is not None:
        ordered_props = props.loc[adata.obs_names]
        adata.obs["area"] = ordered_props["area"].values
        adata.obsm["spatial"] = ordered_props.filter(regex="centroid-").values
        adata.obsm["contour"] = ordered_props["contour"].values
        adata.obsm["bbox"] = ordered_props.filter(regex="bbox-").values

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
