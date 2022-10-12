"""IO functions for STARmap technology.
"""
import os

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import get_points_props


def read_starmap_as_anndata(data_dir: str) -> AnnData:
    """Read STARmap data directory as AnnData.

    Args:
        data_dir: Path to directory containing STARmap files.

    Returns:
        AnnData of cell x genes.
    """
    lm.main_info("Constructing count matrix.")
    X = pd.read_csv(os.path.join(data_dir, "cell_barcode_count.csv"), header=None)
    genes = pd.read_csv(os.path.join(data_dir, "cell_barcode_names.csv"), header=None)

    obs = pd.DataFrame(index=["Cell_" + str(i) for i in range(X.shape[0])])
    var = pd.DataFrame(index=genes[2])

    return AnnData(X=csr_matrix(X, dtype=np.uint16), obs=obs, var=var)


def read_starmap_positions_as_dataframe(path: str) -> pd.DataFrame:
    """Read STARmap cell positions npz as dataframe.

    Args:
        path: Path to file

    Returns:
        DataFrame containing cell positions.
    """
    labels = np.load(path)["labels"]
    labels = csr_matrix(labels).tocoo()
    df_labels = pd.DataFrame({"x": labels.row, "y": labels.col, "label": labels.data})[["x", "y", "label"]]

    # To consist with
    # https://github.com/weallen/STARmap/blob/0b1cddf459a69b73f935aca7f7e0008c349453c0/python/viz.py#L20
    unique_label, label_area = np.unique(df_labels["label"], return_counts=True)
    df_labels = df_labels[df_labels["label"].isin(unique_label[np.logical_and(label_area > 1000, label_area < 100000)])]
    df_labels = df_labels[df_labels["label"] != np.max(df_labels["label"])]

    return df_labels


def read_starmap(
    data_dir: str,
) -> AnnData:
    """Read STARmap data as AnnData.

    Args:
        data_dir: Path to directory containing STARmap files.
    """
    adata = read_starmap_as_anndata(data_dir)
    df_labels = read_starmap_positions_as_dataframe(os.path.join(data_dir, "labels.npz"))
    props = get_points_props(df_labels)

    props.index = adata.obs_names
    ordered_props = props.loc[adata.obs_names]
    adata.obs["area"] = ordered_props["area"].values
    adata.obsm["spatial"] = ordered_props.filter(regex="centroid-").values
    adata.obsm["contour"] = ordered_props["contour"].values
    adata.obsm["bbox"] = ordered_props.filter(regex="bbox-").values

    scale, scale_unit = 1.0, None

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
