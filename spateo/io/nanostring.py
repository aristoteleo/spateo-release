"""IO functions for NanoString CosMx technology.
"""
from typing import List, Optional, Union

import pandas as pd
from anndata import AnnData

from ..configuration import SKM
from ..logging import logger_manager as lm


def read_nanostring_as_dataframe(path: str, label_columns: Optional[List[str]] = None) -> pd.DataFrame:
    dtype = {"target": "category", "CellComp": "category"}
    # Use first 10 rows for validation.
    df = pd.read_csv(path, dtype=dtype, nrows=10)
    if label_columns:
        for column in label_columns:
            if column not in df.columns:
                raise IOError(f"Column `{column}` is not present.")

    df = pd.read_csv(path, dtype=dtype)
    if label_columns:
        labels = df[label_columns[0]].astype(str)
        for label in label_columns[1:]:
            labels += "-" + df[label].astype(str)
    df["label"] = labels
    return df


def read_nanostring(
    path: str,
    meta_path: str,
    scale: float = 1.0,
    scale_unit: Optional[str] = None,
    binsize: Optional[int] = None,
    label_columns: Optional[Union[str, List[str]]] = None,
) -> AnnData:
    """Read NanoString CosMx data as AnnData.

    Args:
        path: Path to transcript detection CSV file.
        meta_path: Path to cell metadata CSV file.
        scale: Physical length per coordinate. For visualization only.
        scale_unit: Scale unit.
        binsize: Size of pixel bins. Should only be provided when labels
            (i.e. the `segmentation_adata` and `labels` arguments) are not used.
        label_columns: Columns that contain already-segmented cell labels. Each
            unique combination is considered a unique cell.
    Returns:
        Bins x genes or labels x genes AnnData.
    """
    if sum([binsize is not None, label_columns is not None]) != 1:
        raise IOError("Exactly one of `binsize`, `label_columns` must be provided.")
    if binsize is not None and abs(int(binsize)) != binsize:
        raise IOError("Positive integer `binsize` must be provided when `segmentation_adata` is not provided.")

    label_columns = [label_columns] if isinstance(label_columns, str) else label_columns
    data = read_nanostring_as_dataframe(path, label_columns)
    metadata = read_nanostring_as_dataframe(meta_path, label_columns)

    if label_columns:
        lm.main_info(f"Using cell labels from `{label_columns}` columns.")
        binsize = 1
        # cell_ID == 0 indicates not assigned
        data = data[data["cell_ID"] > 0]
    elif binsize is not None:
        # Unclear how to deal with binning, since coordinates have decimals.
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    uniq_gene = sorted(data["target"].unique())
    uniq_cell = sorted(data["label"].unique())
    shape = (len(uniq_cell), len(uniq_gene))
    cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))

    label_gene_counts = data.groupby(["label", "target"], observed=True, sort=False)
    label_gene_counts.name = "count"
    label_gene_counts = label_gene_counts.reset_index()
    x_ind = label_gene_counts["label"].map(cell_dict).astype(int).values
    y_ind = label_gene_counts["target"].map(gene_dict).astype(int).values

    lm.main_info("Constructing count matrices.")
    X = csr_matrix((label_gene_counts["count"].values, (x_ind, y_ind)), shape=shape)
    obs = metadata.set_index("label").loc[uniq_cell]
    var = pd.DataFrame(index=uniq_gene)
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].values

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
