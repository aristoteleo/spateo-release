"""IO functions for BGI stereo technology.

Todo:
    * Optimize `read_bgi` and add appropriate functionality for generating an
        AnnData directly from cell labels.
    * Figure out how to appropriately deal with bounding boxes and offsets.
        @Xiaojieqiu
"""
import gzip
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, spmatrix
from shapely.geometry import Polygon, MultiPolygon

from .utils import (
    bin_indices,
    centroids,
    get_bin_props,
    get_label_props,
    in_concave_hull,
)

COUNT_COLUMN_MAPPING = {
    "total": 3,
    "spliced": 4,
    "unspliced": 5,
}


def read_bgi_as_dataframe(path: str) -> pd.DataFrame:
    """Read a BGI read file as a pandas DataFrame.

    Args:
        path: Path to read file.

    Returns:
        Pandas Dataframe with column names `gene`, `x`, `y`, `total` and
        additionally `spliced` and `unspliced` if splicing counts are present.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype={
            0: "category",  # geneID
            1: np.uint32,  # x
            2: np.uint32,  # y
            3: np.uint16,  # total
            4: np.uint16,  # spliced
            5: np.uint16,  # unspliced
        },
    )


def read_bgi_agg(
    path: str,
    x_max: Optional[int] = None,
    y_max: Optional[int] = None,
    binsize: Optional[int] = 1,
) -> Tuple[spmatrix, Optional[spmatrix], Optional[spmatrix]]:
    """Read BGI read file to calculate total number of UMIs observed per
    coordinate.

    Args:
        path: Path to read file.
        x_max: Maximum x coordinate. If not provided, the maximum non-zero x
            will be used.
        y_max: Maximum y coordinate. If not provided, the maximum non-zero y
            will be used.
        binsize: The number of spatial bins to aggregate RNAs captured by DNBs in those bins. By default it is 1.

    Returns:
        Tuple containing 3 sparse matrices corresponding to total, spliced,
        and unspliced counts respectively. If the read file does not contain
        spliced and unspliced counts, the last two elements are None.
    """
    data = read_bgi_as_dataframe(path)
    x, y = data["x"].values, data["y"].values
    x_min, y_min = np.min(x), np.min(y)

    if binsize != 1:
        data["x"] = bin_indices(x, x_min, binsize)
        data["y"] = bin_indices(y, y_min, binsize)

    x_max = max(x_max, data["x"].max()) if x_max else data["x"].max()
    y_max = max(y_max, data["y"].max()) if y_max else data["y"].max()

    matrices = []
    for name, i in COUNT_COLUMN_MAPPING.items():
        if i < len(data.columns):
            matrices.append(
                csr_matrix(
                    (data[data.columns[i]], (data["x"] - x_min, data["y"] - y_min)),
                    shape=(x_max - x_min + 1, y_max - y_min + 1),
                    dtype=np.uint16,
                )
            )
        else:
            matrices.append(None)
    return tuple(matrices)


def read_bgi(
    path: str,
    binsize: int = 50,
    slice: Optional[str] = None,
    label_path: Optional[str] = None,
    alpha_hull: Union[Polygon, MultiPolygon] = None,
    version="stereo_v1",
) -> AnnData:
    """A helper function that facilitates constructing an AnnData object suitable for downstream spateo analysis

    Parameters
    ----------
        filename: `str`
            A string that points to the directory and filename of spatial transcriptomics dataset, produced by the
            stereo-seq method from BGI.
        binsize: `int` (default: 50)
            The number of spatial bins to aggregate RNAs captured by DNBs in those bins. Usually this is 50, which is
            close to 25 uM.
        slice: `str` or None (default: None)
            Name of the slice. Will be used when displaying multiple slices.
        label_path: `str` or None (default: None)
            A string that points to the directory and filename of cell segmentation label matrix(Format:`.npy`).
            If not None, the results of cell segmentation will be used, and param `binsize` will be ignored.
        version: `str`
            The version of technology. Currently not used. But may be useful when the data format changes after we update
            the stero-seq techlogy in future.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An AnnData object. Each row of the AnnData object correspond to a spot (aggregated with multiple bins). The
            `spatial` key in the .obsm corresponds to the x, y coordinates of the centroids of all spot.
    """
    data = read_bgi_as_dataframe(path)
    columns = list(data.columns)
    gene_column, x_column, y_column = columns[:3]
    total_column = columns[COUNT_COLUMN_MAPPING["total"]]

    # get cell name
    if not label_path:
        x, y = data[x_column].values, data[y_column].values
        x_min, y_min = np.min(x), np.min(y)

        if alpha_hull is not None:
            is_inside = in_concave_hull(
                data.loc[:, [x_column, y_column]].values, alpha_hull
            )
            data = data.loc[is_inside, :]

        data["x_ind"] = bin_indices(x, x_min, binsize)
        data["y_ind"] = bin_indices(y, y_min, binsize)
        data["x_centroid"] = centroids(data["x_ind"].values, x_min, binsize)
        data["y_centroid"] = centroids(data["y_ind"].values, y_min, binsize)

        # TODO: This take a long time for many rows! Map each unique x, y pair
        # to a cell name first.
        data["cell_name"] = data["x_ind"].astype(str) + "_" + data["y_ind"].astype(str)

        # aggregate spots with multiple bins
        dedup = ~data.duplicated("cell_name")
        label_props = get_bin_props(data[["x_ind", "y_ind"]][dedup], binsize)
        coor = data[["x_centroid", "y_centroid"]][dedup].values
    else:
        # TODO: Get cell names using labels
        data["cell_name"] = data["cell"].astype(str)

        # Measure properties and get contours of labeled cell regions.
        label_mtx = np.load(label_path)
        label_props = get_label_props(
            label_mtx, properties=("label", "area", "bbox", "centroid")
        )
        # Get centroid from label_props
        if alpha_hull is not None:
            is_inside = in_concave_hull(
                label_props.loc[:, ["centroid-0", "centroid-1"]].values, alpha_hull
            )
            label_props = label_props.loc[is_inside, :]

        coor = label_props[["centroid-0", "centroid-1"]].values

    uniq_cell, uniq_gene = data["cell_name"].unique(), data[gene_column].unique()
    uniq_cell, uniq_gene = list(uniq_cell), list(uniq_gene)

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data[gene_column].map(gene_dict)

    # Important! by default, duplicate entries are summed together in the following which is needed for us!
    X = csr_matrix(
        (data[total_column], (data["csr_x_ind"], data["csr_y_ind"])),
        shape=(len(uniq_cell), len(uniq_gene)),
    )
    layers = {}
    for name, i in COUNT_COLUMN_MAPPING.items():
        if name != "total" and i < len(columns):
            layers[name] = csr_matrix(
                (data[columns[i]], (data["csr_x_ind"], data["csr_y_ind"])),
                shape=(len(uniq_cell), len(uniq_gene)),
            )

    label_props["cell_name"] = uniq_cell
    label_props["slice"] = slice

    var = pd.DataFrame({"gene_short_name": uniq_gene})
    var.set_index("gene_short_name", inplace=True)

    # to GeoDataFrame
    obs = gpd.GeoDataFrame(label_props, geometry="contours")
    obs.set_index("cell_name", inplace=True)

    obsm = {"spatial": coor}

    adata = AnnData(X=X, layers=layers, obs=obs, var=var, obsm=obsm)

    return adata
