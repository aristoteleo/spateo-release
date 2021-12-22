from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from .utils import bin_index, centroid, get_bin_props, get_label_props


def read_bgi(
    filename: str,
    binsize: int = 50,
    slice: Optional[str] = None,
    label_path: Optional[str] = None,
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

    data = pd.read_csv(filename, header=0, delimiter="\t")
    data["geneID"] = data.geneID.astype(str).str.strip('"')

    # get cell name
    if not label_path:
        x, y = data["x"], data["y"]
        x_min, y_min = np.min(x), np.min(y)

        data["x_ind"] = bin_index(data["x"].values, x_min, binsize)
        data["y_ind"] = bin_index(data["y"].values, y_min, binsize)

        data["x_centroid"] = centroid(data["x_ind"].values, x_min, binsize)
        data["y_centroid"] = centroid(data["y_ind"].values, y_min, binsize)

        data["cell_name"] = data["x_ind"].astype(str) + "_" + data["y_ind"].astype(str)
    else:
        data["cell_name"] = data["cell"].astype(str)

    uniq_cell, uniq_gene = data.cell_name.unique(), data.geneID.unique()
    uniq_cell, uniq_gene = list(uniq_cell), list(uniq_gene)

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

    # !Note that in this version, the column name of the gene count value may be 'UMICount' or 'MIDCounts'.
    count_name = "UMICount" if "UMICount" in data.columns else "MIDCounts"

    # Important! by default, duplicate entries are summed together in the following which is needed for us!
    csr_mat = csr_matrix(
        (data[count_name], (data["csr_x_ind"], data["csr_y_ind"])),
        shape=((len(uniq_cell), len(uniq_gene))),
    )

    # get cell
    if not label_path:
        # aggregate spots with multiple bins
        label_props = get_bin_props(
            data[["x_ind", "y_ind"]].drop_duplicates(inplace=False), binsize
        )
        coor = data[["x_centroid", "y_centroid"]].drop_duplicates(inplace=False).values
    else:
        # Measure properties and get contours of labeled cell regions.
        label_mtx = np.load(label_path)
        label_props = get_label_props(
            label_mtx, properties=("label", "area", "bbox", "centroid")
        )
        # Get centroid from label_props
        coor = label_props[["centroid-0", "centroid-1"]].values

    label_props["cell_name"] = uniq_cell
    label_props["slice"] = slice

    var = pd.DataFrame({"gene_short_name": uniq_gene})
    var.set_index("gene_short_name", inplace=True)

    # to GeoDataFrame
    obs = gpd.GeoDataFrame(label_props, geometry="contours")
    obs.set_index("cell_name", inplace=True)

    obsm = {"spatial": coor}

    adata = AnnData(csr_mat, obs=obs.copy(), var=var.copy(), obsm=obsm.copy())

    return adata
