import pandas as pd
import numpy as np
from anndata import AnnData

from scipy.sparse import csr_matrix


def bin_index(coord, coord_min, binsize=50):
    """Take a DNB coordinate, the mimimum coordinate and the binsize, calculate the index of bins for the current
    coordinate.

    Parameters
    ----------
        coord: `float`
            Current x or y coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        binsize: `float`
            Size of the bins to aggregate data.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """

    num = np.floor((coord - coord_min) / binsize)

    return num.astype(int)


def centroid(bin_ind, coord_min, binsize=50):
    """Take a bin index, the mimimum coordinate and the binsize, calculate the centroid of the current bin.

    Parameters
    ----------
        bin_ind: `float`
            The bin index for the current coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        coord_centroids: `float`
            The x or y coordinate for the centroid corresponding to the current bin index.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """

    coord_centroids = coord_min + bin_ind * binsize + binsize / 2

    return coord_centroids


def readBGI(filename, binsize=50, version='stereo_v1'):
    """A helper function that facilitates constructing an AnnData object suitable for downstream spateo analysis

    Parameters
    ----------
        filename: `str`
            A string that points to the directory and filename of spatial transcriptomics dataset, produced by the
            stereo-seq method from BGI.
        binsize: `int` (default: 50)
            The number of spatial bins to aggregate RNAs captured by DNBs in those bins. Usually this is 50, which is
            close to 25 uM.
        version: `str`
            The version of technology. Currently not used. But may be useful when the data format changes after we update
            the stero-seq techlogy in future.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An AnnData object. Each row of the AnnData object correspond to a spot (aggregated with multiple bins). The
            `spatial` key in the .obsm corresponds to the x, y coordinates of the centroids of all spot.
    """

    data = pd.read_csv(filename, header=0, delimiter='\t')

    x, y = data['x'], data['y']
    x_min, y_min = np.min(x), np.min(y)

    data['x_ind'] = bin_index(data['x'].values, x_min, binsize)
    data['y_ind'] = bin_index(data['y'].values, y_min, binsize)

    data['x_centroid'] = centroid(data['x_ind'].values, x_min, binsize)
    data['y_centroid'] = centroid(data['y_ind'].values, y_min, binsize)

    data['cell_name'] = data['x_ind'].astype(str) + '_' + data['y_ind'].astype(str)

    uniq_cell, uniq_gene = data.cell_name.unique(), data.geneID.unique()
    uniq_cell, uniq_gene = list(uniq_cell), list(uniq_gene)

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

    # Important! by default, duplicate entries are summed together in the following which is needed for us!
    csr_mat = csr_matrix((data['UMICount'], (data["csr_x_ind"], data["csr_y_ind"])),
                         shape=((len(uniq_cell), len(uniq_gene))))

    var = pd.DataFrame(
        {"gene_short_name": uniq_gene}
    )
    var.set_index("gene_short_name", inplace=True)

    obs = pd.DataFrame(
        {"cell_name": uniq_cell}
    )
    obs.set_index("cell_name", inplace=True)

    obsm = {
        "spatial": data.loc[:, ['x_centroid', "y_centroid"]].drop_duplicates().values
    }

    adata = AnnData(csr_mat, obs=obs.copy(), var=var.copy(), obsm=obsm.copy())

    return adata



