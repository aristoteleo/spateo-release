import pandas as pd
import numpy as np
from anndata import AnnData
import cv2
from skimage import measure
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd

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

    # !Note that in this version, the column name of the gene count value may be 'UMICount' or 'MIDCounts'.
    count_name = 'UMICount' if 'UMICount' in data.columns else 'MIDCounts'

    # Important! by default, duplicate entries are summed together in the following which is needed for us!
    csr_mat = csr_matrix((data[count_name], (data["csr_x_ind"], data["csr_y_ind"])),
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


def readBGI_cells(filename,
                  label_path,
                  slice: str = None,
                  version='stereo_v1') -> AnnData:
    """A helper function that facilitates constructing an AnnData object suitable for downstream spateo analysis.
    Using the results of cell segmentation.

    Parameters
    ----------
        filename: `str`
            A string that points to the directory and filename of single-cell spatial transcriptomics dataset, produced
            by the stereo-seq method from BGI. Using the results of cell segmentation.
        label_path: `str`
            A string that points to the directory and filename of cell segmentation label matrix(Format:`.npy`).
            Will be used when displaying with exact shapes of cells.
        slice: `str` (default: None)
            Name of the slice. Will be used when displaying multiple slices.
        version: `str`
            The version of technology. Currently not used. But may be useful when the data format changes after we update
            the stereo-seq technology in future.

    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An AnnData object. Each row of the AnnData object correspond to a cell (results of cell segmentation). The
            `spatial` key in the .obsm corresponds to the x, y coordinates of the centroids of all cells. The
            .obs is a `geopandas.GeoDataFrame` with the geometry `contours`.
            The columns in .obs are the region properties, including `label`, `area`, `centroid`, `bbox` and `contours`.
    """
    data = pd.read_csv(filename, header=0, delimiter='\t')

    data['cell_name'] = data['cell'].astype(str)

    uniq_cell, uniq_gene = data.cell_name.unique(), data.geneID.unique()
    uniq_cell, uniq_gene = list(uniq_cell), list(uniq_gene)

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

    # !Note that in this version, the column name of the gene count value may be 'UMICount' or 'MIDCounts'.
    count_name = 'UMICount' if 'UMICount' in data.columns else 'MIDCounts'

    # Important! by default, duplicate entries are summed together in the following which is needed for us!
    csr_mat = csr_matrix((data[count_name], (data["csr_x_ind"], data["csr_y_ind"])),
                         shape=((len(uniq_cell), len(uniq_gene))))

    label_mtx = np.load(label_path)
    # Measure properties and get contours of labeled cell regions.
    label_props = _get_cell_props(label_mtx, properties=('label', 'area', 'bbox', 'centroid'))
    label_props["cell_name"] = uniq_cell

    # Get centroid from label_props
    coor = label_props[["centroid-0", "centroid-1"]].values

    var = pd.DataFrame(
        {"gene_short_name": uniq_gene}
    )
    var.set_index("gene_short_name", inplace=True)

    # to GeoDataFrame
    obs = gpd.GeoDataFrame(label_props, geometry="contours")
    obs.set_index("cell_name", inplace=True)

    obsm = {
        "spatial": coor
    }

    adata = AnnData(csr_mat, obs=obs.copy(), var=var.copy(), obsm=obsm.copy())

    return adata


def _get_cell_props(label_mtx,
                    properties=('label', 'area', 'bbox', 'centroid')):
    """Measure properties of labeled cell regions.

    Parameters
    ----------
        label_mtx: `numpy.ndarray`
            cell segmentation label matrix
        properties: `tuple`
            used properties

    Returns
    -------
        props: `pandas.DataFrame`
            A dataframe with properties and contours

    """
    def contours(mtx):
        mtx = np.pad(mtx, 1)
        mtx[mtx > 0] = 255
        mtx = mtx.astype(np.uint8)
        contour = cv2.findContours(mtx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        contour = contour - np.array([1, 1])
        return contour

    def contour_to_geo(contour):
        n = contour.shape[0]
        contour = np.squeeze(contour)
        if n >= 3:
            geo = Polygon(contour)
        elif n == 2:
            geo = LineString(contour)
        else:
            geo = Point(contour)
        return geo

    props = measure.regionprops_table(label_mtx, properties=properties, extra_properties=[contours])
    props = pd.DataFrame(props)
    props['contours'] = props.apply(lambda x: x['contours'] + x[['bbox-0', 'bbox-1']].to_numpy(), axis=1)
    props['contours'] = props['contours'].apply(contour_to_geo)
    return props


def read_image(adata: AnnData,
               filename: str,
               scale_factor: float,
               slice: str = None,
               img_layer: str = None) -> AnnData:
    """Load an image into the AnnData object.

    Parameters
    ----------
        adata: `AnnData`
            AnnData object
        filename:  `str`
            The path of the image
        scale_factor: `float`
            The scale factor of the image. Define: pixels/DNBs
        slice: `str` (default: None)
            Name of the slice. Will be used when displaying multiple slices.
        img_layer: `str` (default: None)
            Name of the image layer.

    Returns
    -------
        adata: `AnnData`
            :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['images'][img_layer]`
                The stored image
            :attr:`~anndata.AnnData.uns`\\ `['spatial'][slice]['scalefactors'][img_layer]`
                The scale factor for the spots
    """
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Could not find '{filename}'")

    # Create a new dictionary or add to the original slice
    if 'spatial' not in adata.uns_keys():
        adata.uns['spatial'] = dict()
    if slice not in adata.uns['spatial'].keys():
        adata.uns['spatial'][slice] = dict()

    if 'images' not in adata.uns['spatial'][slice]:
        adata.uns['spatial'][slice]['images'] = {img_layer: img}
    else:
        adata.uns['spatial'][slice]['images'][img_layer] = img

    if 'scalefactors' not in adata.uns['spatial'][slice]:
        adata.uns['spatial'][slice]['scalefactors'] = {img_layer: scale_factor}
    else:
        adata.uns['spatial'][slice]['scalefactors'][img_layer] = scale_factor

    return adata
