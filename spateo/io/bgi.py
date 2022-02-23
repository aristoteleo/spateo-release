"""IO functions for BGI stereo technology.

Todo:
    * Optimize `read_bgi` and add appropriate functionality for generating an
        AnnData directly from cell labels.
    * Figure out how to appropriately deal with bounding boxes and offsets.
        @Xiaojieqiu
"""
import gzip
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import skimage.io
from anndata import AnnData
from scipy.sparse import csr_matrix, spmatrix
from shapely.geometry import Polygon, MultiPolygon

from ..configuration import SKM
from .utils import bin_indices, centroids, get_bin_props, get_label_props, in_concave_hull, mapping_label

COUNT_COLUMN_MAPPING = {
    SKM.X_LAYER: 3,
    SKM.SPLICED_LAYER_KEY: 4,
    SKM.UNSPLICED_LAYER_KEY: 5,
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
            "geneID": "category",  # geneID
            "x": np.uint32,  # x
            "y": np.uint32,  # y
            3: np.uint16,  # total
            4: np.uint16,  # spliced
            5: np.uint16,  # unspliced
        },
    )


def read_bgi_agg(
    path: str,
    stain_path: Optional[str] = None,
    scale: Optional[float] = None,
    scale_unit: Optional[str] = "um",
    binsize: int = 1,
    gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
) -> AnnData:
    """Read BGI read file to calculate total number of UMIs observed per
    coordinate.

    Args:
        path: Path to read file.
        stain_path: Path to nuclei staining image. Must have the same coordinate
            system as the read file.
        scale: Physical length per coordinate, in um. For visualization only.
        scale_unit: Scale unit. Defaults to um.
        binsize: Size of pixel bins.
        gene_agg: Dictionary of layer keys to gene names to aggregate. For
            example, `{'mito': ['list', 'of', 'mitochondrial', 'genes']}` will
            yield an AnnData with a layer named "mito" with the aggregate total
            UMIs of the provided gene list.

    Returns:
        An AnnData object containing the UMIs per coordinate and the nucleus
        staining image, if provided. The total UMIs are stored as a sparse matrix in
        `.X`, and spliced and unspliced counts (if present) are stored in
        `.layers['spliced']` and `.layers['unspliced']` respectively.
        The nuclei image is stored as a Numpy array in `.layers['nuclei']`.
    """
    data = read_bgi_as_dataframe(path)
    x_min, y_min = data["x"].min(), data["y"].min()
    data["x"] -= x_min
    data["y"] -= y_min
    x, y = data["x"].values, data["y"].values
    x_max, y_max = x.max(), y.max()
    shape = (x_max + 1, y_max + 1)

    # Read image and update x,y max if appropriate
    layers = {}
    if stain_path:
        image = skimage.io.imread(stain_path)[x_min:, y_min:]
        x_max = max(x_max, image.shape[0])
        y_max = max(y_max, image.shape[1])
        shape = (x_max + 1, y_max + 1)
        # Reshape image to match new x,y max
        if image.shape != shape:
            image = np.pad(image, ((0, shape[0] - image.shape[0]), (0, shape[1] - image.shape[1])))
        # Resize image to match bins
        layers[SKM.STAIN_LAYER_KEY] = image

    if binsize > 1:
        shape = (math.ceil(shape[0] / binsize), math.ceil(shape[1] / binsize))
        x = bin_indices(x, 0, binsize)
        y = bin_indices(y, 0, binsize)

        # Resize image if necessary
        if stain_path:
            layers[SKM.STAIN_LAYER_KEY] = cv2.resize(image, shape[::-1])

        if scale is not None:
            scale *= binsize

    # Put total in X
    X = csr_matrix((data[data.columns[COUNT_COLUMN_MAPPING[SKM.X_LAYER]]].values, (x, y)), shape=shape, dtype=np.uint16)

    for name, i in COUNT_COLUMN_MAPPING.items():
        if name != SKM.X_LAYER and i < len(data.columns):
            layers[name] = csr_matrix((data[data.columns[i]].values, (x, y)), shape=shape, dtype=np.uint16)

    # Aggregate gene lists
    if gene_agg:
        for name, genes in gene_agg.items():
            mask = data["geneID"].isin(genes) if isinstance(genes, list) else data["geneID"].map(genes)
            data_genes = data[mask]
            _x, _y = data_genes["x"].values, data_genes["y"].values
            layers[name] = csr_matrix(
                (data_genes[data.columns[COUNT_COLUMN_MAPPING[SKM.X_LAYER]]].values, (_x, _y)),
                shape=shape,
                dtype=np.uint16,
            )

    adata = AnnData(X=X, layers=layers)

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_AGG_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_XMIN_KEY, x_min)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_YMIN_KEY, y_min)
    if scale:
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
        SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata


def read_bgi(
    path: str,
    binsize: int = 50,
    slice: Optional[str] = None,
    label_path: Optional[str] = None,
    alpha_hull: Optional[Polygon] = None,
    version="stereo_v1",
) -> AnnData:
    """A helper function that facilitates constructing an AnnData object suitable for downstream spateo analysis

    Parameters
    ----------
        path: `str`
            A string that points to the directory and filename of spatial transcriptomics dataset, produced by the
            stereo-seq method from BGI.
        binsize: `int` (default: 50)
            The number of spatial bins to aggregate RNAs captured by DNBs in those bins. Usually this is 50, which is
            close to 25 uM.
        slice: `str` or None (default: None)
            Name of the slice. Will be used when displaying multiple slices.
        label_path: `str` or None (default: None)
            A string that points to the path of cell segmentation label matrix(Format:`.npy` or '.npz').
            If not None, the results of cell segmentation will be used, and param `binsize` will be ignored.
        alpha_hull: `Polygon` or None (default: None)
            The computed concave hull. It must be a Polygon and thus you may need to take one of the Polygon from the
            MultiPolygon object returned from `get_concave_hull` function.
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
    total_column = columns[COUNT_COLUMN_MAPPING["total"]]

    # get cell name
    if not label_path:
        x, y = data["x"].values, data["y"].values
        x_min, y_min = np.min(x), np.min(y)

        if alpha_hull is not None:
            is_inside = in_concave_hull(data.loc[:, ["x", "y"]].values, alpha_hull)
            data = data.loc[is_inside, :]
            x, y = data["x"].values, data["y"].values

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
        # load label file
        label_file = np.load(label_path)
        shifts = None
        # npy (just labels) or npz (['x_min', 'y_min', 'labels'])
        if isinstance(label_file, np.lib.npyio.NpzFile):
            shifts = (label_file["x_min"], label_file["y_min"])
            label_mtx = label_file["labels"]
        else:
            label_mtx = label_file
        data = mapping_label(data, label_mtx, shifts=shifts)

        # Measure properties and get contours of labeled cell regions.
        label_props = get_label_props(label_mtx, properties=("label", "area", "bbox", "centroid"))
        # Get centroid from label_props
        if alpha_hull is not None:
            is_inside = in_concave_hull(label_props.loc[:, ["centroid-0", "centroid-1"]].values, alpha_hull)
            label_props = label_props.loc[is_inside, :]

        coor = label_props[["centroid-0", "centroid-1"]].values

    uniq_cell, uniq_gene = data["cell_name"].unique(), data["geneID"].unique()
    uniq_cell, uniq_gene = list(uniq_cell), list(uniq_gene)

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

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
