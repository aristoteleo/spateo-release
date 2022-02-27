"""IO functions for BGI stereo technology.

Todo:
    * Optimize `read_bgi` and add appropriate functionality for generating an
        AnnData directly from cell labels.
    * Figure out how to appropriately deal with bounding boxes and offsets.
        @Xiaojieqiu
"""
import gzip
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import skimage.io
from anndata import AnnData
from scipy.sparse import csr_matrix, spmatrix
from shapely.geometry import Polygon, MultiPolygon

from .utils import bin_indices, centroids, get_bin_props, get_label_props, get_coords_labels, in_concave_hull
from ..configuration import SKM
from ..warnings import IOWarning

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
    scale: Optional[float] = 1.0,
    scale_unit: Optional[str] = None,
    binsize: int = 1,
    gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
) -> AnnData:
    """Read BGI read file to calculate total number of UMIs observed per
    coordinate.

    Args:
        path: Path to read file.
        stain_path: Path to nuclei staining image. Must have the same coordinate
            system as the read file.
        scale: Physical length per coordinate. For visualization only.
        scale_unit: Scale unit.
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
    x, y = data["x"].values, data["y"].values
    x_max, y_max = x.max(), y.max()
    shape = (x_max + 1, y_max + 1)

    # Read image and update x,y max if appropriate
    layers = {}
    if stain_path:
        image = skimage.io.imread(stain_path)
        x_max = max(x_max, image.shape[0])
        y_max = max(y_max, image.shape[1])
        shape = (x_max + 1, y_max + 1)
        # Reshape image to match new x,y max
        if image.shape != shape:
            image = np.pad(image, ((0, shape[0] - image.shape[0]), (0, shape[1] - image.shape[1])))
        layers[SKM.STAIN_LAYER_KEY] = image

    if binsize > 1:
        shape = (math.ceil(shape[0] / binsize), math.ceil(shape[1] / binsize))
        x = bin_indices(x, 0, binsize)
        y = bin_indices(y, 0, binsize)
        x_min, y_min = x.min(), y.min()

        # Resize image if necessary
        if stain_path:
            layers[SKM.STAIN_LAYER_KEY] = cv2.resize(image, shape[::-1])

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

    adata = AnnData(X=X, layers=layers)[x_min:, y_min:].copy()

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_AGG_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata


def read_bgi(
    path: str,
    scale: float = 1.0,
    scale_unit: Optional[str] = None,
    binsize: int = 1,
    segmentation_adata: Optional[AnnData] = None,
    labels_layer: Optional[str] = None,
    labels: Optional[Union[np.ndarray, str]] = None,
) -> AnnData:
    """Read BGI read file as AnnData.

    Args:
        path: Path to read file.
        scale: Physical length per coordinate. For visualization only.
        scale_unit: Scale unit.
        binsize: Size of pixel bins. Should only be provided when labels
            (i.e. the `segmentation_adata` and `labels` arguments) are not used
        segmentation_adata: AnnData containing segmentation results
        labels_layer: Layer name in `segmentation_adata` containing labels
        labels: Numpy array or path to numpy array saved with `np.save` that
            contains labels

    Returns:
        Bins x genes or labels x genes AnnData.
    """
    # Check inputs
    if (segmentation_adata is None) ^ (labels_layer is None):
        raise IOError("Both `segmentation_adata` and `labels_layer` must be provided")
    if segmentation_adata is not None:
        if labels is not None:
            raise IOError("Only one of `segmentation_adata` or `labels` may be provided")
        if binsize > 1:
            raise IOError("`binsize` argument is not supported when `segmentation_adata` is provided")
        if SKM.get_adata_type(segmentation_adata) != SKM.ADATA_AGG_TYPE:
            raise IOError("Only `AGG` type AnnDatas are supported")
    elif abs(int(binsize)) != binsize:
        raise IOError("Positive integer `binsize` must be provided when `segmentation_adata` is not provided")
    if isinstance(labels, str):
        labels = np.load(labels)

    data = read_bgi_as_dataframe(path)

    # Only binning supported in this case
    if segmentation_adata is None and labels is None:
        if binsize < 2:
            warnings.warn("Using binsize of 1. Please consider using a larger bin size.", IOWarning)

        if binsize > 1:
            x_bin = bin_indices(data["x"].values, 0, binsize)
            y_bin = bin_indices(data["y"].values, 0, binsize)
            data["x"], data["y"] = x_bin, y_bin

        data["label"] = data["x"].astype(str) + "-" + data["y"].astype(str)
        props = get_bin_props(data[["x", "y", "label"]].drop_duplicates(), binsize)

    # Use labels.
    else:
        shape = (data["x"].max(), data["y"].max())
        if labels is not None:
            if labels.shape != shape:
                warnings.warn(f"Labels matrix {labels.shape} has different shape as data matrix {shape}", IOWarning)
        else:
            labels = SKM.select_layer_data(segmentation_adata, labels_layer)
        label_coords = get_coords_labels(labels)

        if labels_layer is not None:
            seg_binsize = SKM.get_uns_spatial_attribute(segmentation_adata, SKM.UNS_SPATIAL_BINSIZE_KEY)
            x_min, y_min = (
                int(segmentation_adata.obs_names[0]) * seg_binsize,
                int(segmentation_adata.var_names[0]) * seg_binsize,
            )
            label_coords["x"] += x_min
            label_coords["y"] += y_min

        # When binning was used for segmentation, need to expand indices to cover
        # every binned pixel.
        if seg_binsize > 1:
            coords_dfs = [label_coords]
            for i in range(seg_binsize):
                for j in range(seg_binsize):
                    coords = label_coords.copy()
                    coords["x"] += i
                    coords["y"] += j
                    coords_dfs.append(coords)
            label_coords = pd.concat(coords_dfs, ignore_index=True)
        data = pd.merge(data, label_coords, on=["x", "y"], how="inner")
        props = get_label_props(labels)

    uniq_cell, uniq_gene = sorted(data["label"].unique()), sorted(data["geneID"].unique())
    shape = (len(uniq_cell), len(uniq_gene))
    cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))
    x_ind = data["label"].map(cell_dict).astype(int).values
    y_ind = data["geneID"].map(gene_dict).astype(int).values

    X = csr_matrix((data[data.columns[COUNT_COLUMN_MAPPING[SKM.X_LAYER]]].values, (x_ind, y_ind)), shape=shape)
    layers = {}
    for name, i in COUNT_COLUMN_MAPPING.items():
        if name != SKM.X_LAYER and i < len(data.columns):
            layers[name] = csr_matrix((data[data.columns[i]].values, (x_ind, y_ind)), shape=shape)

    obs = pd.DataFrame(index=uniq_cell)
    var = pd.DataFrame(index=uniq_gene)
    adata = AnnData(X=X, obs=obs, var=var, layers=layers)
    ordered_props = props.loc[adata.obs_names]
    adata.obs["area"] = ordered_props["area"].values
    adata.obsm["spatial"] = ordered_props.filter(regex="centroid-").values
    # adata.obsm["contour"] = ordered_props["contour"].values
    if segmentation_adata is not None:
        adata.obsm["bbox"] = ordered_props.filter(regex="bbox-").values

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
