"""IO functions for BGI stereo technology.
"""
import gzip
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import ngs_tools as ngs
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import skimage.io

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing_extensions import Literal

from ..configuration import SKM
from ..logging import logger_manager as lm
from .utils import (
    bin_indices,
    centroids,
    get_bin_props,
    get_coords_labels,
    get_label_props,
    get_points_props,
    in_concave_hull,
)

VERSIONS = {
    "stereo": ngs.chemistry.get_chemistry("Stereo-seq"),
}
COUNT_COLUMN_MAPPING = {
    SKM.X_LAYER: 3,
    SKM.SPLICED_LAYER_KEY: 4,
    SKM.UNSPLICED_LAYER_KEY: 5,
}


def read_bgi_as_dataframe(path: str, label_column: Optional[str] = None) -> pd.DataFrame:
    """Read a BGI read file as a pandas DataFrame.

    Args:
        path: Path to read file.
        label_column: Column name containing positive cell labels.

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `gene`: Gene name/ID (whatever was used in the original file)
            * `x`, `y`: X and Y coordinates
            * `total`, `spliced`, `unspliced`: Counts for each RNA species.
                The latter two is only present if they are in the original file.
    """
    dtype = {
        "geneID": "category",  # geneID
        "x": np.uint32,  # x
        "y": np.uint32,  # y
        # Multiple different names for total counts
        "MIDCounts": np.uint16,
        "MIDCount": np.uint16,
        "UMICount": np.uint16,
        "UMICounts": np.uint16,
        "EXONIC": np.uint16,  # spliced
        "INTRONIC": np.uint16,  # unspliced,
    }
    rename = {
        "MIDCounts": "total",
        "MIDCount": "total",
        "UMICount": "total",
        "UMICounts": "total",
        "EXONIC": "spliced",
        "INTRONIC": "unspliced",
    }
    # Use first 10 rows for validation.
    df = pd.read_csv(path, sep="\t", dtype=dtype, comment="#", nrows=10)

    if label_column:
        dtype.update({label_column: np.uint32})
        rename.update({label_column: "label"})

        if label_column not in df.columns:
            raise IOError(f"Column `{label_column}` is not present.")

    # If duplicate columns are provided, we don't know which to use!
    rename_inverse = {}
    for _from, _to in rename.items():
        rename_inverse.setdefault(_to, []).append(_from)
    for _to, _cols in rename_inverse.items():
        if sum(_from in df.columns for _from in _cols) > 1:
            raise IOError(f"Found multiple columns mapping to `{_to}`.")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=dtype,
        comment="#",
    ).rename(columns=rename)


def dataframe_to_labels(df: pd.DataFrame, column: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert a BGI dataframe that contains cell labels to a labels matrix.

    Args:
        df: Read dataframe, as returned by :func:`read_bgi_as_dataframe`.
        columns: Column that contains cell labels as positive integers. Any labels
            that are non-positive are ignored.

    Returns:
        Labels matrix
    """
    shape = shape or (df["x"].max() + 1, df["y"].max() + 1)
    labels = np.zeros(shape, dtype=int)

    for label, _df in df.drop_duplicates(subset=[column, "x", "y"]).groupby(column):
        if label <= 0:
            continue
        labels[(_df["x"].values, _df["y"].values)] = label
    return labels


def dataframe_to_filled_labels(df: pd.DataFrame, column: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert a BGI dataframe that contains cell labels to a (filled) labels matrix.

    Args:
        df: Read dataframe, as returned by :func:`read_bgi_as_dataframe`.
        columns: Column that contains cell labels as positive integers. Any labels
            that are non-positive are ignored.

    Returns:
        Labels matrix
    """
    shape = shape or (df["x"].max() + 1, df["y"].max() + 1)
    labels = np.zeros(shape, dtype=int)
    for label, _df in df.drop_duplicates(subset=[column, "x", "y"]).groupby(column):
        if label <= 0:
            continue
        points = _df[["x", "y"]].values.astype(int)
        min_offset = points.min(axis=0)
        max_offset = points.max(axis=0)
        xmin, ymin = min_offset
        xmax, ymax = max_offset
        points -= min_offset
        hull = cv2.convexHull(points, returnPoints=True)
        mask = cv2.fillConvexPoly(np.zeros((max_offset - min_offset + 1)[::-1], dtype=np.uint8), hull, color=1).T
        labels[xmin : xmax + 1, ymin : ymax + 1][mask == 1] = label
    return labels


def read_bgi_agg(
    path: str,
    stain_path: Optional[str] = None,
    binsize: int = 1,
    gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
    prealigned: bool = False,
    label_column: Optional[str] = None,
    version: Literal["stereo"] = "stereo",
) -> AnnData:
    """Read BGI read file to calculate total number of UMIs observed per
    coordinate.

    Args:
        path: Path to read file.
        stain_path: Path to nuclei staining image. Must have the same coordinate
            system as the read file.
        binsize: Size of pixel bins.
        gene_agg: Dictionary of layer keys to gene names to aggregate. For
            example, `{'mito': ['list', 'of', 'mitochondrial', 'genes']}` will
            yield an AnnData with a layer named "mito" with the aggregate total
            UMIs of the provided gene list.
        prealigned: Whether the stain image is already aligned with the minimum
            x and y RNA coordinates.
        label_column: Column that contains already-segmented cell labels.
        version: BGI technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.

    Returns:
        An AnnData object containing the UMIs per coordinate and the nucleus
        staining image, if provided. The total UMIs are stored as a sparse matrix in
        `.X`, and spliced and unspliced counts (if present) are stored in
        `.layers['spliced']` and `.layers['unspliced']` respectively.
        The nuclei image is stored as a Numpy array in `.layers['nuclei']`.
    """
    lm.main_debug(f"Reading data from {path}.")
    data = read_bgi_as_dataframe(path, label_column)
    x_min, y_min = data["x"].min(), data["y"].min()
    x, y = data["x"].values, data["y"].values
    x_max, y_max = x.max(), y.max()
    shape = (x_max + 1, y_max + 1)

    # Read image and update x,y max if appropriate
    layers = {}
    if stain_path:
        lm.main_debug(f"Reading stain image from {stain_path}.")
        image = skimage.io.imread(stain_path)
        if prealigned:
            lm.main_warning(
                (
                    "Assuming stain image was already aligned with the minimum x and y RNA coordinates. "
                    "(prealinged=True)"
                )
            )
            image = np.pad(image, ((x_min, 0), (y_min, 0)))
        x_max = max(x_max, image.shape[0] - 1)
        y_max = max(y_max, image.shape[1] - 1)
        shape = (x_max + 1, y_max + 1)
        # Reshape image to match new x,y max
        if image.shape != shape:
            lm.main_warning(f"Padding stain image from {image.shape} to {shape} with zeros.")
            image = np.pad(image, ((0, shape[0] - image.shape[0]), (0, shape[1] - image.shape[1])))
        layers[SKM.STAIN_LAYER_KEY] = image

    # Construct labels matrix if present
    labels = None
    if "label" in data.columns:
        lm.main_warning("Using the `label_column` option may result in disconnected labels.")
        labels = dataframe_to_labels(data, "label", shape)
        layers[SKM.LABELS_LAYER_KEY] = labels

    if binsize > 1:
        lm.main_info(f"Binning counts with binsize={binsize}.")
        shape = (math.ceil(shape[0] / binsize), math.ceil(shape[1] / binsize))
        x = bin_indices(x, 0, binsize)
        y = bin_indices(y, 0, binsize)
        x_min, y_min = x.min(), y.min()

        # Resize image if necessary
        if stain_path:
            layers[SKM.STAIN_LAYER_KEY] = cv2.resize(image, shape[::-1])

        if labels is not None:
            lm.main_warning("Cell labels were provided, but `binsize` > 1. There may be slight inconsistencies.")
            layers[SKM.LABELS_LAYER_KEY] = labels[::binsize, ::binsize]

    # See read_bgi_as_dataframe for standardized column names
    lm.main_info("Constructing count matrices.")
    X = csr_matrix((data["total"].values, (x, y)), shape=shape, dtype=np.uint16)
    if "spliced" in data.columns:
        layers[SKM.SPLICED_LAYER_KEY] = csr_matrix((data["spliced"].values, (x, y)), shape=shape, dtype=np.uint16)
    if "unspliced" in data.columns:
        layers[SKM.UNSPLICED_LAYER_KEY] = csr_matrix((data["unspliced"].values, (x, y)), shape=shape, dtype=np.uint16)

    # Aggregate gene lists
    if gene_agg:
        lm.main_info("Aggregating counts for genes provided by `gene_agg`.")
        for name, genes in gene_agg.items():
            mask = data["geneID"].isin(genes) if isinstance(genes, list) else data["geneID"].map(genes)
            data_genes = data[mask]
            _x, _y = data_genes["x"].values, data_genes["y"].values
            layers[name] = csr_matrix(
                (data_genes["total"].values, (_x, _y)),
                shape=shape,
                dtype=np.uint16,
            )

    adata = AnnData(X=X, layers=layers)[x_min:, y_min:].copy()

    scale, scale_unit = 1.0, None
    if version in VERSIONS:
        resolution = VERSIONS[version].resolution
        scale, scale_unit = resolution.scale, resolution.unit

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_AGG_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE, "segmentation_adata", optional=True)
def read_bgi(
    path: str,
    binsize: Optional[int] = None,
    segmentation_adata: Optional[AnnData] = None,
    labels_layer: Optional[str] = None,
    labels: Optional[Union[np.ndarray, str]] = None,
    seg_binsize: int = 1,
    label_column: Optional[str] = None,
    add_props: bool = True,
    version: Literal["stereo"] = "stereo",
) -> AnnData:
    """Read BGI read file as AnnData.

    Args:
        path: Path to read file.
        binsize: Size of pixel bins. Should only be provided when labels
            (i.e. the `segmentation_adata` and `labels` arguments) are not used.
        segmentation_adata: AnnData containing segmentation results.
        labels_layer: Layer name in `segmentation_adata` containing labels.
        labels: Numpy array or path to numpy array saved with `np.save` that
            contains labels.
        seg_binsize: the bin size used in cell segmentation, used in conjunction
            with `labels` and will be overwritten when `labels_layer` and
            `segmentation_adata` are not None.
        label_column: Column that contains already-segmented cell labels. If this
            column is present, this takes prescedence.
        add_props: Whether or not to compute label properties, such as area,
            bounding box, centroid, etc.
        version: BGI technology version. Currently only used to set the scale and
            scale units of each unit coordinate. This may change in the future.

    Returns:
        Bins x genes or labels x genes AnnData.
    """
    # Check inputs
    if sum([binsize is not None, segmentation_adata is not None, labels is not None, label_column is not None]) != 1:
        raise IOError("Exactly one of `segmentation_adata`, `binsize`, `labels`, `label_column` must be provided.")
    if (segmentation_adata is None) ^ (labels_layer is None):
        raise IOError("Both `segmentation_adata` and `labels_layer` must be provided.")
    if segmentation_adata is not None:
        if SKM.get_adata_type(segmentation_adata) != SKM.ADATA_AGG_TYPE:
            raise IOError("Only `AGG` type AnnDatas are supported.")
    if binsize is not None and abs(int(binsize)) != binsize:
        raise IOError("Positive integer `binsize` must be provided when `segmentation_adata` is not provided.")
    if isinstance(labels, str):
        labels = np.load(labels)

    lm.main_debug(f"Reading data from {path}.")
    data = read_bgi_as_dataframe(path, label_column)
    n_columns = data.shape[1]

    # Obtain total genes from raw data, so that the columns always match
    # regardless of what method was used.
    uniq_gene = sorted(data["geneID"].unique())

    props = None
    if label_column is not None:
        lm.main_info(f"Using cell labels from `{label_column}` column.")
        binsize = 1
        data = data[data["label"] > 0]
        if add_props:
            lm.main_warning(
                "Using `label_column` as cell labels with `add_props=True` may result in incorrect contours."
            )
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

    # Use labels.
    else:
        binsize = 1
        shape = (data["x"].max(), data["y"].max())
        if labels is not None:
            lm.main_info(f"Using labels provided with `labels` argument.")
            if labels.shape != shape:
                lm.main_warning(f"Labels matrix {labels.shape} has different shape as data matrix {shape}")
        else:
            labels = SKM.select_layer_data(segmentation_adata, labels_layer)
        label_coords = get_coords_labels(labels)

        if labels_layer is not None:
            lm.main_info(f"Using labels provided with `segmentation_adata` and `labels_layer` arguments.")
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
            lm.main_warning("Binning was used for segmentation.")
            coords_dfs = []
            for i in range(seg_binsize):
                for j in range(seg_binsize):
                    coords = label_coords.copy()
                    coords["x"] += i
                    coords["y"] += j
                    coords_dfs.append(coords)
            label_coords = pd.concat(coords_dfs, ignore_index=True)
        data = pd.merge(data, label_coords, on=["x", "y"], how="inner")
        if add_props:
            props = get_label_props(labels)

    uniq_cell = sorted(data["label"].unique())
    shape = (len(uniq_cell), len(uniq_gene))
    cell_dict = dict(zip(uniq_cell, range(len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(len(uniq_gene))))
    x_ind = data["label"].map(cell_dict).astype(int).values
    y_ind = data["geneID"].map(gene_dict).astype(int).values

    # See read_bgi_as_dataframe for standardized column names
    lm.main_info("Constructing count matrices.")
    X = csr_matrix((data["total"].values, (x_ind, y_ind)), shape=shape)
    layers = {}
    if "spliced" in data.columns:
        layers[SKM.SPLICED_LAYER_KEY] = csr_matrix((data["spliced"].values, (x_ind, y_ind)), shape=shape)
    if "unspliced" in data.columns:
        layers[SKM.UNSPLICED_LAYER_KEY] = csr_matrix((data["unspliced"].values, (x_ind, y_ind)), shape=shape)

    obs = pd.DataFrame(index=uniq_cell)
    var = pd.DataFrame(index=uniq_gene)
    adata = AnnData(X=X, obs=obs, var=var, layers=layers)
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
