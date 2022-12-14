"""IO functions for NanoString CosMx technology.
"""
import glob
import os
import re
import warnings
from typing import List, NamedTuple, Optional, Union

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
from .utils import get_points_props

try:
    import ngs_tools as ngs

    VERSIONS = {
        "cosmx": ngs.chemistry.get_chemistry("CosMx").resolution,
    }
except ModuleNotFoundError:

    class SpatialResolution(NamedTuple):
        scale: float = 1.0
        unit: Optional[Literal["nm", "um", "mm"]] = None

    VERSIONS = {"cosmx": SpatialResolution(0.18, "um")}

FOV_PARSER = re.compile("^.+_F(?P<fov>[0-9]+)\..+$")


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


def stitch_images(stain_dir: str, positions_path: str, labels: bool = False) -> np.ndarray:
    """Stitch multiple FOVs into a single image using position information.

    Args:
        stain_dir: Directory containing JPEG or TIFF files with filenames
            ending in '_FXXX' where XXX indicates the FOV index.
        positions_path: Path to CSV file containing FOV positions.
        labels: Whether these are labels (and therefore should be made unique).

    Returns:
        A numpy array containing the stitched image. May contain multiple channels,
            which is the last dimension of the array.
    """
    # Load all images in stain_dir, indexed by FOV
    stain_fov_paths = {}
    for filename in os.listdir(stain_dir):
        path = os.path.join(stain_dir, filename)
        match = FOV_PARSER.match(filename)
        if match:
            fov = int(match["fov"])
            if fov in stain_fov_paths:
                raise IOError(f"Multiple images for FOV {fov} were found: {stain_fov_paths[fov]}, {path}.")
            stain_fov_paths[fov] = path
    lm.main_debug(f"Found {len(stain_fov_paths)} FOV images.")

    # Read FOV positions and make sure they match exactly with the files.
    fov_df = pd.read_csv(positions_path, dtype={"fov": int}, index_col="fov")
    if set(fov_df.index) != set(stain_fov_paths.keys()):
        raise IOError(f"FOVs defined in {positions_path} do not match exactly with those found in {stain_dir}.")
    fov_x = dict(fov_df["x_global_px"].astype(np.uint32))
    fov_y = dict(fov_df["y_global_px"].astype(np.uint32))

    # Detect the size of the entire image.
    # Also, check that all the images have the same non-XY dimensions.
    xmin, ymin = min(fov_x.values()), min(fov_y.values())
    xmax, ymax = 0, 0
    extra_dims = None
    dtype = None
    stain_fovs = {}
    for fov, path in stain_fov_paths.items():
        x, y = fov_x[fov], fov_y[fov]
        img = skimage.io.imread(path)
        xmax = max(xmax, x + img.shape[1] - 1)
        ymax = max(ymax, y + img.shape[0] - 1)
        stain_fovs[fov] = img

        if extra_dims is None:
            extra_dims = img.shape[2:]
        elif extra_dims != img.shape[2:]:
            raise IOError(f"FOV {path} has inconsistent non-XY dimensions.")
        if dtype is None:
            dtype = img.dtype
        elif dtype != img.dtype:
            raise IOError(f"FOV {path} has inconsistent dtype.")

    if labels:
        dtype = np.uint

    last_label = 0
    img = np.zeros((xmax - xmin + 1, ymax - ymin + 1) + extra_dims, dtype=dtype)
    for fov, _img in stain_fovs.items():
        x, y = fov_x[fov] - xmin, fov_y[fov] - ymin
        if labels:
            _img[_img > 0] += last_label
            last_label = _img.max()
        img[x : x + _img.shape[1], y : y + _img.shape[0]] = np.fliplr(np.swapaxes(_img, 0, 1))
    return img


# def read_nanostring_agg(
#     path: str,
#     stain_path: Optional[str] = None,
#     binsize: int = 1,
#     gene_agg: Optional[Dict[str, Union[List[str], Callable[[str], bool]]]] = None,
#     prealigned: bool = False,
#     label_columns: Optional[Union[str, List[str]]] = None,
#     version: Literal["cosmx"] = "cosmx",
# ) -> AnnData:
#     lm.main_debug(f"Reading data from {path}.")
#     data = read_nanostring_as_dataframe(path, label_columns)
#     x_min, y_min = data["x"].min(), data["y"].min()
#     x, y = data["x"].values, data["y"].values
#     x_max, y_max = x.max(), y.max()
#     shape = (x_max + 1, y_max + 1)
#
#     # Read image and update x,y max if appropriate
#     layers = {}
#     if stain_path:
#         lm.main_debug(f"Reading stain image from {stain_path}.")
#         image = skimage.io.imread(stain_path)
#         if prealigned:
#             lm.main_warning(
#                 (
#                     "Assuming stain image was already aligned with the minimum x and y RNA coordinates. "
#                     "(prealinged=True)"
#                 )
#             )
#             image = np.pad(image, ((x_min, 0), (y_min, 0)))
#         x_max = max(x_max, image.shape[0] - 1)
#         y_max = max(y_max, image.shape[1] - 1)
#         shape = (x_max + 1, y_max + 1)
#         # Reshape image to match new x,y max
#         if image.shape != shape:
#             lm.main_warning(f"Padding stain image from {image.shape} to {shape} with zeros.")
#             image = np.pad(image, ((0, shape[0] - image.shape[0]), (0, shape[1] - image.shape[1])))
#         layers[SKM.STAIN_LAYER_KEY] = image


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
        resolution = VERSIONS[version]
        scale, scale_unit = resolution.scale, resolution.unit

    # Set uns
    SKM.init_adata_type(adata, SKM.ADATA_UMI_TYPE)
    SKM.init_uns_pp_namespace(adata)
    SKM.init_uns_spatial_namespace(adata)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_BINSIZE_KEY, binsize)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_KEY, scale)
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_SCALE_UNIT_KEY, scale_unit)
    return adata
