from typing import Dict, List, Literal, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from pyvista import PolyData
from scipy.sparse import csr_matrix, diags, issparse, lil_matrix, spmatrix
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.stats import norm

from ..configuration import SKM
from ..logging import logger_manager as lm


def rescaling(mat: Union[np.ndarray, spmatrix], new_shape: Union[List, Tuple]) -> Union[np.ndarray, spmatrix]:
    """This function rescale the resolution of the input matrix that represents a spatial domain. For example, if you
    want to decrease the resolution of a matrix by a factor of 2, the new_shape will be `mat.shape / 2`.

    Args:
        mat: The input matrix of the spatial domain (or an image).
        new_shape: The rescaled shape of the spatial domain, each dimension must be an factorial of the original
                    dimension.

    Returns:
        res: the spatial resolution rescaled matrix.
    """
    shape = (new_shape[0], mat.shape[0] // mat[0], new_shape[1], mat.shape[1] // mat[1])

    res = mat.reshape(shape).sum(-1).sum(1)
    return res


def get_mapper(smoothed=True):
    mapper = {
        "X_spliced": "M_s" if smoothed else "X_spliced",
        "X_unspliced": "M_u" if smoothed else "X_unspliced",
        "X_new": "M_n" if smoothed else "X_new",
        "X_old": "M_o" if smoothed else "X_old",
        "X_total": "M_t" if smoothed else "X_total",
        # "X_uu": "M_uu" if smoothed else "X_uu",
        # "X_ul": "M_ul" if smoothed else "X_ul",
        # "X_su": "M_su" if smoothed else "X_su",
        # "X_sl": "M_sl" if smoothed else "X_sl",
        # "X_protein": "M_p" if smoothed else "X_protein",
        "X": "X" if smoothed else "X",
    }
    return mapper


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())

    return dict1


def flatten(arr):
    if type(arr) == pd.core.series.Series:
        ret = arr.values.flatten()
    elif sp.issparse(arr):
        ret = arr.A.flatten()
    else:
        ret = arr.flatten()
    return ret


def compute_corr_ci(
    r: float,
    n: int,
    confidence: float = 95,
    decimals: int = 2,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
):
    """Parametric confidence intervals around a correlation coefficient

    Args:
        r: Correlation coefficient
        n: Length of x vector and y vector (the vectors used to compute the correlation)
        confidence: Confidence level, as a percent (so 95 = 95% confidence interval). Must be between 0 and 100.
        decimals: Number of rounded decimals
        alternative: Defines the alternative hypothesis, or tail for the correlation coefficient. Must be one of
            "two-sided" (default), "greater" or "less"

    Returns:
        ci: Confidence interval
    """
    assert alternative in [
        "two-sided",
        "greater",
        "less",
    ], "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."

    # r-to-z transform:
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)

    if alternative == "two-sided":
        critical_val = np.abs(norm.ppf((1 - confidence) / 2))
        ci_z = np.array([z - critical_val * se, z + critical_val * se])
    elif alternative == "greater":
        critical_val = norm.ppf(confidence)
        ci_z = np.array([z - critical_val * se, np.inf])
    else:
        critical_val = norm.ppf(confidence)
        ci_z = np.array([-np.inf, z + critical_val * se])

    # z-to-r transform:
    ci = np.tanh(ci_z)
    ci = np.round(ci, decimals)
    return ci


def calc_1nd_moment(X, W, normalize_W=True):
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W
        return W @ X, W
    else:
        return W @ X


def affine_transform(X, A, b):
    X = np.array(X)
    A = np.array(A)
    b = np.array(b)
    return (A @ X.T).T + b


def gen_rotation_2d(degree: float):
    from math import cos, radians, sin

    rad = radians(degree)
    R = [
        [cos(rad), -sin(rad)],
        [sin(rad), cos(rad)],
    ]
    return np.array(R)


def compute_smallest_distance(
    coords: np.ndarray, leaf_size: int = 40, sample_num=None, use_unique_coords=True
) -> float:
    """Compute and return smallest distance. A wrapper for sklearn API
    Parameters
    ----------
        coords:
            NxM matrix. N is the number of data points and M is the dimension of each point's feature.
        leaf_size : int, optional
            Leaf size parameter for building Kd-tree, by default 40.
        sample_num:
            The number of cells to be sampled.
        use_unique_coords:
            Whether to remove duplicate coordinates
    Returns
    -------
        min_dist: float
            the minimum distance between points
    """
    if len(coords.shape) != 2:
        raise ValueError("Coordinates should be a NxM array.")
    if use_unique_coords:
        # main_info("using unique coordinates for computing smallest distance")
        coords = [tuple(coord) for coord in coords]
        coords = np.array(list(set(coords)))
    # use cKDTree which is implmented in C++ and is much faster than KDTree
    kd_tree = cKDTree(coords, leafsize=leaf_size)
    if sample_num is None:
        sample_num = len(coords)
    N, _ = min(len(coords), sample_num), coords.shape[1]
    selected_estimation_indices = np.random.choice(len(coords), size=N, replace=False)

    # Note k=2 here because the nearest query is always a point itself.
    distances, _ = kd_tree.query(coords[selected_estimation_indices, :], k=2)
    min_dist = min(distances[:, 1])

    return min_dist


def polyhull(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> PolyData:
    """Create a PolyData object from the convex hull constructed with the input data points.

    scipy's ConvexHull to be 500X faster than using vtkDelaunay3D and vtkDataSetSurfaceFilter because you skip the
    expensive 3D tesselation of the volume.

    Args:
        x: x coordinates of the data points.
        y: y coordinates of the data points.
        z: z coordinates of the data points.

    Returns:
        poly: a PolyData object generated with the convex hull constructed based on the input data points.
    """
    hull = ConvexHull(np.column_stack((x, y, z)))
    faces = np.column_stack((3 * np.ones((len(hull.simplices), 1), dtype=np.int), hull.simplices)).flatten()
    poly = PolyData(hull.points, faces)
    return hull, poly


def in_hull(p: np.ndarray, hull: Tuple[Delaunay, np.ndarray]) -> np.ndarray:
    """Test if points in `p` are in `hull`

    Args:
        p: a `N x K` coordinates of `N` points in `K` dimensions
        hull: either a scipy.spatial.Delaunay object or the `MxK` array of the coordinates of `M` points in `K`
        dimensions for which Delaunay triangulation will be computed.

    Returns:
        res: A numpy array with boolean values indicating whether the input points is in the convex hull.
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    res = hull.find_simplex(p) >= 0
    return res


# ---------------------------------------------------------------------------------------------------
# For filtering dataframe by written instructions
# ---------------------------------------------------------------------------------------------------
def parse_instruction(instruction: str, axis_map: Optional[Dict[str, str]] = None):
    """
    Parses a single filtering instruction and returns the equivalent pandas query string.

    Args:
        instruction: Filtering condition, in a form similar to the following: "x less than 950 and z less than or
            equal to 350". This is equivalent to ((x < 950) & (z <= 350)). Here, x is the name of one dataframe column
            and z is the name of another. For negation, use "not (x less than 950)".
        axis_map: In the case that an alias can be used for the dataframe column names (e.g. "x-axis" -> "x"),
            this dictionary maps these optional aliases to column names.

    Returns:
        query: The equivalent pandas query string.
    """
    # Replace the axis names with the corresponding column names
    for axis, col in axis_map.items():
        instruction = instruction.replace(axis, col)

    # Replace the human-readable operators with their Python equivalents
    instruction = instruction.replace("less than or equal to", "<=")
    instruction = instruction.replace("less than", "<")
    instruction = instruction.replace("greater than or equal to", ">=")
    instruction = instruction.replace("greater than", ">")
    instruction = instruction.replace("equal to", "==")
    instruction = instruction.replace("not (", "~(")

    return instruction


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def filter_adata_spatial(
    adata: AnnData, coords_key: str, instructions: List[str], col_alias_map: Optional[Dict[str, str]] = None
):
    """Filters the AnnData object by spatial coordinates based on the provided instructions list, to be executed
    sequentially.

    Args:
        adata: AnnData object containing spatial coordinates in .obsm
        coords_key: Key in .obsm containing spatial coordinates
        instructions: List of filtering instructions, in a form similar to the following: "x less than 950 and z less
            than or equal to 350". This is equivalent to ((x < 950) & (z <= 350)). Here, x is the name of one dataframe
            column and z is the name of another. For negation, use "not (x less than 950)".
        col_alias_dict: In the case that an alias can be used for the dataframe column names (e.g. "x-axis" is used
            to refer to the dataframe column "x"), this dictionary maps these optional aliases to column names.

    Returns:
        adata: Filtered AnnData object
    """
    logger = lm.get_main_logger()

    # Default alias map will map "x" -> "points_x", "y" -> "points_y", etc.
    if col_alias_map is None:
        col_alias_map = {"x": "points_x", "y": "points_y", "z": "points_z"}

    coordinates = adata.obsm[coords_key]
    if coordinates.shape[1] == 2:
        df = pd.DataFrame(coordinates, index=adata.obs_names, columns=["points_x", "points_y"])
    elif coordinates.shape[1] == 3:
        df = pd.DataFrame(coordinates, index=adata.obs_names, columns=["points_x", "points_y", "points_z"])
    else:
        raise ValueError(f"Coordinates must be 2D or 3D. Given shape: {coordinates.shape}.")

    # Process each instruction:
    for instruction in instructions:
        query = parse_instruction(instruction, col_alias_map)
        df = df.query(query)
    logger.info(f"Filtered {len(adata)} cells to {len(df)} cells.")

    # Filter AnnData object:
    adata = adata[df.index, :].copy()
    return adata


# ---------------------------------------------------------------------------------------------------
# For creating arbitrary axes between two existing axes
# ---------------------------------------------------------------------------------------------------
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def create_new_coordinate(
    adata: anndata.AnnData, position_key: str = "spatial", plane: Literal["xy", "yz", "xz", "-xy", "-yz", "-xz"] = "xy"
):
    """Projects points from an AnnData object onto a specified plane and direction, calculate the distances along this
    projection, and add the results to the AnnData object.

    Args:
        adata: AnnData object containing spatial coordinates in .obsm
        position_key: Key in .obsm containing spatial coordinates. Defaults to "spatial".
        plane: Plane to project points onto. Must be one of "xy", "yz", "xz", "-xy", "-yz", "-xz". The "-" prefix
            indicates that the direction along the first axis is reversed (i.e. instead of starting from the minimum
            value, it starts from the maximum value). Defaults to "xy".

    Returns:
        adata: AnnData object with new column added to .obs
    """
    if "z" in plane and adata.obsm[position_key].shape[1] < 3:
        raise ValueError("Cannot project onto z-axis if there are only 2 spatial dimensions.")

    if position_key in adata.obsm.keys():
        if adata.obsm[position_key].shape[1] == 2:
            pos_df = pd.DataFrame(adata.obsm[position_key], index=adata.obs_names, columns=["X", "Y"])
        else:
            pos_df = pd.DataFrame(adata.obsm[position_key], index=adata.obs_names, columns=["X", "Y", "Z"])

        # Extracting the relevant columns based on the plane
        if plane in ["xy", "-xy"]:
            cols = ["X", "Y"]
        elif plane in ["yz", "-yz"]:
            cols = ["Y", "Z"]
        elif plane in ["xz", "-xz"]:
            cols = ["X", "Z"]
        else:
            raise ValueError("Invalid coord_column")

        # Projection and calculation of distance
        if plane in ["xy", "yz", "xz"]:  # Positive planes
            min_point = pos_df[cols].min()
            max_point = pos_df[cols].max()
        else:  # Negative planes
            min_point = pos_df[cols].copy()
            max_point = pos_df[cols].copy()
            min_point[cols[1]] = pos_df[cols[1]].max()
            max_point[cols[1]] = pos_df[cols[1]].min()
        reverse = "-" in plane
        adata.obs[f"{plane} Coordinate"] = pos_df.apply(
            lambda row: project_and_calculate_distance(row, cols, min_point, max_point, reverse), axis=1
        )

        return adata


def project_and_calculate_distance(
    row: pd.Series, cols: list, min_point: pd.Series, max_point: pd.Series, reverse: bool = False
):
    """Project a point onto a line with a slope of either 1 or -1, and calculate the distance along the new line for
    the projected point to establish its coordinate.

    Args:
        row: A row from the dataframe that contains the spatial coordinates
        cols: The list of two columns that define the plane, e.g. ["X", "Y"]
        min_point: The minimum point that defines the "start" of the new coordinate axis
        max_point: The maximum point that defines the "end" of the new coordinate axis
        reverse: Flag that indicates whether the projection starts from the minimum along the x-axis or the maximum
            along the x-axis (or y-axis or z-axis, depending on which axis is used as the reference).

    Returns:
        distance: The distance along the new coordinate axis for the projected point
    """
    if reverse:
        # For -xy, -yz, -xz (slope -1)
        x_proj = (row[cols[0]] - row[cols[1]]) / 2
        y_proj = -x_proj
        reference_point = max_point
    else:
        # For xy, yz, xz (slope 1)
        x_proj = (row[cols[0]] + row[cols[1]]) / 2
        y_proj = x_proj
        reference_point = min_point

    distance = np.sqrt((x_proj - reference_point[cols[0]]) ** 2 + (y_proj - reference_point[cols[1]]) ** 2)
    return distance
