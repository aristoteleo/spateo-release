from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial import cKDTree

from spateo.logging import logger_manager as lm

####################
# Before Alignment #
####################


def _iteration(n: int, progress_name: str, verbose: bool = True, indent_level=1):
    iteration = (
        lm.progress_logger(range(n), progress_name=progress_name, indent_level=indent_level) if verbose else range(n)
    )
    return iteration


def downsampling(
    models: Union[List[AnnData], AnnData],
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    spatial_key: str = "spatial",
) -> Union[List[AnnData], AnnData]:
    from dynamo.tools.sampling import sample

    models = models if isinstance(models, list) else [models]
    sampling_models = []
    for m in models:
        sampling_model = m.copy()
        if n_sampling > sampling_model.shape[0]:
            n_sampling = sampling_model.shape[0]
        sampling = sample(
            arr=np.asarray(sampling_model.obs_names),
            n=n_sampling,
            method=sampling_method,
            X=sampling_model.obsm[spatial_key],
        )
        sampling_model = sampling_model[sampling, :]
        sampling_models.append(sampling_model)
    return sampling_models


## generate label transfer prior
def generate_label_transfer_prior(cat1, cat2, positive_pairs=None, negative_pairs=None):
    label_transfer_prior = dict()
    if positive_pairs is None:
        positive_pairs = []
    if negative_pairs is None:
        negative_pairs = []
    # let same annotation to have a high value
    if (len(positive_pairs) == 0) and (len(negative_pairs) == 0):
        for c in cat1:
            if c in cat2:
                positive_pairs.append({"left": [c], "right": [c], "value": 10})
    for c2 in cat2:
        cur_transfer_prior = dict()
        for c1 in cat1:
            cur_transfer_prior[c1] = 1
        label_transfer_prior[c2] = cur_transfer_prior
    for p in positive_pairs:
        for l in p["left"]:
            for r in p["right"]:
                label_transfer_prior[r][l] = p["value"]
        # label_transfer_prior[p[1]][p[0]] = p[2]
    for p in negative_pairs:
        for l in p["left"]:
            for r in p["right"]:
                label_transfer_prior[r][l] = p["value"]
        # label_transfer_prior[p[1]][p[0]] = p[2]
    norm_label_transfer_prior = dict()
    for c2 in cat2:
        norm_c = np.array([label_transfer_prior[c2][c1] for c1 in cat1]).sum()
        cur_transfer_prior = dict()
        for c1 in cat1:
            cur_transfer_prior[c1] = label_transfer_prior[c2][c1] / norm_c
        norm_label_transfer_prior[c2] = cur_transfer_prior
    return norm_label_transfer_prior


# group pca
def group_pca(
    adatas: List[ad.AnnData],
    batch_key: str = "batch",
    pca_key: str = "X_pca",
    use_hvg: bool = True,
    hvg_key: str = "highly_variable",
    **args,
) -> None:
    """
    Perform PCA on a concatenated set of AnnData objects and store the results back in each individual AnnData.

    Parameters:
    ----------
    adatas : List[AnnData]
        A list of AnnData objects to be concatenated and processed.
    batch_key : str, optional
        The key to distinguish different batches in the concatenated AnnData object (default is 'batch').
    pca_key : str, optional
        The key under which to store the PCA results in each AnnData's `.obsm` attribute (default is 'X_pca').
    use_hvg : bool, optional
        Whether to perform PCA using only highly variable genes (default is True).
    hvg_key : str, optional
        The key under which highly variable genes are marked in `.var` (default is 'highly_variable').
    **args
        Additional arguments to pass to `sc.tl.pca`.

    Raises:
    ------
    ValueError:
        If the specified batch_key already exists in any of the input AnnData objects.
        If no highly variable genes are found when use_hvg is True.
    """

    import scanpy as sc

    # Check if batch_key already exists in any of the adatas
    for i, adata in enumerate(adatas):
        if batch_key in adata.obs.columns:
            raise ValueError(
                f"batch_key '{batch_key}' already exists in adata.obs for dataset {i}. Please choose a different key."
            )

    # Concatenate all AnnData objects, using batch_key to differentiate them
    adata_pca = ad.concat(adatas, label=batch_key)

    # Identify and use highly variable genes for PCA if requested
    if use_hvg:
        sc.pp.highly_variable_genes(adata_pca, batch_key=batch_key)
        if not adata_pca.var[hvg_key].any():
            raise ValueError(
                "No highly variable genes were found. Please check your data or parameters for highly variable gene selection."
            )

        # Perform PCA using only highly variable genes
        sc.tl.pca(adata_pca, **args, use_highly_variable=True)
    else:
        # Perform PCA without restricting to highly variable genes
        sc.tl.pca(adata_pca, **args)

    # Split the PCA results back into the original AnnData objects
    for i in range(len(adatas)):
        adatas[i].obsm[pca_key] = adata_pca[adata_pca.obs[batch_key] == str(i)].obsm["X_pca"].copy()


###################
# After Alignment #
###################


def get_optimal_mapping_relationship(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    keep_all: bool = False,
):
    X_max_index = np.argwhere((pi.T == pi.T.max(axis=0)).T)
    Y_max_index = np.argwhere(pi == pi.max(axis=0))
    if not keep_all:
        values, counts = np.unique(X_max_index[:, 0], return_counts=True)
        x_index_unique, x_index_repeat = values[counts == 1], values[counts != 1]
        X_max_index_unique = X_max_index[np.isin(X_max_index[:, 0], x_index_unique)]

        for i in x_index_repeat:
            i_max_index = X_max_index[X_max_index[:, 0] == i]
            i_kdtree = cKDTree(Y[i_max_index[:, 1]])
            _, ii = i_kdtree.query(X[i], k=1)
            X_max_index_unique = np.concatenate([X_max_index_unique, i_max_index[ii].reshape(1, 2)], axis=0)

        values, counts = np.unique(Y_max_index[:, 1], return_counts=True)
        y_index_unique, y_index_repeat = values[counts == 1], values[counts != 1]
        Y_max_index_unique = Y_max_index[np.isin(Y_max_index[:, 1], y_index_unique)]

        for i in y_index_repeat:
            i_max_index = Y_max_index[Y_max_index[:, 1] == i]
            i_kdtree = cKDTree(X[i_max_index[:, 0]])
            _, ii = i_kdtree.query(Y[i], k=1)
            Y_max_index_unique = np.concatenate([Y_max_index_unique, i_max_index[ii].reshape(1, 2)], axis=0)

        X_max_index = X_max_index_unique.copy()
        Y_max_index = Y_max_index_unique.copy()

    X_pi_value = pi[X_max_index[:, 0], X_max_index[:, 1]].reshape(-1, 1)
    Y_pi_value = pi[Y_max_index[:, 0], Y_max_index[:, 1]].reshape(-1, 1)
    return X_max_index, X_pi_value, Y_max_index, Y_pi_value


def mapping_aligned_coords(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    keep_all: bool = False,
) -> Tuple[dict, dict]:
    """
    Optimal mapping coordinates between X and Y.

    Args:
        X: Aligned spatial coordinates.
        Y: Aligned spatial coordinates.
        pi: Mapping between the two layers output by PASTE.
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.

    Returns:
        Two dicts of mapping_X, mapping_Y, pi_index, pi_value.
            mapping_X is X coordinates aligned with Y coordinates.
            mapping_Y is the Y coordinate aligned with X coordinates.
            pi_index is index between optimal mapping points in the pi matrix.
            pi_value is the value of optimal mapping points.
    """

    X = X.copy()
    Y = Y.copy()
    pi = pi.copy()

    # Obtain the optimal mapping between points
    (
        X_max_index,
        X_pi_value,
        Y_max_index,
        Y_pi_value,
    ) = get_optimal_mapping_relationship(X=X, Y=Y, pi=pi, keep_all=keep_all)

    mappings = []
    for max_index, pi_value, subset in zip(
        [X_max_index, Y_max_index], [X_pi_value, Y_pi_value], ["index_x", "index_y"]
    ):
        mapping_data = pd.DataFrame(
            np.concatenate([max_index, pi_value], axis=1),
            columns=["index_x", "index_y", "pi_value"],
        ).astype(
            dtype={
                "index_x": np.int32,
                "index_y": np.int32,
                "pi_value": np.float64,
            }
        )
        mapping_data.sort_values(by=[subset, "pi_value"], ascending=[True, False], inplace=True)
        mapping_data.drop_duplicates(subset=[subset], keep="first", inplace=True)
        mappings.append(
            {
                "mapping_X": X[mapping_data["index_x"].values],
                "mapping_Y": Y[mapping_data["index_y"].values],
                "pi_index": mapping_data[["index_x", "index_y"]].values,
                "pi_value": mapping_data["pi_value"].values,
            }
        )

    return mappings[0], mappings[1]


def mapping_center_coords(modelA: AnnData, modelB: AnnData, center_key: str) -> dict:
    """
    Optimal mapping coordinates between X and Y based on intermediate coordinates.

    Args:
        modelA: modelA aligned with center model.
        modelB: modelB aligned with center model.
        center_key: The key in ``.uns`` that corresponds to the alignment info between modelA/modelB and center model.

    Returns:
        A dict of raw_X, raw_Y, mapping_X, mapping_Y, pi_value.
            raw_X is the raw X coordinates.
            raw_Y is the raw Y coordinates.
            mapping_X is the Y coordinates aligned with X coordinates.
            mapping_Y is the X coordinates aligned with Y coordinates.
            pi_value is the value of optimal mapping points.
    """

    modelA_dict = modelA.uns[center_key].copy()
    modelB_dict = modelB.uns[center_key].copy()

    mapping_X_cols = [f"mapping_X_{i}" for i in range(modelA_dict["mapping_Y"].shape[1])]
    raw_X_cols = [f"raw_X_{i}" for i in range(modelA_dict["raw_Y"].shape[1])]
    mapping_Y_cols = [f"mapping_Y_{i}" for i in range(modelB_dict["mapping_Y"].shape[1])]
    raw_Y_cols = [f"raw_Y_{i}" for i in range(modelB_dict["raw_Y"].shape[1])]

    X_cols = mapping_X_cols.copy() + raw_X_cols.copy() + ["mid"]
    X_data = pd.DataFrame(
        np.concatenate(
            [
                modelA_dict["raw_Y"],
                modelA_dict["mapping_Y"],
                modelA_dict["pi_index"][:, [0]],
            ],
            axis=1,
        ),
        columns=X_cols,
    )
    X_data["pi_value_X"] = modelA_dict["pi_value"].astype(np.float64)

    Y_cols = mapping_Y_cols.copy() + raw_Y_cols.copy() + ["mid"]
    Y_data = pd.DataFrame(
        np.concatenate(
            [
                modelB_dict["raw_Y"],
                modelB_dict["mapping_Y"],
                modelB_dict["pi_index"][:, [0]],
            ],
            axis=1,
        ),
        columns=Y_cols,
    )
    Y_data["pi_value_Y"] = modelB_dict["pi_value"].astype(np.float64)

    mapping_data = pd.merge(Y_data, X_data, on=["mid"], how="inner")
    mapping_data["pi_value"] = mapping_data[["pi_value_X"]].values * mapping_data[["pi_value_Y"]].values

    return {
        "raw_X": mapping_data[raw_X_cols].values,
        "raw_Y": mapping_data[raw_Y_cols].values,
        "mapping_X": mapping_data[mapping_X_cols].values,
        "mapping_Y": mapping_data[mapping_Y_cols].values,
        "pi_value": mapping_data["pi_value"].astype(np.float64).values,
    }


def get_labels_based_on_coords(
    model: AnnData,
    coords: np.ndarray,
    labels_key: Union[str, List[str]],
    spatial_key: str = "align_spatial",
) -> pd.DataFrame:
    """Obtain the label information in anndata.obs[key] corresponding to the coords."""

    key = [labels_key] if isinstance(labels_key, str) else labels_key

    cols = ["x", "y", "z"] if coords.shape[1] == 3 else ["x", "y"]
    X_data = pd.DataFrame(model.obsm[spatial_key], columns=cols)
    X_data[key] = model.obs[key].values
    X_data.drop_duplicates(inplace=True, keep="first")

    Y_data = pd.DataFrame(coords.copy(), columns=cols)
    Y_data["map_index"] = Y_data.index
    merge_data = pd.merge(Y_data, X_data, on=cols, how="inner")
    return merge_data


#########################
# Some helper functions #
#########################


def solve_RT_by_correspondence(
    X: np.ndarray, Y: np.ndarray, return_scale: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
    """
    Solve for the rotation matrix R and translation vector t that best align the points in X to the points in Y.

    Args:
        X (np.ndarray): Source points, shape (N, D).
        Y (np.ndarray): Target points, shape (N, D).
        return_scale (bool, optional): Whether to return the scale factor. Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
        If return_scale is False, returns the rotation matrix R and translation vector t.
        If return_scale is True, also returns the scale factor s.
    """

    D = X.shape[1]
    N = X.shape[0]

    # Calculate centroids of X and Y
    tX = np.mean(X, axis=0)
    tY = np.mean(Y, axis=0)

    # Demean the points
    X_demean = X - tX
    Y_demean = Y - tY

    # Compute the covariance matrix
    H = np.dot(Y_demean.T, X_demean)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure the rotation matrix is proper
    # if np.linalg.det(R) < 0:
    #     Vt[-1, :] *= -1
    #     R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t = tX - np.dot(tY, R.T)

    if return_scale:
        # Compute the scale factor
        s = np.trace(np.dot(X_demean.T, X_demean) - np.dot(R.T, np.dot(Y_demean.T, X_demean))) / np.trace(
            np.dot(Y_demean.T, Y_demean)
        )
        return R, t, s
    else:
        return R, t


##############
# Simulation #
##############


def split_slice(
    adata,
    spatial_key,
    split_num=5,
    axis=2,
):
    spatial_points = adata.obsm[spatial_key]
    N = spatial_points.shape[0]
    sorted_points = np.argsort(spatial_points[:, axis])
    points_per_segment = len(sorted_points) // split_num
    split_adata = []
    for slice_id, i in enumerate(range(0, N, points_per_segment)):
        sorted_adata = adata[sorted_points[i : i + points_per_segment], :].copy()
        sorted_adata.obs["slice"] = slice_id
        split_adata.append(sorted_adata)
    return split_adata[:split_num]


# def tps_deformation(
#     adata,
#     spatial_key,
#     key_added,
#     grid_num=2,
#     tps_noise_scale=25,
#     add_corner_points=True,
#     alpha=0.1,
#     inplace=True,
# ):
#     from tps import ThinPlateSpline

#     spatial = adata.obsm[spatial_key]
#     # get min max
#     x_min, x_max = np.min(spatial[:, 0]), np.max(spatial[:, 0])
#     y_min, y_max = np.min(spatial[:, 1]), np.max(spatial[:, 1])

#     # define the length of grid
#     grid_size_x = (x_max - x_min) / grid_num
#     grid_size_y = (y_max - y_min) / grid_num
#     # generate the grid
#     x_grid = np.linspace(x_min, x_max, grid_num + 1)[:-1] + grid_size_x / 2
#     y_grid = np.linspace(y_min, y_max, grid_num + 1)[:-1] + grid_size_y / 2
#     xx, yy = np.meshgrid(x_grid, y_grid)
#     # generate control points
#     src_points = []
#     dst_points = []
#     for i in range(xx.shape[0]):
#         for j in range(xx.shape[1]):
#             x_center, y_center = xx[i, j], yy[i, j]
#             x = x_center
#             y = y_center
#             src_points.append(np.column_stack([x, y]))
#             dst_points.append(
#                 src_points[-1] + np.random.normal(scale=(grid_size_x + grid_size_y) * tps_noise_scale / 2, size=(1, 2))
#             )
#     src_points = np.concatenate(src_points, axis=0)
#     dst_points = np.concatenate(dst_points, axis=0)
#     if add_corner_points:
#         src_points = np.concatenate(
#             [np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]), src_points], 0
#         )
#         dst_points = np.concatenate(
#             [np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]), dst_points], 0
#         )
#     # calculate the TPS deformation
#     tps = ThinPlateSpline(alpha=alpha)  # Regularization
#     tps.fit(src_points, dst_points)
#     # perform tps deformation
#     tps_spatial = tps.transform(spatial)
#     if inplace:
#         adata.obsm[key_added] = tps_spatial
#         return lambda x: tps.transform(x)
#     else:
#         adata_tps = adata.copy()
#         adata_tps.obsm[key_added] = tps_spatial
#         return adata_tps, lambda x: tps.transform(x)


def tps_deformation(
    adata,
    spatial_key,
    key_added,
    grid_num=2,
    tps_noise_scale=25,
    add_corner_points=True,
    alpha=0.1,
    inplace=True,
):
    from tps import ThinPlateSpline

    spatial = adata.obsm[spatial_key]
    # get min max
    x_min, x_max = np.min(spatial[:, 0]), np.max(spatial[:, 0])
    y_min, y_max = np.min(spatial[:, 1]), np.max(spatial[:, 1])

    # define the length of grid
    grid_size_x = (x_max - x_min) / grid_num
    grid_size_y = (y_max - y_min) / grid_num
    # generate the grid
    x_grid = np.linspace(x_min, x_max, grid_num + 1)[:-1] + grid_size_x / 2
    y_grid = np.linspace(y_min, y_max, grid_num + 1)[:-1] + grid_size_y / 2
    xx, yy = np.meshgrid(x_grid, y_grid)
    # generate control points
    src_points = []
    dst_points = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x_center, y_center = xx[i, j], yy[i, j]
            x = x_center
            y = y_center
            src_points.append(np.column_stack([x, y]))
            dst_points.append(
                src_points[-1] + np.random.normal(scale=(grid_size_x + grid_size_y) * tps_noise_scale / 2, size=(1, 2))
            )
    src_points = np.concatenate(src_points, axis=0)
    dst_points = np.concatenate(dst_points, axis=0)
    if add_corner_points:
        src_points = np.concatenate(
            [np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]), src_points], 0
        )
        dst_points = np.concatenate(
            [np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]), dst_points], 0
        )
    # calculate the TPS deformation
    tps = ThinPlateSpline(alpha=alpha)  # Regularization
    tps.fit(src_points, dst_points)
    # perform tps deformation
    tps_spatial = tps.transform(spatial)
    if inplace:
        adata.obsm[key_added] = tps_spatial
        return lambda x: tps.transform(x)
    else:
        adata_tps = adata.copy()
        adata_tps.obsm[key_added] = tps_spatial
        return adata_tps, lambda x: tps.transform(x)
