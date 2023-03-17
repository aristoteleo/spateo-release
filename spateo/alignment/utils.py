from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial import cKDTree

from spateo.logging import logger_manager as lm

####################
# Before Alignment #
####################


def _iteration(n: int, progress_name: str, verbose: bool = True):
    iteration = lm.progress_logger(range(n), progress_name=progress_name) if verbose else range(n)
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
        sampling = sample(
            arr=np.asarray(sampling_model.obs_names),
            n=n_sampling,
            method=sampling_method,
            X=sampling_model.obsm[spatial_key],
        )
        sampling_model = sampling_model[sampling, :]
        sampling_models.append(sampling_model)
    return sampling_models


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
