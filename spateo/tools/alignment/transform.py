from typing import List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData


def rigid_transform(
    coords: np.ndarray,
    coords_refA: np.ndarray,
    coords_refB: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal transformation based on the two sets of points and apply the transformation to other points.

    Args:
        coords: Coordinate matrix needed to be transformed.
        coords_refA: Referential coordinate matrix before transformation.
        coords_refB: Referential coordinate matrix after transformation.

    Returns:
        The coordinate matrix after transformation
    """
    # Check the spatial coordinates

    coords, coords_refA, coords_refB = coords.copy(), coords_refA.copy(), coords_refB.copy()
    assert (
        coords.shape[1] == coords_refA.shape[1] == coords_refA.shape[1]
    ), "The dimensions of the input coordinates must be uniform, 2D or 3D."
    coords_dim = coords.shape[1]
    if coords_dim == 2:
        coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]
        coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
        coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

    # Compute optimal transformation based on the two sets of points.
    coords_refA = coords_refA.T
    coords_refB = coords_refB.T

    centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
    centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

    Am = coords_refA - centroid_A
    Bm = coords_refB - centroid_B
    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # Apply the transformation to other points
    new_coords = (R @ coords.T) + t
    new_coords = np.asarray(new_coords.T)
    return new_coords[:, :2] if coords_dim == 2 else new_coords


def paste_transform(
    adata: AnnData,
    adata_ref: AnnData,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key: str = "models_align",
) -> AnnData:
    """
    Align the space coordinates of the new model with the transformation matrix obtained from PASTE.

    Args:
        adata: The anndata object that need to be aligned.
        adata_ref: The anndata object that have been aligned by PASTE.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinates.
        mapping_key: The key in `.uns` that corresponds to the alignment info from PASTE.

    Returns:
        adata: The anndata object that have been to be aligned.
    """

    assert mapping_key in adata_ref.uns_keys(), "`mapping_key` value is wrong."

    t = adata_ref.uns[mapping_key]["tY"]
    R = adata_ref.uns[mapping_key]["R"]

    adata_coords = adata.obsm[spatial_key].copy() - t
    adata.obsm[key_added] = R.dot(adata_coords.T).T
    return adata
