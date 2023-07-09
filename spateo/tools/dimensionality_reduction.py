"""Tools for dimensionality reduction, sourced from Dynamo: https://github.com/aristoteleo/dynamo-release/
dynamo/tools/dimension_reduction.py"""

from typing import List, Optional

import numpy as np
from anndata import AnnData

from ..logging import logger_manager as lm


def reduceDimension(
    adata: AnnData,
    X_data: np.ndarray = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    basis: Optional[str] = "pca",
    dims: Optional[List[int]] = None,
    n_pca_components: int = 30,
    n_components: int = 2,
    n_neighbors: int = 30,
    reduction_method: str = "umap",
    embedding_key: Optional[str] = None,
    enforce: bool = False,
    cores: int = 1,
    copy: bool = False,
    **kwargs,
) -> Optional[AnnData]:
    """Low dimension reduction projection of an AnnData object first with PCA, followed by non-linear
    dimension reduction methods.

    Args:
        adata: AnnData object
        X_data: The user supplied non-AnnDta data that will be used for dimension reduction directly. Defaults to None.
        genes: The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used. Defaults to None.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        basis: The space that will be used for clustering. Valid names includes, for example, `pca`, `umap`,
            `velocity_pca` (that is, you can use velocity for clustering), etc. Defaults to "pca".
        dims: The list of dimensions that will be selected for clustering. If `None`, all dimensions will be used.
            Defaults to None.
        n_pca_components: Number of input PCs (principle components) that will be used for further non-linear dimension
            reduction. If n_pca_components is larger than the existing #PC in adata.obsm['X_pca'] or input layer's
            corresponding pca space (layer_pca), pca will be rerun with n_pca_components PCs requested. Defaults to 30.
        n_components: The dimension of the space to embed into. Defaults to 2.
        n_neighbors: The number of nearest neighbors when constructing adjacency matrix. Defaults to 30.
        reduction_method: Non-linear dimension reduction method to further reduce dimension based on the top
            n_pca_components PCA components. Currently, tSNE (fitsne instead of traditional tSNE used) or umap are
            supported. Defaults to "umap".
        embedding_key: The str in .obsm that will be used as the key to save the reduced embedding space. By default it
            is None and embedding key is set as layer + reduction_method. If layer is None, it will be "X_neighbors".
            Defaults to None.
        enforce: Whether to re-perform dimension reduction even if there is reduced basis in the AnnData object.
            Defaults to False.
        cores: The number of cores used for calculation. Used only when tSNE reduction_method is used. Defaults to 1.
        copy: Whether to return a copy of the AnnData object or update the object in place. Defaults to False.
        kwargs: Other kwargs that will be passed to umap.UMAP. for umap, min_dist is a noticeable kwargs that would
            significantly influence the reduction result.

    Returns:
        adata: An updated AnnData object updated with reduced dimension data for data from different layers,
            returned if `copy` is true.
    """
    logger = lm.get_main_logger()

    if copy:
        adata = adata.copy()

    logger.info("Retrieving data for dimension reduction ...")

    if X_data is None:
        if layer is None:
            X_data = adata.X
        else:
            X_data = adata.layers[layer]
    if basis[:2] + reduction_method in adata.obsm_keys():
        has_basis = True
    else:
        has_basis = False

    if has_basis and not enforce:
        logger.warning(
            f"Adata already has basis {reduction_method}. Dimension reduction {reduction_method} will be skipped! \n"
            f"set enforce=True to re-perform dimension reduction."
        )

    if embedding_key is None:
        embedding_key = "X_" + reduction_method if layer is None else layer + "_" + reduction_method

    if not has_basis or enforce:
        logger.info(
            f"Performing {reduction_method.upper()} method using {basis} with n_pca_components ="
            f" {n_pca_components} ..."
        )

        if basis == "tsne":
            "FILLER"


# NOTES: dynamo-release/dynamo/tools/connectivity.py umap_conn_indices_dist_embedding for UMAP computation
