"""Tools for dimensionality reduction, adapted from Dynamo: https://github.com/aristoteleo/dynamo-release/
dynamo/tools/dimension_reduction.py"""
import warnings
from copy import deepcopy
from typing import Callable, List, Literal, Optional, Tuple, Union

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scipy
from anndata import AnnData
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.utils.sparsefuncs import mean_variance_axis
from umap import UMAP
from umap.umap_ import (
    find_ab_params,
    fuzzy_simplicial_set,
    nearest_neighbors,
    simplicial_set_embedding,
)

from ..configuration import SKM
from ..logging import logger_manager as lm
from ..preprocessing.transform import log1p
from ..tools.cluster.leiden import calculate_leiden_partition
from ..tools.find_neighbors import adj_to_knn


# ---------------------------------------------------------------------------------------------------
# Master dimensionality reduction function
# ---------------------------------------------------------------------------------------------------
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def perform_dimensionality_reduction(
    adata: AnnData,
    X_data: np.ndarray = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    basis: Optional[str] = "pca",
    dims: Optional[List[int]] = None,
    n_pca_components: int = 30,
    n_components: int = 2,
    n_neighbors: int = 30,
    reduction_method: Literal["pca", "tsne", "umap"] = "umap",
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
        basis: The space that will be used for clustering. If None, will use the data itself, without any other
            processing. Can be None, "pca", or any other key based on PCA.
        dims: The list of dimensions that will be selected for clustering. If `None`, all dimensions will be used.
            Defaults to None.
        n_pca_components: Number of input PCs (principle components) that will be used for further non-linear dimension
            reduction. If n_pca_components is larger than the existing #PC in adata.obsm['X_pca'] or input layer's
            corresponding pca space (layer_pca), pca will be rerun with n_pca_components PCs requested. Defaults to 30.
        n_components: The dimension of the space to embed into. Defaults to 2.
        n_neighbors: The number of nearest neighbors when constructing adjacency matrix. Defaults to 30.
        reduction_method: Non-linear dimension reduction method to further reduce dimension based on the top
            n_pca_components PCA components. Currently, tSNE (fitsne instead of traditional tSNE used), umap or pca are
            supported. If "pca", will search for/compute the PCA representation and then stop. If "tsne" or
            "umap", will compute the PCA representation (or not, if 'basis' is None) and use this to then compute the
            UMAP embedding. Defaults to "umap".
        embedding_key: The str in .obsm that will be used as the key to save the reduced embedding space. By default it
            is None and embedding key is set as layer + reduction_method. If layer is None, it will be "X_neighbors".
            Defaults to None.
        enforce: Whether to re-perform dimension reduction even if there is reduced basis in the AnnData object.
            Defaults to False.
        cores: The number of cores used for calculation. Used only when tSNE reduction_method is used. Defaults to 1.
        copy: Whether to return a copy of the AnnData object or update the object in place. Defaults to False.
        kwargs: Other kwargs that will be passed to umap.UMAP. One notable variable is "densmap",
            for a density-preserving dimensionality reduction. There is also "min_dist", which provides the minimum
            distance apart that points are allowed to be in the low dimensional representation.

    Returns:
        adata: An updated AnnData object updated with reduced dimension data for data from different layers,
            only if `copy` is true.
    """
    logger = lm.get_main_logger()

    if copy:
        adata = adata.copy()

    logger.info("Retrieving data for dimension reduction ...")

    # Prepare dimensionality reduction (i.e. compute basis if not already existing, etc.)
    if X_data is None:
        if genes is not None:
            genes = adata.var_names.intersection(genes).to_list()
            if len(genes) == 0:
                raise ValueError("No genes from your genes list appear in your AnnData object.")

        if basis is None:
            if layer is None:
                if genes is not None:
                    X_data = adata[:, genes].X
                else:
                    X_data = adata.X
            else:
                if genes is not None:
                    X_data = adata[:, genes].layers[layer]
                else:
                    X_data = adata.layers[layer]

            X_data = log1p(X_data)
        else:
            pca_key = "X_pca" if layer is None else layer + "_pca"
            n_pca_components = max(max(dims), n_pca_components) if dims is not None else n_pca_components

            if basis not in adata.obsm.keys():
                if (
                    genes is not None
                    or pca_key not in adata.obsm.keys()
                    or adata.obsm[pca_key].shape[1] < n_pca_components
                ):
                    if layer is None:
                        if genes is not None:
                            CM = adata[:, genes].X
                        else:
                            CM = adata.X
                    else:
                        if genes is not None:
                            CM = adata[:, genes].layers[layer]
                        else:
                            CM = adata.layers[layer]
                        CM = log1p(CM)

                    cm_genesums = CM.sum(axis=0)
                    valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                    valid_ind = np.array(valid_ind).flatten()
                    # Valid genes used for dimensionality reduction:
                    adata.uns["pca_valid_ind"] = valid_ind
                    CM = CM[:, valid_ind]
                    logger.info(f"Computing PCA with {n_pca_components} PCs and storing in {pca_key}...")
                    adata, fit, _ = pca(adata, CM, n_pca_components=n_pca_components, pca_key=pca_key, return_all=True)

            if pca_key in adata.obsm.keys():
                X_data = adata.obsm[pca_key]
            else:
                if genes is not None:
                    CM = adata[:, genes].layers[layer]
                else:
                    CM = adata.layers[layer]
                CM = log1p(CM)

                cm_genesums = CM.sum(axis=0)
                valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                valid_ind = np.array(valid_ind).flatten()
                # Valid genes used for dimensionality reduction:
                adata.uns["pca_valid_ind"] = valid_ind
                CM = CM[:, valid_ind]
                logger.info(f"Computing PCA with {n_pca_components} PCs and storing in {pca_key}...")
                adata, fit, _ = pca(adata, CM, n_pca_components=n_pca_components, pca_key=pca_key, return_all=True)
                X_data = adata.obsm[pca_key]

        if dims is not None:
            X_data = X_data[:, dims]

    if reduction_method != "pca":
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

            if reduction_method == "tsne":
                try:
                    from fitsne import FItSNE
                except ImportError:
                    raise ImportError(
                        "Please first install fitsne to perform accelerated tSNE method. Install instruction is "
                        "provided here: https://pypi.org/project/fitsne/"
                    )

                X_dim = FItSNE(X_data, nthreads=cores)
                logger.info_insert_adata(embedding_key, adata_attr="obsm")
                adata.obsm[embedding_key] = X_dim

            elif reduction_method == "umap":
                # Default UMAP parameters
                umap_kwargs = {
                    "n_components": n_components,
                    "metric": "euclidean",
                    "min_dist": 0.5,
                    "spread": 1.0,
                    "max_iter": None,
                    "alpha": 1.0,
                    "gamma": 1.0,
                    "negative_sample_rate": 5,
                    "init_pos": "spectral",
                    "random_state": 0,
                    "densmap": False,
                    "dens_lambda": 2.0,
                    "dens_frac": 0.3,
                    "dens_var_shift": 0.1,
                    "output_dens": False,
                    "verbose": False,
                }
                umap_kwargs.update(kwargs)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    (
                        mapper,
                        graph,
                        knn_indices,
                        knn_dists,
                        X_dim,
                    ) = umap_conn_indices_dist_embedding(X_data, n_neighbors, **umap_kwargs)

                logger.info_insert_adata(embedding_key, adata_attr="obsm")
                adata.obsm[embedding_key] = X_dim
                adata.uns["umap_fit"] = {
                    "fit": mapper,
                    "n_pca_components": n_pca_components,
                }

            else:
                raise Exception(f"Reduction_method {reduction_method} is not supported.")

    logger.info("Finished computing dimensionality reduction.")
    if copy:
        return adata
    return None


# ---------------------------------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------------------------------
def umap_conn_indices_dist_embedding(
    X: np.ndarray,
    n_neighbors: int = 30,
    n_components: int = 2,
    metric: Union[str, Callable] = "euclidean",
    min_dist: float = 0.1,
    spread: float = 1.0,
    max_iter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: float = 5,
    init_pos: Union[Literal["spectral", "random"], np.ndarray] = "spectral",
    random_state: Union[int, np.random.RandomState, None] = 0,
    densmap: bool = False,
    dens_lambda: float = 2.0,
    dens_frac: float = 0.3,
    dens_var_shift: float = 0.1,
    output_dens: bool = False,
    return_mapper: bool = True,
    **umap_kwargs,
) -> Union[
    Tuple[UMAP, scipy.sparse.coo_matrix, np.ndarray, np.ndarray, np.ndarray],
    Tuple[scipy.sparse.coo_matrix, np.ndarray, np.ndarray, np.ndarray],
]:
    """Compute connectivity graph, matrices for kNN neighbor indices, distance matrix and low dimension embedding with
    UMAP.

    From Dynamo: https://github.com/aristoteleo/dynamo-release/, which in turn derives this code from umap-learn:
    (https://github.com/lmcinnes/umap/blob/97d33f57459de796774ab2d7fcf73c639835676d/umap/umap_.py).

    Args:
        X: The input array for which to perform UMAP
        n_neighbors: The number of nearest neighbors to compute for each sample in `X`. Defaults to 30.
        n_components: The dimension of the space to embed into. Defaults to 2.
        metric: The distance metric to use to find neighbors. Defaults to "euclidean".
        min_dist: The effective minimum distance between embedded points. Smaller values will result in a more
            clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger
            values will result on a more even dispersal of points. The value should be set relative to the `spread`
            value, which determines the scale at which embedded points will be spread out. Defaults to 0.1.
        spread: The effective scale of embedded points. In combination with min_dist this determines how
            clustered/clumped the embedded points are. Defaults to 1.0.
        max_iter: The number of training epochs to be used in optimizing the low dimensional embedding. Larger values
            result in more accurate embeddings. If None is specified a value will be selected based on the size of the
            input dataset (200 for large datasets, 500 for small). This argument was refactored from n_epochs from
            UMAP-learn to account for recent API changes in UMAP-learn 0.5.2. Defaults to None.
        alpha: Initial learning rate for the SGD. Defaults to 1.0.
        gamma: Weight to apply to negative samples. Values higher than one will result in greater weight being given to
            negative samples. Defaults to 1.0.
        negative_sample_rate: The number of negative samples to select per positive sample in the optimization process.
            Increasing this value will result in greater repulsive force being applied, greater optimization cost, but
            slightly more accuracy. The number of negative edge/1-simplex samples to use per positive edge/1-simplex
            sample in optimizing the low dimensional embedding. Defaults to 5.
        init_pos: The method to initialize the low dimensional embedding. Where:
            "spectral": use a spectral embedding of the fuzzy 1-skeleton.
            "random": assign initial embedding positions at random.
            Or an np.ndarray to define the initial position.
            Defaults to "spectral".
        random_state: The method to generate random numbers. If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random number generator; If None, the random number
            generator is the RandomState instance used by `numpy.random`. Defaults to 0.
        densmap: Whether to use the density-augmented objective function to optimize the embedding according to the
            densMAP algorithm. Defaults to False.
        dens_lambda: Controls the regularization weight of the density correlation term in densMAP. Higher values
            prioritize density preservation over the UMAP objective, and vice versa for values closer to zero. Setting
            this parameter to zero is equivalent to running the original UMAP algorithm. Defaults to 2.0.
        dens_frac: Controls the fraction of epochs (between 0 and 1) where the density-augmented objective is used in
            densMAP. The first (1 - dens_frac) fraction of epochs optimize the original UMAP objective before
            introducing the density correlation term. Defaults to 0.3.
        dens_var_shift: A small constant added to the variance of local radii in the embedding when calculating the
            density correlation objective to prevent numerical instability from dividing by a small number. Defaults to
            0.1.
        output_dens: Whether the local radii of the final embedding (an inverse measure of local density) are computed
            and returned in addition to the embedding. If set to True, local radii of the original data are also
            included in the output for comparison; the output is a tuple (embedding, original local radii, embedding
            local radii). This option can also be used when densmap=False to calculate the densities for UMAP
            embeddings. Defaults to False.
        return_mapper: Whether to return the data mapped onto the UMAP space. Defaults to True.

    Returns:
        mapper: Data mapped onto umap space, will be returned only if `return_mapper` is True
        graph: Sparse matrix representing the nearest neighbor graph
        knn_indices: The indices of the nearest neighbors for each sample
        knn_dists: The distances of the nearest neighbors for each sample
        embedding: The embedding of the data in low-dimensional space
    """
    logger = lm.get_main_logger()

    default_epochs = 500 if X.shape[0] <= 10000 else 200
    max_iter = default_epochs if max_iter is None else max_iter
    random_state = check_random_state(random_state)

    _raw_data = X

    if X.shape[0] < 4096:  # 1
        dmat = pairwise_distances(X, metric=metric)
        graph = fuzzy_simplicial_set(
            X=dmat,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric="precomputed",
        )
        if type(graph) == tuple:
            graph = graph[0]

        # extract knn_indices, knn_dist
        g_tmp = deepcopy(graph)
        g_tmp[graph.nonzero()] = dmat[graph.nonzero()]
        knn_indices, knn_dists = adj_to_knn(g_tmp, n_neighbors=n_neighbors)
    else:
        # Standard case
        (knn_indices, knn_dists, rp_forest) = nearest_neighbors(
            X=X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
        )

        graph = fuzzy_simplicial_set(
            X=X,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=rp_forest,
        )

    logger.info("Constructing embedding ...")

    a, b = find_ab_params(spread, min_dist)
    if type(graph) == tuple:
        graph = graph[0]

    dens_lambda = dens_lambda if densmap else 0.0
    dens_frac = dens_frac if densmap else 0.0

    if dens_lambda < 0.0:
        raise ValueError("dens_lambda cannot be negative")
    if dens_frac < 0.0 or dens_frac > 1.0:
        raise ValueError("dens_frac must be between 0.0 and 1.0")
    if dens_var_shift < 0.0:
        raise ValueError("dens_var_shift cannot be negative")

    densmap_kwds = {
        "lambda": dens_lambda,
        "frac": dens_frac,
        "var_shift": dens_var_shift,
        "n_neighbors": n_neighbors,
    }
    embedding_, aux_data = simplicial_set_embedding(
        data=_raw_data,
        graph=graph,
        n_components=n_components,
        initial_alpha=alpha,  # learning_rate
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=max_iter,
        init=init_pos,
        random_state=random_state,
        metric=metric,
        metric_kwds={},
        densmap=densmap,
        densmap_kwds=densmap_kwds,
        output_dens=output_dens,
    )

    if return_mapper:
        import umap.umap_ as umap

        from .utils import update_dict

        _umap_kwargs = {
            "angular_rp_forest": False,
            "local_connectivity": 1.0,
            "metric_kwds": None,
            "set_op_mix_ratio": 1.0,
            "target_metric": "categorical",
            "target_metric_kwds": None,
            "target_n_neighbors": -1,
            "target_weight": 0.5,
            "transform_queue_size": 4.0,
            "transform_seed": 42,
        }
        umap_kwargs = update_dict(_umap_kwargs, umap_kwargs)

        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            min_dist=min_dist,
            spread=spread,
            n_epochs=max_iter,
            learning_rate=alpha,
            repulsion_strength=gamma,
            negative_sample_rate=negative_sample_rate,
            init=init_pos,
            random_state=random_state,
            **umap_kwargs,
        ).fit(X)

        return mapper, graph, knn_indices, knn_dists, embedding_
    else:
        return graph, knn_indices, knn_dists, embedding_


def find_optimal_n_umap_components(X_data: np.ndarray, max_n_components: Optional[int] = None, **umap_params):
    """Determine the optimal number of UMAP components by maximizing the silhouette score for the Leiden partitioning.

    Args:
        X_data: Input data to UMAP
        max_n_components: Maximum number of UMAP components to test. If not given, will use half the number of
            features (half the number of columns of the input array).
        **umap_params: Parameters to pass to the UMAP function. Should not include 'n_components', which will be
            added by this function.

    Returns:
        best_n_components: Number of components resulting in the highest silhouette score for the Leiden partitioning
    """
    best_score = -1
    best_n_components = None
    # A few important keyword arguments to set:
    umap_params["return_mapper"] = False
    umap_params["min_dist"] = 0.5

    for n_components in range(2, max_n_components + 1):
        umap_params["n_components"] = n_components
        _, _, _, embedding = umap_conn_indices_dist_embedding(X_data, **umap_params)
        clusters = calculate_leiden_partition(input_mat=embedding, num_neighbors=20, graph_type="embedding")

        # Compute silhouette score:
        score = silhouette_score(embedding, clusters)

        # if this score is better than the current best, update best score and components
        if score > best_score:
            best_score = score
            best_n_components = n_components

    return best_n_components


# ---------------------------------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------------------------------
def pca(
    adata: AnnData,
    X_data: np.ndarray = None,
    n_pca_components: int = 30,
    pca_key: str = "X_pca",
    pcs_key: str = "PCs",
    layer: Union[List[str], str, None] = None,
    svd_solver: Literal["randomized", "arpack"] = "randomized",
    random_state: int = 0,
    use_truncated_SVD_threshold: int = 500000,
    use_incremental_PCA: bool = False,
    incremental_batch_size: Optional[int] = None,
    return_all: bool = False,
) -> Union[AnnData, Tuple[AnnData, Union[PCA, TruncatedSVD], np.ndarray]]:
    """Perform PCA reduction.

    For large datasets (>1 million samples), incremental PCA is recommended to avoid memory issues. For datasets with
    more than 500,000 samples, truncated SVD will be used. Otherwise, truncated SVD with centering will be used.

    Args:
        adata: AnnData object to store results in
        X_data: Optional data array to perform dimension reduction on
        n_pca_components: Number of PCA components reduced to. Defaults to 30.
        pca_key: The key to store the reduced data. Defaults to "X".
        pcs_key: The key to store the principle axes in feature space. Defaults to "PCs".
        layer: The layer(s) to perform dimension reduction on. Only used if 'X_data' is not provided. Defaults to
            None to use ".X".
        svd_solver: The svd_solver to solve svd decomposition in PCA.
        random_state: The seed used to initialize the random state for PCA.
        use_truncated_SVD_threshold: The threshold of observations to use truncated SVD instead of standard PCA for
            efficiency.
        use_incremental_PCA: whether to use Incremental PCA. Recommended to set True when dataset is too
            large to fit in memory.
        incremental_batch_size: The number of samples to use for each batch when performing incremental PCA. If
            batch_size is None, then batch_size is inferred from the data and set to 5 * n_features.
        return_all: Whether to return the PCA fit model and the reduced array together with the updated AnnData
            object. Defaults to False.

    Returns:
        adata: Updated AnnData object
        fit: The PCA fit model. Returned only if 'return_all' is True.
        X_pca: The reduced data array. Returned only if 'return_all' is True.
    """
    logger = lm.get_main_logger()

    if X_data is None:
        if "use_for_pca" not in adata.var.keys():
            adata.var["use_for_pca"] = True

        if layer is None:
            X_data = adata.X[:, adata.var.use_for_pca.values]
        else:
            if "X" in layer:
                X_data = adata.X[:, adata.var.use_for_pca.values]
            elif "total" in layer:
                X_data = adata.layers["X_total"][:, adata.var.use_for_pca.values]
            elif "spliced" in layer:
                X_data = adata.layers["X_spliced"][:, adata.var.use_for_pca.values]
            elif type(layer) is str:
                X_data = adata.layers["X_" + layer][:, adata.var.use_for_pca.values]
            else:
                raise ValueError(
                    f"Layer {layer} not in AnnData object. Please use 'X', 'total', 'spliced', or a layer name."
                )

        cm_genesums = X_data.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        bad_genes = np.where(adata.var.use_for_pca)[0][~valid_ind]

        adata.var.iloc[bad_genes, adata.var.columns.tolist().index("use_for_pca")] = False
        X_data = X_data[:, valid_ind]

    if use_incremental_PCA:
        fit, X_pca = pca_fit(
            X_data, pca_func=IncrementalPCA, n_components=n_pca_components, batch_size=incremental_batch_size
        )

    else:
        if adata.n_obs < use_truncated_SVD_threshold:
            if not scipy.sparse.issparse(X_data):
                fit, X_pca = pca_fit(
                    X_data,
                    pca_func=PCA,
                    n_components=n_pca_components,
                    svd_solver=svd_solver,
                    random_state=random_state,
                )
            else:
                fit, X_pca = truncated_SVD_with_center(
                    X_data,
                    n_components=n_pca_components,
                    random_state=random_state,
                )
        else:
            fit, X_pca = pca_fit(
                X_data,
                pca_func=TruncatedSVD,
                n_components=n_pca_components + 1,
                random_state=random_state,
            )
            # First component is related to the total counts
            X_pca = X_pca[:, 1:]

    logger.info_insert_adata(pca_key, adata_attr="obsm")
    adata.obsm[pca_key] = X_pca

    if use_incremental_PCA or adata.n_obs < use_truncated_SVD_threshold:
        adata.uns[pcs_key] = fit.components_.T
        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_
    else:
        # first columns is related to the total UMI (or library size)
        adata.uns[pcs_key] = fit.components_.T[:, 1:]
        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

    adata.uns["pca_mean"] = fit.mean_ if hasattr(fit, "mean_") else np.zeros(X_data.shape[1])

    if return_all:
        return adata, fit, X_pca
    else:
        return adata


def pca_fit(
    X: np.ndarray,
    pca_func: Callable,
    n_components: int = 30,
    **kwargs,
) -> Tuple[PCA, np.ndarray]:
    """Apply PCA to the input data array X using the specified PCA function.

    Args:
        X: The input data array of shape (n_samples, n_features).
        pca_func: The PCA function to use, which should have a 'fit' and 'transform' method, such as the PCA class
            or the IncrementalPCA class from sklearn.decomposition.
        n_components: The number of principal components to compute
        **kwargs: Any additional keyword arguments that will be passed to the PCA function

    Returns:
        fit: The fitted PCA object
        X_pca: The reduced data array of shape (n_samples, n_components)
    """
    fit = pca_func(
        n_components=min(n_components, X.shape[1] - 1),
        **kwargs,
    ).fit(X)
    X_pca = fit.transform(X)
    return fit, X_pca


def truncated_SVD_with_center(
    X: np.ndarray,
    n_components: int = 30,
    random_state: Union[int, np.random.RandomState, None] = 0,
) -> Tuple[PCA, np.ndarray]:
    """Apply truncated SVD to the input data array X with centering.

    Args:
        X: The input data array of shape (n_samples, n_features).
        n_components: The number of principal components to compute
        random_state: The seed used to initialize the random state for PCA.

    Returns:
        fit: The fitted truncated SVD object
        X_pca: The reduced data array of shape (n_samples, n_components)
    """
    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    v0 = random_state.uniform(-1, 1, np.min(X.shape))
    n_components = min(n_components, X.shape[1] - 1)

    mean = X.mean(0)
    X_H = X.T.conj()
    mean_H = mean.T.conj()
    ones = np.ones(X.shape[0])[None, :].dot

    # Following callables implement different types of matrix operations.
    def matvec(x):
        """Matrix-vector multiplication. Performs the operation X_centered*x
        where x is a column vector or an 1-D array."""
        return X.dot(x) - mean.dot(x)

    def matmat(x):
        """Matrix-matrix multiplication. Performs the operation X_centered*x
        where x is a matrix or ndarray."""
        return X.dot(x) - mean.dot(x)

    def rmatvec(x):
        """Adjoint matrix-vector multiplication. Performs the operation
        X_centered^H * x where x is a column vector or an 1-d array."""
        return X_H.dot(x) - mean_H.dot(ones(x))

    def rmatmat(x):
        """Adjoint matrix-matrix multiplication. Performs the operation
        X_centered^H * x where x is a matrix or ndarray."""
        return X_H.dot(x) - mean_H.dot(ones(x))

    # Construct the LinearOperator with callables above.
    X_centered = LinearOperator(
        shape=X.shape,
        matvec=matvec,
        matmat=matmat,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
        dtype=X.dtype,
    )

    # Solve SVD without calculating individuals entries in LinearOperator.
    U, Sigma, VT = svds(X_centered, solver="arpack", k=n_components, v0=v0)
    Sigma = Sigma[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])
    X_transformed = U * Sigma
    components_ = VT
    exp_var = np.var(X_transformed, axis=0)
    _, full_var = mean_variance_axis(X, axis=0)
    full_var = full_var.sum()

    result_dict = {
        "X_pca": X_transformed,
        "components_": components_,
        "explained_variance_ratio_": exp_var / full_var,
    }

    fit = PCA(
        n_components=n_components,
        random_state=random_state,
    )
    X_pca = result_dict["X_pca"]
    fit.mean_ = mean.A1.flatten()
    fit.components_ = result_dict["components_"]
    fit.explained_variance_ratio_ = result_dict["explained_variance_ratio_"]

    return fit, X_pca


def find_optimal_pca_components(
    X: np.ndarray,
    pca_func: Callable,
    max_components: Optional[int] = None,
    drop_ratio: float = 0.33,
    **kwargs,
) -> int:
    """Find the optimal number of PCA components using the elbow method.

    Args:
        X: The input data array of shape (n_samples, n_features)
        pca_func: The PCA function to use, which should have a 'fit' and 'transform' method, such as the PCA class
            or the IncrementalPCA class from sklearn.decomposition.
        max_components: The maximum number of principal components to test. If not given, will use half the number of
            features (half the number of columns of the input array).
        drop_ratio: The ratio of the change in explained variance to consider a significant drop
        **kwargs: Any additional keyword arguments that will be passed to the PCA function

    Returns:
        n: Optimal number of components
    """
    if max_components is None:
        max_components = X.shape[1] // 2

    explained_variances = []
    for n_components in range(2, max_components + 1):
        fit, _ = pca_fit(X, pca_func, n_components=n_components, **kwargs)
        explained_variances.append(fit.explained_variance_ratio_.sum())
    explained_variances = np.array(explained_variances)

    # Plot the explained variance as a function of the number of components
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_components + 1), explained_variances, "bo-", linewidth=2)
    plt.xlabel("Number of components")
    plt.ylabel("Total explained variance")
    plt.title("Elbow plot for PCA")
    plt.show()

    # The elbow point is the point of maximum curvature from the end of the curve, but we also want to find the point
    # where the change in explained variance drops significantly
    # We subtract each point from the preceding point (ignoring the first few components because the variance
    # explained by each of the first few components will be the highest):
    start_index = 5
    deltas = np.diff(explained_variances[start_index:][::-1])

    significant_drop = [i for i in range(1, len(deltas)) if deltas[i] < drop_ratio * deltas[i - 1]]
    if significant_drop:
        n = significant_drop[0] + start_index + 1
    else:
        # If there's no significant drop, use the point of maximum curvature
        n = np.argmax(deltas) + start_index + 1

    return n
