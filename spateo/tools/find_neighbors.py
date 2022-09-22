"""
Functions for finding nearest neighbors and the distances between them in spatial transcriptomics data.
"""
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy
import sklearn
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ..configuration import SKM
from ..logging import logger_manager as lm
from ..tools.labels import row_normalize


# ------------------------------------- Wrapper for weighted spatial graph ------------------------------------- #
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def weighted_spatial_graph(
    adata: AnnData,
    spatial_key: str = "spatial",
    fixed: str = "n_neighbors",
    n_neighbors_method: str = "ball_tree",
    n_neighbors: int = 30,
    decay_type: str = "reciprocal",
    p: float = 0.05,
    sigma: float = 100,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, AnnData]:
    """Given an AnnData object, compute distance array with either a fixed number of neighbors for each bucket or a
    fixed search radius for each bucket.

    Args:
        adata: an anndata object.
        spatial_key: Key in .obsm containing coordinates for each bucket.
        fixed: Options: 'n_neighbors', 'radius'- sets either fixed number of neighbors or fixed search radius for each
            bucket.
        n_neighbors_method: Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree". Unused unless 'fixed' is 'n_neighbors'.
        n_neighbors: Number of neighbors each bucket has. Unused unless 'fixed' is 'n_neighbors'.
        decay_type: Sets method by which edge weights are defined. Options: "reciprocal", "ranked", "uniform".
            Unused unless 'fixed' is 'n_neighbors'.
        p: Cutoff for Gaussian (used to find where distribution drops below p * (max_value)). Unused unless 'fixed' is
            'radius'.
        sigma: Standard deviation of the Gaussian. Unused unless 'fixed' is 'radius'.

    Returns:
        out_graph: Weighted nearest neighbors graph with shape [n_samples, n_samples]
        distance_graph: Unweighted graph with shape [n_samples, n_samples]
        adata: Updated AnnData object containing 'spatial_distance' in .obsp and 'spatial_neighbors' in .uns.
    """
    logger = lm.get_main_logger()
    if fixed == "n_neighbors":
        weights_graph, distance_graph, adata = generate_spatial_weights_fixed_nbrs(
            adata,
            spatial_key,
            num_neighbors=n_neighbors,
            method=n_neighbors_method,
            decay_type=decay_type,
        )
    elif fixed == "radius":
        weights_graph, distance_graph, adata = generate_spatial_weights_fixed_radius(
            adata,
            spatial_key,
            method=n_neighbors_method,
            p=p,
            sigma=sigma,
        )
    else:
        logger.error("Invalid argument given to 'fixed'. Options: 'n_neighbors', 'radius'.")
        raise ValueError("Invalid argument given to 'fixed'. Options: 'n_neighbors', 'radius'.")

    return weights_graph, distance_graph, adata


# -------------------------------- Wrapper for weighted transcriptomic space graph -------------------------------- #
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def generate_expr_weights(
    adata: AnnData,
    nbr_object: NearestNeighbors = None,
    basis: str = "pca",
    n_neighbors_method: str = "ball_tree",
    n_pca_components: int = 30,
    num_neighbors: int = 30,
    decay_type: str = "reciprocal",
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, AnnData]:
    """Given an AnnData object, compute distance array in gene expression space.

    Args:
        adata: an anndata object.
        nbr_object: An optional sklearn.neighbors.NearestNeighbors object. Can optionally create a nearest neighbor
            object with custom functionality.
        basis: str, default 'pca'
            The space that will be used for nearest neighbor search. Valid names includes, for example, `pca`, `umap`,
            or `X`
        n_neighbors_method: str, default 'ball_tree'
            Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree".
        n_pca_components: Only used if 'basis' is 'pca'. Sets number of principal components to compute.
        num_neighbors: Number of neighbors for each bucket, used in computing distance graph
        decay_type: Sets method by which edge weights are defined. Options: "reciprocal", "ranked", "uniform".

    Returns:
        out_graph: Weighted k-nearest neighbors graph with shape [n_samples, n_samples].
        distance_graph: Unweighted graph with shape [n_samples, n_samples].
        adata: Updated AnnData object containing 'spatial_distance' in .obsp and 'spatial_neighbors' in .uns.
    """
    logger = lm.get_main_logger()

    nbrs, adata = transcriptomic_connectivity(
        adata, nbr_object, basis, n_neighbors_method, n_pca_components, num_neighbors
    )

    assert isinstance(num_neighbors, int), f"Number of neighbors {num_neighbors} is not an integer."

    distance_graph = nbrs.kneighbors_graph(n_neighbors=num_neighbors, mode="distance")

    if basis == "X":
        X_data = adata.X
    elif "X_" + basis in adata.obsm_keys():
        # Assume basis can be found in .obs under "X_{basis}":
        X_data = adata.obsm["X_" + basis]
    else:
        logger.error("Invalid option given to 'basis'. Options: 'pca', 'umap' or 'X'.")

    distances, knn = nbrs.kneighbors(X_data)
    n_obs, n_neighbors = knn.shape

    logger.info_insert_adata("expression_neighbors", adata_attr="uns")
    logger.info_insert_adata("expression_neighbors.indices", adata_attr="uns")
    logger.info_insert_adata("expression_neighbors.params", adata_attr="uns")

    adata.uns["expression_neighbors"] = {}
    adata.uns["expression_neighbors"]["indices"] = knn
    adata.uns["expression_neighbors"]["params"] = {
        "n_neighbors": n_neighbors,
        "method": n_neighbors_method,
        "metric": "euclidean",
    }

    # Compute nonspatial (gene expression) weights:
    graph_out = distance_graph.copy()

    # Convert distances to weights
    if decay_type == "uniform":
        graph_out.data = np.ones_like(graph_out.data)
    elif decay_type == "reciprocal":
        graph_out.data = 1 / graph_out.data
    elif decay_type == "ranked":
        linear_weights = np.exp(-1 * (np.arange(1, num_neighbors + 1) * 1.5 / num_neighbors) ** 2)

        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):
            start_ptr, end_ptr = indptr[n], indptr[n + 1]

            if end_ptr >= start_ptr:
                # Row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]

                # Assign the weights in ranked order
                weights = np.empty_like(linear_weights)
                weights[np.argsort(nbrs)] = linear_weights

                data[start_ptr:end_ptr] = weights
        graph_out.data = data
    else:
        logger.error(
            f"Weights decay type <{decay_type}> not recognised. Should be 'uniform', 'reciprocal' or 'ranked'."
        )
        raise ValueError(
            f"Weights decay type <{decay_type}> not recognised.\n" f"Should be 'uniform', 'reciprocal' or 'ranked'."
        )

    out_graph = row_normalize(graph_out, verbose=False)
    return out_graph, distance_graph, adata


def transcriptomic_connectivity(
    adata: AnnData,
    nbr_object: NearestNeighbors = None,
    basis: str = "pca",
    n_neighbors_method: str = "ball_tree",
    n_pca_components: int = 30,
    num_neighbors: int = 30,
) -> Tuple[NearestNeighbors, AnnData]:
    """Given an AnnData object, compute pairwise connectivity matrix in transcriptomic space

    Args:
        adata : an anndata object.
        nbr_object: An optional sklearn.neighbors.NearestNeighbors object. Can optionally create a nearest neighbor
            object with custom functionality.
        basis: str, default 'pca'
            The space that will be used for nearest neighbor search. Valid names includes, for example, `pca`, `umap`,
            or `X`
        n_neighbors_method: str, default 'ball_tree'
            Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree".
        n_pca_components: Only used if 'basis' is 'pca'. Sets number of principal components to compute.
        num_neighbors: Number of neighbors for each bucket, used in computing distance graph

    Returns:
        nbrs : Object of class `sklearn.neighbors.NearestNeighbors`
        adata : Modified AnnData object
    """
    logger = lm.get_main_logger()

    if basis == "pca" and "X_pca" not in adata.obsm_keys():
        logger.info("PCA to be used as basis, X_pca not found, computing PCA...", indent_level=2)
        pca = PCA(
            n_components=min(n_pca_components, adata.X.shape[1] - 1),
            svd_solver="arpack",
            random_state=0,
        )
        fit = pca.fit(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else pca.fit(adata.X)
        X_pca = fit.transform(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else fit.transform(adata.X)
        adata.obsm["X_pca"] = X_pca

    if basis == "X":
        X_data = adata.X
    elif "X_" + basis in adata.obsm_keys():
        # Assume basis can be found in .obs under "X_{basis}":
        X_data = adata.obsm["X_" + basis]
    else:
        logger.error("Invalid option given to 'basis'. Options: 'pca', 'umap' or 'X'.")

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm=n_neighbors_method).fit(X_data)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    # Update AnnData to add spatial distances, spatial connectivities and spatial neighbors from the sklearn
    # NearestNeighbors run:
    distances, knn = nbrs.kneighbors(X_data)
    n_obs, n_neighbors = knn.shape
    distances = scipy.sparse.csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    logger.info_insert_adata("expression_connectivities", adata_attr="obsp")
    logger.info_insert_adata("expression_distances", adata_attr="obsp")

    adata.obsp["expression_distances"] = distances
    adata.obsp["expression_connectivities"] = connectivities

    return nbrs, adata


# --------------------------------------- Cell-cell/bucket-bucket distance --------------------------------------- #
def remove_greater_than(
    graph: scipy.sparse.csr_matrix, threshold: float, copy: bool = False, verbose: bool = False
) -> scipy.sparse.csr_matrix:
    """Remove values greater than a threshold from a sparse matrix.

    Args:
        graph: The input scipy matrix of the graph.
        threshold: Upper numerical threshold to avoid filtering.
        copy: Set True to avoid altering original graph.
        verbose: Set True to display messages at runtime- not recommended generally since this will print entire arrays.

    Returns:
        graph: The updated graph with values greater than the threshold removed.
    """
    logger = lm.get_main_logger()

    if copy:
        graph = graph.copy()

    greater_indices = np.where(graph.data > threshold)[0]

    if verbose:
        logger.info(
            f"CSR data field:\n{graph.data}\n" f"compressed indices of values > threshold:\n{greater_indices}\n"
        )

    graph.data = np.delete(graph.data, greater_indices)
    graph.indices = np.delete(graph.indices, greater_indices)

    # Update ptr
    hist, _ = np.histogram(greater_indices, bins=graph.indptr)
    cum_hist = np.cumsum(hist)
    graph.indptr[1:] -= cum_hist

    if verbose:
        logger.info(
            f"\nCumulative histogram:\n{cum_hist}\n"
            f"\n___ New CSR ___\n"
            f"pointers:\n{graph.indptr}\n"
            f"indices:\n{graph.indices}\n"
            f"data:\n{graph.data}\n"
        )

    return graph


def generate_spatial_distance_graph(
    locations: np.ndarray,
    nbr_object: NearestNeighbors = None,
    method: str = "ball_tree",
    num_neighbors: Union[None, int] = None,
    radius: Union[None, float] = None,
) -> Tuple[sklearn.neighbors.NearestNeighbors, scipy.sparse.csr_matrix]:
    """Creates graph based on distance in space.

    Args:
        locations: Spatial coordinates for each bucket with shape [n_samples, 2]
        nbr_object : An optional sklearn.neighbors.NearestNeighbors object. Can optionally create a nearest neighbor
            object with custom functionality.
        method: Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree".
        num_neighbors: Number of neighbors for each bucket.
        radius: Search radius around each bucket.

    Returns:
        nbrs: sklearn NearestNeighbor object.
        graph_out: A sparse matrix of the spatial graph.
    """
    logger = lm.get_main_logger()

    if num_neighbors is None and radius is None:
        logger.error("Number of neighbors or search radius for each bucket must be provided.")
        raise ValueError("Number of neighbors or search radius for each bucket must be provided.")

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm=method).fit(locations)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    if num_neighbors is None:
        # no limit to number of neighbours
        return nbrs.radius_neighbors_graph(radius=radius, mode="distance")

    else:
        assert isinstance(num_neighbors, int), f"Number of neighbors {num_neighbors} is not an integer."

        graph_out = nbrs.kneighbors_graph(n_neighbors=num_neighbors, mode="distance")

        if radius is not None:
            assert isinstance(radius, (float, int)), f"Radius {radius} is not an integer or float."

            graph_out = remove_greater_than(graph_out, radius, copy=False, verbose=False)

        return nbrs, graph_out


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def generate_spatial_weights_fixed_nbrs(
    adata: AnnData,
    spatial_key: str = "spatial",
    num_neighbors: int = 10,
    method: str = "ball_tree",
    decay_type: str = "reciprocal",
    nbr_object: NearestNeighbors = None,
) -> Union[Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, AnnData]]:
    """Starting from a k-nearest neighbor graph, generate a nearest neighbor graph.

    Args:
        spatial_key: Key in .obsm where x- and y-coordinates are stored.
        num_neighbors: Number of neighbors each bucket has.
        method: Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
        "ball_tree" and "kd_tree".
        decay_type: Sets method by which edge weights are defined. Options: "reciprocal", "ranked", "uniform".

    Returns:
        out_graph: Weighted k-nearest neighbors graph with shape [n_samples, n_samples].
        distance_graph: Unweighted graph with shape [n_samples, n_samples].
        adata: Updated AnnData object containing 'spatial_distance' in .obsp and 'spatial_neighbors' in .uns.
    """
    logger = lm.get_main_logger()

    if method not in ["ball_tree", "kd_tree"]:
        logger.error("Invalid argument passed to 'method'. Options: 'ball_tree', 'kd_tree'.")

    # Get locations from AnnData:
    locations = adata.obsm[spatial_key]

    # Define the nearest-neighbors graph
    nbrs, distance_graph = generate_spatial_distance_graph(
        locations,
        nbr_object=nbr_object,
        method=method,
        num_neighbors=num_neighbors,
        radius=None,
    )

    # Update AnnData to add spatial distances, spatial connectivities and spatial neighbors from the sklearn
    # NearestNeighbors run:
    distances, knn = nbrs.kneighbors(locations)
    n_obs, n_neighbors = knn.shape
    distances = scipy.sparse.csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    logger.info_insert_adata("spatial_connectivities", adata_attr="obsp")
    logger.info_insert_adata("spatial_distances", adata_attr="obsp")

    adata.obsp["spatial_distances"] = distances
    adata.obsp["spatial_connectivies"] = connectivities

    logger.info_insert_adata("spatial_neighbors", adata_attr="uns")
    logger.info_insert_adata("spatial_neighbors.indices", adata_attr="uns")
    logger.info_insert_adata("spatial_neighbors.params", adata_attr="uns")

    adata.uns["spatial_neighbors"] = {}
    adata.uns["spatial_neighbors"]["indices"] = knn
    adata.uns["spatial_neighbors"]["params"] = {"n_neighbors": n_neighbors, "method": method, "metric": "euclidean"}

    # Compute spatial weights:
    graph_out = distance_graph.copy()

    # Convert distances to weights
    if decay_type == "uniform":
        graph_out.data = np.ones_like(graph_out.data)
    elif decay_type == "reciprocal":
        graph_out.data = 1 / graph_out.data
    elif decay_type == "ranked":
        linear_weights = np.exp(-1 * (np.arange(1, num_neighbors + 1) * 1.5 / num_neighbors) ** 2)

        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):
            start_ptr, end_ptr = indptr[n], indptr[n + 1]

            if end_ptr >= start_ptr:
                # Row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]

                # Assign the weights in ranked order
                weights = np.empty_like(linear_weights)
                weights[np.argsort(nbrs)] = linear_weights

                data[start_ptr:end_ptr] = weights
        graph_out.data = data
    else:
        logger.error(
            f"Weights decay type <{decay_type}> not recognised. Should be 'uniform', 'reciprocal' or 'ranked'."
        )
        raise ValueError(
            f"Weights decay type <{decay_type}> not recognised.\n" f"Should be 'uniform', 'reciprocal' or 'ranked'."
        )

    out_graph = row_normalize(graph_out, verbose=False)
    return out_graph, distance_graph, adata


def gaussian_weight_2d(distance: float, sigma: float):
    """Calculate normalized gaussian value for a given distance from central point
    Normalized by 2*pi*sigma-squared
    """
    sigma_squared = float(sigma) ** 2
    return np.exp(-0.5 * distance**2 / sigma_squared) / np.sqrt(sigma_squared * 2 * np.pi)


def p_equiv_radius(p: float, sigma: float):
    """Find radius at which you eliminate fraction p of a radial Gaussian probability distribution with standard
    deviation sigma.
    """
    assert p < 1.0, f"p was {p}, must be less than 1"
    return np.sqrt(-2 * sigma**2 * np.log(p))


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def generate_spatial_weights_fixed_radius(
    adata: AnnData,
    spatial_key: str = "spatial",
    p: float = 0.05,
    sigma: float = 100,
    nbr_object: NearestNeighbors = None,
    method: str = "ball_tree",
    max_num_neighbors: int = None,
    verbose: bool = False,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, AnnData]:
    """Starting from a radius-based neighbor graph, generate a sparse graph (csr format) with weighted edges, where edge
    weights decay with distance.

    Note that decay is assumed to follow a Gaussian distribution.

    Args:
        spatial_key: Key in .obsm where x- and y-coordinates are stored.
        p: Cutoff for Gaussian (used to find where distribution drops below p * (max_value)).
        sigma: Standard deviation of the Gaussian.
        method: Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree".
        max_num_neighbors: Sets upper threshold on number of neighbors a bucket can have.

    Returns:
        out_graph: Weighted nearest neighbors graph with shape [n_samples, n_samples].
        distance_graph: Unweighted graph with shape [n_samples, n_samples].
        adata: Updated AnnData object containing 'spatial_distance' in .obsp and 'spatial_neighbors' in .uns.
    """
    logger = lm.get_main_logger()

    # Get locations from AnnData:
    locations = adata.obsm[spatial_key]

    # Selecting r for neighbor graph construction:
    r = p_equiv_radius(p, sigma)
    if verbose:
        logger.info(f"Equivalent radius for removing {p} of " f"gaussian distribution with sigma {sigma} is: {r}\n")

    # Define the nearest neighbor graph:
    nbrs, distance_graph = generate_spatial_distance_graph(
        locations, nbr_object=nbr_object, num_neighbors=max_num_neighbors, radius=r
    )

    # Update AnnData to add spatial distances, spatial connectivities and spatial neighbors from the sklearn
    # NearestNeighbors run:
    distances, knn = nbrs.kneighbors(locations)
    n_obs, n_neighbors = knn.shape
    distances = scipy.sparse.csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    logger.info_insert_adata("spatial_connectivities", adata_attr="obsp")
    logger.info_insert_adata("spatial_distances", adata_attr="obsp")

    adata.obsp["spatial_distances"] = distances
    adata.obsp["spatial_connectivies"] = connectivities

    logger.info_insert_adata("spatial_neighbors", adata_attr="uns")
    logger.info_insert_adata("spatial_neighbors.indices", adata_attr="uns")
    logger.info_insert_adata("spatial_neighbors.params", adata_attr="uns")

    adata.uns["spatial_neighbors"] = {}
    adata.uns["spatial_neighbors"]["indices"] = knn
    adata.uns["spatial_neighbors"]["params"] = {"n_neighbors": n_neighbors, "method": method, "metric": "euclidean"}

    graph_out = distance_graph.copy()

    # Convert distances to weights
    graph_out.data = gaussian_weight_2d(graph_out.data, sigma)

    out_graph = row_normalize(graph_out, verbose=verbose)
    return out_graph, distance_graph, adata


# ----------- (Identical to generate_spatial_weights_fixed_nbrs, but specific to STGNN) ---------- #
def calculate_distance(position: np.ndarray) -> np.ndarray:
    """Given array of x- and y-coordinates, compute pairwise distances between all samples"""
    distance_matrix = squareform(pdist(position, metric="euclidean"))
    """
    n_bucket = position.shape[0]
    distance_matrix = np.zeros([n_bucket, n_bucket])

    for i in range(n_bucket):
        x = position[i, :]
        for j in range(i + 1, n_bucket):
            y = position[j, :]
            d = np.sqrt(np.sum(np.square(x - y)))
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d"""

    return distance_matrix


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def construct_pairwise(
    adata: AnnData,
    spatial_key: str = "spatial",
    n_neighbors: int = 8,
    exclude_self: bool = True,
    save_id: Union[None, str] = None,
) -> None:
    """Constructing bucket-to-bucket nearest neighbors graph.

    Args:
        adata: An anndata object.
        spatial_key: Key in .obsm in which x- and y-coordinates are stored.
        n_neighbors: Number of nearest neighbors to compute for each bucket.
        exclude_self: Set True to set elements along the diagonal to zero.
        save_id: Optional string; if not None, will save distance matrix and neighbors matrix to path:
        './neighbors/{save_id}_distance.csv' and path: './neighbors/{save_id}_neighbors.csv', respectively.
    """
    position = adata.obsm[spatial_key]
    # calculate distance matrix
    distance_matrix = calculate_distance(position)
    n_bucket = distance_matrix.shape[0]

    adata.obsm["distance_matrix"] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_bucket, n_bucket])
    for i in range(n_bucket):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    if save_id is not None:
        if not os.path.exists(os.path.join(os.getcwd(), "neighbors")):
            os.makedirs(os.path.join(os.getcwd(), "neighbors"))
        dist_df = pd.DataFrame(distance_matrix, index=list(adata.obs_names), columns=list(adata.obs_names))
        dist_df.to_csv(os.path.join(os.getcwd(), f"neighbors/{save_id}_distance.csv"))
        neighbors_df = pd.DataFrame(interaction, index=list(adata.obs_names), columns=list(adata.obs_names))
        neighbors_df.to_csv(os.path.join(os.getcwd(), f"./neighbors/{save_id}_neighbors.csv"))

    adata.obsm["graph_neigh"] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    if exclude_self:
        np.fill_diagonal(adj, 0)

    adata.obsm["adj"] = adj


def normalize_adj(adj: np.ndarray, exclude_self: bool = True) -> np.ndarray:
    """Symmetrically normalize adjacency matrix, set diagonal to 1 and return processed adjacency array.

    Args:
        adj: Pairwise distance matrix of shape [n_samples, n_samples].
        exclude_self: Set True to set diagonal of adjacency matrix to 1.

    Returns:
        adj_proc: The normalized adjacency matrix.
    """
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_proc = adj.toarray() if exclude_self else adj.toarray() + np.eye(adj.shape[0])

    return adj_proc
