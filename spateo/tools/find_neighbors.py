"""
Functions for finding nearest neighbors, the distances between them and the spatial weighting between points in
spatial transcriptomics data.
"""
import os
import sys
from typing import Iterable, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------------------------------
def calculate_distance(position: np.ndarray, dist_metric: str = "euclidean") -> np.ndarray:
    """Given array of x- and y-coordinates, compute pairwise distances between all samples using Euclidean distance."""
    distance_matrix = squareform(pdist(position, metric=dist_metric))

    return distance_matrix


def local_dist(coords_i: np.ndarray, coords: np.ndarray):
    """For single sample, compute distance between that sample and each other sample in the data.

    Args:
        coords_i: Array of shape (n, ), where n is the dimensionality of the data; the coordinates of a single point
        coords: Array of shape (m, n), where n is the dimensionality of the data and m is an arbitrary number of
            samples; pairwise distances from `coords_i`.

    Returns:
        distances: array-like, shape (m, ), where m is an arbitrary number of samples. The pairwise distances
            between `coords_i` and each point in `coords`.
    """
    distances = np.sqrt(np.sum((coords_i - coords) ** 2, axis=1))
    return distances


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


def adj_to_knn(adj: np.ndarray, n_neighbors: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Given an adjacency matrix, convert to KNN graph.

    Args:
        adj: Adjacency matrix of shape (n_samples, n_samples)
        n_neighbors: Number of nearest neighbors to include in the KNN graph

    Returns:
        indices: Array (n_samples x n_neighbors) storing the indices for each node's nearest neighbors in the
            knn graph.
        weights: Array (n_samples x n_neighbors) storing the edge weights for each node's nearest neighbors in
            the knn graph.
    """
    n_obs = adj.shape[0]
    indices = np.zeros((n_obs, n_neighbors), dtype=int)
    weights = np.zeros((n_obs, n_neighbors), dtype=float)

    for i in range(n_obs):
        neighbors = adj[i, :].nonzero()
        # Set self as nearest neighbor
        indices[i, :] = i
        weights[i, :] = 0.0

        # there could be more or less than n_neighbors because of an approximate search
        n_neighbors = len(neighbors[1])

        if n_neighbors > n_neighbors - 1:
            sorted_indices = np.argsort(adj[i][:, neighbors[1]].A)[0][: (n_neighbors - 1)]
            indices[i, 1:] = neighbors[1][sorted_indices]
            weights[i, 1:] = adj[i][0, neighbors[1][sorted_indices]].A
        else:
            idx_ = np.arange(1, (n_neighbors + 1))
            indices[i, idx_] = neighbors[1]
            weights[i, idx_] = adj[i][:, neighbors[1]].A

    return indices, weights


def knn_to_adj(knn_indices: np.ndarray, knn_weights: np.ndarray) -> scipy.sparse.csr_matrix:
    """Given the indices and weights of a KNN graph, convert to adjacency matrix.

    Args:
        knn_indices: Array (n_samples x n_neighbors) storing the indices for each node's nearest neighbors in the
            knn graph.
        knn_weights: Array (n_samples x n_neighbors) storing the edge weights for each node's nearest neighbors in
            the knn graph.

    Returns:
        adj: The adjacency matrix corresponding to the KNN graph
    """
    adj = scipy.sparse.csr_matrix(
        (
            knn_weights.flatten(),
            (
                np.repeat(knn_indices[:, 0], knn_indices.shape[1]),
                knn_indices.flatten(),
            ),
        )
    )
    adj.eliminate_zeros()
    return adj


def compute_distances_and_connectivities(
    knn_indices: np.ndarray, distances: np.ndarray
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """Computes connectivity and sparse distance matrices

    Args:
        knn_indices: Array of shape (n_samples, n_samples) containing the indices of the nearest neighbors for each
            sample.
        distances: The distances to the n_neighbors the closest points in knn graph

    Returns:
        distances: Sparse distance matrix
        connectivities: Sparse connectivity matrix
    """
    n_obs, n_neighbors = knn_indices.shape
    distances = scipy.sparse.csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn_indices.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    return distances, connectivities


# ---------------------------------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------------------------------
class Kernel(object):
    """
    Spatial weights for regression models are learned using kernel functions.

    Args:
        i: Index of the point for which to estimate the density
        data: Array of shape (n_samples, n_features) representing the data. If aiming to derive weights from spatial
            distance, this should be the array of spatial positions.
        bw: Bandwidth parameter for the kernel density estimation
        cov: Optional array of shape (n_samples, ). Can be used to adjust the distance calculation to look only at
            samples of interest, which is determined from nonzero values in this vector. This can be, e.g. a one-hot
            vector for a particular cell type.
        expr_mat: Can be used together with 'cov' (so will only be used if 'cov' is not None)- if the spatial neighbors
            are not consistent with the sample in question (determined by assessing similarity by "cov"),
            there may be different mechanisms at play. In this case, will instead search for nearest neighbors in
            the gene expression space if given.
        fixed: If True, `bw` is treated as a fixed bandwidth parameter. Otherwise, it is treated as the number
            of nearest neighbors to include in the bandwidth estimation.
        exclude_self: If True, ignore each sample itself when computing the kernel density estimation
        function: The name of the kernel function to use. Valid options are as follows (note that in equations,
            any "1" can be substituted for any other value(s)):
            - 'triangular': linearly decaying kernel,
                :math K(u) =
                    \begin{cases}
                        1-|u| & \text{if} |u| \leq 1 \ 0 & \text{otherwise}
                    \end{cases},
            - 'quadratic': quadratically decaying kernel,
                :math K(u) =
                    \begin{cases}
                        \dfrac{3}{4}(1-u^2)
                    \end{cases},
            - 'gaussian': decays following normal distribution, :math K(u) = \dfrac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}u^2},
            - 'uniform': AKA the tophat kernel, sets weight of all observations within the bandwidth to the same value,
                :math K(u) =
                    \begin{cases}
                        1 & \text{if} |u| \leq 1 \\ 0 & \text{otherwise}
                    \end{cases},
            - 'exponential': exponentially decaying kernel, :math K(u) = e^{-|u|},
            - 'bisquare': assigns a weight of zero to observations outside of the bandwidth, and decays within the
                bandwidth following equation
                    :math K(u) =
                        \begin{cases}
                            \dfrac{15}{16}(1-u^2)^2 & \text{if} |u| \leq 1 \\ 0 & \text{otherwise}
                        \end{cases}.
        threshold: Threshold for the kernel density estimation. If the density is below this threshold, the density
            will be set to zero.
        eps: Error-correcting factor to avoid division by zero
        sparse_array: If True, the kernel will be converted to sparse array. Recommended for large datasets.
        normalize_weights: If True, the weights will be normalized to sum to 1.
        use_expression_neighbors_only: If True, will only use the expression matrix to find nearest neighbors.
    """

    def __init__(
        self,
        i: int,
        data: Union[np.ndarray, scipy.sparse.spmatrix],
        bw: Union[int, float],
        cov: Optional[np.ndarray] = None,
        expr_mat: Optional[np.ndarray] = None,
        fixed: bool = True,
        exclude_self: bool = False,
        function: str = "triangular",
        threshold: float = 1e-5,
        eps: float = 1.0000001,
        sparse_array: bool = False,
        normalize_weights: bool = False,
        use_expression_neighbors_only: bool = False,
    ):

        if use_expression_neighbors_only:
            self.dist_vector = local_dist(expr_mat[i], expr_mat).reshape(-1)
            self.function = "uniform"
            if cov is not None:
                max_dist = np.max(self.dist_vector)
                self.dist_vector[cov == 0] = max_dist
        else:
            self.dist_vector = local_dist(data[i], data).reshape(-1)
            self.function = function.lower()
            if cov is not None:
                max_dist = np.max(self.dist_vector)
                self.dist_vector[cov == 0] = max_dist

                # Indices of nearest neighbors- if fewer than 1/3 of the neighbors (given by bw for not "fixed" and
                # using the ten nearest neighbors for "fixed") are consistent with "cov" of the sample in question,
                # will instead search for nearest neighbors in the gene expression space:
                n = 10 if fixed else int(bw)
                neighbor_dist_vector = self.dist_vector[self.dist_vector > 0]
                neighbor_dists = np.argpartition(neighbor_dist_vector, n)[:n]
                neighbor_indices = np.where(np.isin(self.dist_vector, neighbor_dist_vector[neighbor_dists]))[0]
                n_neighbor_threshold = 3 if fixed else int(n / 3)
                if np.sum(cov[neighbor_indices]) < n_neighbor_threshold and expr_mat is not None:
                    self.dist_vector = local_dist(expr_mat[i], expr_mat).reshape(-1)
                    if cov is not None:
                        max_dist = np.max(self.dist_vector)
                        self.dist_vector[cov == 0] = max_dist
                    # Set kernel to uniform for expression neighbors:
                    self.function = "uniform"

        if fixed:
            self.bandwidth = float(bw)
        else:
            if exclude_self:
                self.bandwidth = np.partition(self.dist_vector, int(bw) + 1)[int(bw) + 1] * eps
            else:
                self.bandwidth = np.partition(self.dist_vector, int(bw))[int(bw)] * eps

        bw_dist = self.dist_vector / self.bandwidth
        # Exclude self as a neighbor:
        if exclude_self:
            bw_dist = np.where(bw_dist == 0.0, np.max(bw_dist), bw_dist)
        self.kernel = self._kernel_functions(bw_dist)

        # Bisquare and uniform need to be truncated if the sample is outside of the provided bandwidth:
        if self.function in ["bisquare", "uniform"]:
            self.kernel[bw_dist > 1] = 0

        # Set density to zero if below threshold:
        self.kernel[self.kernel < threshold] = 0

        # Normalize the kernel by the number of non-zero neighbors, if applicable:
        if normalize_weights:
            self.kernel = self.kernel / np.count_nonzero(self.kernel)

        if sparse_array:
            self.kernel = scipy.sparse.csr_matrix(self.kernel)

    def _kernel_functions(self, x):
        if self.function == "triangular":
            return 1 - x
        elif self.function == "uniform":
            return np.ones(x.shape) * 0.5
        elif self.function == "quadratic":
            return (3.0 / 4) * (1 - x**2)
        # elif self.function == "bisquare":
        #     return (15.0 / 16) * (1 - x**2) ** 2
        elif self.function == "bisquare":
            return (1 - (x) ** 2) ** 2
        elif self.function == "gaussian":
            return np.exp(-0.5 * (x) ** 2)
        elif self.function == "exponential":
            return np.exp(-x)
        else:
            raise ValueError(
                f'Unsupported kernel function. Valid options: "triangular", "uniform", "quadratic", '
                f'"bisquare", "gaussian" or "exponential". Got {self.function}.'
            )


def get_wi(
    i: int,
    n_samples: int,
    coords: np.ndarray,
    cov: Optional[np.ndarray] = None,
    expr_mat: Optional[np.ndarray] = None,
    fixed_bw: bool = True,
    exclude_self: bool = False,
    kernel: str = "gaussian",
    bw: Union[float, int] = 100,
    threshold: float = 1e-5,
    sparse_array: bool = False,
    normalize_weights: bool = False,
    use_expression_neighbors_only: bool = False,
) -> scipy.sparse.csr_matrix:
    """Get spatial weights for an individual sample, given the coordinates of all samples in space.

    Args:
        i: Index of sample for which weights are to be calculated to all other samples in the dataset
        n_samples: Total number of samples in the dataset
        coords: Array of shape (n_samples, 2) or (n_samples, 3) representing the spatial coordinates of each sample
        cov: Optional array of shape (n_samples, ). Can be used to adjust the distance calculation to look only at
            samples of interest, which is determined from nonzero values in this vector. This can be, e.g. a one-hot
            vector for a particular cell type.
        expr_mat: Can be used together with 'cov'- if the spatial neighbors are not consistent with the sample in
            question (determined by assessing similarity by "cov"), there may be different mechanisms at play. In this
            case, will instead search for nearest neighbors in the gene expression space if given.
        fixed_bw: If True, `bw` is treated as a spatial distance for computing spatial weights. Otherwise,
            it is treated as the number of neighbors.
        exclude_self: If True, ignore each sample itself when computing the kernel density estimation
        kernel: The name of the kernel function to use. Valid options: "triangular", "uniform", "quadratic",
            "bisquare", "gaussian" or "exponential"
        bw: Bandwidth for the spatial kernel
        threshold: Threshold for the kernel density estimation. If the density is below this threshold, the density
            will be set to zero.
        sparse_array: If True, the kernel will be converted to sparse array. Recommended for large datasets.
        normalize_weights: If True, the weights will be normalized to sum to 1.
        use_expression_neighbors_only: If True, will only use expression neighbors to determine the bandwidth.

    Returns:
        wi: Array of weights for sample of interest
    """

    if bw == np.inf:
        wi = np.ones(n_samples)
        return wi

    wi = Kernel(
        i,
        coords,
        bw,
        cov=cov,
        expr_mat=expr_mat,
        fixed=fixed_bw,
        exclude_self=exclude_self,
        function=kernel,
        threshold=threshold,
        sparse_array=sparse_array,
        normalize_weights=normalize_weights,
        use_expression_neighbors_only=use_expression_neighbors_only,
    ).kernel

    return wi


# ---------------------------------------------------------------------------------------------------
# Construct nearest neighbor graphs
# ---------------------------------------------------------------------------------------------------
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def construct_nn_graph(
    adata: AnnData,
    spatial_key: str = "spatial",
    dist_metric: str = "euclidean",
    n_neighbors: int = 8,
    exclude_self: bool = True,
    make_symmetrical: bool = False,
    save_id: Union[None, str] = None,
) -> None:
    """Constructing bucket-to-bucket nearest neighbors graph.

    Args:
        adata: An anndata object.
        spatial_key: Key in .obsm in which x- and y-coordinates are stored.
        dist_metric: Distance metric to use. Options: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
            ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’,
            ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
            ‘sqeuclidean’, ‘yule’.
        n_neighbors: Number of nearest neighbors to compute for each bucket.
        exclude_self: Set True to set elements along the diagonal to zero.
        make_symmetrical: Set True to make sure adjacency matrix is symmetrical (i.e. ensure that if A is a neighbor
            of B, B is also included among the neighbors of A)
        save_id: Optional string; if not None, will save distance matrix and neighbors matrix to path:
        './neighbors/{save_id}_distance.csv' and path: './neighbors/{save_id}_neighbors.csv', respectively.
    """
    position = adata.obsm[spatial_key]
    # calculate distance matrix
    distance_matrix = calculate_distance(position, dist_metric)
    n_bucket = distance_matrix.shape[0]

    adata.obsp["distance_matrix"] = distance_matrix

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

    # transform adj to symmetrical adj
    adj = interaction

    if make_symmetrical:
        adj = adj + adj.T
        adj = np.where(adj > 1, 1, adj)

    if exclude_self:
        np.fill_diagonal(adj, 0)

    adata.obsp["adj"] = scipy.sparse.csr_matrix(adj)


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def neighbors(
    adata: AnnData,
    nbr_object: NearestNeighbors = None,
    basis: str = "pca",
    spatial_key: str = "spatial",
    n_neighbors_method: str = "ball_tree",
    n_pca_components: int = 30,
    n_neighbors: int = 10,
) -> Tuple[NearestNeighbors, AnnData]:
    """Given an AnnData object, compute pairwise connectivity matrix in transcriptomic space

    Args:
        adata : an anndata object.
        nbr_object: An optional sklearn.neighbors.NearestNeighbors object. Can optionally create a nearest neighbor
            object with custom functionality.
        basis: str, default 'pca'
            The space that will be used for nearest neighbor search. Valid names includes, for example, `pca`, `umap`,
            or `X` for gene expression neighbors, 'spatial' for neighbors in the physical space.
        spatial_key: Optional, can be used to specify .obsm entry in adata that contains spatial coordinates. Only
            used if basis is 'spatial'.
        n_neighbors_method: str, default 'ball_tree'
            Specifies algorithm to use in computing neighbors using sklearn's implementation. Options:
            "ball_tree" and "kd_tree".
        n_pca_components: Only used if 'basis' is 'pca'. Sets number of principal components to compute (if PCA has
            not already been computed for this dataset).
        n_neighbors: Number of neighbors for kneighbors queries.

    Returns:
        nbrs : Object of class `sklearn.neighbors.NearestNeighbors`
        adata : Modified AnnData object
    """
    logger = lm.get_main_logger()

    if basis == "pca" and "X_pca" not in adata.obsm_keys():
        logger.info(
            "PCA to be used as basis for :func `transcriptomic_connectivity`, X_pca not found, " "computing PCA...",
            indent_level=2,
        )
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
    elif basis == "spatial":
        X_data = adata.obsm[spatial_key]
    elif "X_" + basis in adata.obsm_keys():
        # Assume basis can be found in .obs under "X_{basis}":
        X_data = adata.obsm["X_" + basis]
    else:
        raise ValueError("Invalid option given to 'basis'. Options: 'pca', 'umap', 'spatial' or 'X'.")

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm=n_neighbors_method, n_neighbors=n_neighbors, metric="euclidean").fit(X_data)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    # Update AnnData to add spatial distances, spatial connectivities and spatial neighbors from the sklearn
    # NearestNeighbors run:
    distances, knn = nbrs.kneighbors(X_data)
    distances, connectivities = compute_distances_and_connectivities(knn, distances)

    if basis != "spatial":
        logger.info_insert_adata("expression_connectivities", adata_attr="obsp")
        logger.info_insert_adata("expression_distances", adata_attr="obsp")

        adata.obsp["expression_distances"] = distances
        adata.obsp["expression_connectivities"] = connectivities
    else:
        logger.info_insert_adata("spatial_distances", adata_attr="obsp")
        logger.info_insert_adata("spatial_connectivities", adata_attr="obsp")

        adata.obsp["spatial_distances"] = distances
        adata.obsp["spatial_connectivities"] = connectivities

    return nbrs, adata


# ---------------------------------------------------------------------------------------------------
# Compute affinity matrix
# ---------------------------------------------------------------------------------------------------
def calculate_affinity(position: np.ndarray, dist_metric: str = "euclidean", n_neighbors: int = 10) -> np.ndarray:
    """Given array of x- and y-coordinates, compute affinity matrix between all samples using Euclidean distance.
    Math from: Zelnik-Manor, L., & Perona, P. (2004). Self-tuning spectral clustering.
    Advances in neural information processing systems, 17.
    https://proceedings.neurips.cc/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf
    """
    # Calculate euclidian distance matrix
    dists = squareform(pdist(position, metric=dist_metric))

    # For each row, sort the distances in ascending order and take the index of the n-th position (nearest neighbors)
    knn_distances = np.sort(dists, axis=0)[n_neighbors]
    knn_distances = knn_distances[np.newaxis].T

    # Calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale

    # Divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # Apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix
