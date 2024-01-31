"""
Functions for finding nearest neighbors, the distances between them and the spatial weighting between points in
spatial transcriptomics data.
"""
import os
import sys
from functools import partial
from typing import Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
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


def jaccard_index(row_i: np.ndarray, array: np.ndarray):
    """Compute the Jaccard index between a row of a binary array and all other rows.

    Args:
        row_i: 1D binary array representing the row for which to compute the Jaccard index.
        array: 2D binary array containing the rows to compare against.

    Returns:
        jaccard_indices: 1D array of Jaccard indices between `row_i` and each row in `array`.
    """
    intersect = np.logical_and(row_i, array)
    union = np.logical_or(row_i, array)
    jaccard_scores = np.sum(intersect, axis=1) / np.sum(union, axis=1)
    return jaccard_scores


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
        current_neighbors = adj[i, :].nonzero()
        # Set self as nearest neighbor
        indices[i, :] = i
        weights[i, :] = 0.0

        # there could be more or less than n_neighbors because of an approximate search
        current_n_neighbors = len(current_neighbors[1])

        if current_n_neighbors > n_neighbors - 1:
            sorted_indices = np.argsort(adj[i][:, current_neighbors[1]].A)[0][: (n_neighbors - 1)]
            indices[i, 1:] = current_neighbors[1][sorted_indices]
            weights[i, 1:] = adj[i][0, current_neighbors[1][sorted_indices]].A
        else:
            idx_ = np.arange(1, (current_n_neighbors + 1))
            indices[i, idx_] = current_neighbors[1]
            weights[i, idx_] = adj[i][:, current_neighbors[1]].A

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


def calculate_distances_chunk(
    coords_chunk: np.ndarray,
    chunk_start_idx: int,
    coords: np.ndarray,
    n_nonzeros: Optional[dict] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Pairwise distance computation, coupled with :func `find_bw`.

    Args:
        coords_chunk: Array of shape (n_samples_chunk, n_features) containing coordinates of the chunk of interest.
        chunk_start_idx: Index of the first sample in the chunk. Required if `n_nonzeros` is not None.
        coords: Array of shape (n_samples, n_features) containing the coordinates of all points.
        n_nonzeros: Optional dictionary containing the number of non-zero columns for each row in the distance matrix.
        metric: Distance metric to use for pairwise distance computation, can be any of the metrics supported by
            :func `sklearn.metrics.pairwise_distances`.
    """
    distances_chunk = pairwise_distances(coords_chunk, coords, metric=metric)

    # If n_nonzeros is not None, find the number of columns that are nonzero across both rows:
    if n_nonzeros is not None:
        # Normalization factors:
        paired_nonzeros = np.zeros_like(distances_chunk)
        for i in range(distances_chunk.shape[0]):
            for j in range(distances_chunk.shape[1]):
                paired_nonzeros[i, j] = len(n_nonzeros[chunk_start_idx + i] & n_nonzeros[j])
        normalized_chunk = distances_chunk / paired_nonzeros
        distances_chunk = normalized_chunk

    return distances_chunk


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def find_bw_for_n_neighbors(
    adata: anndata.AnnData,
    coords_key: str = "spatial",
    n_anchors: Optional[int] = None,
    target_n_neighbors: int = 6,
    initial_bw: Optional[float] = None,
    chunk_size: int = 1000,
    exclude_self: bool = False,
    normalize_distances: bool = False,
    verbose: bool = True,
    max_iterations: int = 100,
    alpha: float = 0.5,
) -> float:
    """Finds the bandwidth such that on average, cells in the sample have n neighbors.

    Args:
        adata: AnnData object containing coordinates for all cells
        coords_key: Key in adata.obsm where the spatial coordinates are stored
        target_n_neighbors: Target average number of neighbors per cell
        initial_bw: Can optionally be used to set the starting distance for the bandwidth search
        chunk_size: Number of cells to compute pairwise distance for at once
        exclude_self: Whether to exclude self from the list of neighbors
        normalize_distances: Whether to normalize the distances by the number of nonzero columns (should be used only
            if the entry in .obs[coords_key] contains something other than x-, y-, z-coordinates).
        verbose: Whether to print the bandwidth at each iteration. If False, will only print the final bandwidth.
        max_iterations: Will stop the process and return the bandwidth that results in the closest number of neighbors
            to the specified target if it takes more than this number of iterations.
        alpha: Factor used in determining the new bandwidth- ratio of found neighbors to target neighbors will be
            raised to this power.

    Returns:
        bandwidth: Bandwidth in distance units
    """
    coords = adata.obsm[coords_key]
    # Select n_anchors random indices if applicable:
    if n_anchors is not None:
        np.random.seed(0)  # Seed for reproducibility
        anchor_indices = np.random.choice(coords.shape[0], size=n_anchors, replace=False)
        anchor_coords = coords[anchor_indices]
        chunk_size = min(chunk_size, anchor_coords.shape[0])
    else:
        anchor_indices = np.arange(coords.shape[0])
        anchor_coords = coords

    metric = "jaccard" if "jaccard" in coords_key else "euclidean"

    # If normalize_distances is True, get the indices of nonzero columns for each row in the distance matrix- only
    # used if metric is Euclidean distance:
    if normalize_distances and metric == "euclidean":
        n_nonzeros = {}
        for i in range(coords.shape[0]):
            n_nonzeros[i] = set(np.nonzero(coords[i, :])[0])
    else:
        n_nonzeros = None

    # Compute distances in chunks, include start and end indices:
    chunks_with_indices = [
        (anchor_coords[i : i + chunk_size], anchor_indices[i]) for i in range(0, anchor_coords.shape[0], chunk_size)
    ]
    # Calculate pairwise distances for each chunk in parallel
    if metric == "jaccard":
        partial_func = partial(calculate_distances_chunk, coords=coords, metric=metric)
    else:
        partial_func = partial(calculate_distances_chunk, coords=coords, n_nonzeros=n_nonzeros, metric=metric)
    distances = Parallel(n_jobs=-1)(delayed(partial_func)(chunk, start_idx) for chunk, start_idx in chunks_with_indices)
    # Concatenate the results to get the full pairwise distance matrix
    distances = np.concatenate(distances, axis=0)

    # Initialize bandwidth and iteration counter:
    bandwidth = 88 if initial_bw is None else initial_bw
    iteration = 0
    if verbose:
        print(f"Initial bandwidth: {bandwidth}")

    closest_bw = bandwidth
    lower_bound = 0.9 * target_n_neighbors
    upper_bound = 1.1 * target_n_neighbors

    closest_avg_neighbors = float("inf")

    while iteration < max_iterations:
        iteration += 1
        bw_dist = distances / bandwidth
        if exclude_self:
            neighbor_counts = np.sum(bw_dist <= 1, axis=1) - 1
        else:
            neighbor_counts = np.sum(bw_dist <= 1, axis=1)

        # Check if the average number of neighbors is close to the target
        avg_neighbors = np.mean(neighbor_counts)
        if verbose:
            print(f"For bandwidth {bandwidth}, found {avg_neighbors} neighbors on average.")

        # Store the closest bandwidth and closest average neighbors:
        if abs(target_n_neighbors - closest_avg_neighbors) > abs(target_n_neighbors - avg_neighbors):
            closest_bw = bandwidth
            closest_avg_neighbors = avg_neighbors

        # Check for exit condition
        if lower_bound <= avg_neighbors <= upper_bound:
            if verbose:
                print(f"Final bandwidth: {bandwidth}")
            return bandwidth

        # Dynamic adjustment of bandwidth
        adjustment_factor = target_n_neighbors / avg_neighbors
        bandwidth *= adjustment_factor**alpha

        if verbose:
            print(f"Iteration {iteration}, new bandwidth: {bandwidth}")

    if verbose:
        print(
            f"Max iterations reached. Returning closest bandwidth: {closest_bw}, with average neighbors: "
            f"{closest_avg_neighbors}"
        )

    return closest_bw


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def find_threshold_distance(
    adata: anndata.AnnData,
    coords_key: str = "X_pca",
    n_neighbors: int = 10,
    chunk_size: int = 1000,
    normalize_distances: bool = False,
) -> float:
    """Finds threshold distance beyond which there is a dramatic increase in the average distance to remaining
    nearest neighbors.

    Args:
        adata: AnnData object containing coordinates for all cells
        coords_key: Key in adata.obsm where the spatial coordinates are stored
        n_neighbors: Will first compute the number of nearest neighbors as a comparison for querying additional
            distance values.
        chunk_size: Number of cells to compute pairwise distance for at once
        normalize_distances: Whether to normalize the distances by the number of nonzero columns (should be used only
            if the entry in .obs[coords_key] contains something other than x-, y-, z-coordinates).

    Returns:
        bandwidth: Bandwidth in distance units
    """
    coords = adata.obsm[coords_key]

    # If normalize_distances is True, get the indices of nonzero columns for each row in the distance matrix:
    if normalize_distances:
        n_nonzeros = {}
        for i in range(coords.shape[0]):
            n_nonzeros[i] = set(np.nonzero(coords[i, :])[0])
    else:
        n_nonzeros = None

    # Compute distances in chunks, include start and end indices:
    chunks_with_indices = [(coords[i : i + chunk_size], i) for i in range(0, coords.shape[0], chunk_size)]
    # Calculate pairwise distances for each chunk in parallel
    partial_func = partial(calculate_distances_chunk, coords=coords, n_nonzeros=n_nonzeros)
    distances = Parallel(n_jobs=-1)(delayed(partial_func)(chunk, start_idx) for chunk, start_idx in chunks_with_indices)
    # Concatenate the results to get the full pairwise distance matrix
    distances = np.concatenate(distances, axis=0)

    # Find the k nearest neighbors for each sample
    k_nearest_distances = np.sort(distances)[:, :n_neighbors]

    # Compute the mean and standard deviation of the k nearest distances
    mean_k_distances = np.mean(k_nearest_distances, axis=1)
    std_k_distances = np.std(k_nearest_distances, axis=1)

    # Find the distance where there is a dramatic increase compared to the k nearest neighbors
    threshold_distance = np.max(mean_k_distances + 3 * std_k_distances)

    return threshold_distance


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
            samples of interest vs. samples not of interest, which is determined from nonzero values in this vector.
            This can be used to modify the modeling process based on factors thought to reflect biological differences,
            for example, to condition on histological classification, passing a distance threshold, etc. If 'ct' is
            also given, will look for samples of interest that are also of the same cell type.
        ct: Optional array of shape (n_samples, ), containing vector where cell types are encoded as integers. Can be
            used to condition nearest neighbor finding on cell type or other category.
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
        use_expression_neighbors: If True, will only use the expression matrix to find nearest neighbors.
    """

    def __init__(
        self,
        i: int,
        data: Union[np.ndarray, scipy.sparse.spmatrix],
        bw: Union[int, float],
        cov: Optional[np.ndarray] = None,
        ct: Optional[np.ndarray] = None,
        expr_mat: Optional[np.ndarray] = None,
        fixed: bool = True,
        exclude_self: bool = False,
        function: str = "triangular",
        threshold: float = 1e-5,
        eps: float = 1.0000001,
        sparse_array: bool = False,
        normalize_weights: bool = False,
        use_expression_neighbors: bool = False,
    ):
        if use_expression_neighbors:
            self.dist_vector = local_dist(expr_mat[i], expr_mat).reshape(-1)
            self.function = "uniform"
        else:
            self.dist_vector = local_dist(data[i], data).reshape(-1)
            self.function = function.lower()

        if fixed:
            self.bandwidth = float(bw)
        else:
            if exclude_self:
                self.bandwidth = np.partition(self.dist_vector, int(bw) + 1)[int(bw) + 1] * eps
            else:
                self.bandwidth = np.partition(self.dist_vector, int(bw))[int(bw)] * eps

        max_dist = np.max(self.dist_vector)
        if cov is not None and ct is not None:
            # If condition is met, compare to samples of the same cell type:
            if cov[i] == 1:
                self.dist_vector[ct != ct[i]] = max_dist
        elif cov is not None and ct is None:
            # Ignore samples that do not meet the condition:
            self.dist_vector[cov == 0] = max_dist
        elif ct is not None:
            # Compare to samples of the same cell type:
            self.dist_vector[ct != ct[i]] = max_dist

        bw_dist = self.dist_vector / self.bandwidth
        # Exclude self as a neighbor:
        if exclude_self:
            bw_dist = np.where(bw_dist == 0.0, np.max(bw_dist), bw_dist)
        self.kernel = self._kernel_functions(bw_dist)

        # Bisquare and uniform need to be truncated if the sample is outside of the provided bandwidth:
        self.kernel[bw_dist > 1] = 0

        # Set density to zero if below threshold:
        self.kernel[self.kernel < threshold] = 0
        n_neighbors = np.count_nonzero(self.kernel)

        # Normalize the kernel by the number of non-zero neighbors, if applicable:
        if normalize_weights:
            self.kernel = self.kernel / n_neighbors

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
    ct: Optional[np.ndarray] = None,
    expr_mat: Optional[np.ndarray] = None,
    fixed_bw: bool = True,
    exclude_self: bool = False,
    kernel: str = "gaussian",
    bw: Union[float, int] = 100,
    threshold: float = 1e-5,
    sparse_array: bool = False,
    normalize_weights: bool = False,
    use_expression_neighbors: bool = False,
) -> scipy.sparse.csr_matrix:
    """Get spatial weights for an individual sample, given the coordinates of all samples in space.

    Args:
        i: Index of sample for which weights are to be calculated to all other samples in the dataset
        n_samples: Total number of samples in the dataset
        coords: Array of shape (n_samples, 2) or (n_samples, 3) representing the spatial coordinates of each sample
        cov: Optional array of shape (n_samples, ). Can be used to adjust the distance calculation to look only at
            samples of interest vs. samples not of interest, which is determined from nonzero values in this vector.
            This can be used to modify the modeling process based on factors thought to reflect biological differences,
            for example, to condition on histological classification, passing a distance threshold, etc. If 'ct' is
            also given, will look for samples of interest that are also of the same cell type.
        ct: Optional array of shape (n_samples, ), containing vector where cell types are encoded as integers. Can be
            used to condition nearest neighbor finding on cell type or other category.
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
        use_expression_neighbors: If True, will only use expression neighbors to determine the bandwidth.

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
        ct=ct,
        expr_mat=expr_mat,
        fixed=fixed_bw,
        exclude_self=exclude_self,
        function=kernel,
        threshold=threshold,
        sparse_array=sparse_array,
        normalize_weights=normalize_weights,
        use_expression_neighbors=use_expression_neighbors,
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
    """Given an AnnData object, compute pairwise connectivity matrix in transcriptomic or physical space

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
        logger.info_insert_adata("expression_connectivities.indices", adata_attr="uns")
        logger.info_insert_adata("expression_connectivities.params", adata_attr="uns")

        adata.obsp["expression_distances"] = distances
        adata.obsp["expression_connectivities"] = connectivities
        adata.uns["expression_connectivities"]["indices"] = knn
        adata.uns["expression_connectivities"]["params"] = {"n_neighbors": n_neighbors, "metric": "euclidean"}
    else:
        logger.info_insert_adata("spatial_distances", adata_attr="obsp")
        logger.info_insert_adata("spatial_connectivities", adata_attr="obsp")
        logger.info_insert_adata("spatial_connectivities.indices", adata_attr="uns")
        logger.info_insert_adata("spatial_connectivities.params", adata_attr="uns")

        adata.obsp["spatial_distances"] = distances
        adata.obsp["spatial_connectivities"] = connectivities
        adata.uns["spatial_connectivities"]["indices"] = knn
        adata.uns["spatial_connectivities"]["params"] = {"n_neighbors": n_neighbors, "metric": "euclidean"}

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
