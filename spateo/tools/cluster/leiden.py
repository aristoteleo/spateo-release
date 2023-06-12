from typing import Literal

import igraph
import leidenalg
import numpy as np
from sklearn.neighbors import kneighbors_graph


def distance_knn_graph(dist: np.ndarray, num_neighbors: int):
    """Construct a k-nearest neighbor graph from a distance matrix.

    Args:
        dist: Pairwise distance matrix
        num_neighbors: Number of nearest neighbors

    Returns:
        G: K-nearest neighbor graph
    """
    n = dist.shape[0]
    G = igraph.Graph()
    G.add_vertices(n)
    edges = []
    weights = []
    for i in range(n):
        sorted_ind = np.argsort(dist[i, :])
        for j in range(1, 1 + num_neighbors):
            # if i < sorted_ind[j]:
            edges.append((i, sorted_ind[j]))
            weights.append(dist[i, sorted_ind[j]])
            # weights.append(1.)
    G.add_edges(edges)
    G.es["weight"] = weights
    return G


def embedding_knn_graph(X: np.ndarray, num_neighbors: int):
    """Construct a k-nearest neighbor graph from an arbitrary array, of shape [n_samples, n_features]

    Args:
        X: Embedding matrix
        num_neighbors: Number of nearest neighbors

    Returns:
        G: K-nearest neighbor graph
    """
    adj = kneighbors_graph(X, num_neighbors, include_self=False, mode="distance")
    G = igraph.Graph.Weighted_Adjacency(adj)
    return G


def calculate_leiden_partition(
    input_mat: np.ndarray,
    num_neighbors: int,
    resolution: float = 1.0,
    n_iterations: int = -1,
    graph_type: Literal["distance", "embedding"] = "distance",
):
    """Performs Leiden clustering on a given dataset.

    Args:
        input_mat: The input data, will be interepreted as either a distance matrix (if :param `graph_type` is
            "distance" or an embedding matrix (if :param `graph_type` is "embedding")
        num_neighbors: The number of nearest neighbors for constructing the graph
        resolution: The resolution parameter for the Leiden algorithm
        n_iterations: The number of iterations for the Leiden algorithm (-1 for unlimited iterations)
        graph_type: Specifies the input type, either 'distance' or 'embedding'

    Returns:
        clusters: Array containing cluster assignments for each data point
    """
    if graph_type == "distance":
        G = distance_knn_graph(input_mat, num_neighbors)
    elif graph_type == "embedding":
        G = embedding_knn_graph(input_mat, num_neighbors)
    partition_kwargs = {"resolution_parameter": resolution, "seed": 888, "n_iterations": n_iterations}

    partition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, **partition_kwargs)
    clusters = np.array(partition.membership, dtype=int)
    return clusters
