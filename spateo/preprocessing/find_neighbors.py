"""
Functions for finding nearest neighbors and the distances between them in spatial transcriptomics data.
"""
from typing import Optional, Union
import anndata
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from torch import Tensor, FloatTensor

from ..configuration import SKM



# ------------------------------------------ Cell-cell/spot-spot distance ------------------------------------------ #
def calculate_distance(position: np.ndarray) -> np.ndarray:
    """Given array of x- and y-coordinates, compute pairwise distances between all samples"""
    n_spot = position.shape[0]
    distance_matrix = np.zeros([n_spot, n_spot])

    for i in range(n_spot):
        x = position[i, :]
        for j in range(i + 1, n_spot):
            y = position[j, :]
            d = np.sqrt(np.sum(np.square(x - y)))
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    return distance_matrix


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def construct_pairwise(adata: anndata.AnnData,
                       spatial_key: str = 'spatial',
                       n_neighbors: int = 3):
    """
    Constructing spot-to-spot nearest neighbors graph

    Args:
        adata : class `anndata.AnnData`
        spatial_key : str, default 'spatial'
            Key in .obsm in which x- and y-coordinates are stored
        n_neighbors : int, default 3
            Number of nearest neighbors to compute for each spot
    """
    position = adata.obsm['spatial']
    # calculate distance matrix
    distance_matrix = calculate_distance(position)
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj


def normalize_adj(adj: np.ndarray,
                  exclude_self: bool = True) -> np.ndarray:
    """
    Symmetrically normalize adjacency matrix, set diagonal to 1 and return processed adjacency array.

    Args:
        adj : np.ndarray, shape [n_samples, n_samples]
            Pairwise distance matrix
        exclude_self : bool, default True
            Set True to set diagonal of adjacency matrix to 1
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_proc = adj.toarray() + np.eye(adj.shape[0])
    return adj_proc


# Constructing weighted distance matrices:
def generate_spatial_distance_graph(locations: np.ndarray,
                                    nbr_object: NearestNeighbors = None,
                                    num_neighbors: Union[None, int] = None,
                                    radius: Union[None, float] = None,
                                    ) -> csr_matrix:
    """
    Creates graph based on distance in space.

    Parameters
    ----------
    locations : np.ndarray, shape [n_samples, 2]
        Spatial coordinates for each spot
    nbr_object : optional sklearn.neighbors.NearestNeighbors object
        Can optionally create a nearest neighbor object with custom functionality
    num_neighbors : optional int
        Number of neighbors for each spot
    radius : optional float
        Search radius around each spot
    """
    assert num_neighbors is not None or radius is not None, \
        "Number of neighbors or search radius for each spot must be provided."

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm='ball_tree').fit(locations)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    if num_neighbors is None:
        # no limit to number of neighbours
        return nbrs.radius_neighbors_graph(radius=radius, mode="distance")

    else:
        assert isinstance(num_neighbors, int), (
            f"Number of neighbors {num_neighbors} is not an integer"
        )

        graph_out = nbrs.kneighbors_graph(n_neighbors=num_neighbors, mode="distance")

        if radius is not None:
            assert isinstance(radius, (float, int)), (
                f"Radius {radius} is not an integer or float"
            )

            graph_out = remove_greater_than(graph_out, radius, copy=False, verbose=False)

        return graph_out





