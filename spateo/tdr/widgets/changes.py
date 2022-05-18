from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import DataSet, MultiBlock, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .ddrtree import DDRTree, cal_ncenter
from .slice import euclidean_distance, three_d_slice

####################################
# Changes along a vector direction #
####################################


def changes_along_line(
    model: Union[PolyData, UnstructuredGrid],
    key: Union[str, list] = None,
    n_points: int = 100,
    vec: Union[tuple, list] = (1, 0, 0),
    center: Union[tuple, list] = None,
) -> Tuple[np.ndarray, np.ndarray, MultiBlock, MultiBlock]:
    slices, line_points, line = three_d_slice(model=model, method="line", n_slices=n_points, vec=vec, center=center)

    x, y = [], []
    x_length = 0
    for slice, (point_i, point) in zip(slices, enumerate(line_points)):
        change_value = np.asarray(slice[key]).sum()
        y.append(change_value)

        if point_i == 0:
            x.append(0)
        else:
            point1 = line_points[point_i - 1].points.flatten()
            point2 = line_points[point_i].points.flatten()

            ed = euclidean_distance(instance1=point1, instance2=point2, dimension=3)

            x_length += ed
            x.append(x_length)

    return np.asarray(x), np.asarray(y), slices, line


#################################
# Changes along the model shape #
#################################


def changes_along_shape(
    model: Union[PolyData, UnstructuredGrid],
    spatial_key: Optional[str] = None,
    key_added: Optional[str] = "rd_spatial",
    dim: int = 2,
    inplace: bool = False,
    **kwargs,
):
    model = model.copy() if not inplace else model
    X = model.points if spatial_key is None else model[spatial_key]

    DDRTree_kwargs = {
        "maxIter": 10,
        "sigma": 0.001,
        "gamma": 10,
        "eps": 0,
        "dim": dim,
        "Lambda": 5 * X.shape[1],
        "ncenter": cal_ncenter(X.shape[1]),
    }
    DDRTree_kwargs.update(kwargs)
    Z, Y, stree, R, W, Q, C, objs = DDRTree(X, **DDRTree_kwargs)

    # Obtain the real part of the complex argument
    model[key_added] = np.real(W).astype(np.float64)

    return model if not inplace else None


##############################
# Changes along the branches #
##############################


def ElPiGraph_tree(
    X: np.ndarray,
    NumNodes: int = 50,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a principal elastic tree.
    Reference: Albergante et al. (2020), Robust and Scalable Learning of Complex Intrinsic Dataset Geometry via ElPiGraph.

    Args:
        X: DxN, data matrix list.
        NumNodes: The number of nodes of the principal graph. Use a range of 10 to 100 for ElPiGraph approach.
        **kwargs: Other parameters used in elpigraph.computeElasticPrincipalTree. For details, please see:
                  https://github.com/j-bac/elpigraph-python/blob/master/elpigraph/_topologies.py

    Returns:
        nodes: The nodes in the principal tree.
        edges: The edges between nodes in the principal tree.
    """
    try:
        import elpigraph
    except ImportError:
        raise ImportError(
            "You need to install the package `elpigraph-python`."
            "\nInstall elpigraph-python via `pip install git+https://github.com/j-bac/elpigraph-python.git`."
        )

    ElPiGraph_kwargs = {
        "alpha": 0.01,
        "FinalEnergy": "Penalized",
        "StoreGraphEvolution": True,
        "GPU": False,
    }
    ElPiGraph_kwargs.update(kwargs)
    if ElPiGraph_kwargs["GPU"] is True:
        try:
            import cupy
        except ImportError:
            raise ImportError(
                "You need to install the package `cupy`." "\nInstall cupy via `pip install cupy-cuda113`."
            )

    elpi_tree = elpigraph.computeElasticPrincipalTree(X=np.asarray(X), NumNodes=NumNodes, **ElPiGraph_kwargs)

    nodes = elpi_tree[0]["NodePositions"]  # ['AllNodePositions'][k]
    matrix_edges_weights = elpi_tree[0]["ElasticMatrix"]  # ['AllElasticMatrices'][k]
    matrix_edges_weights = np.triu(matrix_edges_weights, 1)
    edges = np.array(np.nonzero(matrix_edges_weights), dtype=int).transpose()

    return nodes, edges


def SimplePPT_tree(
    X: np.ndarray,
    NumNodes: int = 50,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple principal tree.
    Reference: Mao et al. (2015), SimplePPT: A simple principal tree algorithm, SIAM International Conference on Data Mining.

    Args:
        X: DxN, data matrix list.
        NumNodes: The number of nodes of the principal graph. Use a range of 100 to 2000 for PPT approach.
        **kwargs: Other parameters used in simpleppt.ppt. For details, please see:
                  https://github.com/LouisFaure/simpleppt/blob/main/simpleppt/ppt.py

    Returns:
        nodes: The nodes in the principal tree.
        edges: The edges between nodes in the principal tree.
    """
    try:
        import igraph
        import simpleppt
    except ImportError:
        raise ImportError(
            "You need to install the package `simpleppt` and `igraph`."
            "\nInstall simpleppt via `pip install -U simpleppt`."
            "\nInstall igraph via `pip install -U igraph`"
        )

    SimplePPT_kwargs = {
        "seed": 1,
        "lam": 10,
    }
    SimplePPT_kwargs.update(kwargs)

    X = np.asarray(X)
    ppt_tree = simpleppt.ppt(X=X, Nodes=NumNodes, **SimplePPT_kwargs)

    R = ppt_tree.R
    nodes = (np.dot(X.T, R) / R.sum(axis=0)).T

    B = ppt_tree.B
    edges = np.array(igraph.Graph.Adjacency((B > 0).tolist(), mode="undirected").get_edgelist())

    return nodes, edges


def map_points_to_branch(
    model: Union[PolyData, UnstructuredGrid],
    nodes: np.ndarray,
    spatial_key: Optional[str] = None,
    key_added: Optional[str] = "nodes",
    inplace: bool = False,
    **kwargs,
):
    """
    Find the closest principal tree node to any point in the model through KDTree.

    Args:
        model: A reconstruct model.
        nodes: The nodes in the principal tree.
        spatial_key: The key that corresponds to the coordinates of the point in the model. If spatial_key is None,
                     the coordinates are model.points.
        key_added: The key under which to add the nodes labels.
        inplace: Updates model in-place.
        kwargs: Other parameters used in scipy.spatial.KDTree.

    Returns:
        A model, which contains the following properties:
            `model.point_data[key_added]`, the nodes labels array.
    """
    from scipy.spatial import KDTree

    model = model.copy() if not inplace else model
    X = model.points if spatial_key is None else model[spatial_key]

    nodes_kdtree = KDTree(np.asarray(nodes), **kwargs)
    _, ii = nodes_kdtree.query(np.asarray(X), k=1)
    model.point_data[key_added] = ii

    return model if not inplace else None


def map_gene_to_branch(
    model: Union[PolyData, UnstructuredGrid],
    tree: PolyData,
    key: Union[str, list],
    nodes_key: Optional[str] = "nodes",
    inplace: bool = False,
):
    """
    Find the closest principal tree node to any point in the model through KDTree.

    Args:
        model: A reconstruct model contains the gene expression label.
        tree: A three-dims principal tree model contains the nodes label.
        key: The key that corresponds to the gene expression.
        nodes_key: The key that corresponds to the coordinates of the nodes in the tree.
        inplace: Updates tree model in-place.

    Returns:
        A tree, which contains the following properties:
            `tree.point_data[key]`, the gene expression array.
    """
    model = model.copy()

    model_data = pd.DataFrame(model[nodes_key], columns=["nodes_id"])
    key = [key] if isinstance(key, str) else key
    for sub_key in key:
        model_data[sub_key] = np.asarray(model[sub_key])
    model_data = model_data.groupby(by="nodes_id").sum()
    model_data["nodes_id"] = model_data.index
    model_data.index = range(len(model_data.index))

    tree = tree.copy() if not inplace else tree

    tree_data = pd.DataFrame(tree[nodes_key], columns=["nodes_id"])
    tree_data = pd.merge(tree_data, model_data, how="outer", on="nodes_id")
    tree_data.fillna(value=0, inplace=True)
    for sub_key in key:
        tree.point_data[sub_key] = tree_data[sub_key].values

    return tree if not inplace else None


def construct_tree_model(
    nodes: np.ndarray,
    edges: np.ndarray,
    key_added: Optional[str] = "nodes",
) -> PolyData:
    """
    Construct a principal tree model.

    Args:
        nodes: The nodes in the principal tree.
        edges: The edges between nodes in the principal tree.
        key_added: The key under which to add the nodes labels.

    Returns:
         A three-dims principal tree model, which contains the following properties:
            `tree_model.point_data[key_added]`, the nodes labels array.
    """

    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T
    tree_model = pv.PolyData(nodes, edges_w_padding)
    tree_model.point_data[key_added] = np.arange(0, len(nodes), 1)

    return tree_model


def changes_along_branch(
    model: Union[PolyData, UnstructuredGrid],
    spatial_key: Optional[str] = None,
    map_key: Union[str, list] = None,
    key_added: Optional[str] = "nodes",
    rd_method: Literal["ElPiGraph", "SimplePPT"] = "ElPiGraph",
    NumNodes: int = 50,
    inplace: bool = False,
    **kwargs,
) -> Tuple[Union[DataSet, PolyData, UnstructuredGrid], PolyData]:

    model = model.copy() if not inplace else model
    X = model.points if spatial_key is None else model[spatial_key]

    if rd_method == "ElPiGraph":
        nodes, edges = ElPiGraph_tree(X=X, NumNodes=NumNodes, **kwargs)
    elif rd_method == "SimplePPT":
        nodes, edges = SimplePPT_tree(X=X, NumNodes=NumNodes, **kwargs)
    else:
        raise ValueError("`rd_method` value is wrong." "\nAvailable `rd_method` are: `'ElPiGraph'`, `'SimplePPT'`.")

    map_points_to_branch(model=model, nodes=nodes, spatial_key=spatial_key, key_added=key_added, inplace=True)
    tree_model = construct_tree_model(nodes=nodes, edges=edges)

    if not (map_key is None):
        map_gene_to_branch(model=model, tree=tree_model, key=map_key, nodes_key=key_added, inplace=True)

    return model if not inplace else None, tree_model
