from typing import Optional, Union

import numpy as np
import pandas as pd
from pyvista import PolyData, UnstructuredGrid
from scipy.spatial import KDTree

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def map_points_to_backbone(
    model: Union[PolyData, UnstructuredGrid],
    backbone_model: PolyData,
    nodes_key: str = "nodes",
    key_added: Optional[str] = "nodes",
    inplace: bool = False,
    **kwargs,
):
    """
    Find the closest principal tree node to any point in the model through KDTree.

    Args:
        model: The reconstructed model.
        backbone_model: The constructed backbone model.
        nodes_key: The key that corresponds to the coordinates of the nodes in the backbone.
        key_added: The key under which to add the nodes labels.
        inplace: Updates model in-place.
        **kwargs: Additional parameters that will be passed to ``scipy.spatial.KDTree.`` function.

    Returns:
        A model, which contains the following properties:
            `model.point_data[key_added]`, the nodes labels array.
    """
    model = model.copy() if not inplace else model

    nodes_data = pd.DataFrame(np.asarray(backbone_model.points), columns=["x", "y", "z"], dtype=float)
    nodes_data[nodes_key] = backbone_model.point_data[nodes_key].astype(int)
    nodes_data = nodes_data.sort_values(by=nodes_key)
    backbone_nodes = nodes_data.loc[:, ["x", "y", "z"]].values

    nodes_kdtree = KDTree(np.asarray(backbone_nodes), **kwargs)
    _, ii = nodes_kdtree.query(np.asarray(model.points), k=1)
    model.point_data[key_added] = ii

    return model if not inplace else None


def map_gene_to_backbone(
    model: Union[PolyData, UnstructuredGrid],
    tree: PolyData,
    key: Union[str, list],
    nodes_key: Optional[str] = "nodes",
    inplace: bool = False,
):
    """
    Find the closest principal tree node to any point in the model through KDTree.

    Args:
        model: A reconstructed model contains the gene expression label.
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
