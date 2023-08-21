from typing import Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from pyvista import PolyData, UnstructuredGrid
from scipy.spatial.distance import cdist


def construct_backbone(
    model: Union[PolyData, UnstructuredGrid],
    spatial_key: Optional[str] = None,
    nodes_key: str = "nodes",
    rd_method: Literal["ElPiGraph", "SimplePPT", "PrinCurve"] = "ElPiGraph",
    num_nodes: int = 50,
    color: str = "gainsboro",
    **kwargs,
) -> Tuple[PolyData, float, Optional[str]]:
    """
    Organ's backbone construction based on 3D point cloud model.

    Args:
        model:  A point cloud model.
        spatial_key: If spatial_key is None, the spatial coordinates are in model.points, otherwise in model[spatial_key].
        nodes_key: The key that corresponds to the coordinates of the nodes in the backbone.
        rd_method: The method of constructing a backbone model. Available ``rd_method`` are:

                * ``'ElPiGraph'``: Generate a principal elastic tree.
                * ``'SimplePPT'``: Generate a simple principal tree.
                * ``'PrinCurve'``: This is the global module that contains principal curve and nonlinear principal
                                   component analysis algorithms that work to optimize a line over an entire dataset.
        num_nodes: Number of nodes for the backbone model.
        color: Color to use for plotting backbone model.
        **kwargs: Additional parameters that will be passed to ``ElPiGraph_method``, ``SimplePPT_method`` or ``PrinCurve_method`` function.

    Returns:
        backbone_model: A three-dims backbone model.
        backbone_length: The length of the backbone model.
        plot_cmap: Recommended colormap parameter values for plotting.
    """
    model = model.copy()
    X = model.points if spatial_key is None else model[spatial_key]

    if rd_method == "ElPiGraph":
        from .backbone_methods import ElPiGraph_method

        nodes, edges = ElPiGraph_method(X=X, NumNodes=num_nodes, **kwargs)
    elif rd_method == "SimplePPT":
        from .backbone_methods import SimplePPT_method

        nodes, edges = SimplePPT_method(X=X, NumNodes=num_nodes, **kwargs)
    elif rd_method == "PrinCurve":
        from .backbone_methods import PrinCurve_method

        nodes, edges = PrinCurve_method(X=X, NumNodes=num_nodes, **kwargs)
    else:
        raise ValueError(
            "`rd_method` value is wrong." "\nAvailable `rd_method` are: `'ElPiGraph'`, `'SimplePPT'`, `'PrinCurve'`."
        )

    # Construct the backbone model
    from ..models_migration import construct_lines

    backbone_model, plot_cmap = construct_lines(
        points=nodes, edges=edges, key_added="backbone", label="backbone", color=color
    )
    backbone_model.point_data[nodes_key] = np.arange(0, len(nodes), 1)

    # Calculate the length of the backbone
    s_points, e_points = nodes[edges[:, 0], :], nodes[edges[:, 1], :]
    backbone_length = cdist(XA=np.asarray(s_points), XB=np.asarray(e_points), metric="euclidean").diagonal().sum()

    return backbone_model, backbone_length, plot_cmap
