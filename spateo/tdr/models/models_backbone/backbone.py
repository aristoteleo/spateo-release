from typing import Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from pyvista import PolyData, UnstructuredGrid
from scipy.sparse import issparse
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


def update_backbone(
    backbone: PolyData,
    nodes_key: str = "nodes",
    key_added: str = "updated_nodes",
    select_nodes: Optional[Union[list, np.ndarray]] = None,
    interactive: Optional[bool] = True,
    model_size: Union[float, list] = 8.0,
    colormap: str = "Spectral",
) -> Union[PolyData, UnstructuredGrid]:
    """
    Update the bakcbone through interaction or input of selected nodes.

    Args:
        backbone: The backbone model.
        nodes_key: The key that corresponds to the coordinates of the nodes in the backbone.
        key_added: The key under which to add the labels of new nodes.
        select_nodes: Nodes that need to be retained.
        interactive: Whether to delete useless nodes interactively. When ``interactive`` is True, ``select_nodes`` is invalid.
        model_size: Thickness of backbone. When ``interactive`` is False, ``model_size`` is invalid.
        colormap: Colormap of backbone. When ``interactive`` is False, ``colormap`` is invalid.

    Returns:
        updated_backbone: The updated backbone model.
    """
    model = backbone.copy()
    if interactive is True:
        from ...widgets.clip import _interactive_rectangle_clip
        from ...widgets.utils import _interactive_plotter

        p = _interactive_plotter()
        p.add_point_labels(
            model,
            labels=nodes_key,
            font_size=18,
            font_family="arial",
            text_color="white",
            shape_color="black",
            always_visible=True,
        )

        picked_models, picking_r_list = [], []
        if f"{nodes_key}_rgba" in model.array_names:
            p.add_mesh(
                model,
                scalars=f"{nodes_key}_rgba",
                rgba=True,
                style="wireframe",
                render_lines_as_tubes=True,
                line_width=model_size,
            )
        else:
            p.add_mesh(
                model,
                scalars=nodes_key,
                style="wireframe",
                render_lines_as_tubes=True,
                line_width=model_size,
                cmap=colormap,
            )
        _interactive_rectangle_clip(
            plotter=p,
            model=model,
            picking_list=picked_models,
            picking_r_list=picking_r_list,
        )
        p.show(cpos="iso")
        updated_backbone = picking_r_list[0]
    else:
        updated_backbone = model.extract_cells(np.isin(np.asarray(model.point_data[nodes_key]), select_nodes))

    updated_backbone.point_data[key_added] = np.arange(0, updated_backbone.n_points, 1)
    return updated_backbone


def backbone_scc(
    adata: AnnData,
    backbone: PolyData,
    genes: Optional[list] = None,
    adata_nodes_key: str = "backbone_nodes",
    backbone_nodes_key: str = "updated_nodes",
    key_added: Optional[str] = "backbone_scc",
    layer: Optional[str] = None,
    e_neigh: int = 10,
    s_neigh: int = 6,
    cluster_method: Literal["leiden", "louvain"] = "leiden",
    resolution: Optional[float] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Spatially constrained clustering (scc) along the backbone.

    Args:
        adata: The anndata object.
        backbone: The backbone model.
        genes: The list of genes that will be used to subset the data for clustering. If ``genes = None``, all genes will be used.
        adata_nodes_key: The key that corresponds to the nodes in the adata.
        backbone_nodes_key: The key that corresponds to the nodes in the backbone.
        key_added: adata.obs key under which to add the cluster labels.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If ``layer = None``, ``.X`` is used.
        e_neigh: the number of nearest neighbor in gene expression space.
        s_neigh: the number of nearest neighbor in physical space.
        cluster_method: the method that will be used to cluster the cells.
        resolution: the resolution parameter of the louvain clustering algorithm.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` attribute, storing the clustering results.
    """
    import dynamo as dyn
    from dynamo.tools.utils import fetch_X_data

    from ....tools import scc

    adata = adata if inplace else adata.copy()
    if "pp" not in adata.uns.keys():
        adata.uns["pp"] = {}
    genes, X_data = fetch_X_data(adata, genes, layer)
    X_data = X_data.A if issparse(X_data) else X_data
    X_data = pd.DataFrame(X_data, columns=genes)
    X_data[adata_nodes_key] = adata.obs[adata_nodes_key].values
    X_data = pd.DataFrame(X_data.groupby(by=adata_nodes_key).mean())
    backbone_nodes = X_data.index

    X_spatial = pd.DataFrame(backbone.points, index=backbone.point_data[backbone_nodes_key])
    X_spatial = X_spatial.loc[backbone_nodes, :].values

    backbone_adata = ad.AnnData(
        X=X_data.values,
        var=pd.DataFrame(index=X_data.columns),
        obs=pd.DataFrame(backbone_nodes, columns=[adata_nodes_key]),
        obsm={"spatial": X_spatial},
        uns={"__type": "UMI", "pp": {}},
    )

    dyn.pp.normalize(backbone_adata)
    dyn.pp.log1p(backbone_adata)
    backbone_adata.obsm["X_backbone"] = backbone_adata.X
    scc(
        backbone_adata,
        spatial_key="spatial",
        pca_key="X_backbone",
        e_neigh=e_neigh,
        s_neigh=s_neigh,
        resolution=resolution,
        key_added="scc",
        cluster_method=cluster_method,
    )

    cluster_dict = {i: c for i, c in zip(backbone_adata.obs[adata_nodes_key], backbone_adata.obs["scc"])}
    adata.obs[key_added] = adata.obs[adata_nodes_key].map(lambda x: cluster_dict[x])
    return None if inplace else adata
