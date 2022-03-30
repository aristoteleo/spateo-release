import random
from typing import Optional, Tuple

import anndata
import cv2
import numpy as np
import pandas as pd
import torch

from ..preprocessing.filter import filter_cells, filter_genes
from .find_clusters_utils import *


def find_cluster_spagcn(
    adata: anndata.AnnData,
    n_clusters: int,
    p: float = 0.5,
    s: int = 1,
    b: int = 49,
    refine_shape: Optional[str] = None,
    his_img_path: Optional[str] = None,
    total_umi: Optional[str] = None,
    x_pixel: str = None,
    y_pixel: str = None,
    x_array: str = None,
    y_array: str = None,
    seed: int = 100,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """Function to find clusters with spagcn.

    Args:
        adata: an Anndata object, after normalization.
        n_clusters: Desired number of clusters.
        p: parameter `p` in spagcn algorithm. See `SpaGCN` for details. Defaults to 0.5.
        s: alpha to control the color scale in calculating adjacent matrix. Defaults to 1.
        b: beta to control the range of neighbourhood when calculate grey value for one spot in calculating adjacent matrix. Defaults to 49.
        refine_shape: Smooth the spatial domains with given spatial topology, "hexagon" for Visium data, "square" for ST data. Defaults to None.
        his_img_path: The file path of histology image used to calculate adjacent matrix in spagcn algorithm. Defaults to None.
        total_umi: By providing the key(colname) in `adata.obs` which contains total UMIs(counts) for each spot, the function use the total counts as
                                a grayscale image when histology image is not provided. Ignored if his_img_path is not `None`. Defaults to "total_umi".
        x_pixel: The key(colname) in `adata.obs` which contains corresponding x-pixels in histology image. Defaults to None.
        y_pixel: The key(colname) in `adata.obs` which contains corresponding y-pixels in histology image. Defaults to None.
        x_array: The key(colname) in `adata.obs` which contains corresponding x-coordinates. Defaults to None.
        y_array: The key(colname) in `adata.obs` which contains corresponding y-coordinates. Defaults to None.
        seed: Global seed for `random`, `torch`, `numpy`. Defaults to 100.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments. Defaults to False.

    Returns:
        class:`~anndata.AnnData`: An `~anndata.AnnData` object with cluster info in "spagcn_pred", and in "spagcn_pred_refined" if `refine_shape` is set.
                                The adjacent matrix used in spagcn algorithm is saved in `adata.uns["adj_spagcn"]`.
    """

    if x_array is None:
        x_array = [i[0] for i in adata.obsm["X_spatial"]]
    else:
        x_array = adata.obs[x_array].tolist()

    if y_array is None:
        y_array = [i[1] for i in adata.obsm["X_spatial"]]
    else:
        y_array = adata.obs[y_array].tolist()

    if x_pixel is None:
        x_pixel = [int(i) for i in x_array]
    else:
        x_pixel = adata.obs[x_pixel].tolist()

    if y_pixel is None:
        y_pixel = [int(i) for i in y_array]
    else:
        y_pixel = adata.obs[y_pixel].tolist()

    s = 1
    b = 49

    if his_img_path is None:
        if total_umi is None:
            adj = calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        else:
            total_umi = adata.obs[total_umi].tolist()
            total_umi = [int(x / max(total_umi) * 254 + 1) for x in total_umi]
            total_umi_mtx = pd.DataFrame({"x_pos": x_pixel, "y_pos": y_pixel, "n_umis": total_umi})
            total_umi_mtx = total_umi_mtx.pivot(index="x_pos", columns="y_pos", values="n_umis").fillna(1).to_numpy()
            umi_gs_img = np.dstack((total_umi_mtx, total_umi_mtx, total_umi_mtx)).astype(int)
            adj = calculate_adj_matrix(
                x=x_array,
                y=y_array,
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                image=umi_gs_img,
                beta=b,
                alpha=s,
                histology=True,
            )
    else:
        img = cv2.imread(his_img_path)
        adj = calculate_adj_matrix(
            x=x_array,
            y=y_array,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            image=img,
            beta=b,
            alpha=s,
            histology=True,
        )

    adata.uns["adj_spagcn"] = adj

    l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # Set seed
    r_seed = t_seed = n_seed = seed

    # Seaech for suitable resolution
    res = search_res(
        adata,
        adj,
        l,
        n_clusters,
        start=0.7,
        step=0.1,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=r_seed,
        t_seed=t_seed,
        n_seed=n_seed,
    )

    clf = SpaGCN()
    clf.set_l(l)

    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    # Run
    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=200,
    )
    y_pred, prob = clf.predict()
    adata.obs["spagcn_pred"] = y_pred
    adata.obs["spagcn_pred"] = adata.obs["spagcn_pred"].astype("category")
    adata.obs["spagcn_pred"] = [str(i) for i in adata.obs["spagcn_pred"]]

    if refine_shape is not None:
        # Do cluster refinement(optional)
        adj_2d = calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        refined_pred = refine(
            sample_id=adata.obs.index.tolist(),
            pred=adata.obs["spagcn_pred"].tolist(),
            dis=adj_2d,
            shape=refine_shape,
        )
        adata.obs["spagcn_pred_refined"] = refined_pred
        adata.obs["spagcn_pred_refined"] = adata.obs["spagcn_pred_refined"].astype("category")

    if copy:
        return adata
    return None


def find_cluster_scc(
    adata: anndata.AnnData,
    min_cells: int = 100,
    spatial_key: str = "spatial",
    e_neigh: int = 30,
    s_neigh: int = 6,
    resolution: Optional[float] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """Spatially constrained clustering (scc) to identify continuous tissue domains.

    Args:
        adata: an Anndata object, after normalization.
        min_cells: minimal number of cells the gene expressed.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        e_neigh: the number of nearest neighbor in gene expression space.
        s_neigh: the number of nearest neighbor in physical space.
        resolution: the resolution parameter of the leiden clustering algorithm.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments.
            Defaults to False.

    Returns:
        Depends on the argument `copy` return either an `~anndata.AnnData` object with cluster info in "scc_e_{a}_s{b}"
        or None.
    """

    import dynamo as dyn

    filter_genes(adata, min_cells=min_cells)
    adata.uns["pp"] = {}
    dyn.pp.normalize_cell_expr_by_size_factors(adata, layers="X")
    dyn.pp.log1p(adata)
    dyn.pp.pca_monocle(adata, n_pca_components=30, pca_key="X_pca")

    dyn.tl.neighbors(adata, n_neighbors=e_neigh)
    dyn.tl.neighbors(adata, n_neighbors=s_neigh, basis=spatial_key, result_prefix="spatial")
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1
    dyn.tl.leiden(adata, adj_matrix=adj, resolution=resolution, result_key="scc_e" + str(e_neigh) + "_s" + str(s_neigh))

    if copy:
        return adata
    return None
