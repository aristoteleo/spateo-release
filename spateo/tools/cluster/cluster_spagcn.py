import random
from typing import Optional

import numpy as np
import torch
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ...configuration import SKM
from .utils import compute_pca_components


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def spagcn_vanilla(
    adata: AnnData,
    spatial_key: str = "spatial",
    key_added: Optional[str] = "spagcn_pred",
    n_pca_components: Optional[int] = None,
    e_neigh: int = 10,
    resolution: float = 0.4,
    n_clusters: Optional[int] = None,
    refine_shape: Literal["hexagon", "square"] = "hexagon",
    p: float = 0.5,
    seed: int = 100,
    numIterMaxSpa: int = 2000,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Integrating gene expression and spatial location to identify spatial domains via SpaGCN.
    Original Code Repository: https://github.com/jianhuupenn/SpaGCN

    Reference:
        Jian Hu, Xiangjie Li, Kyle Coleman, Amelia Schroeder, Nan Ma, David J. Irwin, Edward B. Lee,
        Russell T. Shinohara & Mingyao Li. SpaGCN: Integrating gene expression, spatial location and histology to
        identify spatial domains and spatially variable genes by graph convolutional network. Nature Methods volume 18,
        pages1342–1351 (2021)

    Args:
        adata: An Anndata object after normalization.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        key_added: adata.obs key under which to add the cluster labels.
                   The initial clustering results of SpaGCN are under `key_added`,
                   and the refined clustering results are under `f'{key_added}_refined'`.
        n_pca_components: Number of principal components to compute.
                          If `n_pca_components` == None, the value at the inflection point of the PCA curve is
                          automatically calculated as n_comps.
        e_neigh: Number of nearest neighbor in gene expression space.
            Used in dyn.pp.neighbors(adata, n_neighbors=e_neigh).
        resolution: Resolution in the Louvain clustering method. Used when `n_clusters`==None.
        n_clusters: Number of spatial domains wanted.
                    If `n_clusters` != None, the suitable resolution in the initial Louvain clustering method
                    will be automatically searched based on n_clusters.
        refine_shape: Smooth the spatial domains with given spatial topology, "hexagon" for Visium data, "square" for ST
            data. Defaults to None.
        p: Percentage of total expression contributed by neighborhoods.
        seed: Global seed for `random`, `torch`, `numpy`. Defaults to 100.
        numIterMaxSpa: SpaGCN maximum number of training iterations.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
        Depending on the parameter `copy`, when True return an updates adata with the field ``adata.obs[key_added]`` and
        ``adata.obs[f'{key_added}_refined']``, containing the cluster result based on SpaGCN; else inplace update the
        adata object.
    """
    try:
        import SpaGCN as spg
    except ImportError:
        raise ImportError("\nplease install SpaGCN:\n\n\tpip install SpaGCN")

    adata = adata.copy() if copy else adata

    # Spatial coordinates.
    coords_x = adata.obsm[spatial_key][:, 0]
    coords_y = adata.obsm[spatial_key][:, 1]

    # Calculate the adjacent matrix.
    adj = spg.calculate_adj_matrix(x=coords_x, y=coords_y, histology=False)

    # Find the l value given p to control p.
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # Set seed.
    r_seed = t_seed = n_seed = seed

    # Set the resolution in the initial Louvain's Clustering methods.
    if n_clusters is None:
        res = resolution
    else:
        # Search for suitable resolution based on n_clusters.
        res = spg.search_res(
            adata,
            adj,
            l,
            n_clusters,
            start=resolution,
            step=0.1,
            tol=5e-3,
            lr=0.05,
            max_epochs=200,
            r_seed=r_seed,
            t_seed=t_seed,
            n_seed=n_seed,
        )

    # Run SpaGCN
    clf = spg.SpaGCN()
    clf.set_l(l)

    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    if n_pca_components is None:
        pcs, n_pca_components, _ = compute_pca_components(adata.X, save_curve_img=None)
    clf.train(
        adata,
        adj,
        num_pcs=n_pca_components,
        n_neighbors=e_neigh,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=numIterMaxSpa,
    )

    # SpaGCN Cluster result.
    y_pred, prob = clf.predict()
    adata.obs[key_added] = y_pred
    adata.obs[key_added] = adata.obs[key_added].astype("category")

    # Do cluster refinement
    adj_2d = spg.calculate_adj_matrix(x=coords_x, y=coords_y, histology=False)
    refined_pred = spg.refine(
        sample_id=adata.obs.index.tolist(),
        pred=adata.obs[key_added].tolist(),
        dis=adj_2d,
        shape=refine_shape,
    )
    adata.obs[f"{key_added}_refined"] = refined_pred
    adata.obs[f"{key_added}_refined"] = adata.obs[f"{key_added}_refined"].astype("category")

    return adata if copy else None
