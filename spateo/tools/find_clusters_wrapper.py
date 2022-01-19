from .find_clusters import *


def find_clusters(adata, method="spagcn", copy=False, **kwargs):
    """Find Clusters for given adata object

    Args:
        adata (class:`~anndata.AnnData`): An Annodata object.
        method (str, optional): supported clustering methods are, "spagcn". Defaults to "spagcn".
        copy (bool): Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments. Defaults to False.

    Returns:
        class:`~anndata.AnnData`: An `~anndata.AnnData` object with cluster info.
    """

    if method == "spagcn":
        find_cluster_spagcn(adata, **kwargs)

    if copy:
        return adata
    return None
