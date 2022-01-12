from .find_clusters import *


def find_clusters(adata, method="spagcn", **kwargs):
    """Find Clusters for given adata object

    Args:
        adata (class:`~anndata.AnnData`): An Annodata object.
        method (str, optional): supported clustering methods are, "spagcn". Defaults to "spagcn".

    Returns:
        class:`~anndata.AnnData`: An `~anndata.AnnData` object with cluster info.
    """

    if method == "spagcn":
        adata = find_cluster_spagcn(adata, kwargs)
        return adata
