from .clustering import *

def find_clusters(
    adata,
    method="spagcn",
    **kwargs
):
    '''Cluster finding for given adata object.
    Parameters
    ----------
        adata: class:`~anndata.AnnData`
            an Annodata object. 
        method: `str` (default: "spagcn")
            supported clustering methods are: 
            "spagcn".
        **kwargs: other arguments for chosen clustering method.
    Returns
    -------
        An `~anndata.AnnData` object with cluster info.

    '''

    if (method=="spagcn"):
        adata = find_cluster_spagcn(adata, kwargs)
        return adata
