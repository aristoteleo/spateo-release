from anndata import AnnData
from dynamo.plot.scatters import scatters


def position(adata: AnnData, skey: str = "position", **kwargs):
    """"""
    if skey in adata.obsm_keys():
        x, y = adata.obsm[skey][:, 0], adata.obsm[skey][:, 1]

    scatters(adata, x=x, y=y, **kwargs)
