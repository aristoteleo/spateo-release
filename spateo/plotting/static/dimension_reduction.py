# code adapted from https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/plot/dimension_reduction.py
from anndata import AnnData
from typing import Optional, Union


from .scatters import scatters

# from .scatters import docstrings

# docstrings.delete_params("scatters.parameters", "adata", "basis")


# @docstrings.with_indent(4)
def pca(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with pca basis.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s
    Returns
    -------
    Nothing but plots the pca embedding of the adata object.

    """

    scatters(adata, "pca", *args, **kwargs)


# @docstrings.with_indent(4)
def umap(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with umap basis.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s
    Returns
    -------
    Nothing but plots the umap embedding of the adata object.

    """

    return scatters(adata, "umap", *args, **kwargs)


# @docstrings.with_indent(4)
def trimap(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with trimap basis.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s
    Returns
    -------
    Nothing but plots the pca embedding of the adata object.

    """
    return scatters(adata, "trimap", *args, **kwargs)


# @docstrings.with_indent(4)
def tsne(adata: AnnData, *args, **kwargs):
    """\
    Scatter plot with tsne basis.
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object.
        %(scatters.parameters.no_adata|basis)s
    Returns
    -------
    Nothing but plots the tsne embedding of the adata object.

    """
    return scatters(adata, "tsne", *args, **kwargs)
