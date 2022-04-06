from anndata import AnnData

from .logging import logger_manager as lm


def copy_adata(adata: AnnData) -> AnnData:
    """wrapper for deep copy adata and log copy operation since it is memory intensive.

    Args:
        adata: An adata object that will be deep copied.
        logger: Whether to report logging info

    Examples
    --------
    >>> import dynamo as dyn
    >>> adata = dyn.sample_data.hgForebrainGlutamatergic()
    >>> original_adata = copy_adata(adata)
    >>> # now after this statement, adata "points" to a new object, copy of the original
    >>> adata = copy_adata(adata)
    >>> adata.X[0, 1] = -999
    >>> # original_adata unchanged
    >>> print(original_adata.X[0, 1])
    >>> # we can use adata = copy_adata(adata) inside a dynammo function when we want to create a adata copy
    >>> # without worrying about changing the original copy.
    """
    logger = lm.get_main_logger()
    logger.info(
        "Deep copying AnnData object and working on the new copy. Original AnnData object will not be modified.",
        indent_level=1,
    )
    data = adata.copy()
    return data
