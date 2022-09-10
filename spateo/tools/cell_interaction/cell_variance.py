"""
Characterizing cell-to-cell variability within spatial domains
"""
import pandas as pd
import scipy
from anndata import AnnData

from ...configuration import SKM


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def compute_variance_decomposition(adata: AnnData, spatial_label_id: str, celltype_label_id: str):
    """
    Within spatial regions, determines the proportion of the total variation that occurs within the same cell type,
    the proportion of the variation that occurs between cell types in the region, and the proportion of the variation
    that comes from baseline differences in the expression levels of the genes in the data. The within-cell type
    variation could potentially come from differences in cell-cell communication.

    Args:
        adata: class `anndata.AnnData`
        spatial_label_id: Key in .obs containing spatial domain labels
        celltype_label_id: Key in .obs containing cell type labels

    Returns:
        var_decomposition : pd.DataFrame
            Dataframe containing four columns, for the category label, celltype variation,
    """
    adata_copy = adata.copy()

    # Dataframe containing gene expression, cell type labels and spatial domain labels:
    data = adata_copy.X.toarray() if scipy.sparse.issparse(adata_copy.X) else adata_copy.X
    df = pd.DataFrame(data, columns=adata_copy.var_names)
    df["spatial_domain"] = pd.Series(list(adata.obs[spatial_label_id]), dtype="category")
