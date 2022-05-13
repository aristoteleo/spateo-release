"""Written by @Jinerhal, adapted by @Xiaojieqiu.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData


def digi_exp_plot(
    adata: AnnData,
    gene_dict: dict,
    region_key: str,
    mode: str = "density",
):
    """Simple function to visualize expression level varies along regions.

    Args:
        adata (AnnData): _description_
        gene_dict (dict): _description_
        region_key (str): _description_
        mode (str, optional): _description_. Defaults to "density".

    Returns:
        _type_: _description_
    """

    digi_region = []
    gene_list = []
    gene_mean = []

    for i in np.unique(adata.obs[region_key]):
        adata_tmp = adata[adata.obs[region_key] == i, :]
        for anno in list(gene_dict.keys()):
            for gene in gene_dict[anno]:
                digi_region.append(i)
                gene_list.append(gene + " " + anno)
                gene_mean.append(np.mean(adata_tmp[:, gene].X))

    df_plt = pd.DataFrame({region_key: digi_region, "Gene": gene_list, "Mean expression": gene_mean})

    if mode == "exp":
        p = sns.lineplot(data=df_plt, x=region_key, y="Mean expression", hue="Gene")
    elif mode == "density":
        p = sns.kdeplot(data=df_plt, x=region_key, weights="Mean expression", hue="Gene")
        p.set_xlim(0, max(adata.obs[region_key]))

    return df_plt
