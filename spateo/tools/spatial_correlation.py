from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from esda.moran import Moran_BV, Moran_Local_BV
from libpysal.weights import WSP

from .find_neighbors import neighbors


def spatial_bv_moran_obs_genes(
    adata: AnnData,
    obs_key: str,
    connectivity_key: str = "spatial_connectivities",
    genes: Union[str, int, Sequence[str], Sequence[int], None] = None,
    n_neighbors: int = 10,
    mode: str = "moran",
    transformation: str = "r",
    permutations: Optional[int] = 999,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Calculate global bivariate Moran's I between a spatial variable and gene expression

    Parameters
    ----------
    adata
        AnnData object containing spatial data
    obs_key
        Key in `adata.obs` for the variable
    connectivity_key
        Key in `adata.obsp` for spatial connectivity matrix (default: 'spatial_connectivities')
    genes
        Genes to calculate (names or indices). If None, use all genes.
    mode
        Spatial correlation mode (only 'moran' supported)
    transformation
        Weight transformation method ('r' for row-standardization)
    permutations
        Number of permutations for significance testing
    copy
        Return a DataFrame instead of storing in AnnData

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:
        I             : float
                        value of bivariate Moran's I
        sim           : array
                        (if permutations>0)
                        vector of I values for permuted samples
        p_sim         : float
                        (if permutations>0)
                        p-value based on permutations (one-sided)
                        null: spatial randomness
                        alternative: the observed I is extreme
                        it is either extremely high or extremely low
        z_sim         : array
                        (if permutations>0)
                        standardized I based on permutations
        p_z_sim       : float
                        (if permutations>0)
                        p-value based on standard normal approximation from
                        permutations

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['{obs_key}_gene_bv_moranI']`` - the above mentioned dataframe``.
    """
    # Validate inputs
    if mode != "moran":
        raise ValueError(f"Unsupported mode: {mode}. Only 'moran' is currently supported")

    if obs_key not in adata.obs:
        raise KeyError(f"'{obs_key}' not found in adata.obs")

    if connectivity_key not in adata.obsp:
        neighbors(
            adata,
            basis="spatial",
            spatial_key="spatial",
            n_neighbors_method="ball_tree",
            n_neighbors=n_neighbors,
        )
        connectivity_key = "spatial_connectivities"

    # Extract target variable and spatial weights
    y = np.array(adata.obs[obs_key].values, dtype=np.float64)
    spatial_weights = adata.obsp[connectivity_key]
    wsp = WSP(spatial_weights)
    w = wsp.to_W()

    # Identify genes to process
    if genes is None:
        gene_names = adata.var_names.tolist()
        gene_indices = range(adata.n_vars)
    elif isinstance(genes, (str, int)):
        gene_names = [genes] if isinstance(genes, str) else [adata.var_names[genes]]
        gene_indices = [adata.var_names.get_loc(genes)] if isinstance(genes, str) else [genes]
    else:
        gene_names = []
        gene_indices = []
        for gene in genes:
            if isinstance(gene, str):
                gene_names.append(gene)
                gene_indices.append(adata.var_names.get_loc(gene))
            else:
                gene_names.append(adata.var_names[gene])
                gene_indices.append(gene)

    # Prepare results storage
    results = {
        "I": [],
    }
    if permutations is not None:
        results.update(
            {
                "EI_sim": [],
                "pval_sim": [],
                "pval_z_sim": [],
                "z_sim": [],
            }
        )

    # Process each gene
    for idx in gene_indices:
        # Extract gene expression
        x = adata.X[:, idx]
        if hasattr(x, "toarray"):
            x = x.toarray().flatten()
        x = x.astype(np.float64)

        # Calculate bivariate Moran's I
        moran_bv = Moran_BV(
            x,
            y,
            w,
            transformation=transformation,
            permutations=permutations,
        )

        # Store results
        results["I"].append(moran_bv.I)

        if permutations is not None:
            results["EI_sim"].append(moran_bv.EI_sim)
            results["pval_sim"].append(moran_bv.p_sim)
            results["pval_z_sim"].append(moran_bv.p_z_sim)
            results["z_sim"].append(moran_bv.z_sim)

    # Create results DataFrame
    df = pd.DataFrame(results, index=gene_names)

    # Store or return results
    if copy:
        return df
    else:
        adata.uns[f"{obs_key}_gene_bv_moranI"] = df
        return None


def spatial_bv_local_moran(
    adata: AnnData,
    feature1_key: str,
    feature2_key: str,
    connectivity_key: str = "spatial_connectivities",
    n_neighbors: int = 10,
    mode: str = "moran",
    transformation: str = "r",
    permutations: Optional[int] = 999,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Calculate global bivariate Moran's I between a spatial variable and gene expression

    Parameters
    ----------
    adata
        AnnData object containing spatial data
    feature1_key
        Key in `adata.obs` for the first variable or gene_name
    feature2_key
        Key in `adata.obs` for the seconda variable or gene_name
    connectivity_key
        Key in `adata.obsp` for spatial connectivity matrix (default: 'spatial_connectivities')
    mode
        Spatial correlation mode (only 'moran' supported)
    transformation
        Weight transformation method ('r' for row-standardization)
    permutations
        Number of permutations for significance testing
    copy
        Return a DataFrame instead of storing in AnnData

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the following keys:
        I             : float
                        value of bivariate Moran's I
        q             : array
                        (if permutations>0) values indicate quandrant location 1 HH, 2 LH, 3 LL, 4 HL
        sim           : array
                        (if permutations>0)
                        vector of I values for permuted samples
        p_sim         : float
                        (if permutations>0)
                        p-value based on permutations (one-sided)
                        null: spatial randomness
                        alternative: the observed I is extreme
                        it is either extremely high or extremely low
        z_sim         : array
                        (if permutations>0)
                        standardized I based on permutations
        p_z_sim       : float
                        (if permutations>0)
                        p-value based on standard normal approximation from
                        permutations

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['{feature1_key}_{feature2_key}_bv_local_moranI']`` - the above mentioned dataframe``.
    """
    # Validate inputs
    if mode != "moran":
        raise ValueError(f"Unsupported mode: {mode}. Only 'moran' is currently supported")

    if feature1_key not in adata.obs and feature1_key not in adata.var_names:
        raise KeyError(f"'{feature1_key}' not found in adata.obs and a gene name")

    if feature2_key not in adata.obs and feature2_key not in adata.var_names:
        raise KeyError(f"'{feature2_key}' not found in adata.obs and a gene name")

    if connectivity_key not in adata.obsp:
        neighbors(
            adata,
            basis="spatial",
            spatial_key="spatial",
            n_neighbors_method="ball_tree",
            n_neighbors=n_neighbors,
        )
        connectivity_key = "spatial_connectivities"

    # Extract target variable and spatial weights
    if feature1_key in adata.obs:
        x = np.array(adata.obs[feature1_key].values, dtype=np.float64)
    else:
        idx = adata.var_names.get_loc(feature1_key)
        x = adata.X[:, idx]
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = x.squeeze().astype(np.float64)

    if feature2_key in adata.obs:
        y = np.array(adata.obs[feature2_key].values, dtype=np.float64)
    else:
        idx = adata.var_names.get_loc(feature2_key)
        print(hasattr(adata.X[:, idx], "toarray"))
        y = adata.X[:, idx]
        if hasattr(y, "toarray"):
            y = y.toarray()
        y = y.squeeze().astype(np.float64)

    spatial_weights = adata.obsp[connectivity_key]
    wsp = WSP(spatial_weights)
    w = wsp.to_W()

    local_moran_bv = Moran_Local_BV(x, y, w, transformation=transformation, permutations=permutations)

    df = pd.DataFrame(index=adata.obs_names)
    df["I"] = local_moran_bv.Is
    if permutations is not None:
        df["q"] = local_moran_bv.q
        df["EI_sim"] = local_moran_bv.EI_sim
        df["pval_sim"] = local_moran_bv.p_sim
        df["pval_z_sim"] = local_moran_bv.p_z_sim
        df["z_sim"] = local_moran_bv.z_sim

    # Store or return results
    if copy:
        return df
    else:
        adata.uns[f"{feature1_key}_{feature2_key}_bv_local_moranI"] = df
        return None
