from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import csr_matrix, isspmatrix, spmatrix
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ...configuration import SKM
from ...logging import logger_manager as lm

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if isspmatrix(X) else np.asarray(X)


def compute_pca_components(
    matrix: Union[np.ndarray, spmatrix], random_state: Optional[int] = 1, save_curve_img: Optional[str] = None
) -> Tuple[Any, int, float]:
    """
    Calculate the inflection point of the PCA curve to
    obtain the number of principal components that the PCA should retain.

    Args:
        matrix: A dense or sparse matrix.
        save_curve_img: If save_curve_img != None, save the image of the PCA curve and inflection points.
    Returns:
        new_n_components: The number of principal components that PCA should retain.
        new_components_stored: Percentage of variance explained by the retained principal components.
    """
    # Convert sparse matrix to dense matrix.
    matrix = to_dense_matrix(matrix)
    matrix[np.isnan(matrix)] = 0

    # Principal component analysis (PCA).
    pca = PCA(n_components=None, random_state=random_state)
    pcs = pca.fit_transform(matrix)

    # Percentage of variance explained by each of the selected components.
    # If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.
    raw_components_ratio = pca.explained_variance_ratio_
    raw_n_components = np.arange(1, raw_components_ratio.shape[0] + 1)

    # Calculate the inflection point of the PCA curve.
    kl = KneeLocator(raw_n_components, raw_components_ratio, curve="convex", direction="decreasing")
    new_n_components = int(kl.knee)
    new_components_stored = round(float(np.sum(raw_components_ratio[:new_n_components])), 3)

    # Whether to save the image of PCA curve and inflection point.
    if save_curve_img is not None:

        kl.plot_knee()
        plt.tight_layout()
        plt.savefig(save_curve_img, dpi=100)

    return pcs, new_n_components, new_components_stored


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def pca_spateo(
    adata: AnnData,
    X_data=None,
    n_pca_components: Optional[int] = None,
    pca_key: Optional[str] = "X_pca",
    genes: Union[list, None] = None,
    layer: Union[str, None] = None,
    random_state: Optional[int] = 1,
):
    """
    Do PCA for dimensional reduction.

    Args:
        adata:
            An Anndata object.
        X_data:
            The user supplied data that will be used for dimension reduction directly.
        n_pca_components:
            The number of principal components that PCA will retain. If none, will Calculate the inflection point
            of the PCA curve to obtain the number of principal components that the PCA should retain.
        pca_key:
            Add the PCA result to :attr:`obsm` using this key.
        genes:
            The list of genes that will be used to subset the data for dimension reduction and clustering. If `None`,
            all genes will be used.
        layer:
            The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, will use
            ``adata.X``.
    Returns:
        adata_after_pca: The processed AnnData, where adata.obsm[pca_key] stores the PCA result.
    """
    if X_data is None:
        if genes is not None:
            genes = adata.var_names.intersection(genes).to_list()
            lm.main_info("Using user provided gene set...")
            if len(genes) == 0:
                raise ValueError("no genes from your genes list appear in your adata object.")
        else:
            genes = adata.var_names
        if layer is not None:
            matrix = adata[:, genes].layers[layer].copy()
            lm.main_info('Runing PCA on adata.layers["' + layer + '"]...')
        else:
            matrix = adata[:, genes].X.copy()
            lm.main_info("Runing PCA on adata.X...")
    else:
        matrix = X_data.copy()
        lm.main_info("Runing PCA on user provided data...")

    if n_pca_components is None:
        pcs, n_pca_components, _ = compute_pca_components(adata.X, random_state=random_state, save_curve_img=None)
    else:
        matrix = to_dense_matrix(matrix)
        pca = PCA(n_components=n_pca_components, random_state=random_state)
        pcs = pca.fit_transform(matrix)

    adata.obsm[pca_key] = pcs[:, :n_pca_components]


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def sctransform(
    adata: AnnData,
    rlib_path: str,
    n_top_genes: Optional[int] = 3000,
    save_sct_img_1: Optional[str] = None,
    save_sct_img_2: Optional[str] = None,
    **kwargs,
):
    """
    Use sctransform with an additional flag vst.flavor="v2" to perform normalization and dimensionality reduction
    Original Code Repository: https://github.com/saketkc/pySCTransform

    Installation:
    Conda:
        ```conda install R```
    R:
        ```if (!require("BiocManager", quietly = TRUE))
            install.packages("BiocManager")```
        ```BiocManager::install(version = "3.14")```
        ```BiocManager::install("glmGamPoi")```
    Python:
        ```pip install rpy2```
        ```pip install git+https://github.com/saketkc/pysctransform```

    Examples:
        >>> sctransform(adata=adata, rlib_path="/Users/jingzehua/opt/anaconda3/envs/spateo/lib/R")

    Args:
        adata: An Anndata object.
        rlib_path: library path for R environment.
        n_top_genes: Number of highly-variable genes to keep.
        save_sct_img_1: If save_sct_img_1 != None, save the image of the GLM model parameters.
        save_sct_img_2: If save_sct_img_2 != None, save the image of the final residual variances.
        **kwargs: Additional keyword arguments to ``pysctransform.SCTransform``.

    Returns:
        Updates adata with the field ``adata.obsm["pearson_residuals"]``, containing pearson_residuals.
    """
    import os

    os.environ["R_HOME"] = rlib_path

    try:
        from pysctransform import SCTransform, vst
        from pysctransform.plotting import plot_fit, plot_residual_var
    except ImportError:
        raise ImportError(
            "You need to install the package `pysctransform`."
            "Install pysctransform via `pip install git+https://github.com/saketkc/pysctransform.git`"
        )

    residuals = SCTransform(adata, var_features_n=n_top_genes, **kwargs)
    adata.obsm["pearson_residuals"] = residuals

    # Plot model characteristics.
    if save_sct_img_1 is not None or save_sct_img_2 is not None:
        vst_out = vst(
            adata.X.T,
            gene_names=adata.var_names.tolist(),
            cell_names=adata.obs_names.tolist(),
            method="fix-slope",
            exclude_poisson=True,
        )
        # Visualize the GLM model parameters.
        if save_sct_img_1 is not None:
            _ = plot_fit(vst_out)
            plt.savefig(save_sct_img_1, dpi=100)
        # Visualize the final residual variances with respect to mean and highlight highly variable genes.
        if save_sct_img_2 is not None:
            _ = plot_residual_var(vst_out)
            plt.savefig(save_sct_img_2, dpi=100)


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def pearson_residuals(
    adata: AnnData,
    n_top_genes: Optional[int] = 3000,
    subset: bool = False,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
):
    """
    Preprocess UMI count data with analytic Pearson residuals.

    Pearson residuals transform raw UMI counts into a representation where three aims are achieved:
        1.Remove the technical variation that comes from differences in total counts between cells;
        2.Stabilize the mean-variance relationship across genes, i.e. ensure that biological signal from both low and
          high expression genes can contribute similarly to downstream processing
        3.Genes that are homogeneously expressed (like housekeeping genes) have small variance, while genes that are
          differentially expressed (like marker genes) have high variance

    Args:
        adata: An anndata object.
        n_top_genes: Number of highly-variable genes to keep.
        subset: Inplace subset to highly-variable genes if `True` otherwise merely indicate highly variable genes.
        theta: The negative binomial overdispersion parameter theta for Pearson residuals.
               Higher values correspond to less overdispersion (var = mean + mean^2/theta), and `theta=np.Inf`
               corresponds to a Poisson model.
        clip: Determines if and how residuals are clipped:
                * If `None`, residuals are clipped to the interval [-sqrt(n), sqrt(n)], where n is the number of cells
                  in the dataset (default behavior).
                * If any scalar c, residuals are clipped to the interval [-c, c]. Set `clip=np.Inf` for no clipping.
        check_values: Check if counts in selected layer are integers. A Warning is returned if set to True.

    Returns:
        Updates adata with the field ``adata.obsm["pearson_residuals"]``, containing pearson_residuals.
    """
    from dynamo.external.pearson_residual_recipe import (
        compute_highly_variable_genes,
        compute_pearson_residuals,
    )

    if not (n_top_genes is None):
        compute_highly_variable_genes(
            adata, n_top_genes=n_top_genes, recipe="pearson_residuals", inplace=True, subset=subset
        )

    X = adata.X.copy()
    residuals = compute_pearson_residuals(X, theta=theta, clip=clip, check_values=check_values)
    adata.obsm["pearson_residuals"] = residuals


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adatas")
def integrate(
    adatas: List[AnnData],
    batch_key: str = "slices",
    fill_value: Union[int, float] = 0,
) -> AnnData:
    """
    Concatenating all anndata objects.

    Args:
        adatas: AnnData matrices to concatenate with.
        batch_key: Add the batch annotation to :attr:`obs` using this key.
        fill_value: Scalar value to fill newly missing values in arrays with.
    Returns:
        integrated_adata: The concatenated AnnData, where adata.obs[batch_key] stores a categorical variable labeling the batch.
    """

    batch_ca = [adata.obs[batch_key][0] for adata in adatas]

    # Merge the obsm, varm and uns data of all anndata objcets separately.
    obsm_dict, varm_dict, uns_dict = {}, {}, {}
    obsm_keys, varm_keys, uns_keys = [], [], []
    for adata in adatas:
        obsm_keys.extend(list(adata.obsm.keys()))
        varm_keys.extend(list(adata.varm.keys()))
        uns_keys.extend(list(adata.uns_keys()))

    obsm_keys, varm_keys, uns_keys = list(set(obsm_keys)), list(set(varm_keys)), list(set(uns_keys))
    n_obsm_keys, n_varm_keys, n_uns_keys = len(obsm_keys), len(varm_keys), len(uns_keys)

    if n_obsm_keys > 0:
        for key in obsm_keys:
            obsm_dict[key] = np.concatenate([to_dense_matrix(adata.obsm[key]) for adata in adatas], axis=0)
    if n_varm_keys > 0:
        for key in varm_keys:
            varm_dict[key] = np.concatenate([to_dense_matrix(adata.varm[key]) for adata in adatas], axis=0)
    if n_uns_keys > 0:
        for key in uns_keys:
            if "__type" in uns_keys and key == "__type":
                uns_dict["__type"] = adatas[0].uns["__type"]
            else:
                uns_dict[key] = {
                    ca: adata.uns[key] if key in adata.uns_keys() else None for ca, adata in zip(batch_ca, adatas)
                }

    # Delete obsm, varm and uns data.
    for adata in adatas:
        del adata.obsm, adata.varm, adata.uns

    # Concatenating obs and var data which will ignore the uns, obsm, varm attributes.
    integrated_adata = adatas[0].concatenate(
        *adatas[1:],
        batch_key=batch_key,
        batch_categories=batch_ca,
        join="outer",
        fill_value=fill_value,
        uns_merge=None,
    )

    # Add Concatenated obsm data and varm data to integrated anndata object.
    if n_obsm_keys > 0:
        for key, value in obsm_dict.items():
            integrated_adata.obsm[key] = value
    if n_varm_keys > 0:
        for key, value in varm_dict.items():
            integrated_adata.varm[key] = value
    if n_uns_keys > 0:
        for key, value in uns_dict.items():
            integrated_adata.uns[key] = value

    return integrated_adata


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def harmony_debatch(
    adata: AnnData,
    key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    max_iter_harmony: int = 10,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Use harmonypy [Korunsky19]_ to remove batch effects.
    This function should be run after performing PCA but before computing the neighbor graph.
    Original Code Repository: https://github.com/slowkow/harmonypy
    Interesting example: https://slowkow.com/notes/harmony-animation/

    Args:
        adata: An Anndata object.
        key: The name of the column in ``adata.obs`` that differentiates among experiments/batches.
        basis: The name of the field in ``adata.obsm`` where the PCA table is stored.
        adjusted_basis: The name of the field in ``adata.obsm`` where the adjusted PCA table will be stored after
            running this function.
        max_iter_harmony: Maximum number of rounds to run Harmony. One round of Harmony involves one clustering and one
            correction step.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
        Updates adata with the field ``adata.obsm[adjusted_basis]``, containing principal components adjusted by
        Harmony.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")

    adata = adata.copy() if copy else adata

    # Convert sparse matrix to dense matrix.
    matrix = to_dense_matrix(adata.obsm[basis])

    # Use Harmony to adjust the PCs.
    harmony_out = harmonypy.run_harmony(matrix, adata.obs, key, max_iter_harmony=max_iter_harmony)
    adjusted_matrix = harmony_out.Z_corr.T

    # Convert dense matrix to sparse matrix.
    if isspmatrix(adata.obsm[basis]):
        adjusted_matrix = csr_matrix(adjusted_matrix)

    adata.obsm[adjusted_basis] = adjusted_matrix

    return adata if copy else None


def ecp_silhouette(
    matrix: Union[np.ndarray, spmatrix],
    cluster_labels: np.ndarray,
) -> float:
    """
    Here we evaluate the clustering performance by calculating the Silhouette Coefficient.
    The silhouette analysis is used to choose an optimal value for clustering resolution.

    The Silhouette Coefficient is a widely used method for evaluating clustering performance,
    where a higher Silhouette Coefficient score relates to a model with better defined clusters and
    indicates a good separation between the celltypes.

    Advantages of the Silhouette Coefficient:
        * The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
          Scores around zero indicate overlapping clusters.
        * The score is higher when clusters are dense and well separated,
          which relates to a standard concept of a cluster.

    Original Code Repository: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

    Args:
        matrix: A dense or sparse matrix of feature.
        cluster_labels: A array of labels for each cluster.

    Returns:
        Mean Silhouette Coefficient for all clusters.

    Examples:
        >>> silhouette_score(matrix=adata.obsm["X_pca"], cluster_labels=adata.obs["leiden"].values)
    """
    return silhouette_score(matrix, cluster_labels, metric="euclidean")


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def spatial_adj_dyn(
    adata: AnnData,
    spatial_key: str = "spatial",
    pca_key: str = "pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
    n_pca_components: int = 30,
):
    """
    Calculate the adjacent matrix based on a neighborhood graph of gene expression space
    and a neighborhood graph of physical space.
    """
    import dynamo as dyn

    # Compute a neighborhood graph of gene expression space.
    dyn.tl.neighbors(adata, X_data=adata.obsm[pca_key], n_neighbors=e_neigh, n_pca_components=n_pca_components)

    # Compute a neighborhood graph of physical space.
    dyn.tl.neighbors(
        adata,
        X_data=adata.obsm[spatial_key],
        n_neighbors=s_neigh,
        result_prefix="spatial",
        n_pca_components=n_pca_components,
    )

    # Calculate the adjacent matrix.
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1
    return adj
