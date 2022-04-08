from typing import List, Optional, Tuple, Union

import lack
import numpy as np
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import csr_matrix, isspmatrix, spmatrix
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

slog = lack.LoggerManager(namespace="spateo")

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if isspmatrix(X) else X


def compute_pca_components(
    matrix: Union[np.ndarray, spmatrix], save_curve_img: Optional[str] = None
) -> Tuple[int, float]:
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

    # Principal component analysis (PCA).
    pca = PCA(n_components=None)
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
        import matplotlib.pyplot as plt

        kl.plot_knee()
        plt.tight_layout()
        plt.savefig(save_curve_img, dpi=100)

    return pcs, new_n_components, new_components_stored


def pca_spateo(
    adata: AnnData,
    n_pca_components: Optional[int] = None,
    pca_key: Optional[str] = "X_pca",
    basis: str = None,
) -> Optional[AnnData]:
    """
    Do PCA for dimensional reduction.

    Args:
        adata: An Anndata object.
        n_pca_components: The number of principal components that PCA will retain. If none, will Calculate the
            inflection point of the PCA curve to obtain the number of principal components that the PCA should retain.
        pca_key: Add the PCA result to :attr:`obsm` using this key.
        basis: The name of the field in ``adata.obsm`` where the PCA should be performed on. If none, will perform
            PCA on ``adata.X``
    Returns:
        adata_after_pca: The processed AnnData, where adata.obsm[pca_key] stores the PCA result.
    """
    if basis is None:
        matrix = adata.X.copy()
        slog.main_info("Runing PCA on X data...")
    else:
        matrix = adata.obsm[basis].copy()
        slog.main_info('Runing PCA on obsm["' + basis + '"]...')

    if n_pca_components is None:
        pcs, n_pca_components, _ = compute_pca_components(adata.X, save_curve_img=None)
    else:
        matrix = to_dense_matrix(matrix)
        pca = PCA(n_components=n_pca_components)
        pcs = pca.fit_transform(matrix)

    adata.obsm[pca_key] = pcs[:, :n_pca_components]


def sctransform(
    adata: AnnData,
    rlib_path: str,
    n_top_genes: int = 3000,
) -> Optional[AnnData]:
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

    Returns:
        Updates adata with the field ``adata.obsm["pearson_residuals"]``, containing pearson_residuals.
    """
    import os

    os.environ["R_HOME"] = rlib_path

    try:
        from pysctransform import SCTransform
    except ImportError:
        raise ImportError(
            "You need to install the package `pysctransform`."
            "Install pysctransform via `pip install git+https://github.com/saketkc/pysctransform.git`"
        )

    residuals = SCTransform(adata, var_features_n=n_top_genes)
    adata.obsm["pearson_residuals"] = residuals


def integrate(
    adatas: List[AnnData],
    batch_key: str = "slice",
) -> AnnData:
    """Concatenate multiple different anndata objects.

    Args:
        adatas: AnnData matrices to concatenate with.
        batch_key: Add the batch annotation to :attr:`obs` using this key.

    Returns:
        integrated_adata: The concatenated AnnData, where adata.obs[batch_key] stores a categorical variable labeling
            the batch.
    """
    batch_ca = [adata.obs[batch_key][0] for adata in adatas]
    integrated_adata = adatas[0].concatenate(adatas[1:], batch_key=batch_key, batch_categories=batch_ca, join="outer")
    return integrated_adata


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


def spatial_adj_dyn(
    adata: AnnData,
    spatial_key: str = "spatial",
    pca_key: str = "pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
):
    """
    Calculate the adjacent matrix based on a neighborhood graph of gene expression space
    and a neighborhood graph of physical space.
    """
    import dynamo as dyn

    # Compute a neighborhood graph of gene expression space.
    dyn.tl.neighbors(adata, n_neighbors=e_neigh, basis=pca_key)

    # Compute a neighborhood graph of physical space.
    dyn.tl.neighbors(adata, n_neighbors=s_neigh, basis=spatial_key, result_prefix="spatial")

    # Calculate the adjacent matrix.
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1
    return adj
