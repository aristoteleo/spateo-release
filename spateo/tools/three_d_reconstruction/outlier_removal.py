import anndata as ad
import pandas as pd

from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KernelDensity


def om_kde(
    adata: ad.AnnData,
    coordsby: str = "spatial",
    threshold: float = 0.2,
    kernel: str = "gaussian",
    bandwidth: float = 1.0,
):
    """Outlier detection based on kernel density estimation."""

    coords = adata.obsm[coordsby]
    adata.obs["coords_kde"] = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(coords).score_samples(coords)

    CV = adata.obs["coords_kde"].describe(percentiles=[threshold])[f"{int(threshold*100)}%"]

    return adata[adata.obs["coords_kde"] > CV, :]


def om_EllipticEnvelope(
    adata: ad.AnnData,
    coordsby: str = "spatial",
    threshold: float = 0.05,
):
    """Outlier detection based on EllipticEnvelope algorithm."""

    coords = pd.DataFrame(adata.obsm[coordsby])
    adata.obs["outlier"] = EllipticEnvelope(contamination=threshold).fit(coords).predict(coords)

    return adata[adata.obs["outlier"] != -1, :]
