"""
Wrapper function to run generative modeling for count denoising and imputation.
"""
from typing import List, Optional, Tuple, Union

import anndata
import numpy as np
import scipy
import spreg
from libpysal import weights

from ...configuration import SKM
from ...logging import logger_manager as lm
from ...plotting.static.space import space
from ...preprocessing.filter import filter_genes
from ...preprocessing.normalize import normalize_total
from ...preprocessing.transform import log1p
from ...tools.spatial_degs import moran_i
from .impute import STGNN


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def impute_and_downsample(
    adata: anndata.AnnData,
    filter_by_moran: bool = False,
    spatial_key: str = "spatial",
    positive_ratio_cutoff: float = 0.1,
    imputation: bool = True,
    n_ds: Optional[int] = None,
    to_visualize: Union[None, str, List[str]] = None,
    cmap: str = "magma",
    device: str = "cpu",
    **kwargs,
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Smooth gene expression distributions and downsample a spatial sample by selecting representative points from
    this smoothed slice.

    Args:
        adata: AnnData object to model
        filter_by_moran: Set True to split - for samples with highly uniform expression patterns, simple spatial
            smoothing will be used. For samples with localized patterns, graph neural network will be used for
            smoothing. If False, graph neural network will be applied to all genes.
        spatial_key: Only used if 'filter_by_moran' is True; key in .obsm where x- and y-coordinates are stored.
        positive_ratio_cutoff: Filter condition for genes- each gene must be present in higher than this proportion
            of the total number of cells to be retained
        imputation: Set True to perform imputation. If False, will only downsample.
        n_ds: Optional number of cells to downsample to- if not given, will not perform downsampling
        kwargs: Additional arguments that can be provided to :func `STGNN.train_STGNN`. Options for kwargs:
            - learn_rate: Float, controls magnitude of gradient for network learning
            - dropout: Float between 0 and 1, proportion of weights in each layer to set to 0
            - act: String specifying activation function for each encoder layer. Options: "sigmoid", "tanh", "relu",
                "elu"
            - clip: Float between 0 and 1, threshold below which imputed feature values will be set to 0,
                    as a percentile. Recommended between 0 and 0.1.
            - weight_decay: Float, controls degradation rate of parameters
            - epochs: Int, number of iterations of training loop to perform
            - dim_output: Int, dimensionality of the output representation
            - alpha: Float, controls influence of reconstruction loss in representation learning
            - beta: Float, weight factor to control the influence of contrastive loss in representation learning
            - theta: Float, weight factor to control the influence of the regularization term in representation learning
            - add_regularization: Bool, adds penalty term to representation learning

    Returns:
        adata_orig: Input AnnData object
        (optional) adata_rex:
        (optional) adata: AnnData subsetted down to downsampled buckets.
    """
    logger = lm.get_main_logger()
    if n_ds is None and not imputation:
        logger.error(
            "Neither downsampling nor imputation will be done (no integer has been provided to 'n_ds' and "
            "'imputation' is currently False)- exiting program."
        )

    adata_orig = adata.copy()
    normalize_total(adata_orig, 1e4)
    log1p(adata_orig)
    min_cells = int(adata_orig.n_obs * positive_ratio_cutoff)
    filter_genes(adata_orig, min_cells=min_cells)

    if imputation:
        if filter_by_moran:
            # Keep genes with significant Moran's I q-value (threshold = 0.05):
            m_degs = moran_i(adata_orig)
            m_uniform = m_degs[m_degs.moran_q_val >= 0.05].index
            m_degs = m_degs[m_degs.moran_q_val < 0.05].index

            adata_m_filt_out = adata_orig[:, m_uniform]
            adata_m_filt = adata_orig[:, m_degs]

            # For the genes with nonsignificant Moran's index, perform spatial smoothing:
            adata_m_filt_out_rex = adata_m_filt_out.copy()
            n_neighbors = int(0.01 * adata_m_filt_out.n_obs)
            w = weights.distance.KNN.from_array(adata_m_filt_out.obsm[spatial_key], k=n_neighbors)
            rec_lag = scipy.sparse.csr_matrix(spreg.utils.lag_spatial(w, adata_m_filt_out.X))
            rec_lag.eliminate_zeros()
            adata_m_filt_out_rex.X = rec_lag

            # For the genes with significant Moran's index, perform smoothing w/ generative modeling:
            model = STGNN(adata_m_filt, spatial_key, random_seed=50, add_regularization=False, device=device)
            adata_m_filt_rex = model.train_STGNN(**kwargs)
            # Set default layer to 'X_smooth_gcn' (the reconstruction):
            adata_m_filt_rex.X = adata_m_filt_rex.layers["X_smooth_gcn"]

            # Final smoothing:
            w = weights.distance.KNN.from_array(adata_m_filt_rex.obsm["spatial"], k=n_neighbors)
            rec_lag = scipy.sparse.csr_matrix(spreg.utils.lag_spatial(w, adata_m_filt_rex.X))
            rec_lag.eliminate_zeros()
            adata_m_filt_rex.X = rec_lag

            adata_rex = anndata.concat([adata_m_filt_rex, adata_m_filt_out_rex], axis=1)
            # .uns, .varm and .obsm are ignored by the concat operation- add back to the concatenated object:
            adata_rex.uns = adata_m_filt_rex.uns
            adata_rex.varm = adata_m_filt_rex.varm
            adata_rex.obsm = adata_m_filt_rex.obsm

            if to_visualize is not None:
                for feat in to_visualize:
                    # For plotting, normalize all columns of imputed and original data such that the maximum value is 1:
                    feat_idx = adata_orig.var_names.get_loc(feat)
                    adata_orig.X[:, feat_idx] /= np.max(adata_orig.X[:, feat_idx])
                    adata_rex.X[:, feat_idx] /= np.max(adata_rex.X[:, feat_idx])

                    # Generate two plots: one for observed data and one for imputed:
                    print(f"{feat} Observed")
                    size = 100 / adata_orig.n_obs
                    space(adata_orig, color=feat, cmap=cmap, figsize=(2.5, 2.5), dpi=300, pointsize=size, alpha=0.9)

                    print(f"{feat} Imputed")
                    size = 100 / adata_orig.n_obs
                    space(adata_rex, color=feat, cmap=cmap, figsize=(2.5, 2.5), dpi=300, pointsize=size, alpha=0.9)

        else:
            # Smooth all genes using generative modeling:
            model = STGNN(adata_orig, spatial_key, random_seed=50, add_regularization=False, device=device)
            adata_rex = model.train_STGNN(**kwargs)
            # Set default layer to 'X_smooth_gcn' (the reconstruction):
            adata_rex.X = adata_rex.layers["X_smooth_gcn"]

            if to_visualize is not None:
                for feat in to_visualize:
                    # Generate two plots: one for observed data and one for imputed:
                    print(f"{feat} Observed")
                    size = 100 / adata_orig.n_obs
                    space(adata_orig, color=feat, cmap=cmap, figsize=(5, 5), dpi=300, pointsize=size, alpha=0.9)

                    print(f"{feat} Imputed")
                    size = 100 / adata_orig.n_obs
                    space(adata_rex, color=feat, cmap=cmap, figsize=(5, 5), dpi=300, pointsize=size, alpha=0.9)

    # Add downsampling later:

    if imputation:
        return adata_rex, adata_orig
