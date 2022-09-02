"""
Wrapper function to run generative modeling for count denoising and imputation.
"""
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from impute import STGNN

from ...configuration import SKM
from ...plotting.static.space import space


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def run_denoise_impute(
    adata: AnnData,
    spatial_key: str = "spatial",
    group_key: Union[None, str] = None,
    device: str = "cpu",
    to_visualize: Union[None, str, List[str]] = None,
    clip: Union[None, float] = None,
    cmap: str = "magma",
):
    """
    Given AnnData object, perform gene expression denoising and imputation using a generative model. Assumes AnnData
    has been processed beforehand.

    Args:
        adata : class `anndata.AnnData`
            AnnData object to model
        spatial_key : str, default 'spatial'
            Key in .obsm where x- and y-coordinates are stored
        group_key : optional str
            Key in .obs where labels are stored. If given, can plot categories alongside gene expression,
            i.e. to visualize marker genes with respect to label boundaries.
        device : str, default 'cpu'
            Options: 'cpu', 'cuda:_', to run on either CPU or GPU. If running on GPU, provide the label of the device
            to run on.
        to_visualize : optional str or list of str
            If not None, will plot the observed gene expression values in addition to the expression values resulting
            from the reconstruction
        clip : optional float
            Threshold below which imputed feature values will be set to 0, as a percentile
        cmap : str, default 'magma'
            Colormap to use for visualization
    """
    # Copy original AnnData:
    adata_orig = adata.copy()
    model = STGNN(adata, spatial_key, random_seed=50, add_regularization=False, device=device)
    adata_rex = model.train_STGNN(clip=clip)
    # Set default layer to 'X_smooth_gcn' (the reconstruction):
    adata_rex.X = adata_rex.layers["X_smooth_gcn"]

    if to_visualize is not None:
        for feat in to_visualize:
            # Generate two plots: one for observed data and one for imputed:
            # To be able to visualize side-by-side, temporarily append the smoothed feature to the original under a
            # different index in .var:
            adata_to_vis = adata_orig.copy()
            adata_to_vis.obs[feat + " denoised"] = adata_rex[:, feat].X

            size = 0.3 if len(adata_to_vis) < 3000 else 0.1
            if group_key is not None:
                space(adata_to_vis, color=[feat, feat + " denoised", group_key], ncols=3, cmap=cmap, dpi=300,
                      pointsize=size, alpha=1)
            else:
                space(adata_to_vis, color=[feat, feat + " denoised"], ncols=2, cmap=cmap, dpi=300, pointsize=size,
                      alpha=1)
