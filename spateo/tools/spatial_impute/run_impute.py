"""
Wrapper function to run generative modeling for count denoising and imputation.
"""
from impute import STGNN

from typing import Union, List
import anndata
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from spateo.configuration import SKM


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def run_denoise_impute(adata: anndata.AnnData,
                       spatial_key: str = 'spatial',
                       device: str = 'cpu',
                       to_visualize: Union[None, str, List[str]] = None,
                       cmap: str = 'magma'):
    """
    Given AnnData object, perform gene expression denoising and imputation using a generative model. Assumes AnnData
    has been processed beforehand.

    Args:
        adata : class `anndata.AnnData`
            AnnData object to model
        spatial_key : str, default 'spatial'
            Key in .obsm where x- and y-coordinates are stored
        device : str, default 'cpu'
            Options: 'cpu', 'cuda:_', to run on either CPU or GPU. If running on GPU, provide the label of the device
            to run on.
        to_visualize : optional str or list of str
            If not None, will plot the observed gene expression values in addition to the expression values resulting
            from the reconstruction
        cmap : str, default 'magma'
            Colormap to use for visualization
    """
    # Copy original AnnData:
    adata_orig = adata.copy()
    model = STGNN(adata, spatial_key, random_seed=50, add_regularization=False, device=device)
    adata_rex = model.train_STGNN()
    # Set default layer to 'ReX' (the reconstruction):
    adata_rex.X = adata_rex.obsm['ReX']


    if to_visualize is not None:
        for feat in to_visualize:
            # Generate two plots: one for observed data and one for imputed:
            to_plot_orig = adata_orig[:, feat].X
            if sp.issparse(to_plot_orig):
                to_plot_orig = to_plot_orig.toarray()
            to_plot_rex = adata_rex[:, feat].X
            # For visualization, set max colormap value to the 99th percentile values:
            orig_vmax = np.percentile(to_plot_orig, 99)
            rex_vmax = np.percentile(to_plot_rex, 99)

            fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5), constrained_layout=True)
            size = 100000 / adata.n_obs
            scatterplot = ax.scatter(adata.obsm[spatial_key][:, 0],
                                          adata.obsm[spatial_key][:, 1],
                                          c=to_plot_orig,
                                          cmap=cmap,
                                          vmin=0, vmax=orig_vmax,
                                          s=size, alpha=1.0)
            ax.set_aspect('equal', 'datalim')
            ax.set_title(f'{feat.title()} Observed',
                         fontsize=14,
                         fontweight="bold",
                         )
            ax.set_ylim(ax.get_ylim()[::-1])
            cbar = fig.colorbar(scatterplot)
            # https://stackoverflow.com/questions/62812792/adding-colorbar-to-scatterplot-after-loop
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5), constrained_layout=True)
            size = 100000 / adata.n_obs
            scatterplot = ax.scatter(adata.obsm[spatial_key][:, 0],
                                     adata.obsm[spatial_key][:, 1],
                                     c=to_plot_rex,
                                     cmap=cmap,
                                     vmin=0, vmax=rex_vmax,
                                     s=size, alpha=1.0)
            ax.set_aspect('equal', 'datalim')
            ax.set_title(f'{feat.title()} Enhanced',
                         fontsize=14,
                         fontweight="bold",
                         )
            ax.set_ylim(ax.get_ylim()[::-1])
            cbar = fig.colorbar(scatterplot)
            plt.show()