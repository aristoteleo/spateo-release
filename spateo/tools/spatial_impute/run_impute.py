"""
Wrapper function to run generative modeling for count denoising and imputation.
"""
from typing import List, Union

from anndata import AnnData

from ...configuration import SKM
from ...plotting.static.space import space
from .impute import STGNN


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def run_denoise_impute(
    adata: AnnData,
    spatial_key: str = "spatial",
    device: str = "cpu",
    to_visualize: Union[None, str, List[str]] = None,
    cmap: str = "magma",
    **kwargs,
):
    """
    Given AnnData object, perform gene expression denoising and imputation using a generative model. Assumes AnnData
    has been processed beforehand.

    Args:
        adata: AnnData object to model
        spatial_key: Key in .obsm where x- and y-coordinates are stored
        device: Options: 'cpu', 'cuda:_', to run on either CPU or GPU. If running on GPU, provide the label of the
            device to run on.
        to_visualize: If not None, will plot the observed gene expression values in addition to the expression values
            resulting from the reconstruction
        cmap: Colormap to use for visualization
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
    """
    # Copy original AnnData:
    adata_orig = adata.copy()
    model = STGNN(adata, spatial_key, random_seed=50, add_regularization=False, device=device)
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
