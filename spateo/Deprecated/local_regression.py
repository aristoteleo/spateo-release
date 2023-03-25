"""
Functions to perform regression over smaller subsets of the tissue.
"""
import inspect
import os
from typing import List, Literal, Optional, Union

from anndata import AnnData
from sklearn.cluster import KMeans

from spateo.Deprecated.generalized_lm import GLMCV
from spateo.Deprecated.spatial_regression import Niche_LR_Model, Niche_Model
from spateo.logging import logger_manager as lm
from spateo.plotting.static.space import space
from spateo.tools.ST_regression.regression_utils import compute_kmeans
from spateo.utils import remove_kwargs


# ---------------------------------------------------------------------------------------------------
# Functions for rationally-selected regions using clustering
# ---------------------------------------------------------------------------------------------------
def compute_all_niche():
    "filler"


def compute_all_niche_lr(
    adata: AnnData,
    lig: Union[None, str, List[str]],
    rec: Union[None, str, List[str]] = None,
    spatial_key: str = "spatial",
    spatial_regions_key: Optional[str] = None,
    n_spatial_regions: Optional[int] = None,
    cluster_with: Literal["position", "niche", "target", "spectral_position"] = "niche",
    distr: Union[None, Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"]] = None,
    group_key: Union[None, str] = None,
    targets: Union[None, List] = None,
    layer: Union[None, str] = None,
    cci_dir: Union[None, str] = None,
    species: Literal["human", "mouse", "axolotl"] = "human",
    normalize: bool = True,
    smooth: bool = False,
    log_transform: bool = False,
    niche_compute_indicator: bool = True,
    weights_mode: str = "knn",
    data_id: Union[None, str] = None,
    regression_gs_params: Union[None, dict] = None,
    regression_n_gs_cv: Union[None, int] = None,
    subset_key: Union[None, str] = None,
    subset_categories: Union[None, str, List[str]] = None,
    **kwargs,
):
    """Wraps models for spatially-aware regression using the prevalence of and connections between categories within
    spatial neighborhoods and the cell type-specific expression of ligands and receptors, applying our modeling
    approach to each of many possible tissue regions.

    Args:
        adata: object of class `anndata.AnnData`
        lig: Name(s) of ligands to use as predictors
        rec: Name(s) of receptors to use as regression targets. If not given, will search through database for all
            genes that correspond to the provided genes from 'ligands'
        group_key: Key in .obs where group (e.g. cell type) information can be found
        spatial_key: Key in .obsm where x- and y-coordinates are stored
        spatial_regions_key: Optional key in .obs where spatially-constrained cluster information can be found. If
            given, 'n_spatial_regions' will not be used.
        n_spatial_regions: Number of spatial regions to divide tissue section into. If not given, will attempt to
            select the best number of regions at runtime. Note that if this argument is one, will apply model to all
            spots in entire tissue section.
        cluster_with: Selects the method used to group cells- options: "position" to group based on similarity of the
            positions of cells, "niche" to group based on similarity of the niche composition of the cells,
            "target" to group based on similarity in expression of the downstream targets, "spectral_position" to use
            spectral clustering and cellular position for grouping.
        distr: Can optionally provide distribution family to specify the type of model that should be fit at the time
            of initializing this class rather than after calling :func `GLMCV_fit_predict`- can be "gaussian",
            "poisson", "softplus", "neg-binomial", or "gamma". Case sensitive.
        targets: Subset to genes of interest: will be used as dependent variables
        layer: Entry in .layers to use instead of .X when fitting model- all other operations will use .X.
        cci_dir: Full path to the directory containing cell-cell communication databases. Only used in the case of
            models that use ligands for prediction.
        species: Specifies L:R database to use, options: "human", "mouse", or "axolotl". Note that in the case of the
            axolotl, the human database will be used, so the input data needs to contain the human equivalents for
            axolotl gene names.
        normalize: Perform library size normalization, to set total counts in each cell to the same number (adjust
            for cell size)
        smooth: To correct for dropout effects, leverage gene expression neighborhoods to smooth expression
        log_transform: Set True if log-transformation should be applied to expression (otherwise, will assume
            preprocessing/log-transform was computed beforehand)
        niche_compute_indicator: For the cell type pair interactions array, threshold all nonzero values to 1 to reflect
            the presence of an interaction between the two cell types within each niche. Otherwise, will fit on
            normalized data.
        weights_mode: Options "knn", "kernel", "band"; sets whether to use K-nearest neighbors, a kernel-based
            method, or distance band to compute spatial weights, respectively.
        data_id: If given, will save pairwise distance arrays & nearest neighbor arrays to folder in the working
            directory, under './neighbors/{data_id}_distance.csv' and './neighbors/{data_id}_adj.csv'. Will also
            check for existing files under these names to avoid re-computing these arrays. If not given, will not save.
        regression_gs_params: Optional dictionary where keys are variable names for the regressor and
            values are lists of potential values for which to find the best combination using grid search.
        regression_n_gs_cv: Number of folds for grid search cross-validation, will only be used if gs_params is not
            None. If None, will default to a 5-fold cross-validation.
        subset_key: Optional, name of key in .obs containing categorical (e.g. cell type) information to be used for
            potentially subsetting AnnData object. Not used if 'subset_categories' is not provided.
        subset_categories: Optional, names of categories to subset to for the regression. In cases where the exogenous
            block is exceptionally heterogenous, can be used to narrow down the search space.
        kwargs : Provides additional spatial weight-finding arguments or arguments to :class `~GLMCV` or :class
            `~sklearn.cluster.KMeans`. Note that these must specifically match the name that the function will look
            for (case sensitive). For reference:
             - spatial weight-finding arguments:
                - n_neighbors : int
                    Number of nearest neighbors for KNN
                - p : int
                    Minkowski p-norm for KNN and distance band methods
                - distance_metric : str
                    Pairwise distance metric for KNN
                - bandwidth : float or array-like of floats
                    Sets kernel width for kernel method
                - fixed : bool
                    Allow bandwidth to vary across observations for kernel method
                - n_neighbors_bandwidth : int
                    Number of nearest neighbors for determining bandwidth for kernel method
                - kernel_function : str
                    "triangular", "uniform", "quadratic", "quartic" or "gaussian". Rule for setting how spatial
                    weight decays with distance
                - threshold : float
                    Distance for which to consider spots "neighbors" for each spot in distance band method (typically
                    in units of pixels)
                - alpha : float
                    Should be less than 0; can be used to set weights to decay with distance for distance band method

             - GLMCV (generalized linear model) class arguments:
                - alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.) term of the loss function
                - Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression.
                    If not provided, Tau will default to the identity matrix.
                - reg_lambda: Regularization parameter :math:`\\lambda` of penalty term
                - n_lambdas: Number of lambdas along the regularization path. Defaults to 25.
                - cv: Number of cross-validation repeats
                - learning_rate: Governs the magnitude of parameter updates for the gradient descent algorithm
                - max_iter: Maximum number of iterations for the solver
                - tol: Convergence threshold or stopping criteria. Optimization loop will stop when relative change in
                    parameter norm is below the threshold.
                - eta: A threshold parameter that linearizes the exp() function above eta
                - clip_coeffs: Absolute value below which to set coefficients to zero
                - score_metric: Scoring metric. Options:
                - "deviance": Uses the difference between the saturated (perfectly predictive) model and the true model.
                - "pseudo_r2": Uses the coefficient of determination b/w the true and predicted values.
                - fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision
                    function
                - random_seed: Seed of the random number generator used to initialize the solution. Default: 888
                - theta: Shape parameter of the negative binomial distribution (number of successes before the first
                    failure). It is used only if 'distr' is equal to "neg-binomial", otherwise it is ignored.

             - sklearn KMeans class arguments:
                - init: Method for initialization: options: 'k-means++', which selects initial cluster centroids
                    using sampling based on an empirical probability distribution of the points’ contribution to the
                    overall inertia, 'random', which chooses observations (rows) at random from data for the initial
                    centroids
                - n_init: Number of times the k-means algorithm is run with different centroid seeds. The final
                    result is the best output of n_init consecutive runs in terms of inertia.
                - max_iter: Maximum number of iterations of the k-means algorithm for a single run
                - tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of
                    two consecutive iterations to declare convergence
                - algorithm: K-means algorithm to use. Options: "lloyd", "elkan".

    Returns:

    """
    logger = lm.get_main_logger()

    # Split kwargs up based on destination:
    glmcv_kwargs_keys = inspect.signature(GLMCV).parameters
    glmcv_keys = [key for key in kwargs if key in glmcv_kwargs_keys]

    glmcv_kwargs = remove_kwargs(kwargs, glmcv_keys)
    glmcv_kwargs = dict(glmcv_kwargs)

    kmeans_kwargs_keys = inspect.signature(KMeans).parameters
    kmeans_keys = [key for key in kwargs if key in kmeans_kwargs_keys]

    kmeans_kwargs = remove_kwargs(kwargs, kmeans_keys)
    kmeans_kwargs = dict(kmeans_kwargs)

    # First, compute the complete matrix from the complete AnnData object- just to record the list of columns
    # Note that there is an option to subset to particular categories at the stage of running :func
    # `model.GLMCV_fit_predict()`- pass None to these arguments (note that this doesn't matter for this particular
    # model instance, but will apply to future ones):
    niche_lr_model = Niche_LR_Model(
        lig,
        rec,
        rec_ds=targets,
        species=species,
        niche_lr_r_lag=False,
        adata=adata,
        group_key=group_key,
        spatial_key=spatial_key,
        n_spatial_regions=n_spatial_regions,
        distr=distr,
        layer=layer,
        cci_dir=cci_dir,
        normalize=normalize,
        smooth=smooth,
        log_transform=log_transform,
        niche_compute_indicator=niche_compute_indicator,
        weights_mode=weights_mode,
        data_id=data_id,
        regression_gs_params=regression_gs_params,
        regression_n_gs_cv=regression_n_gs_cv,
        cat_key=None,
        categories=None,
    )
    all_var = niche_lr_model.X.columns

    # If spatial region boundaries are not defined beforehand, identify spatial regions based on niche characteristics:
    if spatial_regions_key is not None:
        regions = adata.obs[spatial_regions_key]
    else:
        niche_model = Niche_Model(
            adata=adata,
            group_key=group_key,
            genes=targets,
            normalize=normalize,
            smooth=smooth,
            log_transform=log_transform,
            niche_compute_indicator=niche_compute_indicator,
            weights_mode=weights_mode,
            data_id=data_id,
            regression_gs_params=regression_gs_params,
            regression_n_gs_cv=regression_n_gs_cv,
            cat_key=None,
            categories=None,
        )

        # Perform subsetting- note that there is an option to do so also at the stage of running :func
        # `model.GLMCV_fit_predict()`- pass None to these arguments:
        if subset_categories is not None:
            if not isinstance(subset_categories, list):
                subset_categories = [subset_categories]

            if subset_key is None:
                logger.info(
                    "Argument to 'subset_categories' were given, but not 'subset_key' specifying where in .obs to "
                    "look. Defaulting to 'group_key'."
                )

            # Filter adata for rows annotated as being any category in 'categories', and X block for columns annotated
            # with any of the categories in 'categories'.
            adata = adata[adata.obs[subset_key].isin(subset_categories)]
            logger.info(f"Subsetted to categories {','.join(subset_categories)}")

            cell_names = adata.obs_names
            niche_data = niche_model.X.loc[cell_names]
        else:
            niche_data = niche_model.X

        # Perform K-means clustering to split tissue sample into regions:
        if cluster_with == "position":
            adata.obs["spatial_region"] = compute_kmeans(
                data=adata.obsm[spatial_key], k_custom=n_spatial_regions, plot_knee=True, **kmeans_kwargs
            )
        elif cluster_with == "niche":
            adata.obs["spatial_region"] = compute_kmeans(
                data=niche_data, k_custom=n_spatial_regions, plot_knee=True, **kmeans_kwargs
            )
        elif cluster_with == "target":
            adata.obs["spatial_region"] = compute_kmeans(
                data=adata[:, niche_lr_model.genes].X, k_custom=n_spatial_regions, plot_knee=True, **kmeans_kwargs
            )
        else:
            logger.error("Invalid option given to argument 'cluster_with'. Options: 'position', 'niche', 'target'.")

        adata.obs["spatial_region"] = [f"Region {i}" for i in adata.obs["spatial_region"]]
        spatial_regions_key = "spatial_region"
        regions = adata.obs["spatial_region"]
        n_regions = len(set(adata.obs["spatial_region"]))
        logger.info(f"Identified {n_regions} subsets to create models for.")

    # Plot of spatial regions:
    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    space(
        adata,
        color=[spatial_regions_key],
        pointsize=0.05,
        show_legend="upper left",
        save_show_or_return="save",
        figsize=(3, 3),
        save_kwargs={"prefix": f"./figures/{data_id}_spatial_regions", "ext": "png"},
    )

    # For each AnnData subset, perform regression- note at runtime that function will suggest that an argument be
    # provided to 'subset_categories' if not already the case:
    for region in set(adata.obs[spatial_regions_key]):
        "filler"

        # Add all-zero columns for the unused cell types:

    # Compile the results as an entry in .obsm- for regions where certain cell type-cell type pairs are not present,
    # add lists of all zeros

    # Use coefficients for arrow drawing:
    # coefficients = model.coefficients for each of the models

    # Can return the complete coefficients matrix


# class ...():
# coefficients = model.coefficients


# Main GWR classes
