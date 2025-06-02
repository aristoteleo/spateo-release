from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix

# Import libraries needed for spatial graph processing
from scipy.spatial import Delaunay
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    ArchesMixin,
    BaseMinifiedModeModelClass,
    EmbeddingMixin,
    RNASeqMixin,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.module import VAE
from scvi.utils import setup_anndata_dsp
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class SpatialSCVI(
    EmbeddingMixin,
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass,
):
    """Single-cell Variational Inference model.

    This model uses a variational autoencoder (VAE) to learn low-dimensional representations of single-cell data.
    It can be used for batch effect correction, differential expression analysis, data imputation, and various other downstream tasks.

    Parameters
    ----------
    adata
        AnnData object containing single-cell data, which must be registered through the setup_anndata method.
    n_hidden
        Number of nodes in hidden layers.
    n_latent
        Dimension of the latent space (dimension of low-dimensional representation).
    n_spatial
        Dimension of spatial features. Effective when use_spatial=True.
    n_layers
        Number of hidden layers in encoder and decoder networks.
    dropout_rate
        Ratio of neurons randomly dropped during training.
    dispersion
        How to model variance parameters:
        * 'gene' - one parameter per gene
        * 'gene-batch' - different parameters for each gene in each batch
        * 'gene-label' - different parameters for each gene in each label group
        * 'gene-cell' - different parameters for each gene in each cell
    gene_likelihood
        Distribution used to model gene expression:
        * 'nb' - negative binomial distribution (for count data with overdispersion)
        * 'zinb' - zero-inflated negative binomial distribution (for sparse count data)
        * 'poisson' - Poisson distribution (for count data)
        * 'normal' - normal distribution (experimental)
    use_observed_lib_size
        Whether to use observed library size as scaling factor.
    latent_distribution
        Type of latent distribution:
        * 'normal' - standard normal distribution
        * 'ln' - logistic normal distribution
    use_spatial
        Whether to use spatial information. If True, spatial coordinates will be read from adata.obsm['spatial'].
    spatial_graph_type
        Method for spatial graph construction:
        * 'knn' - K-nearest neighbors graph
        * 'delaunay' - Delaunay triangulation
    n_neighbors
        Number of neighbors for KNN graph when spatial_graph_type='knn'.
    attention_heads
        Number of attention heads in graph attention network.
    spatial_kl_weight
        Weight for KL divergence of spatial latent variables.
    **kwargs
        Additional parameters passed to the VAE module.

    Examples
    --------
    >>> import anndata
    >>> import scvi
    >>> adata = anndata.read_h5ad("my_data.h5ad")
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> model = scvi.model.SCVI(adata, use_spatial=True)
    >>> model.train()
    >>> adata.obsm["X_scVI"] = model.get_latent_representation()
    """

    _module_cls = VAE
    # Key names for latent variable mean and variance
    latent_mean_key = "scvi_latent_qzm"
    latent_var_key = "scvi_latent_qzv"

    def __init__(
        self,
        adata: AnnData | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_spatial: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_spatial: bool = False,
        spatial_graph_type: Literal["knn", "delaunay"] = "knn",
        n_neighbors: int = 10,
        attention_heads: int = 1,
        spatial_kl_weight: float = 0.01,
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(adata)

        # Store model configuration parameters
        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_spatial": n_spatial,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs,
        }

        # Create model summary string
        self._model_summary_string = (
            "SpatialSCVI model parameter summary: \n"
            f"Hidden layer nodes: {n_hidden}, Latent dimensions: {n_latent}, Spatial dimensions: {n_spatial}, "
            f"Layers: {n_layers}, Dropout rate: {dropout_rate}, "
            f"Dispersion type: {dispersion}, Gene likelihood: {gene_likelihood}, "
            f"Use spatial info: {use_spatial}, Spatial KL weight: {spatial_kl_weight}"
        )

        # Spatial related parameters
        self.use_spatial = use_spatial
        self.spatial_graph_type = spatial_graph_type
        self.n_neighbors = n_neighbors
        self.attention_heads = attention_heads
        self.spatial_kl_weight = spatial_kl_weight
        self.edge_index = None  # Edge indices for spatial graph

        # If using spatial information, set up spatial graph
        if use_spatial and adata is not None:
            if "spatial" not in adata.obsm:
                warnings.warn(
                    "Using spatial information but no spatial coordinates found in adata.obsm['spatial']. "
                    "Please provide coordinates in adata.obsm['spatial'].",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            else:
                self.setup_spatial_graph(adata)

        # Handle case where adata is not provided during initialization
        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "No adata provided during model initialization. Module will be initialized when train() is called. "
                "This is an experimental feature and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            # Get categorical covariate information (if available)
            cat_covs_per_group = (
                self.adata_manager.get_state_registry("categorical_covariates").n_cats_per_key
                if "categorical_covariates" in self.adata_manager.data_registry
                else None
            )

            # Get number of batches from data
            num_batches = self.summary_stats.n_batch

            # Check if size factor is provided in data
            use_size_factor = "size_factor" in self.adata_manager.data_registry

            # Initialize library size parameters if needed
            lib_mean, lib_var = None, None
            if (
                not use_size_factor
                and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
                and not use_observed_lib_size
            ):
                lib_mean, lib_var = _init_library_size(self.adata_manager, num_batches)

            # CreateVAE module - add spatial related parameters
            from .scvi_spatial_module import SpatialVAE  # Import new spatial VAE module

            if use_spatial:
                # Use VAE module with spatial information
                self.module = SpatialVAE(
                    n_input=self.summary_stats.n_vars,
                    n_batch=num_batches,
                    n_labels=self.summary_stats.n_labels,
                    n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                    n_cats_per_cov=cat_covs_per_group,
                    n_hidden=n_hidden,
                    n_latent=n_latent,
                    n_spatial=n_spatial,
                    n_layers=n_layers,
                    dropout_rate=dropout_rate,
                    dispersion=dispersion,
                    gene_likelihood=gene_likelihood,
                    use_observed_lib_size=use_observed_lib_size,
                    latent_distribution=latent_distribution,
                    use_size_factor_key=use_size_factor,
                    library_log_means=lib_mean,
                    library_log_vars=lib_var,
                    edge_index=self.edge_index,
                    attention_heads=attention_heads,
                    spatial_kl_weight=spatial_kl_weight,
                    **kwargs,
                )
            else:
                # Use standard VAE module
                self.module = self._module_cls(
                    n_input=self.summary_stats.n_vars,
                    n_batch=num_batches,
                    n_labels=self.summary_stats.n_labels,
                    n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                    n_cats_per_cov=cat_covs_per_group,
                    n_hidden=n_hidden,
                    n_latent=n_latent,
                    n_layers=n_layers,
                    dropout_rate=dropout_rate,
                    dispersion=dispersion,
                    gene_likelihood=gene_likelihood,
                    use_observed_lib_size=use_observed_lib_size,
                    latent_distribution=latent_distribution,
                    use_size_factor_key=use_size_factor,
                    library_log_means=lib_mean,
                    library_log_vars=lib_var,
                    **kwargs,
                )

            self.module.minified_data_type = self.minified_data_type

        # Store initialization parameters
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """Set up AnnData object for SpatialSCVI model training.

        This method prepares for training by registering necessary data fields.

        Parameters
        ----------
        adata
            AnnData object containing single-cell data
        layer
            If provided, use this layer's expression values instead of adata.X
        batch_key
            Key in adata.obs representing batch information. If None, assumes all cells are from same batch.
        labels_key
            Key in adata.obs representing cell type or other label information
        size_factor_key
            Key in adata.obs representing size factor information. Used for normalization.
        categorical_covariate_keys
            List of keys in adata.obs representing categorical covariates (like experimental conditions)
        continuous_covariate_keys
            List of keys in adata.obs representing continuous covariates (like quality control metrics)
        """
        # Store setup parameters
        setup_args = cls._get_setup_method_args(**locals())

        # Define all data fields needed by the model
        data_fields = [
            # Main expression data (X)
            LayerField("X", layer, is_count_data=True),
            # Batch information, used for handling batch effects
            CategoricalObsField("batch", batch_key),
            # Cell type or other label information
            CategoricalObsField("labels", labels_key),
            # Size factor for normalization (optional)
            NumericalObsField("size_factor", size_factor_key, required=False),
            # Additional categorical and continuous covariates
            CategoricalJointObsField("categorical_covariates", categorical_covariate_keys),
            NumericalJointObsField("continuous_covariates", continuous_covariate_keys),
        ]

        # Check if data is in minified format and add appropriate fields
        data_type = _get_adata_minify_type(adata)
        if data_type is not None:
            # Add required fields for minified data format
            data_fields += cls._get_fields_for_adata_minification(data_type)

        # Create manager for data fields
        adata_manager = AnnDataManager(fields=data_fields, setup_method_args=setup_args)

        # Register all fields to AnnData object
        adata_manager.register_fields(adata, **kwargs)

        # Register manager to model class
        cls.register_manager(adata_manager)

    def setup_spatial_graph(self, adata: AnnData):
        """Set up spatial graph for spatial information processing.

        Build spatial graph based on spatial coordinates in adata.obsm['spatial'],
        choosing between K-nearest neighbors graph or Delaunay triangulation.

        Parameters
        ----------
        adata
            AnnData object containing spatial coordinates in adata.obsm['spatial']
        """
        try:
            # Get spatial coordinates
            if "spatial" not in adata.obsm:
                raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")

            coordinates = adata.obsm["spatial"]
            logger.info(
                f"Retrieved spatial coordinates for {len(coordinates)} cells, dimension: {coordinates.shape[1]}"
            )

            # Check validity of coordinate data
            if np.isnan(coordinates).any():
                # Handle NaN values
                logger.warning("Spatial coordinates contain NaN values, will use mean filling")
                coordinates = np.nan_to_num(coordinates, nan=np.nanmean(coordinates, axis=0))

            # Build spatial graph based on chosen method
            if self.spatial_graph_type == "knn":
                # Build K-nearest neighbors graph
                logger.info(f"Building spatial graph using KNN method, neighbors: {self.n_neighbors}")
                adj_matrix = kneighbors_graph(
                    coordinates,
                    n_neighbors=min(self.n_neighbors, len(coordinates) - 1),
                    mode="connectivity",
                    include_self=False,
                    n_jobs=-1,  # Use all available CPU cores to speed up computation
                )
                # Ensure graph is symmetric
                adj_matrix = adj_matrix + adj_matrix.T
                adj_matrix.data = np.ones(adj_matrix.data.shape)
                # Remove duplicate edges
                adj_matrix = adj_matrix.tocsr()
                adj_matrix.data = np.ones_like(adj_matrix.data)

            elif self.spatial_graph_type == "delaunay":
                # Build Delaunay triangulation
                logger.info("Building spatial graph using Delaunay triangulation")

                # Ensure coordinates are 2D or 3D, Delaunay needs at least 2D
                if coordinates.shape[1] == 1:
                    # If only 1D coordinates, add a fake second dimension
                    logger.warning(
                        "Coordinates are only 1D, adding random small noise as second dimension for Delaunay triangulation"
                    )
                    coordinates_2d = np.column_stack(
                        [coordinates, np.random.normal(0, 0.01, size=(len(coordinates), 1))]
                    )
                else:
                    coordinates_2d = coordinates

                # Perform triangulation
                tri = Delaunay(coordinates_2d)
                # Get adjacency from triangulation
                edges = set()
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            # Add undirected edge (both directions)
                            edges.add((simplex[i], simplex[j]))
                            edges.add((simplex[j], simplex[i]))

                # Create sparse adjacency matrix
                if edges:
                    rows, cols = zip(*edges)
                    data = np.ones(len(rows))
                    adj_matrix = coo_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs)).tocsr()
                else:
                    logger.warning("Delaunay triangulation did not generate any edges, will use fully connected graph")
                    # Create a sparse graph with a few random connections
                    adj_matrix = kneighbors_graph(
                        coordinates, n_neighbors=min(10, len(coordinates) - 1), mode="connectivity"
                    )
            else:
                raise ValueError(f"Unknown spatial graph type: {self.spatial_graph_type}")

            # Check graph connectivity
            n_components = 1  # Assume connected
            if adj_matrix.nnz == 0:
                logger.warning("Generated graph has no edges, will use fully connected graph")
                # Create fully connected graph
                rows, cols = np.where(np.ones((adata.n_obs, adata.n_obs)) - np.eye(adata.n_obs))
                data = np.ones(len(rows))
                adj_matrix = coo_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs))

            # Convert to PyTorch Geometric edge index format
            edge_index, _ = from_scipy_sparse_matrix(adj_matrix)

            # Move to appropriate device
            if hasattr(self, "device"):
                device = self.device
            elif hasattr(self, "module") and hasattr(self.module, "device"):
                device = self.module.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.edge_index = edge_index.to(device)

            logger.info(
                f"Successfully built spatial graph, type: {self.spatial_graph_type}, "
                f"nodes: {adata.n_obs}, edges: {edge_index.shape[1] // 2}, "
                f"device: {device}"
            )

        except Exception as e:
            # Catch any errors, provide detailed error information
            import traceback

            logger.error(f"Error building spatial graph: {str(e)}\n{traceback.format_exc()}")
            warnings.warn(
                f"Error building spatial graph: {str(e)}. Will not use spatial information.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            self.edge_index = None
            self.use_spatial = False


class MERFISHVI(SpatialSCVI):
    """MERFISH spatial multimodal variational inference model.

    This model extends SCVI to simultaneously process MERFISH data with spatial information and
    other modality data (such as scRNA-seq) without spatial information. It uses a shared
    latent space to jointly model both modalities.

    Parameters
    ----------
    adata_spatial
        AnnData object containing spatial modality data
    adata_nonspatial
        AnnData object containing non-spatial modality data, optional
    n_hidden
        Number of nodes in hidden layers
    n_latent
        Dimension of latent space
    n_spatial
        Dimension of spatial features
    n_layers
        Number of hidden layers in encoder and decoder networks
    dropout_rate
        Dropout rate during training
    dispersion
        How to model variance parameters
    gene_likelihood
        Distribution used to model gene expression
    latent_distribution
        Type of latent distribution
    spatial_graph_type
        Spatial graph construction method, 'knn' or 'delaunay'
    n_neighbors
        Number of neighbors when spatial_graph_type='knn'
    attention_heads
        Number of attention heads in graph attention network
    spatial_kl_weight
        Weight for spatial feature KL divergence
    modality_weights
        Modality weights to balance contribution of each modality in the loss function
    **kwargs
        Additional parameters passed to VAE module
    """

    def __init__(
        self,
        adata_spatial: AnnData,
        adata_nonspatial: Optional[AnnData] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_spatial: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        spatial_graph_type: Literal["knn", "delaunay"] = "knn",
        n_neighbors: int = 10,
        attention_heads: int = 1,
        spatial_kl_weight: float = 0.01,
        modality_weights: Dict[str, float] = {"spatial": 1.0, "nonspatial": 1.0},
        **kwargs,
    ):
        # Store non-spatial modality data
        self.adata_nonspatial = adata_nonspatial
        self.modality_weights = modality_weights

        # Initialize parent class using spatial modality data
        super().__init__(adata_spatial)

        # Store model configuration parameters
        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_spatial": n_spatial,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs,
        }

        # Create model summary string
        self._model_summary_string = (
            "MERFISHVI model parameter summary: \n"
            f"Hidden layer nodes: {n_hidden}, Latent dimensions: {n_latent}, Spatial dimensions: {n_spatial}, "
            f"Layers: {n_layers}, Dropout rate: {dropout_rate}, "
            f"Dispersion type: {dispersion}, Gene likelihood: {gene_likelihood}, "
            f"Spatial graph type: {spatial_graph_type}, Spatial KL weight: {spatial_kl_weight}, "
            f"Modality weights: {modality_weights}"
        )

        # Spatial related parameters
        self.spatial_graph_type = spatial_graph_type
        self.n_neighbors = n_neighbors
        self.attention_heads = attention_heads
        self.spatial_kl_weight = spatial_kl_weight
        self.edge_index = None

        # Set up spatial graph for spatial modality
        self.setup_spatial_graph(adata_spatial)

        # Set up non-spatial modality data manager
        if adata_nonspatial is not None:
            self.setup_nonspatial_anndata(adata_nonspatial)

        # Initialize model
        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "No adata provided during model initialization. Module will be initialized when train() is called. "
                "This is an experimental feature and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            # Create model
            self._create_module()
            self.module.minified_data_type = self.minified_data_type

        # Store initialization parameters
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_nonspatial_anndata(cls, adata: AnnData, **kwargs):
        """Set up non-spatial modality AnnData object.

        Parameters
        ----------
        adata
            Non-spatial modality AnnData object
        """
        # Create separate data manager for non-spatial modality

        # Store setup parameters
        setup_args = cls._get_setup_method_args(**locals())

        # Define all data fields needed by the model
        data_fields = [
            LayerField("X", None, is_count_data=True),
            CategoricalObsField("batch", None),
            CategoricalObsField("labels", None),
            NumericalObsField("size_factor", None, required=False),
            CategoricalJointObsField("categorical_covariates", None),
            NumericalJointObsField("continuous_covariates", None),
        ]

        # Check if data is in minified format and add appropriate fields
        data_type = _get_adata_minify_type(adata)
        if data_type is not None:
            # Add required fields for minified data format
            data_fields += cls._get_fields_for_adata_minification(data_type)

        # Create manager for data fields
        adata_manager = AnnDataManager(fields=data_fields, setup_method_args=setup_args)

        # Register all fields to AnnData object
        adata_manager.register_fields(adata, **kwargs)

        # Register manager to model class
        cls.register_manager(adata_manager)

        logger.info(f"Successfully set up non-spatial modality data, shape: {adata.shape}")

    def _create_module(self):
        """Create model module"""
        # Get spatial modality data information
        n_vars_spatial = self.summary_stats.n_vars
        n_batch_spatial = self.summary_stats.n_batch
        n_labels_spatial = self.summary_stats.n_labels

        # Get spatial modality categorical covariate information
        cat_covs_per_group_spatial = (
            self.adata_manager.get_state_registry("categorical_covariates").n_cats_per_key
            if "categorical_covariates" in self.adata_manager.data_registry
            else None
        )

        # Use size factor
        use_size_factor_spatial = "size_factor" in self.adata_manager.data_registry

        # Initialize library size parameters
        lib_mean_spatial, lib_var_spatial = None, None
        if (
            not use_size_factor_spatial
            and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
            and not self._module_kwargs.get("use_observed_lib_size", True)
        ):
            lib_mean_spatial, lib_var_spatial = _init_library_size(self.adata_manager, n_batch_spatial)

        # Check if non-spatial modality is available
        has_nonspatial = hasattr(self, "adata_manager_nonspatial")

        if has_nonspatial:
            # Get non-spatial modality data information
            n_vars_nonspatial = self.adata_manager_nonspatial.summary_stats.n_vars
            n_batch_nonspatial = self.adata_manager_nonspatial.summary_stats.n_batch
            n_labels_nonspatial = self.adata_manager_nonspatial.summary_stats.n_labels

            # Get non-spatial modality categorical covariate information
            cat_covs_per_group_nonspatial = (
                self.adata_manager_nonspatial.get_state_registry("categorical_covariates").n_cats_per_key
                if "categorical_covariates" in self.adata_manager_nonspatial.data_registry
                else None
            )

            # Use size factor
            use_size_factor_nonspatial = "size_factor" in self.adata_manager_nonspatial.data_registry

            # Initialize library size parameters
            lib_mean_nonspatial, lib_var_nonspatial = None, None
            if (
                not use_size_factor_nonspatial
                and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
                and not self._module_kwargs.get("use_observed_lib_size", True)
            ):
                lib_mean_nonspatial, lib_var_nonspatial = _init_library_size(
                    self.adata_manager_nonspatial, n_batch_nonspatial
                )

        # Import multimodal spatial VAE module
        from .multimodal_spatial_vae import MultiModalSpatialVAE

        # Create model module
        if has_nonspatial:
            self.module = MultiModalSpatialVAE(
                n_input_spatial=n_vars_spatial,
                n_input_nonspatial=n_vars_nonspatial,
                n_batch_spatial=n_batch_spatial,
                n_batch_nonspatial=n_batch_nonspatial,
                n_labels_spatial=n_labels_spatial,
                n_labels_nonspatial=n_labels_nonspatial,
                n_hidden=self._module_kwargs["n_hidden"],
                n_latent=self._module_kwargs["n_latent"],
                n_spatial=self._module_kwargs["n_spatial"],
                n_layers=self._module_kwargs["n_layers"],
                dropout_rate=self._module_kwargs["dropout_rate"],
                dispersion=self._module_kwargs["dispersion"],
                gene_likelihood=self._module_kwargs["gene_likelihood"],
                latent_distribution=self._module_kwargs["latent_distribution"],
                use_observed_lib_size=self._module_kwargs.get("use_observed_lib_size", True),
                edge_index=self.edge_index,
                attention_heads=self.attention_heads,
                spatial_kl_weight=self.spatial_kl_weight,
                modality_weights=self.modality_weights,
                cats_per_cov_spatial=cat_covs_per_group_spatial,
                cats_per_cov_nonspatial=cat_covs_per_group_nonspatial,
                use_size_factor_spatial=use_size_factor_spatial,
                use_size_factor_nonspatial=use_size_factor_nonspatial,
                library_log_means_spatial=lib_mean_spatial,
                library_log_vars_spatial=lib_var_spatial,
                library_log_means_nonspatial=lib_mean_nonspatial,
                library_log_vars_nonspatial=lib_var_nonspatial,
                **{
                    k: v
                    for k, v in self._module_kwargs.items()
                    if k
                    not in [
                        "n_hidden",
                        "n_latent",
                        "n_spatial",
                        "n_layers",
                        "dropout_rate",
                        "dispersion",
                        "gene_likelihood",
                        "latent_distribution",
                        "use_observed_lib_size",
                    ]
                },
            )
        else:
            # Only use spatial modality
            from .scvi_spatial_module import SpatialVAE

            self.module = SpatialVAE(
                n_input=n_vars_spatial,
                n_batch=n_batch_spatial,
                n_labels=n_labels_spatial,
                n_hidden=self._module_kwargs["n_hidden"],
                n_latent=self._module_kwargs["n_latent"],
                n_spatial=self._module_kwargs["n_spatial"],
                n_layers=self._module_kwargs["n_layers"],
                dropout_rate=self._module_kwargs["dropout_rate"],
                dispersion=self._module_kwargs["dispersion"],
                gene_likelihood=self._module_kwargs["gene_likelihood"],
                latent_distribution=self._module_kwargs["latent_distribution"],
                use_observed_lib_size=self._module_kwargs.get("use_observed_lib_size", True),
                edge_index=self.edge_index,
                attention_heads=self.attention_heads,
                spatial_kl_weight=self.spatial_kl_weight,
                n_cats_per_cov=cat_covs_per_group_spatial,
                use_size_factor_key=use_size_factor_spatial,
                library_log_means=lib_mean_spatial,
                library_log_vars=lib_var_spatial,
                **{
                    k: v
                    for k, v in self._module_kwargs.items()
                    if k
                    not in [
                        "n_hidden",
                        "n_latent",
                        "n_spatial",
                        "n_layers",
                        "dropout_rate",
                        "dispersion",
                        "gene_likelihood",
                        "latent_distribution",
                        "use_observed_lib_size",
                    ]
                },
            )

    # Add methods for getting multimodal representations
    @torch.inference_mode()
    def get_spatial_representation(self, adata=None, indices=None, batch_size=None):
        """Get spatial feature representation"""
        if hasattr(self.module, "get_spatial_representation"):
            return self.module.get_spatial_representation(adata, indices, batch_size)
        else:
            raise AttributeError("Current model does not support getting spatial representation")

    @torch.inference_mode()
    def get_latent_representation(self, adata=None, indices=None, batch_size=None, modality="spatial"):
        """Get latent representation, can choose from spatial modality, non-spatial modality, or fused representation"""
        if hasattr(self.module, "get_latent_representation_by_modality"):
            return self.module.get_latent_representation_by_modality(adata, indices, batch_size, modality)
        else:
            return super().get_latent_representation(adata, indices, batch_size)

    @torch.inference_mode()
    def get_fused_representation(self, adata=None, indices=None, batch_size=None):
        """Get fused latent representation"""
        if hasattr(self.module, "get_fused_representation"):
            return self.module.get_fused_representation(adata, indices, batch_size)
        else:
            return self.get_latent_representation(adata, indices, batch_size)
