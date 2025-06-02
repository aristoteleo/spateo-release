from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, kl_divergence

# Import GATv2Conv with fallback
try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    try:
        from torch_geometric.nn.conv import GATv2Conv
    except ImportError:
        raise ImportError("Failed to import GATv2Conv, please install PyTorch Geometric")

from scvi import REGISTRY_KEYS

# Import auto_move_data with fallback options
try:
    from scvi.module.base import LossOutput, auto_move_data
except ImportError:
    try:
        from scvi.module.base import LossOutput
        from scvi.nn.base import auto_move_data
    except ImportError:
        try:
            from scvi.model.base import LossOutput
            from scvi.nn import auto_move_data
        except ImportError:
            raise ImportError("Failed to import auto_move_data and LossOutput, please check scvi-tools version")

# Import unsupported_if_adata_minified
try:
    from scvi.utils import unsupported_if_adata_minified
except ImportError:
    try:
        from scvi.model.base import unsupported_if_adata_minified
    except ImportError:
        # If not available, create a dummy decorator
        def unsupported_if_adata_minified(fn):
            return fn


from scvi.nn import Encoder

# Import VAE with fallbacks
try:
    from scvi.module import VAE
except ImportError:
    try:
        from scvi.module.base import VAE
    except ImportError:
        try:
            from scvi.nn import VAE
        except ImportError:
            raise ImportError("Failed to import VAE class, please check scvi-tools version")

from anndata import AnnData

# Import AnnTorchDataset with fallbacks
try:
    from scvi.data import AnnTorchDataset
except ImportError:
    try:
        from scvi.data._anntorchdataset import AnnTorchDataset
    except ImportError:
        try:
            from scvi.dataloaders import AnnTorchDataset
        except ImportError:
            # Define a placeholder that will raise a more helpful error when used
            class AnnTorchDataset:
                def __init__(self, *args, **kwargs):
                    raise ImportError("Failed to import AnnTorchDataset class, please check scvi-tools version")


logger = logging.getLogger(__name__)


class SpatialEncoder(nn.Module):
    """Spatial encoder that uses graph attention networks to process spatial information.

    Applies graph attention network to latent representations to obtain spatial features.

    Parameters
    ----------
    n_latent
        Dimension of the latent space
    n_spatial
        Dimension of the spatial features
    attention_heads
        Number of attention heads
    dropout_rate
        Dropout ratio
    var_eps
        Minimum value for variance to ensure numerical stability
    """

    def __init__(
        self,
        n_latent: int,
        n_spatial: int,
        attention_heads: int = 1,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
    ):
        super().__init__()

        # Graph attention network to transform latent representations into spatial features
        self.gat = GATv2Conv(
            in_channels=n_latent, out_channels=n_spatial, heads=attention_heads, dropout=dropout_rate, concat=False
        )

        # Fully connected layers for computing spatial feature distribution parameters
        self.mean_encoder = nn.Linear(n_spatial, n_spatial)
        self.var_encoder = nn.Linear(n_spatial, n_spatial)
        self.n_spatial = n_spatial

        self.var_eps = var_eps

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass, calculate spatial feature distribution.

        Parameters
        ----------
        z
            Latent representation, shape [batch_size, n_latent]
        edge_index
            Graph edge indices, shape [2, num_edges]

        Returns
        -------
        tuple
            Mean, variance and sampled value of the spatial feature distribution
        """
        batch_size = z.size(0)
        device = z.device

        # Check if edge index is valid
        if edge_index is None or edge_index.shape[1] == 0:
            # If no edges, return zero tensors
            spatial_mean = torch.zeros(batch_size, self.n_spatial, device=device)
            spatial_var = torch.ones(batch_size, self.n_spatial, device=device) * self.var_eps
            spatial_sample = torch.zeros(batch_size, self.n_spatial, device=device)

            warnings.warn("Edge index is empty or has no edges, returning zero tensor as spatial features", UserWarning)
            return spatial_mean, spatial_var, spatial_sample

        # Ensure edge_index is on the correct device
        if edge_index.device != device:
            edge_index = edge_index.to(device)

        # Check if edge_index is valid (indices not exceeding node count)
        max_index = torch.max(edge_index)
        if max_index >= batch_size:
            warnings.warn(
                f"Edge index contains out-of-range indices, max index is {max_index.item()}, node count is {batch_size}. Will trim edge indices.",
                UserWarning,
            )
            # Select only edges with valid nodes
            valid_edges = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
            if valid_edges.sum() == 0:
                # If no valid edges, return zero tensors
                spatial_mean = torch.zeros(batch_size, self.n_spatial, device=device)
                spatial_var = torch.ones(batch_size, self.n_spatial, device=device) * self.var_eps
                spatial_sample = torch.zeros(batch_size, self.n_spatial, device=device)
                return spatial_mean, spatial_var, spatial_sample

            edge_index = edge_index[:, valid_edges]

        try:
            # Process latent representation with GATv2Conv
            # print('Processing latent representation with GATv2Conv')
            spatial_features = self.gat(z, edge_index)

            # Use mean and variance encoders
            spatial_mean = self.mean_encoder(spatial_features)
            log_var = self.var_encoder(spatial_features)

            # Calculate variance, ensure non-negative
            spatial_var = torch.exp(log_var) + self.var_eps

            # Create normal distribution
            dist = Normal(spatial_mean, spatial_var.sqrt())

            # Sample from the distribution
            spatial_sample = dist.rsample()

            return spatial_mean, spatial_var, spatial_sample
        except Exception as e:
            warnings.warn(
                f"GATv2Conv processing failed: {str(e)}. Will return zero tensor as spatial features.", RuntimeWarning
            )
            # Return zero tensors as fallback
            spatial_mean = torch.zeros(batch_size, self.n_spatial, device=device)
            spatial_var = torch.ones(batch_size, self.n_spatial, device=device) * self.var_eps
            spatial_sample = torch.zeros(batch_size, self.n_spatial, device=device)

            return spatial_mean, spatial_var, spatial_sample


class SpatialVAE(VAE):
    """Variational autoencoder with spatial information support.

    Extends standard VAE to include spatial information processing. Uses graph attention networks
    to capture spatial relationships between cells.

    Parameters
    ----------
    n_input
        Number of input features
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes in hidden layers
    n_latent
        Dimension of latent space
    n_spatial
        Dimension of spatial features
    n_layers
        Number of hidden layers
    dropout_rate
        Dropout rate
    dispersion
        Dispersion parameter type
    gene_likelihood
        Gene likelihood distribution type
    latent_distribution
        Latent distribution type
    **kwargs
        Additional parameters
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_spatial: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        edge_index: Optional[torch.Tensor] = None,
        attention_heads: int = 1,
        spatial_kl_weight: float = 0.01,
        var_eps: float = 1e-4,
        **kwargs,
    ):
        """Initialize SpatialVAE model.

        Parameters
        ----------
        n_input
            Input feature dimension
        n_batch
            Number of batches
        n_labels
            Number of labels
        n_hidden
            Hidden layer dimension
        n_latent
            Latent space dimension
        n_spatial
            Spatial feature dimension
        n_layers
            Number of layers
        dropout_rate
            Dropout rate
        dispersion
            Dispersion type
        gene_likelihood
            Gene likelihood function
        latent_distribution
            Latent distribution type
        edge_index
            Edge indices for establishing spatial relationships
        attention_heads
            Number of attention heads
        spatial_kl_weight
            Weight for spatial feature KL divergence
        var_eps
            Small constant for variance
        """
        # Initialize base VAE
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **kwargs,
        )

        self.n_spatial = n_spatial
        self.spatial_kl_weight = spatial_kl_weight
        self.spatial_encoder = SpatialEncoder(
            n_latent=n_latent,
            n_spatial=n_spatial,
            attention_heads=attention_heads,
            dropout_rate=dropout_rate,
            var_eps=var_eps,
        )

        # Validate and process edge_index
        if edge_index is not None:
            if not isinstance(edge_index, torch.Tensor):
                try:
                    edge_index = torch.tensor(edge_index, dtype=torch.long)
                except Exception as e:
                    warnings.warn(f"Failed to convert edge_index to tensor: {str(e)}, will set to None", UserWarning)
                    edge_index = None
            elif edge_index.dtype != torch.long:
                try:
                    edge_index = edge_index.long()
                except Exception as e:
                    warnings.warn(f"Failed to convert edge_index to long type: {str(e)}, will set to None", UserWarning)
                    edge_index = None

            # Check edge_index shape
            if edge_index is not None and (len(edge_index.shape) != 2 or edge_index.shape[0] != 2):
                warnings.warn(
                    f"Incorrect edge_index shape: {edge_index.shape}, should be [2, num_edges], will set to None",
                    UserWarning,
                )
                edge_index = None

        self.edge_index = edge_index
        self.register_buffer("_edge_index", edge_index)

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        cont_covariates: torch.Tensor | None = None,
        cat_covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Inference process, computes latent representation and spatial features.

        Parameters
        ----------
        x
            Input data
        batch_index
            Batch indices
        cont_covs
            Continuous covariates (VAE parameter naming)
        cat_covs
            Categorical covariates (VAE parameter naming)
        cont_covariates
            Continuous covariates (compatible format)
        cat_covariates
            Categorical covariates (compatible format)

        Returns
        -------
        dict
            Dictionary containing latent representation and spatial features
        """
        # Parameter name unification
        if cont_covs is None and cont_covariates is not None:
            cont_covs = cont_covariates
        if cat_covs is None and cat_covariates is not None:
            cat_covs = cat_covariates

        # Remove redundant parameters to avoid conflicting parameter names
        if "cont_covariates" in kwargs:
            del kwargs["cont_covariates"]
        if "cat_covariates" in kwargs:
            del kwargs["cat_covariates"]

        # Call base VAE inference with correct parameter names
        inference_outputs = super().inference(x, batch_index, cont_covs=cont_covs, cat_covs=cat_covs, **kwargs)

        # Get latent representation
        z = inference_outputs["z"]

        # Ensure edge_index is on the correct device
        if self.edge_index is not None and z.device != self.edge_index.device:
            self.edge_index = self.edge_index.to(z.device)

        # Calculate spatial features
        try:
            spatial_mean, spatial_var, spatial_sample = self.spatial_encoder(z, self.edge_index)

            # Add spatial features to output
            inference_outputs.update(
                {
                    "spatial_mean": spatial_mean,
                    "spatial_var": spatial_var,
                    "spatial_sample": spatial_sample,
                }
            )
        except Exception as e:
            # If spatial encoder fails, add warning log and return zero tensors
            warnings.warn(
                f"Spatial encoder processing failed: {str(e)}. Will return zero tensors as spatial features.",
                UserWarning,
            )
            batch_size = z.size(0)
            device = z.device

            # Create zero tensors as spatial features
            spatial_mean = torch.zeros(batch_size, self.n_spatial, device=device)
            spatial_var = torch.ones(batch_size, self.n_spatial, device=device) * self.spatial_encoder.var_eps
            spatial_sample = torch.zeros(batch_size, self.n_spatial, device=device)

            # Add spatial features to output
            inference_outputs.update(
                {
                    "spatial_mean": spatial_mean,
                    "spatial_var": spatial_var,
                    "spatial_sample": spatial_sample,
                }
            )

        return inference_outputs

    @auto_move_data
    def forward(self, tensors, inference_kwargs=None, compute_loss=True, **kwargs):
        """Forward pass process.

        Parameters
        ----------
        tensors
            Input tensor dictionary
        inference_kwargs
            Parameters passed to inference function
        compute_loss
            Whether to compute loss

        Returns
        -------
        tuple
            Inference outputs, generative outputs and loss
        """
        # Ensure no duplicate parameters will be passed
        # Remove items from kwargs that might conflict with parameters in parent's forward
        if "get_inference_input_kwargs" in kwargs:
            del kwargs["get_inference_input_kwargs"]

        # Call base VAE's forward pass
        if inference_kwargs is None:
            inference_kwargs = {}

        # Fix parameter name mismatch issues
        if "cont_covariates" in tensors and "cont_covs" not in tensors:
            tensors["cont_covs"] = tensors.pop("cont_covariates")

        # Fix categorical covariate parameter name mismatch issues
        if "cat_covariates" in tensors and "cat_covs" not in tensors:
            tensors["cat_covs"] = tensors.pop("cat_covariates")

        # return inference_outputs, generative_outputs, losses

        if compute_loss:
            inference_outputs, generative_outputs, losses = super().forward(
                tensors, inference_kwargs=inference_kwargs, compute_loss=compute_loss, **kwargs
            )
            # losses = module.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
            return inference_outputs, generative_outputs, losses
        else:
            inference_outputs, generative_outputs = super().forward(
                tensors, inference_kwargs=inference_kwargs, compute_loss=compute_loss, **kwargs
            )
            return inference_outputs, generative_outputs

    @unsupported_if_adata_minified
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.tensor | float = 1.0,
    ) -> LossOutput:
        """Calculate loss function, including KL divergence of spatial features.

        Parameters
        ----------
        tensors
            Input tensors
        inference_outputs
            Inference process outputs
        generative_outputs
            Generative process outputs
        kl_weight
            KL divergence weight

        Returns
        -------
        LossOutput
            Loss output object
        """
        # Get base VAE loss
        base_loss = super().loss(tensors, inference_outputs, generative_outputs, kl_weight)

        # If no spatial features, return base loss directly
        if "spatial_mean" not in inference_outputs:
            return base_loss

        # Calculate KL divergence for spatial features
        spatial_mean = inference_outputs["spatial_mean"]
        spatial_var = inference_outputs["spatial_var"]

        # Create distribution objects
        q_s = Normal(spatial_mean, spatial_var.sqrt())
        p_s = Normal(torch.zeros_like(spatial_mean), torch.ones_like(spatial_var.sqrt()))

        # Calculate KL divergence
        kl_divergence_s = kl_divergence(q_s, p_s).sum(dim=-1)

        # Apply weight
        weighted_kl_s = self.spatial_kl_weight * kl_divergence_s

        # Update total loss
        loss = base_loss.loss + torch.mean(weighted_kl_s)

        # Update KL divergence dictionary
        kl_local = base_loss.kl_local
        kl_local.update({"kl_divergence_s": kl_divergence_s})

        return LossOutput(
            loss=loss,
            reconstruction_loss=base_loss.reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=base_loss.extra_metrics,
        )

    def get_latent_representation(self, adata, indices, batch_size):
        return super().get_latent_representation(adata, indices, batch_size)

    @torch.inference_mode()
    def get_spatial_representation(
        self,
        adata=None,
        indices=None,
        batch_size=None,
    ) -> np.ndarray:
        """Get spatial feature representation.

        Parameters
        ----------
        adata
            AnnData object, optional
        indices
            Indices to get representation for, optional
        batch_size
            Batch size, optional

        Returns
        -------
        np.ndarray
            Spatial feature representation
        """
        # Ensure model has spatial encoder
        if not hasattr(self, "spatial_encoder"):
            raise ValueError("Model does not have a spatial encoder")

        try:
            # Get latent representation
            latent = self.get_latent_representation(adata, indices, batch_size)

            # If no edge_index, return zero matrix
            if self.edge_index is None:
                logger.warning("No edge index found, will return zero matrix as spatial representation")
                return np.zeros((latent.shape[0], self.n_spatial))

            # Convert to PyTorch tensor and ensure on correct device
            device = next(self.parameters()).device
            latent_tensor = torch.tensor(latent, dtype=torch.float32, device=device)

            # Ensure edge_index is on correct device
            if self.edge_index.device != device:
                self.edge_index = self.edge_index.to(device)

            # Process in batches to avoid memory overflow
            result_chunks = []
            chunk_size = 2048 if batch_size is None else batch_size

            for i in range(0, latent_tensor.shape[0], chunk_size):
                # Get current batch
                chunk = latent_tensor[i : i + chunk_size]
                # Calculate spatial features
                _, _, spatial_chunk = self.spatial_encoder(chunk, self.edge_index)
                # Add to result list
                result_chunks.append(spatial_chunk.cpu().numpy())

            # Merge all batch results
            spatial_representation = np.concatenate(result_chunks, axis=0)

            return spatial_representation

        except Exception as e:
            # Catch any exceptions to ensure function doesn't crash
            import traceback

            logger.error(f"Error getting spatial representation: {str(e)}\n{traceback.format_exc()}")

            # Return zero matrix as fallback
            n_cells = len(adata) if adata is not None else (len(indices) if indices is not None else 0)
            if n_cells == 0 and hasattr(self, "adata") and self.adata is not None:
                n_cells = self.adata.n_obs

            return np.zeros((n_cells, self.n_spatial))

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get tensors needed for inference process, overrides parent method to fix parameter name mismatch issues.

        Args:
            tensors: Input data tensors
            full_forward_pass: Whether to perform full forward pass

        Returns:
            Dictionary of inputs for inference process
        """
        # Call parent method to get basic inputs
        inputs = super()._get_inference_input(tensors, full_forward_pass)

        # Fix parameter name mismatch issue: rename cont_covariates to cont_covs
        if "cont_covariates" in inputs and "cont_covs" not in inputs:
            inputs["cont_covs"] = inputs.pop("cont_covariates")

        # Fix parameter name mismatch issue: rename cat_covariates to cat_covs
        if "cat_covariates" in inputs and "cat_covs" not in inputs:
            inputs["cat_covs"] = inputs.pop("cat_covariates")

        return inputs

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get tensors for generative process, overrides parent method to fix parameter name mismatch issues.

        Args:
            tensors: Original data tensors
            inference_outputs: Outputs from inference process

        Returns:
            Dictionary of inputs needed for generative process
        """
        # Call parent method to get basic generative inputs
        generative_inputs = super()._get_generative_input(tensors, inference_outputs)

        # Fix continuous covariate parameter name mismatch issue
        if "cont_covariates" in generative_inputs and "cont_covs" not in generative_inputs:
            generative_inputs["cont_covs"] = generative_inputs.pop("cont_covariates")

        # Fix categorical covariate parameter name mismatch issue
        if "cat_covariates" in generative_inputs and "cat_covs" not in generative_inputs:
            generative_inputs["cat_covs"] = generative_inputs.pop("cat_covariates")

        return generative_inputs

    def setup_spatial_graph(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        batch_key: Optional[str] = None,
        method: str = "knn",
        n_neighbors: int = 10,
    ):
        """Set up spatial graph for spatial information processing.

        Constructs a spatial graph based on spatial coordinates in adata.obsm[spatial_key],
        using either K-nearest neighbors or Delaunay triangulation.

        Parameters
        ----------
        adata
            AnnData object containing spatial coordinates in adata.obsm[spatial_key]
        spatial_key
            obsm key storing spatial coordinates, default is 'spatial'
        batch_key
            obs key for batch information, if provided, graph will be constructed per batch
        method
            Method for constructing the graph, can be 'knn' or 'delaunay'
        n_neighbors
            Number of neighbors for KNN method
        """
        try:
            # Get spatial coordinates
            if spatial_key not in adata.obsm:
                raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")

            coordinates = adata.obsm[spatial_key]
            logger.info(
                f"Retrieved spatial coordinates for {len(coordinates)} cells, dimension: {coordinates.shape[1]}"
            )

            # Check validity of coordinate data
            if np.isnan(coordinates).any():
                # Handle NaN values
                nan_count = np.isnan(coordinates).any(axis=1).sum()
                logger.warning(
                    f"Detected {nan_count} cells with NaN coordinates, these cells will be excluded from spatial analysis"
                )

                # Create mask without NaNs
                valid_mask = ~np.isnan(coordinates).any(axis=1)
                coordinates = coordinates[valid_mask]

                if len(coordinates) == 0:
                    raise ValueError("No valid coordinate data after processing NaN values")

            # Process by batch
            if batch_key is not None and batch_key in adata.obs:
                j = 0  # Initialize counter

                # Process each batch separately
                for batch in adata.obs[batch_key].unique():
                    # Get data for current batch
                    batch_mask = adata.obs[batch_key] == batch
                    batch_coords = coordinates[batch_mask]

                    if len(batch_coords) < 3:
                        logger.warning(f"Batch {batch} has only {len(batch_coords)} cells, skipping this batch")
                        continue

                    if method.lower() == "knn":
                        # Use K-nearest neighbors to build graph
                        from scipy.sparse import csr_matrix
                        from sklearn.neighbors import kneighbors_graph
                        from torch_geometric.utils import from_scipy_sparse_matrix

                        # Build KNN graph
                        A = kneighbors_graph(batch_coords, n_neighbors=min(n_neighbors, len(batch_coords) - 1))

                        # Convert to PyTorch edge index format
                        edge_index_batch, _ = from_scipy_sparse_matrix(A)

                        # Get indices in original dataset
                        batch_indices = np.where(batch_mask)[0]
                        batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long)

                        # Map edge indices back to original dataset indices
                        edge_index_batch = batch_indices_tensor[edge_index_batch]

                    else:  # Use Delaunay triangulation
                        from scipy.spatial import Delaunay

                        # Create Delaunay triangles
                        try:
                            tri = Delaunay(batch_coords)
                            triangles = tri.simplices
                        except Exception as e:
                            logger.warning(
                                f"Delaunay triangulation failed for batch {batch}: {str(e)}, skipping this batch"
                            )
                            continue

                        # Extract edges
                        edges = set()
                        for triangle in triangles:
                            for i in range(3):
                                # Create ordered edge pairs
                                edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                                edges.add(edge)

                        # Check if edges were extracted
                        if len(edges) == 0:
                            logger.warning(f"No valid edges extracted for batch {batch}, skipping this batch")
                            continue

                        # Convert to PyTorch edge index format
                        edge_list = torch.tensor(list(edges), dtype=torch.long).t()

                        # Get indices in original dataset
                        batch_indices = np.where(batch_mask)[0]
                        batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long)

                        # Map edge indices back to original dataset indices
                        edge_index_batch = batch_indices_tensor[edge_list]

                    # Merge edge indices
                    if j == 0:
                        edge_index = edge_index_batch
                        j = 1
                    else:
                        edge_index = torch.cat((edge_index, edge_index_batch), dim=1)

                # Check if edges were successfully built
                if j == 0:
                    raise ValueError("Failed to build valid edges for any batch")

            else:  # Process without batches
                if method.lower() == "knn":
                    # Use K-nearest neighbors to build graph
                    from sklearn.neighbors import kneighbors_graph
                    from torch_geometric.utils import from_scipy_sparse_matrix

                    # Build KNN graph
                    A = kneighbors_graph(coordinates, n_neighbors=min(n_neighbors, len(coordinates) - 1))

                    # Convert to PyTorch edge index format
                    edge_index, _ = from_scipy_sparse_matrix(A)

                else:  # Use Delaunay triangulation
                    from scipy.spatial import Delaunay

                    # Create Delaunay triangles
                    tri = Delaunay(coordinates)
                    triangles = tri.simplices

                    # Extract edges
                    edges = set()
                    for triangle in triangles:
                        for i in range(3):
                            # Create ordered edge pairs
                            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                            edges.add(edge)

                    # Convert to PyTorch edge index format
                    edge_index = torch.tensor(list(edges), dtype=torch.long).t()

            # Store edge index
            self.edge_index = edge_index

            # Register as buffer
            if hasattr(self, "_edge_index"):
                self._edge_index = edge_index
            else:
                self.register_buffer("_edge_index", edge_index)

            logger.info(
                f"Successfully built spatial graph with {edge_index.shape[1]} edges connecting {adata.n_obs} cells"
            )

            return edge_index

        except Exception as e:
            logger.error(f"Failed to build spatial graph: {str(e)}")
            # Set to None
            self.edge_index = None
            if hasattr(self, "_edge_index"):
                self._edge_index = None
            else:
                self.register_buffer("_edge_index", None)

            raise ValueError(f"Failed to build spatial graph: {str(e)}")

    def process_in_batches(
        self,
        edge_index: torch.Tensor,
        max_edges_per_batch: int = 100000,
        combine_results: bool = True,
        adata: Optional[AnnData] = None,
    ):
        """Process edge indices in batches, suitable for processing large graph structures.

        Divides edge_index into smaller batches for processing to avoid memory overflow errors.

        Parameters
        ----------
        edge_index
            Edge index tensor, shape [2, num_edges]
        max_edges_per_batch
            Maximum number of edges per batch
        combine_results
            Whether to combine results from all batches
        adata
            AnnData object, if None uses the AnnData object from training

        Returns
        -------
        dict or list
            If combine_results is True, returns combined result dictionary;
            otherwise returns a list of results for each batch
        """
        if edge_index is None or edge_index.shape[1] == 0:
            warnings.warn("Edge index is empty, cannot process in batches", UserWarning)
            return None

        # Get number of edges
        num_edges = edge_index.shape[1]

        # If edge count is less than or equal to max_edges_per_batch, process directly
        if num_edges <= max_edges_per_batch:
            return self.process_edges(edge_index, adata=adata)

        # Calculate batch count
        num_batches = (num_edges + max_edges_per_batch - 1) // max_edges_per_batch

        logger.info(f"Dividing {num_edges} edges into {num_batches} batches, max {max_edges_per_batch} edges per batch")

        # Process in batches
        batch_results = []
        for i in range(num_batches):
            # Calculate start and end indices for current batch
            start_idx = i * max_edges_per_batch
            end_idx = min((i + 1) * max_edges_per_batch, num_edges)

            # Get edge indices for current batch
            batch_edge_index = edge_index[:, start_idx:end_idx]

            try:
                # Process current batch
                batch_result = self.process_edges(batch_edge_index, adata=adata)
                batch_results.append(batch_result)

                logger.debug(f"Completed batch {i+1}/{num_batches}, processed {end_idx-start_idx} edges")

            except Exception as e:
                logger.error(f"Error processing batch {i+1}/{num_batches}: {str(e)}")
                continue

        # If no batches were processed successfully, return None
        if not batch_results:
            warnings.warn("All batch processing failed", UserWarning)
            return None

        # If results don't need to be combined, return list of batch results
        if not combine_results:
            return batch_results

        # Combine results from all batches
        combined_result = {}

        # Get all result keys
        all_keys = set()
        for result in batch_results:
            all_keys.update(result.keys())

        # Combine results for each key
        for key in all_keys:
            # Find all results containing this key
            valid_results = [result[key] for result in batch_results if key in result]

            if not valid_results:
                continue

            # Check data type to decide how to combine
            sample_data = valid_results[0]

            if isinstance(sample_data, torch.Tensor):
                # If tensor, check if it can be concatenated along some dimension
                if len(sample_data.shape) > 0 and sample_data.shape[0] > 0:
                    try:
                        combined_result[key] = torch.cat(valid_results, dim=0)
                    except:
                        # If cannot concatenate, take mean
                        combined_result[key] = torch.stack(valid_results).mean(dim=0)
                else:
                    # If scalar tensor, take mean
                    combined_result[key] = torch.stack(valid_results).mean()
            elif isinstance(sample_data, (int, float)):
                # If numeric, take mean
                combined_result[key] = sum(valid_results) / len(valid_results)
            elif isinstance(sample_data, dict):
                # If dictionary, recursively combine
                combined_result[key] = {}
                all_subkeys = set()
                for result in valid_results:
                    all_subkeys.update(result.keys())

                for subkey in all_subkeys:
                    sub_valid_results = [result[subkey] for result in valid_results if subkey in result]
                    if sub_valid_results:
                        if isinstance(sub_valid_results[0], torch.Tensor):
                            try:
                                combined_result[key][subkey] = torch.cat(sub_valid_results, dim=0)
                            except:
                                combined_result[key][subkey] = torch.stack(sub_valid_results).mean(dim=0)
                        else:
                            combined_result[key][subkey] = sub_valid_results[0]  # Simply take first one
            else:
                # Other types, simply take first one
                combined_result[key] = valid_results[0]

        return combined_result

    def process_edges(self, edge_index: torch.Tensor, adata: Optional[AnnData] = None):
        """Process a single batch of edge indices.

        This is a utility method for processing a single batch of edge indices in batch processing.

        Parameters
        ----------
        edge_index
            Edge index tensor, shape [2, num_edges]
        adata
            AnnData object, if None uses the AnnData object from training

        Returns
        -------
        dict
            Processing results
        """
        # Try different locations for AnnTorchDataset
        try:
            from scvi.data import AnnTorchDataset
        except ImportError:
            try:
                from scvi.data._anntorchdataset import AnnTorchDataset
            except ImportError:
                try:
                    from scvi.dataloaders import AnnTorchDataset
                except ImportError:
                    raise ImportError("Failed to import AnnTorchDataset class, please check scvi-tools version")

        from scvi.data import AnnDataManager

        # Get AnnData object
        if adata is None:
            try:
                # Try different ways to get associated AnnData
                if hasattr(self, "_model") and hasattr(self._model, "_adata_manager"):
                    adata_manager = self._model._adata_manager
                elif hasattr(self, "adata_manager"):
                    adata_manager = self.adata_manager
                elif hasattr(self, "module") and hasattr(self.module, "adata_manager"):
                    adata_manager = self.module.adata_manager
                else:
                    raise ValueError("Cannot find AnnData manager")
            except Exception as e:
                raise ValueError(f"Need to provide adata parameter when processing edge indices: {str(e)}")
        else:
            # Create temporary manager for provided AnnData
            adata_manager = AnnDataManager(fields=[], setup_method_args={})
            adata_manager.register_fields(adata)

        # Create data tensors
        try:
            data = AnnTorchDataset(adata_manager)
            data = data[np.arange(data.data["X"].shape[0])]
        except Exception as e:
            raise ValueError(f"Cannot create AnnTorchDataset: {str(e)}")

        # Move data to model's device
        device = next(self.parameters()).device
        for key, value in data.items():
            if value is not None:
                if key == REGISTRY_KEYS.BATCH_KEY and value is not None:
                    data[key] = torch.LongTensor(value).to(device)
                else:
                    data[key] = torch.Tensor(value).to(device)

        # Ensure edge index is on correct device
        if edge_index.device != device:
            edge_index = edge_index.to(device)

        # Run inference
        with torch.no_grad():
            inference_outputs = self.inference(
                x=data[REGISTRY_KEYS.X_KEY], batch_index=data.get(REGISTRY_KEYS.BATCH_KEY, None), edge_index=edge_index
            )

        return inference_outputs
