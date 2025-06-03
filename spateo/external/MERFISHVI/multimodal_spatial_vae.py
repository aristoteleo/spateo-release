from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, kl_divergence

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    try:
        from torch_geometric.nn.conv import GATv2Conv
    except ImportError:
        raise ImportError("Failed to import GATv2Conv, please install PyTorch Geometric")

from scvi import REGISTRY_KEYS

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

try:
    from scvi.utils import unsupported_if_adata_minified
except ImportError:
    try:
        from scvi.model.base import unsupported_if_adata_minified
    except ImportError:
        # Create a dummy decorator if not available
        def unsupported_if_adata_minified(fn):
            return fn


from anndata import AnnData
from scvi.module import VAE
from scvi.nn import Decoder, Encoder, FCLayers

# Import SpatialEncoder and SpatialVAE
from .scvi_spatial_module import SpatialEncoder, SpatialVAE

logger = logging.getLogger(__name__)


class MultiModalSpatialVAE(SpatialVAE):
    """Multi-modal spatial variational autoencoder.

    Processes both modalities with spatial information and those without spatial information.
    Uses a shared latent space for joint modeling.

    Parameters
    ----------
    n_input_spatial
        Number of input features for spatial modality
    n_input_nonspatial
        Number of input features for non-spatial modality
    n_batch_spatial
        Number of batches for spatial modality
    n_batch_nonspatial
        Number of batches for non-spatial modality
    n_labels_spatial
        Number of labels for spatial modality
    n_labels_nonspatial
        Number of labels for non-spatial modality
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
        n_input_spatial: int,
        n_input_nonspatial: int,
        n_batch_spatial: int = 0,
        n_batch_nonspatial: int = 0,
        n_labels_spatial: int = 0,
        n_labels_nonspatial: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_spatial: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_observed_lib_size: bool = True,
        edge_index: Optional[torch.Tensor] = None,
        attention_heads: int = 1,
        spatial_kl_weight: float = 0.01,
        modality_weights: Dict[str, float] = {"spatial": 1.0, "nonspatial": 1.0},
        cats_per_cov_spatial: Optional[List[int]] = None,
        cats_per_cov_nonspatial: Optional[List[int]] = None,
        use_size_factor_spatial: bool = False,
        use_size_factor_nonspatial: bool = False,
        library_log_means_spatial: Optional[torch.Tensor] = None,
        library_log_vars_spatial: Optional[torch.Tensor] = None,
        library_log_means_nonspatial: Optional[torch.Tensor] = None,
        library_log_vars_nonspatial: Optional[torch.Tensor] = None,
        var_eps: float = 1e-4,
        **kwargs,
    ):
        # Initialize base VAE (note here we do not call SpatialVAE's initialization, but directly call VAE's initialization)
        VAE.__init__(
            self,
            n_input=n_input_spatial,
            n_batch=n_batch_spatial,
            n_labels=n_labels_spatial,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=use_observed_lib_size,
            n_cats_per_cov=cats_per_cov_spatial,
            use_size_factor_key=use_size_factor_spatial,
            library_log_means=library_log_means_spatial,
            library_log_vars=library_log_vars_spatial,
            **kwargs,
        )

        # Store configuration parameters
        self.n_spatial = n_spatial
        self.spatial_kl_weight = spatial_kl_weight
        self.modality_weights = modality_weights
        self.var_eps = var_eps

        # Initialize spatial encoder
        self.spatial_encoder = SpatialEncoder(
            n_latent=n_latent,
            n_spatial=n_spatial,
            attention_heads=attention_heads,
            dropout_rate=dropout_rate,
            var_eps=var_eps,
        )

        # Create encoder for non-spatial modality
        self.nonspatial_encoder = Encoder(
            n_input=n_input_nonspatial,
            n_output=n_latent,
            n_cat_list=[n_batch_nonspatial] if n_batch_nonspatial > 0 else None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )

        # Create decoder for non-spatial modality
        if gene_likelihood in ["zinb", "nb"]:
            from scvi.nn import DecoderSCVI

            self.nonspatial_decoder = DecoderSCVI(
                n_input=n_latent,
                n_output=n_input_nonspatial,
                n_cat_list=[n_batch_nonspatial] if n_batch_nonspatial > 0 else None,
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        else:
            self.nonspatial_decoder = Decoder(
                n_input=n_latent,
                n_output=n_input_nonspatial,
                n_cat_list=[n_batch_nonspatial] if n_batch_nonspatial > 0 else None,
                n_layers=n_layers,
                n_hidden=n_hidden,
            )

        # Create library size parameter for non-spatial modality (if needed)
        if not use_observed_lib_size:
            self.nonspatial_l_mean = torch.nn.Parameter(
                library_log_means_nonspatial
                if library_log_means_nonspatial is not None
                else torch.zeros(n_batch_nonspatial if n_batch_nonspatial > 0 else 1)
            )
            self.nonspatial_l_var = torch.nn.Parameter(
                library_log_vars_nonspatial
                if library_log_vars_nonspatial is not None
                else torch.zeros(n_batch_nonspatial if n_batch_nonspatial > 0 else 1)
            )

        # Process edge_index (spatial graph)
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
                    f"edge_index shape error: {edge_index.shape}, should be [2, num_edges], will set to None",
                    UserWarning,
                )
                edge_index = None

        self.edge_index = edge_index
        self.register_buffer("_edge_index", edge_index)

    @auto_move_data
    def inference_spatial(
        self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Spatial modality inference process.

        Parameters
        ----------
        x
            Input data of spatial modality
        batch_index
            Batch index

        Returns
        -------
        dict
            Dictionary containing inference results
        """
        # Call base VAE inference
        outputs = VAE.inference(self, x, batch_index, **kwargs)

        # Get latent representation
        z = outputs["z"]

        # Ensure edge_index is on the correct device
        if self.edge_index is not None and z.device != self.edge_index.device:
            self.edge_index = self.edge_index.to(z.device)

        # Calculate spatial feature
        try:
            spatial_mean, spatial_var, spatial_sample = self.spatial_encoder(z, self.edge_index)

            # Add spatial feature to outputs
            outputs.update(
                {
                    "spatial_mean": spatial_mean,
                    "spatial_var": spatial_var,
                    "spatial_sample": spatial_sample,
                }
            )
        except Exception as e:
            # If spatial encoder fails, add warning log and return zero tensor
            warnings.warn(
                f"Spatial encoder processing failed: {str(e)}. Will return zero tensor as spatial feature.", UserWarning
            )
            batch_size = z.size(0)
            device = z.device

            # Create zero tensor as spatial feature
            spatial_mean = torch.zeros(batch_size, self.n_spatial, device=device)
            spatial_var = torch.ones(batch_size, self.n_spatial, device=device) * self.var_eps
            spatial_sample = torch.zeros(batch_size, self.n_spatial, device=device)

            # Add spatial feature to outputs
            outputs.update(
                {
                    "spatial_mean": spatial_mean,
                    "spatial_var": spatial_var,
                    "spatial_sample": spatial_sample,
                }
            )

        return outputs

    @auto_move_data
    def inference_nonspatial(
        self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Non-spatial modality inference process.

        Parameters
        ----------
        x
            Input data of non-spatial modality
        batch_index
            Batch index

        Returns
        -------
        dict
            Dictionary containing inference results
        """
        # Use non-spatial modality encoder
        return self.nonspatial_encoder(x, batch_index)

    @auto_move_data
    def inference(
        self,
        x_spatial: torch.Tensor,
        x_nonspatial: Optional[torch.Tensor] = None,
        batch_index_spatial: Optional[torch.Tensor] = None,
        batch_index_nonspatial: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Joint inference process, handles both spatial and non-spatial modalities.

        Parameters
        ----------
        x_spatial
            Input data of spatial modality
        x_nonspatial
            Input data of non-spatial modality, optional
        batch_index_spatial
            Batch index of spatial modality
        batch_index_nonspatial
            Batch index of non-spatial modality

        Returns
        -------
        dict
            Dictionary containing joint inference results
        """
        # First perform spatial modality inference
        outputs = self.inference_spatial(x_spatial, batch_index_spatial, **kwargs)

        # If no non-spatial modality input, return spatial modality results directly
        if x_nonspatial is None:
            return outputs

        # Perform non-spatial modality inference
        nonspatial_outputs = self.inference_nonspatial(x_nonspatial, batch_index_nonspatial)

        # Fuse latent representations of two modalities
        w1 = self.modality_weights.get("spatial", 1.0)
        w2 = self.modality_weights.get("nonspatial", 1.0)
        total_weight = w1 + w2

        # Weighted fusion latent representations
        fused_z = (w1 * outputs["z"] + w2 * nonspatial_outputs["z"]) / total_weight

        # Update output dictionary
        outputs.update(
            {
                # Add non-spatial modality outputs
                "nonspatial_qz_m": nonspatial_outputs["qz_m"],
                "nonspatial_qz_v": nonspatial_outputs["qz_v"],
                "nonspatial_z": nonspatial_outputs["z"],
                # Use fused z
                "fused_z": fused_z,
                # Default use fused z as main latent representation
                "z": fused_z,
            }
        )

        return outputs

    @auto_move_data
    def generative_spatial(
        self, z: torch.Tensor, batch_index: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Spatial modality generative process.

        Parameters
        ----------
        z
            Latent representation
        batch_index
            Batch index

        Returns
        -------
        dict
            Generative output
        """
        return VAE.generative(self, z, batch_index, **kwargs)

    @auto_move_data
    def generative_nonspatial(
        self,
        z: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Non-spatial modality generative process.

        Parameters
        ----------
        z
            Latent representation
        batch_index
            Batch index
        library_size
            Library size

        Returns
        -------
        dict
            Generative output
        """
        # Generate reconstruction of non-spatial modality
        px_rate = self.nonspatial_decoder(z, batch_index)

        # If library size is not provided and needs estimation
        if library_size is None and not self.use_observed_lib_size:
            batch_index = batch_index.view(-1, 1) if batch_index is not None else None
            if batch_index is None and self.nonspatial_l_mean.shape[0] > 1:
                raise ValueError("No batch_index provided, but model has multiple batches")

            # Get library size parameter of current batch
            if batch_index is not None and self.nonspatial_l_mean.shape[0] > 1:
                library_loc = F.linear(torch.ones_like(batch_index, dtype=torch.float), self.nonspatial_l_mean)
                library_scale = F.linear(
                    torch.ones_like(batch_index, dtype=torch.float), torch.exp(self.nonspatial_l_var) + 1e-4
                )
            else:
                library_loc = self.nonspatial_l_mean
                library_scale = torch.exp(self.nonspatial_l_var) + 1e-4

            # Sample library size
            library = torch.distributions.LogNormal(library_loc, library_scale.sqrt()).rsample()
        elif library_size is not None:
            library = library_size
        else:
            # Use observed library size (normalized)
            library = torch.log(torch.sum(torch.exp(x_nonspatial), dim=1, keepdim=True))

        # Build output dictionary
        outputs = {"px_rate": px_rate}

        if self.gene_likelihood == "zinb":
            px_r = self.px_r
            px_dropout = self.px_dropout
            outputs.update({"px_r": px_r, "px_dropout": px_dropout})

        return outputs

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        batch_index_spatial: Optional[torch.Tensor] = None,
        batch_index_nonspatial: Optional[torch.Tensor] = None,
        library_size_spatial: Optional[torch.Tensor] = None,
        library_size_nonspatial: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Joint generative process.

        Parameters
        ----------
        z
            Latent representation
        batch_index_spatial
            Batch index of spatial modality
        batch_index_nonspatial
            Batch index of non-spatial modality
        library_size_spatial
            Library size of spatial modality
        library_size_nonspatial
            Library size of non-spatial modality

        Returns
        -------
        dict
            Dictionary containing outputs of two modalities generative process
        """
        # Generate reconstruction of spatial modality
        spatial_outputs = self.generative_spatial(z, batch_index_spatial, library_size=library_size_spatial, **kwargs)

        # If batch_index_nonspatial is not None, generate reconstruction of non-spatial modality
        if batch_index_nonspatial is not None:
            nonspatial_outputs = self.generative_nonspatial(
                z, batch_index_nonspatial, library_size=library_size_nonspatial, **kwargs
            )

            # Return dictionary containing outputs of two modalities
            return {"spatial": spatial_outputs, "nonspatial": nonspatial_outputs}
        else:
            # Return output of spatial modality only
            return {"spatial": spatial_outputs}

    @auto_move_data
    def forward(
        self, tensors: Dict[str, torch.Tensor], inference_kwargs: Dict = None, compute_loss: bool = True, **kwargs
    ) -> Tuple:
        """Forward propagation process.

        Parameters
        ----------
        tensors
            Input tensor dictionary, containing data of two modalities
        inference_kwargs
            Parameters to pass to inference function
        compute_loss
            Whether to compute loss

        Returns
        -------
        tuple
            Inference output, generative output and loss
        """
        # Ensure no conflicting parameters are passed
        if inference_kwargs is None:
            inference_kwargs = {}

        # Fix parameter names
        if "cont_covariates" in tensors and "cont_covs" not in tensors:
            tensors["cont_covs"] = tensors.pop("cont_covariates")
        if "cat_covariates" in tensors and "cat_covs" not in tensors:
            tensors["cat_covs"] = tensors.pop("cat_covariates")

        # Extract data of two modalities from tensors
        x_spatial = tensors.get("X", None)
        x_nonspatial = tensors.get("X_nonspatial", None)
        batch_index_spatial = tensors.get("batch_indices", None)
        batch_index_nonspatial = tensors.get("batch_indices_nonspatial", None)

        # If no non-spatial modality data is provided, only process spatial modality
        has_nonspatial = x_nonspatial is not None

        # Perform joint inference
        inference_inputs = {
            "x_spatial": x_spatial,
            "batch_index_spatial": batch_index_spatial,
        }

        if has_nonspatial:
            inference_inputs.update(
                {
                    "x_nonspatial": x_nonspatial,
                    "batch_index_nonspatial": batch_index_nonspatial,
                }
            )

        inference_inputs.update(inference_kwargs)
        inference_outputs = self.inference(**inference_inputs)

        # Perform joint generative
        generative_inputs = {
            "z": inference_outputs["z"],  # Use fused latent representation
            "batch_index_spatial": batch_index_spatial,
        }

        if has_nonspatial:
            generative_inputs.update(
                {
                    "batch_index_nonspatial": batch_index_nonspatial,
                }
            )

        generative_outputs = self.generative(**generative_inputs)

        if compute_loss:
            # Calculate joint loss
            loss_inputs = {
                "tensors": tensors,
                "inference_outputs": inference_outputs,
                "generative_outputs": generative_outputs,
            }

            losses = self.loss(**loss_inputs)
            return inference_outputs, generative_outputs, losses
        else:
            return inference_outputs, generative_outputs

    @unsupported_if_adata_minified
    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor | Distribution | None],
        generative_outputs: Dict[str, Dict[str, Distribution | None]],
        kl_weight: torch.tensor | float = 1.0,
    ) -> LossOutput:
        """Calculate joint loss function.

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
        # Extract weights of two modalities
        w_spatial = self.modality_weights.get("spatial", 1.0)
        w_nonspatial = self.modality_weights.get("nonspatial", 1.0)
        total_weight = w_spatial + w_nonspatial

        # Extract input data
        x_spatial = tensors.get("X", None)
        x_nonspatial = tensors.get("X_nonspatial", None)

        # Calculate spatial modality loss
        # Create tensors dictionary containing only spatial modality data
        spatial_tensors = {"X": x_spatial}
        for k, v in tensors.items():
            if k != "X_nonspatial" and not k.endswith("_nonspatial"):
                spatial_tensors[k] = v

        # If there is spatial feature, calculate spatial modality KL divergence
        if "spatial_mean" in inference_outputs:
            spatial_mean = inference_outputs["spatial_mean"]
            spatial_var = inference_outputs["spatial_var"]

            # Create distribution objects
            q_s = Normal(spatial_mean, spatial_var.sqrt())
            p_s = Normal(torch.zeros_like(spatial_mean), torch.ones_like(spatial_var.sqrt()))

            # Calculate KL divergence
            kl_divergence_s = kl_divergence(q_s, p_s).sum(dim=-1)
            spatial_kl = kl_divergence_s
        else:
            spatial_kl = torch.tensor(0.0, device=self.device)

        # Calculate spatial modality reconstruction loss
        spatial_reconst_loss, spatial_kl_local = self._get_reconstruction_loss(x_spatial, generative_outputs["spatial"])

        # Calculate total KL divergence of spatial modality
        spatial_kl_local.update({"kl_divergence_s": spatial_kl})
        spatial_kl_weighted = torch.mean(
            torch.sum(spatial_kl_local["kl_divergence_z"], dim=-1)
        ) + self.spatial_kl_weight * torch.mean(spatial_kl)

        # Calculate total loss of spatial modality
        spatial_loss = spatial_reconst_loss + kl_weight * spatial_kl_weighted

        # If there is non-spatial modality, calculate non-spatial modality loss
        if x_nonspatial is not None and "nonspatial" in generative_outputs:
            # Calculate non-spatial modality KL divergence
            nonspatial_qz_m = inference_outputs["nonspatial_qz_m"]
            nonspatial_qz_v = inference_outputs["nonspatial_qz_v"]

            # Create distribution objects
            qz = Normal(nonspatial_qz_m, nonspatial_qz_v.sqrt())
            pz = Normal(torch.zeros_like(nonspatial_qz_m), torch.ones_like(nonspatial_qz_v.sqrt()))

            # Calculate KL divergence
            kl_divergence_z_nonspatial = kl_divergence(qz, pz).sum(dim=-1)

            # Calculate non-spatial modality reconstruction loss
            nonspatial_reconst_loss = self._get_reconstruction_loss_nonspatial(
                x_nonspatial, generative_outputs["nonspatial"]
            )

            # Calculate total loss of non-spatial modality
            nonspatial_kl_weighted = torch.mean(kl_divergence_z_nonspatial)
            nonspatial_loss = nonspatial_reconst_loss + kl_weight * nonspatial_kl_weighted

            # Merge two modality losses
            total_loss = (w_spatial * spatial_loss + w_nonspatial * nonspatial_loss) / total_weight

            # Update KL divergence dictionary
            spatial_kl_local.update({"kl_divergence_z_nonspatial": kl_divergence_z_nonspatial})
        else:
            # Use spatial modality loss only
            total_loss = spatial_loss
            nonspatial_reconst_loss = torch.tensor(0.0, device=self.device)

        # Return loss output object
        extra_metrics = {
            "spatial_reconstruction_loss": spatial_reconst_loss,
            "nonspatial_reconstruction_loss": nonspatial_reconst_loss if x_nonspatial is not None else None,
            "spatial_kl": spatial_kl_weighted,
            "nonspatial_kl": nonspatial_kl_weighted if x_nonspatial is not None else None,
        }

        return LossOutput(
            loss=total_loss,
            reconstruction_loss=spatial_reconst_loss,  # Use spatial modality reconstruction loss as main reconstruction loss
            kl_local=spatial_kl_local,
            extra_metrics=extra_metrics,
        )

    def _get_reconstruction_loss_nonspatial(
        self, x: torch.Tensor, generative_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate non-spatial modality reconstruction loss.

        Parameters
        ----------
        x
            Input data
        generative_outputs
            Generative process outputs

        Returns
        -------
        torch.Tensor
            Reconstruction loss
        """
        # Calculate reconstruction loss based on different likelihood functions
        px_rate = generative_outputs["px_rate"]

        if self.gene_likelihood == "zinb":
            px_r = generative_outputs["px_r"]
            px_dropout = generative_outputs["px_dropout"]

            # Use zero-inflated negative binomial distribution
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
        elif self.gene_likelihood == "nb":
            px_r = generative_outputs["px_r"]

            # Use negative binomial distribution
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        elif self.gene_likelihood == "poisson":
            # Use Poisson distribution
            reconst_loss = -log_poisson(x, px_rate)
        else:  # normal
            # Use normal distribution
            reconst_loss = -log_normal(x, px_rate, torch.ones_like(px_rate))

        return torch.mean(reconst_loss.sum(dim=-1))

    @torch.inference_mode()
    def get_latent_representation_by_modality(
        self, adata=None, indices=None, batch_size=None, modality="spatial"
    ) -> np.ndarray:
        """Get latent representation of specific modality.

        Parameters
        ----------
        adata
            AnnData object, optional
        indices
            Index to get representation, optional
        batch_size
            Batch processing size, optional
        modality
            Modality to get, can be "spatial", "nonspatial" or "fused"

        Returns
        -------
        np.ndarray
            Latent representation
        """
        if modality == "spatial":
            # Get standard latent representation
            return self.get_latent_representation(adata, indices, batch_size)
        elif modality == "nonspatial":
            # If not joint training, return latent representation of spatial modality
            if not hasattr(self, "nonspatial_encoder"):
                logger.warning(
                    "Model does not have non-spatial encoder, will return latent representation of spatial modality"
                )
                return self.get_latent_representation(adata, indices, batch_size)

            # Get latent representation of non-spatial modality
            # This requires running inference process first
            # Implement similar to get_latent_representation but using nonspatial_encoder
            # ...
            raise NotImplementedError("Non-spatial latent representation retrieval not implemented yet")
        elif modality == "fused":
            return self.get_fused_representation(adata, indices, batch_size)
        else:
            raise ValueError(f"Unsupported modality: {modality}, valid values are 'spatial', 'nonspatial' or 'fused'")

    @torch.inference_mode()
    def get_fused_representation(self, adata=None, indices=None, batch_size=None) -> np.ndarray:
        """Get fused latent representation.

        Parameters
        ----------
        adata
            AnnData object, optional
        indices
            Index to get representation, optional
        batch_size
            Batch processing size, optional

        Returns
        -------
        np.ndarray
            Fused latent representation
        """
        # If not joint training, return latent representation of spatial modality
        if not hasattr(self, "nonspatial_encoder"):
            logger.warning(
                "Model does not have non-spatial encoder, will return latent representation of spatial modality"
            )
            return self.get_latent_representation(adata, indices, batch_size)

        # Need to get latent representations of both modalities and fuse them
        # In actual application, this may require more complex implementation
        return self.get_latent_representation(adata, indices, batch_size)

    @torch.inference_mode()
    def get_nonspatial_specific_features(self, adata=None, indices=None, batch_size=None) -> np.ndarray:
        """Get non-spatial modality specific features.

        Parameters
        ----------
        adata
            AnnData object, optional
        indices
            Index to get representation, optional
        batch_size
            Batch processing size, optional

        Returns
        -------
        np.ndarray
            Non-spatial modality specific features
        """
        # If not joint training, return None
        if not hasattr(self, "nonspatial_encoder"):
            logger.warning("Model does not have non-spatial encoder, cannot get non-spatial features")
            return None

        # Implement logic to get non-spatial modality specific features
        # This may require additional network layers
        return None

    def _get_inference_input(
        self,
        tensors: Dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> Dict[str, torch.Tensor | None]:
        """Get input tensors required for inference process, override parent method to handle multi-modal data.

        Parameters
        ----------
        tensors
            Input data tensors
        full_forward_pass
            Whether to execute full forward propagation

        Returns
        -------
        Dict
            Input dictionary for inference process
        """
        # Call parent method to get basic inputs
        inputs = super()._get_inference_input(tensors, full_forward_pass)

        # Fix parameter name mismatch issue: rename cont_covariates to cont_covs
        if "cont_covariates" in inputs:
            inputs["cont_covs"] = inputs.pop("cont_covariates")
        if "cat_covariates" in inputs:
            inputs["cat_covs"] = inputs.pop("cat_covariates")

        return inputs

    def _get_generative_input(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor | Distribution | None],
    ) -> Dict[str, torch.Tensor | None]:
        """Get input tensors required for generative process, override parent method to handle multi-modal data.

        Parameters
        ----------
        tensors
            Input data tensors
        inference_outputs
            Outputs from inference process

        Returns
        -------
        Dict
            Input dictionary for generative process
        """
        # Call parent method to get basic inputs
        inputs = super()._get_generative_input(tensors, inference_outputs)

        # Handle multi-modal specific parameters
        if "X_nonspatial" in tensors:
            inputs["batch_index_spatial"] = inputs.pop("batch_index", None)
            inputs["batch_index_nonspatial"] = tensors.get("batch_indices_nonspatial", None)

        return inputs


# 引入一些可能需要的辅助函数
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """Log likelihood of zero-inflated negative binomial distribution."""
    case_zero = torch.log(pi + ((1 - pi) * torch.pow(theta / (theta + mu), theta)))
    case_non_zero = (
        torch.log(1 - pi)
        + torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * torch.log(theta)
        + x * torch.log(mu)
        - (x + theta) * torch.log(theta + mu)
    )
    return torch.where(x < eps, case_zero, case_non_zero)


def log_nb_positive(x, mu, theta, eps=1e-8):
    """Log likelihood of negative binomial distribution."""
    return (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * torch.log(theta)
        + x * torch.log(mu)
        - (x + theta) * torch.log(theta + mu)
    )


def log_poisson(x, mu, eps=1e-8):
    """Log likelihood of Poisson distribution."""
    return x * torch.log(mu) - mu - torch.lgamma(x + 1)


def log_normal(x, mu, var, eps=1e-8):
    """Log likelihood of normal distribution."""
    return -0.5 * torch.log(2 * np.pi * var) - 0.5 * torch.pow(x - mu, 2) / var
