"""SpateoVIProtein — SpateoVI with an optional protein (antibody) modality.

Usage::

    from spateo.external.SpateoVI import SpateoVIProtein

    SpateoVIProtein.setup_anndata(
        adata, batch_key='batch', layer='counts',
        protein_expression_obsm_key='antibody_quantification',
    )
    m = SpateoVIProtein(adata, n_latent=20, smooth_weight=0.4, adv_weight=0.1,
                         protein_loss_weight=1.0)
    m.train(max_epochs=200, batch_size=1024)
    z = m.get_latent_representation()
    p = m.get_protein_expression(transform_batch=...)

Mosaic mask: cells whose ``obsm[protein_expression_obsm_key]`` is all-zero are
treated as 'protein missing' and excluded from the protein loss, so the
modality is genuinely OPTIONAL — you can train on a mix of RNA-only and
RNA+protein cells.
"""
from __future__ import annotations
import logging
from typing import Literal, Optional

import numpy as np
import torch
import pandas as pd
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField, CategoricalObsField, ObsmField, ProteinObsmField,
    NumericalJointObsField, CategoricalJointObsField, NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.utils import setup_anndata_dsp

from ._model import SpateoVI, _build_graph_with_distances
from ._module_protein import SpatialVAEProtein

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local fix for scvi-tools >= 1.3.0 upstream bug.
# ProteinObsmField.transfer_field unconditionally calls
# _get_batch_mask_protein_data which dereferences self.batch_field even when
# use_batch_mask=False — crashes with AttributeError ('NoneType').
# Copied from scvi.data.fields._protein and fixed by guarding on use_batch_mask.
# ---------------------------------------------------------------------------
class _ProteinObsmFieldFixed(ProteinObsmField):
    """Drop-in replacement for ProteinObsmField that survives transfer when
    `use_batch_mask=False` (no batch_field configured)."""

    def transfer_field(self, state_registry, adata_target, **kwargs):
        # Skip ProteinFieldMixin.transfer_field (the buggy one); go to its
        # grandparent's implementation directly.
        from scvi.data.fields._arraylike_field import ObsmField as _ObsmField
        transfer_state_registry = _ObsmField.transfer_field(
            self, state_registry, adata_target, **kwargs
        )
        if self.use_batch_mask:
            batch_mask = self._get_batch_mask_protein_data(adata_target)
            if batch_mask is not None:
                transfer_state_registry[self.PROTEIN_BATCH_MASK] = batch_mask
        return transfer_state_registry


class SpateoVIProtein(SpateoVI):
    """V2 + optional protein head. Inherits spatial machinery from V2."""

    def __init__(
        self,
        adata: AnnData | None = None,
        n_hidden: int = 128,
        n_latent: int = 20,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene","gene-batch","gene-label","gene-cell"] = "gene",
        gene_likelihood: Literal["zinb","nb","poisson","normal"] = "zinb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal","ln"] = "normal",
        spatial_graph_type: Literal["knn","delaunay"] = "knn",
        n_neighbors: int = 10,
        per_batch_graph: bool = True,
        n_spatial_layers: int = 2,
        attention_heads: int = 4,
        edge_feat_dim: int = 8,
        smooth_weight: float = 0.4,
        smooth_sigma: float = 1.0,
        adv_weight: float = 0.1,
        # NEW — protein modality
        protein_loss_weight: float = 1.0,
        protein_decoder_layers: int = 2,
        protein_decoder_hidden: int = 128,
        **kwargs,
    ):
        # bypass SpateoVI.__init__ (which builds the gene-only SpatialVAE) and
        # call the BaseModelClass init directly — we will build SpatialVAEProtein
        # below.
        from scvi.model.base import BaseModelClass
        BaseModelClass.__init__(self, adata)

        self._module_kwargs = dict(
            n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
            dropout_rate=dropout_rate, dispersion=dispersion,
            gene_likelihood=gene_likelihood, latent_distribution=latent_distribution,
        )
        self._model_summary_string = (
            f"SpateoVIProtein — n_latent={n_latent} heads={attention_heads} "
            f"smooth={smooth_weight} adv={adv_weight} "
            f"n_proteins={'auto' if adata is not None else '?'}"
        )

        # --- spatial graph ----------------------------------------------------
        if adata is None or "spatial" not in adata.obsm:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_dist  = torch.zeros(0, dtype=torch.float32)
        else:
            coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
            batch_arr = None
            if per_batch_graph:
                if "_scvi_batch" in adata.obs.columns:
                    batch_arr = adata.obs["_scvi_batch"].values
                elif "batch" in adata.obs.columns:
                    batch_arr = adata.obs["batch"].values
            edge_index, edge_dist = _build_graph_with_distances(
                coords, batch_arr, method=spatial_graph_type, n_neighbors=n_neighbors,
            )
            logger.info(f"SpateoVIProtein graph: {edge_index.shape[1]} edges")
        self.edge_index = edge_index
        self.edge_dist  = edge_dist

        if edge_dist.numel() > 0 and smooth_sigma == 1.0:
            smooth_sigma = max(float(edge_dist.median().item()), 1e-3)

        # --- detect protein modality from registry ---------------------------
        n_proteins = 0
        try:
            n_proteins = int(self.summary_stats.get("n_proteins", 0))
        except Exception:
            n_proteins = 0
        logger.info(f"SpateoVIProtein detected n_proteins = {n_proteins}")

        cat_covs_per_group = (
            self.adata_manager.get_state_registry("categorical_covariates").n_cats_per_key
            if "categorical_covariates" in self.adata_manager.data_registry else None
        )
        num_batches = self.summary_stats.n_batch
        use_size_factor = "size_factor" in self.adata_manager.data_registry
        lib_mean, lib_var = None, None
        minified = getattr(self, "minified_data_type", None)
        if (not use_size_factor
                and minified != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
                and not use_observed_lib_size):
            lib_mean, lib_var = _init_library_size(self.adata_manager, num_batches)

        self.module = SpatialVAEProtein(
            n_input=self.summary_stats.n_vars,
            n_batch=num_batches,
            n_proteins=n_proteins,
            protein_loss_weight=protein_loss_weight,
            protein_decoder_layers=protein_decoder_layers,
            protein_decoder_hidden=protein_decoder_hidden,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=cat_covs_per_group,
            n_hidden=n_hidden, n_latent=n_latent,
            n_spatial_layers=n_spatial_layers, attention_heads=attention_heads,
            edge_feat_dim=edge_feat_dim,
            n_layers=n_layers, dropout_rate=dropout_rate,
            dispersion=dispersion, gene_likelihood=gene_likelihood,
            use_observed_lib_size=use_observed_lib_size,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor,
            library_log_means=lib_mean, library_log_vars=lib_var,
            edge_index=edge_index, edge_dist=edge_dist,
            smooth_weight=smooth_weight, smooth_sigma=smooth_sigma,
            adv_weight=adv_weight,
            **kwargs,
        )
        self.module.minified_data_type = minified
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------
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
        protein_expression_obsm_key: str | None = None,
        protein_names_uns_key: str | None = None,
        **kwargs,
    ):
        """Register adata for SpateoVIProtein — adds optional ProteinObsmField."""
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        if protein_expression_obsm_key is not None:
            anndata_fields.append(
                _ProteinObsmFieldFixed(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY,
                    protein_expression_obsm_key,
                    use_batch_mask=False,         # we mask all-zero rows ourselves
                    colnames_uns_key=protein_names_uns_key,
                    is_count_data=True,
                )
            )
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def get_protein_expression(self, adata=None, indices=None, transform_batch=None,
                               batch_size: int = 4096):
        """Predict protein values for cells in adata."""
        assert self.module.n_proteins > 0, "model has no protein head"
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        out = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            inf = self.module.inference(x.to(self.module.device), batch_index.to(self.module.device))
            z_s = inf["z"]
            if transform_batch is not None:
                # map transform_batch to its integer code
                state = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
                cats = list(state["categorical_mapping"])
                code = cats.index(transform_batch)
                bi_use = torch.full_like(batch_index, code).to(self.module.device)
            else:
                bi_use = batch_index.to(self.module.device)
            rate = self.module.protein_decoder(z_s, bi_use)
            out.append(rate.cpu().numpy())
        return np.concatenate(out, 0)

    # ------------------------------------------------------------------
    # Legacy-name-tolerant load (accepts MERFISHVI3 / SpatialVI3 checkpoints).
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, dir_path: str, adata: AnnData | None = None,
             accelerator: str = "auto", device: int | str = "auto",
             prefix: str | None = None, backup_url: str | None = None, **kwargs):
        """Like :meth:`scvi.model.base.BaseModelClass.load`, but transparently
        accepts checkpoints saved under our older class names (``MERFISHVI3``,
        ``SpatialVI3``)."""
        from ._model import _load_with_legacy_names
        return _load_with_legacy_names(
            cls, dir_path, adata=adata, accelerator=accelerator, device=device,
            prefix=prefix, backup_url=backup_url,
            legacy_names=("MERFISHVI3", "SpatialVI3"),
            **kwargs,
        )


