"""SpateoVI — spatially-aware single-cell VAE for MERFISH / Xenium data.

This is the V3 implementation, which:
  * builds the spatial graph AND stores per-edge Euclidean distances, so the
    GAT layers in the module receive distance-encoded edge features;
  * feeds a spatially-refined latent ``z_s = z + Δz`` into the decoder
    (Δz comes from a multi-layer GATv2 + zero-init Linear residual);
  * adds a spatial smoothness penalty and an adversarial batch-removal loss.

Use ``SpateoVI`` for pure-RNA modelling, or ``SpateoVIProtein`` (in
``._model_protein``) for the optional protein modality (TotalVI-style mosaic
mask). ``SpateoVI`` returns the latent representation ``z_s`` and supports an
optional Harmony post-correction via ``use_harmony=True`` in
``get_latent_representation``.
"""
from __future__ import annotations
import logging, warnings
from typing import Literal, Optional

import numpy as np
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField, CategoricalObsField,
    NumericalJointObsField, CategoricalJointObsField, NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    BaseModelClass, VAEMixin, UnsupervisedTrainingMixin, RNASeqMixin,
    ArchesMixin,
)
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.utils import setup_anndata_dsp

from ._module import SpatialVAE

logger = logging.getLogger(__name__)


def _build_graph_with_distances(
    coords: np.ndarray,
    batch_assignments: Optional[np.ndarray] = None,
    method: str = "knn",
    n_neighbors: int = 10,
):
    """Return (edge_index [2,E] long tensor, edge_dist [E] float tensor).

    If batch_assignments is given, the graph is built within each batch and
    edges across batches are never created.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial import Delaunay

    N = coords.shape[0]
    src_list, dst_list, d_list = [], [], []

    def _add_block(idx_global: np.ndarray):
        c = coords[idx_global]
        if c.shape[0] < 3:
            return
        if method == "knn":
            k = min(n_neighbors + 1, len(c))   # +1 because self is included
            nn = NearestNeighbors(n_neighbors=k).fit(c)
            d, idx = nn.kneighbors(c)
            # drop self (column 0)
            d = d[:, 1:]; idx = idx[:, 1:]
            rows = np.repeat(np.arange(len(c)), idx.shape[1])
            cols = idx.ravel()
            dists = d.ravel().astype(np.float32)
        else:  # delaunay
            tri = Delaunay(c)
            edges = set()
            for s in tri.simplices:
                m = len(s)
                for i in range(m):
                    for j in range(i + 1, m):
                        edges.add(tuple(sorted((s[i], s[j]))))
            edges = np.array(list(edges), dtype=np.int64)
            rows, cols = edges[:, 0], edges[:, 1]
            dists = np.linalg.norm(c[rows] - c[cols], axis=1).astype(np.float32)
            # store both directions for symmetry
            rows, cols = np.r_[rows, cols], np.r_[cols, rows]
            dists = np.r_[dists, dists]
        src_list.append(idx_global[rows]); dst_list.append(idx_global[cols])
        d_list.append(dists)

    if batch_assignments is not None:
        for b in np.unique(batch_assignments):
            _add_block(np.where(batch_assignments == b)[0])
    else:
        _add_block(np.arange(N))

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    src = np.concatenate(src_list); dst = np.concatenate(dst_list)
    dist = np.concatenate(d_list).astype(np.float32)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_dist  = torch.tensor(dist, dtype=torch.float32)
    return edge_index, edge_dist


class SpateoVI(VAEMixin, UnsupervisedTrainingMixin, RNASeqMixin, ArchesMixin, BaseModelClass):
    """Spatially-aware single-cell VAE for MERFISH / Xenium data.

    Pure-RNA model. For the multi-modal RNA + protein variant see
    :class:`SpateoVIProtein` (in ``spateo.external.SpateoVI``).

    Parameters
    ----------
    adata
        AnnData registered via :meth:`SpateoVI.setup_anndata`. Must have a
        3-D spatial coordinate in ``adata.obsm['spatial']``.
    n_hidden
        Hidden layer width of the encoder/decoder MLP.
    n_latent
        Latent dimensionality.
    n_spatial_layers
        Number of GATv2 layers in the spatial-refinement stack.
    attention_heads
        Number of attention heads per GATv2 layer.
    edge_feat_dim
        Output dim of the small ``Dist→MLP`` that encodes Euclidean distance
        into edge features fed to the GATv2.
    smooth_weight
        Weight α of the spatial smoothness loss
        ``L_smooth = mean exp(-d²/σ²) ‖z_s[i]-z_s[j]‖²``.
    adv_weight
        GRL strength λ of the adversarial batch-removal discriminator.
    spatial_graph_type
        ``'knn'`` or ``'delaunay'``.
    n_neighbors
        kNN neighbour count (for ``spatial_graph_type='knn'``).
    per_batch_graph
        If True, build the kNN graph independently per batch (no cross-batch
        edges). Strongly recommended for multi-section MERFISH data.
    """

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
        # spatial graph
        spatial_graph_type: Literal["knn","delaunay"] = "knn",
        n_neighbors: int = 10,
        per_batch_graph: bool = True,
        # spatial refinement
        n_spatial_layers: int = 2,
        attention_heads: int = 4,
        edge_feat_dim: int = 8,
        # losses
        smooth_weight: float = 0.5,
        smooth_sigma: float = 1.0,
        adv_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(adata)

        # store config
        self._module_kwargs = dict(
            n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
            dropout_rate=dropout_rate, dispersion=dispersion,
            gene_likelihood=gene_likelihood, latent_distribution=latent_distribution,
        )
        self._model_summary_string = (
            f"SpateoVI — n_hidden={n_hidden} n_latent={n_latent} "
            f"n_spatial_layers={n_spatial_layers} heads={attention_heads} "
            f"smooth={smooth_weight} adv={adv_weight}"
        )

        # ---- build edge_index + distances ------------------------------------
        if adata is None or "spatial" not in adata.obsm:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_dist  = torch.zeros(0, dtype=torch.float32)
            warnings.warn("SpateoVI: no obsm['spatial'] — running without spatial graph.")
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
            logger.info(
                f"SpateoVI graph: {edge_index.shape[1]} edges, "
                f"d med={edge_dist.median().item():.2f}  "
                f"d max={edge_dist.max().item():.2f}"
            )
        self.edge_index = edge_index
        self.edge_dist  = edge_dist

        # smoothness sigma adaptive to median edge distance, if user kept default 1.0
        if edge_dist.numel() > 0:
            smooth_sigma = float(edge_dist.median().item()) if smooth_sigma == 1.0 else smooth_sigma
            smooth_sigma = max(smooth_sigma, 1e-3)

        # ---- module ----------------------------------------------------------
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

        self.module = SpatialVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=num_batches,
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
        **kwargs,
    ):
        """Register an AnnData object for SpateoVI.

        Parameters
        ----------
        adata
            AnnData with raw counts in ``adata.X`` or ``adata.layers[layer]``.
            Must have a 3-D spatial coordinate in ``adata.obsm['spatial']``.
        layer
            If given, read counts from ``adata.layers[layer]``.
        batch_key
            Column in ``adata.obs`` identifying batches (samples / sections).
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    # Optional Harmony post-hoc batch correction on top of get_latent_representation.
    # The VAE already pushes batch effects out of z_s via GRL; Harmony is a fast,
    # well-validated second pass on the 20-dim latent that further tightens
    # cross-batch alignment. Off by default — only run when explicitly requested.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Optional[np.ndarray] = None,
        give_mean: bool = True,
        mc_samples: int = 1,
        batch_size: Optional[int] = None,
        return_dist: bool = False,
        dataloader=None,
        # NEW — harmony post-processing
        use_harmony: bool = False,
        harmony_batch_key: str | None = None,
        harmony_kwargs: dict | None = None,
    ):
        """Return the spatially-refined latent z_s.

        Parameters
        ----------
        use_harmony
            If True, run harmonypy on the returned latent using `harmony_batch_key`
            (defaults to the batch key configured in setup_anndata). Returns the
            harmony-corrected matrix instead of the raw z_s. Requires `harmonypy`.
        harmony_batch_key
            Name of the column in adata.obs to use for harmony. If None, uses the
            batch_key from setup_anndata's registry.
        harmony_kwargs
            Extra keyword args forwarded to harmonypy.run_harmony (e.g. theta=2,
            max_iter_harmony=20).
        """
        # call the VAEMixin's implementation for the raw latent
        z = super().get_latent_representation(
            adata=adata, indices=indices, give_mean=give_mean,
            mc_samples=mc_samples, batch_size=batch_size,
            return_dist=return_dist, dataloader=dataloader,
        )
        if not use_harmony:
            return z
        if return_dist:
            raise ValueError("use_harmony is incompatible with return_dist=True")

        # resolve batch labels
        target = self._validate_anndata(adata) if adata is not None else self.adata
        if indices is not None:
            target = target[indices]
        if harmony_batch_key is None:
            from scvi import REGISTRY_KEYS
            state = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
            # the original column name used in setup_anndata
            harmony_batch_key = (
                state["original_key"] if "original_key" in state
                else "batch" if "batch" in target.obs.columns
                else None
            )
        if harmony_batch_key is None or harmony_batch_key not in target.obs.columns:
            raise ValueError(
                "harmony_batch_key must be set or 'batch' must be in adata.obs"
            )

        try:
            import harmonypy as hm
        except ImportError as e:
            raise ImportError(
                "use_harmony=True needs the harmonypy package. "
                "Install with `pip install harmonypy`."
            ) from e

        meta = target.obs[[harmony_batch_key]].copy()
        meta[harmony_batch_key] = meta[harmony_batch_key].astype(str)
        kwargs = harmony_kwargs or {}
        kwargs.setdefault("verbose", False)
        logger.info(
            f"Running Harmony on {z.shape[0]} cells × {z.shape[1]} dims "
            f"(batch key {harmony_batch_key!r})"
        )
        ho = hm.run_harmony(z, meta, harmony_batch_key, **kwargs)
        zc = np.asarray(ho.Z_corr, dtype=np.float32)
        # harmonypy versions disagree on Z_corr orientation — coerce to (N, D)
        if zc.shape[0] != z.shape[0] and zc.shape == (z.shape[1], z.shape[0]):
            zc = zc.T
        return zc

    # ------------------------------------------------------------------
    # Legacy-name-tolerant load (so old MERFISHVI/SpatialVI checkpoints work).
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, dir_path: str, adata: AnnData | None = None,
             accelerator: str = "auto", device: int | str = "auto",
             prefix: str | None = None, backup_url: str | None = None, **kwargs):
        """Like :meth:`scvi.model.base.BaseModelClass.load`, but transparently
        accepts checkpoints saved under our older class names (``MERFISHVI``,
        ``MERFISHVI2``, ``SpatialVI``, ``SpatialVI2``)."""
        return _load_with_legacy_names(
            cls, dir_path, adata=adata, accelerator=accelerator, device=device,
            prefix=prefix, backup_url=backup_url,
            legacy_names=("MERFISHVI", "MERFISHVI2", "SpatialVI", "SpatialVI2"),
            **kwargs,
        )


def _load_with_legacy_names(cls, dir_path, *, adata, accelerator, device,
                            prefix, backup_url, legacy_names, **kwargs):
    """Shared helper that patches ``registry_['model_name']`` to
    ``cls.__name__`` in memory before delegating to the underlying scvi load.

    We have to call ``BaseModelClass.load.__func__`` directly so that the
    inner ``cls.__name__`` check sees *our* class (e.g. ``SpateoVIProtein``)
    rather than ``BaseModelClass``."""
    import os, shutil, tempfile

    # the *unbound* classmethod function of scvi's load — calling it with our
    # ``cls`` keeps the inner ``cls.__name__`` check pointing at SpateoVI(...)
    _scvi_load = BaseModelClass.load.__func__

    model_pt = os.path.join(dir_path, (prefix or "") + "model.pt")
    if os.path.exists(model_pt):
        ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
        saved = ckpt.get("attr_dict", {}).get("registry_", {}).get("model_name")
        if saved is not None and saved != cls.__name__ and saved in legacy_names:
            logger.info(f"Migrating legacy checkpoint class name {saved!r} -> {cls.__name__!r}")
            ckpt["attr_dict"]["registry_"]["model_name"] = cls.__name__
            tmp = tempfile.mkdtemp(prefix=f"{cls.__name__.lower()}_load_")
            try:
                shutil.copytree(dir_path, os.path.join(tmp, "m"), dirs_exist_ok=True)
                torch.save(ckpt, os.path.join(tmp, "m", (prefix or "") + "model.pt"))
                return _scvi_load(
                    cls,
                    os.path.join(tmp, "m"), adata=adata,
                    accelerator=accelerator, device=device,
                    prefix=prefix, backup_url=backup_url, **kwargs,
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return _scvi_load(
        cls,
        dir_path, adata=adata, accelerator=accelerator, device=device,
        prefix=prefix, backup_url=backup_url, **kwargs,
    )


# Backwards-compatibility alias for the original spateo SpateoVI name.
# Old code using ``from spateo.external.SpateoVI import SpatialVI`` keeps working.
SpatialVI = SpateoVI
