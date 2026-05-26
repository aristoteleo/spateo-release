"""Spatially-aware VAE module for SpateoVI.

Three key design choices:

1) Spatially refined latent fed to the decoder. The encoder produces ``z``;
   a stack of GATv2 layers (with distance-encoded edge features) plus a
   zero-init Linear yields ``z_s = z + Δz``. ``z_s`` is what the decoder sees,
   so the reconstruction loss directly trains the spatial refinement.

2) Explicit spatial smoothness loss:
       ``L_smooth = mean_{(i,j) in E} exp(-d_ij² / σ²) ‖z_s[i] - z_s[j]‖²``
   Encourages neighbouring cells to share latent state (denoising prior).

3) Adversarial batch removal. A gradient-reversal layer + small MLP
   discriminator predicts batch from ``z_s``; the encoder is trained to
   fool it. This directly attacks integration quality (residual batch
   effect in ``z_s``).
"""
from __future__ import annotations
import logging, warnings
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch_geometric.nn import GATv2Conv

from scvi.module import VAE
from scvi.module.base import auto_move_data, LossOutput

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- GRL ------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


# ------------------------------------------------ spatial refinement --------
class SpatialRefine(nn.Module):
    """Multi-layer GATv2 with distance-encoded edge features that refines z to z_s.

    Output is a residual update on z: z_s = LayerNorm(z + dropout(g(z))).
    """

    def __init__(
        self,
        n_latent: int,
        n_layers: int = 2,
        edge_dim: int = 8,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        # tiny MLP that turns scalar distance into edge_dim features
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, edge_dim), nn.GELU(), nn.Linear(edge_dim, edge_dim)
        )
        self.gat_layers = nn.ModuleList([
            GATv2Conv(in_channels=n_latent, out_channels=n_latent,
                      heads=heads, concat=False, edge_dim=edge_dim,
                      dropout=dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # zero-init projection: at start z_s = z + 0 = z, so the model
        # behaves exactly like scVI until the spatial refinement is learned
        self.out_proj = nn.Linear(n_latent, n_latent)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor,
                edge_index: Optional[torch.Tensor],
                edge_dist: Optional[torch.Tensor]) -> torch.Tensor:
        """z: [N, n_latent]; edge_index: [2, E]; edge_dist: [E] (scalar distances).
        Returns z_s = z + delta where delta is the spatial refinement (zero at init)."""
        if edge_index is None or edge_index.numel() == 0:
            return z
        n = z.size(0)
        ok = (edge_index[0] < n) & (edge_index[1] < n)
        ei = edge_index[:, ok]
        if ei.size(1) == 0:
            return z
        d = edge_dist[ok].view(-1, 1) if edge_dist is not None else \
            torch.zeros(ei.size(1), 1, device=z.device)
        eattr = self.dist_mlp(d)
        # GAT stack (with internal residual via gelu)
        h = z
        for gat in self.gat_layers:
            try:
                h_new = gat(h, ei, edge_attr=eattr)
            except Exception as e:
                warnings.warn(f"GATv2 step failed: {e}; returning z unchanged")
                return z
            h = h + self.dropout(F.gelu(h_new))   # residual GAT
        # zero-init projection -> z_s starts identical to z, learns refinement
        delta = self.out_proj(h)
        return z + delta


# ----------------------------------------------- batch discriminator --------
class BatchDiscriminator(nn.Module):
    def __init__(self, n_latent: int, n_batch: int, n_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(n_hidden, max(n_batch, 1)),
        )

    def forward(self, z):
        return self.net(z)


# ====================================================== SpatialVAE V2 ======
class SpatialVAE(VAE):
    """SpatialVAE V2 — feeds spatially-refined latent into the decoder, with
    explicit smoothness loss + adversarial batch-removal loss."""

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 20,
        n_spatial_layers: int = 2,
        attention_heads: int = 4,
        edge_feat_dim: int = 8,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene","gene-batch","gene-label","gene-cell"] = "gene",
        gene_likelihood: Literal["zinb","nb","poisson","normal"] = "zinb",
        latent_distribution: Literal["normal","ln"] = "normal",
        edge_index: Optional[torch.Tensor] = None,
        edge_dist:  Optional[torch.Tensor] = None,
        smooth_weight: float = 0.5,
        smooth_sigma: float = 1.0,
        adv_weight: float = 0.2,
        adv_warmup_epochs: int = 5,
        **kwargs,
    ):
        super().__init__(
            n_input=n_input, n_batch=n_batch, n_labels=n_labels,
            n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
            dropout_rate=dropout_rate, dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution, **kwargs,
        )
        self.n_batch_v2 = n_batch
        self.smooth_weight = smooth_weight
        self.smooth_sigma  = smooth_sigma
        self.adv_weight    = adv_weight
        self.adv_warmup_epochs = adv_warmup_epochs

        self.spatial_refine = SpatialRefine(
            n_latent=n_latent, n_layers=n_spatial_layers,
            edge_dim=edge_feat_dim, heads=attention_heads, dropout=dropout_rate,
        )
        self.batch_disc = BatchDiscriminator(n_latent=n_latent, n_batch=max(n_batch,1))

        # store graph as buffers
        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        if edge_dist is None:
            edge_dist = torch.zeros(edge_index.shape[1], dtype=torch.float32)
        self.register_buffer("_edge_index", edge_index)
        self.register_buffer("_edge_dist",  edge_dist)

    # ------------ inference: produce z_s and substitute z for decoder -------
    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None,
                  cont_covariates=None, cat_covariates=None, **kwargs):
        if cont_covs is None and cont_covariates is not None: cont_covs = cont_covariates
        if cat_covs  is None and cat_covariates  is not None: cat_covs  = cat_covariates
        for k in ("cont_covariates", "cat_covariates"):
            kwargs.pop(k, None)
        out = super().inference(x, batch_index, cont_covs=cont_covs, cat_covs=cat_covs, **kwargs)
        z = out["z"]
        if z.device != self._edge_index.device:
            self._edge_index = self._edge_index.to(z.device)
            self._edge_dist  = self._edge_dist.to(z.device)
        # spatial refinement
        z_s = self.spatial_refine(z, self._edge_index, self._edge_dist)
        out["z_orig"] = z
        out["z"] = z_s                 # decoder will use the spatially-refined latent
        return out

    # ------------ loss: + smoothness + adversarial batch -------------------
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0):
        base = super().loss(tensors, inference_outputs, generative_outputs, kl_weight)
        z_s    = inference_outputs["z"]            # decoder input (post-refinement)
        device = z_s.device

        # (a) spatial smoothness on z_s (the decoder input). The encoder z stays
        # free to be sharp; the GAT refinement carries the smoothing burden,
        # so the model learns a spatially-coherent decoder input for integration
        ei = self._edge_index
        loss_smooth = torch.zeros((), device=device)
        if ei.numel() > 0:
            n = z_s.size(0)
            ok = (ei[0] < n) & (ei[1] < n)
            if ok.any():
                src = ei[0, ok]; dst = ei[1, ok]
                d   = self._edge_dist[ok]
                w   = torch.exp(-(d**2) / (2 * self.smooth_sigma ** 2 + 1e-8))
                diff2 = ((z_s[src] - z_s[dst]) ** 2).sum(-1)
                loss_smooth = (diff2 * w).mean()

        # (b) adversarial batch removal via gradient reversal
        loss_adv = torch.zeros((), device=device)
        loss_adv_disc = torch.zeros((), device=device)
        if self.n_batch_v2 > 1:
            from scvi import REGISTRY_KEYS
            bk = tensors.get(REGISTRY_KEYS.BATCH_KEY)
            if bk is not None:
                bk_long = bk.view(-1).long().to(device)
                # encoder/decoder side: GRL → discriminator → predict batch → encoder gets reversed gradient
                z_rev = grad_reverse(z_s, lambd=self.adv_weight)
                logits = self.batch_disc(z_rev)
                loss_adv = F.cross_entropy(logits, bk_long)
                # also a "free" discriminator-only loss (non-reversed) so the disc actually learns
                with torch.no_grad():
                    pass
                logits_d = self.batch_disc(z_s.detach())
                loss_adv_disc = F.cross_entropy(logits_d, bk_long)

        total = base.loss + self.smooth_weight * loss_smooth + loss_adv + loss_adv_disc
        kl_local = base.kl_local
        # store auxiliary metrics
        extra = dict(base.extra_metrics) if base.extra_metrics is not None else {}
        extra.update({
            "loss_smooth": loss_smooth.detach(),
            "loss_adv_grl": loss_adv.detach(),
            "loss_adv_disc": loss_adv_disc.detach(),
        })
        return LossOutput(
            loss=total,
            reconstruction_loss=base.reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=extra,
        )

    # passthrough param fixers (same as original SpatialVAE)
    def _get_inference_input(self, tensors, full_forward_pass=False):
        inp = super()._get_inference_input(tensors, full_forward_pass)
        if "cont_covariates" in inp and "cont_covs" not in inp:
            inp["cont_covs"] = inp.pop("cont_covariates")
        if "cat_covariates" in inp and "cat_covs" not in inp:
            inp["cat_covs"] = inp.pop("cat_covariates")
        return inp

    def _get_generative_input(self, tensors, inference_outputs):
        out = super()._get_generative_input(tensors, inference_outputs)
        if "cont_covariates" in out and "cont_covs" not in out:
            out["cont_covs"] = out.pop("cont_covariates")
        if "cat_covariates" in out and "cat_covs" not in out:
            out["cat_covs"] = out.pop("cat_covariates")
        return out
