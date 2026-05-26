"""SpateoVI v3 — V2 (spatial-aware) + optional protein (antibody) head.

Inspired by TOTALVI's mosaic protein decoder, but kept simple:
  * a small MLP decoder z_s → protein rate (with batch one-hot conditioning)
  * NegativeBinomial protein likelihood with learnable per-protein dispersion
  * cells without measured proteins (e.g. non-250multi batches) are masked out
    of the protein loss, so the modality is genuinely OPTIONAL

When n_proteins == 0 the model is identical to SpatialVAE.
"""
from __future__ import annotations
import warnings
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial as TorchNB

from scvi import REGISTRY_KEYS
from scvi.module.base import auto_move_data, LossOutput

from ._module import SpatialVAE


class _ProteinDecoder(nn.Module):
    """Tiny MLP: z_s (+ batch one-hot) -> softplus mean for protein NB."""

    def __init__(self, n_latent: int, n_proteins: int, n_batch: int,
                 n_hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_batch = n_batch
        in_dim = n_latent + (n_batch if n_batch > 1 else 0)
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, n_hidden), nn.LayerNorm(n_hidden), nn.GELU(), nn.Dropout(dropout)]
            d = n_hidden
        self.trunk = nn.Sequential(*layers)
        self.head_mean = nn.Linear(d, n_proteins)

    def forward(self, z: torch.Tensor, batch_index: Optional[torch.Tensor]) -> torch.Tensor:
        if self.n_batch > 1 and batch_index is not None:
            bi = batch_index.view(-1).long()
            bi = torch.clamp(bi, 0, self.n_batch - 1)
            onehot = F.one_hot(bi, num_classes=self.n_batch).to(z.dtype)
            h = torch.cat([z, onehot], dim=-1)
        else:
            h = z
        h = self.trunk(h)
        rate = F.softplus(self.head_mean(h)) + 1e-4              # [N, P], positive
        return rate


class SpatialVAEProtein(SpatialVAE):
    """V2 + optional protein NB head."""

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_proteins: int = 0,
        protein_loss_weight: float = 1.0,
        protein_decoder_layers: int = 2,
        protein_decoder_hidden: int = 128,
        **kwargs,
    ):
        super().__init__(n_input=n_input, n_batch=n_batch, **kwargs)
        self.n_proteins = int(n_proteins)
        self.protein_loss_weight = float(protein_loss_weight)

        if self.n_proteins > 0:
            # per-protein NB dispersion (learnable)
            self.py_log_theta = nn.Parameter(torch.zeros(self.n_proteins))
            self.protein_decoder = _ProteinDecoder(
                n_latent=kwargs.get("n_latent", 20),
                n_proteins=self.n_proteins,
                n_batch=max(n_batch, 1),
                n_hidden=protein_decoder_hidden,
                n_layers=protein_decoder_layers,
                dropout=kwargs.get("dropout_rate", 0.1),
            )

    # ------------------------------------------------------------------
    @auto_move_data
    def predict_protein(self, z_s: torch.Tensor, batch_index: Optional[torch.Tensor]) -> torch.Tensor:
        return self.protein_decoder(z_s, batch_index)

    # ------------------------------------------------------------------
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight=1.0):
        base = super().loss(tensors, inference_outputs, generative_outputs, kl_weight)
        if self.n_proteins == 0:
            return base
        y = tensors.get(REGISTRY_KEYS.PROTEIN_EXP_KEY, None)
        if y is None:
            return base                                          # protein modality not registered

        z_s = inference_outputs["z"]                              # decoder uses z_s
        batch_index = tensors.get(REGISTRY_KEYS.BATCH_KEY)
        rate = self.protein_decoder(z_s, batch_index)             # [N, P]
        theta = F.softplus(self.py_log_theta) + 1e-4              # [P]
        # NegativeBinomial parameterised by mean (rate) and dispersion (theta)
        # — using scvi's parameterisation: variance = rate + rate^2 / theta
        # convert mu, theta -> probs, total_count for torch.distributions.NegativeBinomial
        total_count = theta.unsqueeze(0).expand_as(rate)
        probs = rate / (rate + theta.unsqueeze(0))
        nb = TorchNB(total_count=total_count, probs=probs)
        nll = -nb.log_prob(y.float()).sum(-1)                     # [N]

        # mosaic mask — cells with all-zero protein vector are treated as missing
        mask = (y.float().sum(-1) > 0).float()
        if mask.sum() < 1:
            return base
        protein_loss = (nll * mask).sum() / mask.sum().clamp_min(1)

        total = base.loss + self.protein_loss_weight * protein_loss
        extra = dict(base.extra_metrics) if base.extra_metrics is not None else {}
        extra.update({"loss_protein": protein_loss.detach(),
                      "frac_protein_cells": mask.mean().detach()})
        return LossOutput(
            loss=total,
            reconstruction_loss=base.reconstruction_loss,
            kl_local=base.kl_local,
            extra_metrics=extra,
        )
