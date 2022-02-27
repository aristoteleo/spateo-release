"""Functions to refine staining and RNA alignments.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from tqdm import tqdm

from ...configuration import SKM
from ...errors import PreprocessingError


class AlignmentRefiner(nn.Module):
    """Pytorch module to refine alignment between two images.
    Performs Autograd on the affine transformation matrix.
    """

    def __init__(self, reference: np.ndarray, to_align: np.ndarray, theta: Optional[np.ndarray] = None):
        if reference.dtype != np.dtype(bool) or to_align.dtype != np.dtype(bool):
            raise PreprocessingError("`AlignmentRefiner` only supports boolean arrays.")

        super().__init__()
        self.reference = torch.tensor(reference)[None][None].float()
        self.to_align = torch.tensor(to_align)[None][None].float()

        self.weight = torch.tensor(
            np.where(reference, reference.size / (2 * reference.sum()), reference.size / (2 * (~reference).sum()))
        )[None][None]

        # Affine matrix
        if theta is not None:
            self.theta = nn.Parameter(torch.tensor(theta))
        else:
            self.theta = nn.Parameter(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                )
            )

        self.__optimizer = None
        self.history = {}

    def forward(self):
        return self.transform(self.theta, self.to_align, train=True)

    def loss(self, pred):
        return torch.sum(self.weight * (pred - self.reference) ** 2) / self.weight.numel()

    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = torch.optim.Adam(self.parameters())
        return self.__optimizer

    def train(self, n_epochs: int = 100):
        optimizer = self.optimizer()

        with tqdm(total=n_epochs) as pbar:
            for i in range(n_epochs):
                pred = self()
                loss = self.loss(pred)
                self.history.setdefault("loss", []).append(loss.item())

                pbar.set_description(f"Loss {loss.item():.4f}")
                pbar.update(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @staticmethod
    def transform(theta, x, train=False):
        """This method should be used when applying the learned affine
        transformation to an arbitrary image.
        """
        if not train:
            theta = torch.tensor(theta).float()
            x = torch.tensor(x)[None][None].float()
        grid = F.affine_grid(theta.unsqueeze(0), x.size(), align_corners=False)
        t = F.grid_sample(x, grid, align_corners=False)
        return t if train else t.detach().numpy()

    def affine(self):
        return self.theta.detach().numpy()


def refine_alignment(
    adata: AnnData,
    stain_layer: str = SKM.STAIN_LAYER_KEY,
    rna_layer: str = SKM.UNSPLICED_LAYER_KEY,
    n_epochs: int = 100,
    transform_layers: Optional[List[str]] = None,
):
    """Refine the alignment between the staining image and RNA coordinates.

    There are often small misalignments between the staining image and RNA, which
    results in incorrect aggregation of pixels into cells based on staining.
    This function attempts to refine these alignments based on the staining and
    (unspliced) RNA masks.

    Args:
        adata: Input Anndata
        stain_layer: Layer containing staining image. First will look for layer
            `{stain_layer}_mask`. Otherwise, this will be taken as a literal.
        rna_layer: Layer containing (unspliced) RNA. First, will look for layer
            `{rna_layer}_mask`. Otherwise, this will be taken as a literal.
        n_epochs: Number of epochs to run optimization
        transform_layers: Layers to transform and overwrite inplace.
    """
    layer = SKM.gen_new_layer_key(stain_layer, SKM.MASK_SUFFIX)
    if layer not in adata.layers:
        layer = stain_layer
    stain_mask = SKM.select_layer_data(adata, layer)

    layer = SKM.gen_new_layer_key(rna_layer, SKM.MASK_SUFFIX)
    if layer not in adata.layers:
        layer = rna_layer
    rna_mask = SKM.select_layer_data(adata, layer)

    aligner = AlignmentRefiner(rna_mask, stain_mask)
    aligner.train(n_epochs)
    theta = aligner.affine()

    uns_alignment = {"theta": theta}
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_ALIGNMENT_KEY, uns_alignment)

    if transform_layers:
        for layer in transform_layers:
            data = SKM.select_layer_data(adata, layer)
            transformed = aligner.transform(theta, data)
            if data.dtype == np.dtype(bool):
                transformed = transformed > 0.5
            SKM.set_layer_data(adata, layer, transformed)
