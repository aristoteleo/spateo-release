"""Functions to refine staining and RNA alignments.
"""
import math
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from kornia.geometry.transform import thin_plate_spline as tps
from tqdm import tqdm
from typing_extensions import Literal

from ..configuration import SKM
from ..errors import SegmentationError
from ..logging import logger_manager as lm
from . import utils


class AlignmentRefiner(nn.Module):
    def __init__(self, reference: np.ndarray, to_align: np.ndarray):
        super().__init__()
        reference = reference.astype(float) / reference.max()
        to_align = to_align.astype(float) / to_align.max()
        self.reference = torch.tensor(reference)[None][None].float()
        self.to_align = torch.tensor(to_align)[None][None].float()
        self.__optimizer = None
        self.weight = self.reference + 1
        self.history = {}

    def loss(self, pred):
        return -torch.mean(self.weight * (pred * self.reference))

    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = torch.optim.Adam(self.parameters())
        return self.__optimizer

    def forward(self):
        return self.transform(self.to_align, self.get_params(True), train=True)

    def train(self, n_epochs: int = 100):
        optimizer = self.optimizer()

        with tqdm(total=n_epochs) as pbar:
            for _ in range(n_epochs):
                pred = self()
                loss = self.loss(pred)
                self.history.setdefault("loss", []).append(loss.item())

                pbar.set_description(f"Loss {loss.item():.4e}")
                pbar.update(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_params(self, train=False):
        raise NotImplementedError()

    @staticmethod
    def transform(x, params, train=False):
        raise NotImplementederror()


class NonRigidAlignmentRefiner(AlignmentRefiner):
    """Pytorch module to refine alignment between two images by evaluating the
    thin-plate-spline (TPS) for non-rigid alignment.
    Performs Autograd on the displacement matrix between source and destination
    points.
    """

    def __init__(self, reference: np.ndarray, to_align: np.ndarray, meshsize: Optional[int] = None):
        meshsize = meshsize or min(to_align.shape) // 3
        meshes = (math.ceil(to_align.shape[0] / meshsize), math.ceil(to_align.shape[1] / meshsize))
        if meshes[0] <= 1 or meshes[1] <= 1:
            raise SegmentationError(
                f"Using `meshsize` {meshsize} for image of shape {to_align.shape} "
                f"results in {meshes} meshes. Please reduce `meshsize`."
            )
        super().__init__(reference, to_align)
        self.src_points = torch.cartesian_prod(
            torch.linspace(-1, 1, meshes[1]),
            torch.linspace(-1, 1, meshes[0]),
        )
        self.displacement = nn.Parameter(torch.zeros(self.src_points.shape))

    def get_params(self, train=False):
        src_points, displacement = self.src_points, self.displacement
        if not train:
            src_points = src_points.detach().numpy()
            displacement = displacement.detach().numpy()
        return dict(src_points=src_points, displacement=displacement)

    @staticmethod
    def transform(x, params, train=False):
        """This method should be used when applying the learned affine
        transformation to an arbitrary image.
        """
        src_points, displacement = params["src_points"], params["displacement"]
        dst_points = src_points + displacement
        if not train:
            src_points = torch.tensor(src_points).float()
            dst_points = torch.tensor(dst_points).float()
            x = torch.tensor(x)[None][None].float()
        dst_points = dst_points.unsqueeze(0)
        src_points = src_points.unsqueeze(0)
        kernel_weights, affine_weights = tps.get_tps_transform(dst_points, src_points)
        t = tps.warp_image_tps(x, src_points, kernel_weights, affine_weights).squeeze()
        return t if train else t.detach().numpy()


class RigidAlignmentRefiner(AlignmentRefiner):
    """Pytorch module to refine alignment between two images.
    Performs Autograd on the affine transformation matrix.
    """

    def __init__(self, reference: np.ndarray, to_align: np.ndarray, theta: Optional[np.ndarray] = None):
        super().__init__(reference, to_align)
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

    @staticmethod
    def transform(x, params, train=False):
        """This method should be used when applying the learned affine
        transformation to an arbitrary image.
        """
        theta = params["theta"]
        if not train:
            theta = torch.tensor(theta).float()
            x = torch.tensor(x)[None][None].float()
        grid = F.affine_grid(theta.unsqueeze(0), x.size(), align_corners=False)
        t = F.grid_sample(x, grid, align_corners=False)
        return t if train else t.detach().numpy().squeeze()

    def get_params(self, train=False):
        theta = self.theta
        if not train:
            theta = theta.detach().numpy()
        return dict(theta=theta)


MODULES = {"rigid": RigidAlignmentRefiner, "non-rigid": NonRigidAlignmentRefiner}


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def refine_alignment(
    adata: AnnData,
    stain_layer: str = SKM.STAIN_LAYER_KEY,
    rna_layer: str = SKM.UNSPLICED_LAYER_KEY,
    mode: Literal["rigid", "non-rigid"] = "rigid",
    downscale: float = 1,
    k: int = 5,
    n_epochs: int = 100,
    transform_layers: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """Refine the alignment between the staining image and RNA coordinates.

    There are often small misalignments between the staining image and RNA, which
    results in incorrect aggregation of pixels into cells based on staining.
    This function attempts to refine these alignments based on the staining and
    (unspliced) RNA masks.

    Args:
        adata: Input Anndata
        stain_layer: Layer containing staining image.
        rna_layer: Layer containing (unspliced) RNA.
        mode: The alignment mode. Two modes are supported:
            * rigid: A global alignment method that finds a rigid (affine)
                transformation matrix
            * non-rigid: A semi-local alignment method that finds a thin-plate-spline
                with a mesh of certain size. By default, each cell in the mesh
                consists of 1000 x 1000 pixels. This value can be modified
                by providing a `binsize` argument to this function (specifically,
                as part of additional **kwargs).
        downscale: Downscale matrices by this factor to reduce memory and runtime.
        k: Kernel size for Gaussian blur of the RNA matrix.
        n_epochs: Number of epochs to run optimization
        transform_layers: Layers to transform and overwrite inplace.
        **kwargs: Additional keyword arguments to pass to the Pytorch module.
    """
    if mode not in MODULES.keys():
        raise SegmentationError('`mode` must be one of "rigid" and "non-rigid"')
    if adata.shape[0] * downscale > 10000 or adata.shape[1] * downscale > 10000:
        lm.main_warning(
            "Input has dimension > 10000. This may take a while and a lot of memory. "
            "Consider downscaling using the `downscale` option."
        )

    stain = SKM.select_layer_data(adata, stain_layer, make_dense=True)
    rna = SKM.select_layer_data(adata, rna_layer, make_dense=True)
    if k > 1 and rna.dtype != np.dtype(bool):
        lm.main_debug(f"Applying Gaussian blur with k={k}.")
        rna = utils.conv2d(rna, k, mode="gauss")
    if downscale < 1:
        lm.main_debug(f"Downscaling by a factor of {downscale}.")
        stain = cv2.resize(stain.astype(float), (0, 0), fx=downscale, fy=downscale)
        rna = cv2.resize(rna.astype(float), (0, 0), fx=downscale, fy=downscale)

    lm.main_info(f"Refining alignment in {mode} mode.")
    module = MODULES[mode]
    # NOTE: we find a transformation FROM the stain coordinates TO the RNA coordinates
    aligner = module(rna, stain, **kwargs)
    aligner.train(n_epochs)

    params = aligner.get_params()
    SKM.set_uns_spatial_attribute(adata, SKM.UNS_SPATIAL_ALIGNMENT_KEY, params)

    if transform_layers:
        if isinstance(transform_layers, str):
            transform_layers = [transform_layers]
        lm.main_info(f"Transforming layers {transform_layers}")
        for layer in transform_layers:
            data = SKM.select_layer_data(adata, layer)
            transformed = aligner.transform(data, params)
            if data.dtype == np.dtype(bool):
                transformed = transformed > 0.5
            # NOTE: transformed dtypes are implicitly cast to the original dtype
            SKM.set_layer_data(adata, layer, transformed)
