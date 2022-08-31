"""
Denoising and imputation of sparse spatial transcriptomics data


Note that this functionality requires PyTorch >= 1.8.0
Also note that this provides an alternative method for finding spatial domains (not yet fully implemented)
"""
import os
import random
from typing import Union

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from anndata import AnnData
from impute_model import Encoder
from torch import FloatTensor, Tensor, nn
from torch.backends import cudnn
from tqdm import tqdm

from ...configuration import SKM
from ..find_neighbors import construct_pairwise, normalize_adj


# -------------------------------------------- Tensor operations -------------------------------------------- #
def permutation(feature: FloatTensor) -> Tensor:
    """Given counts matrix in tensor form, return counts matrix with scrambled rows/spot names"""
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def get_aug_feature(adata: AnnData, highly_variable: bool = False):
    """
    From AnnData object, get counts matrix, augment it and store both as .obsm entries

    Args:
        adata : class `anndata.AnnData`
            Source AnnData object
        highly_variable : bool, default False
            Set True to subset to highly-variable genes
    """
    if highly_variable:
        adata_Vars = adata[:, adata.var["highly_variable"]]
    else:
        adata_Vars = adata

    if isinstance(adata_Vars.X, scipy.sparse.csc_matrix) or isinstance(adata_Vars.X, scipy.sparse.csr_matrix):
        feat = adata_Vars.X.toarray()[
            :,
        ]
    else:
        feat = adata_Vars.X[
            :,
        ]

    # Data augmentation:
    feat_a = permutation(feat)

    adata.obsm["feat"] = feat
    adata.obsm["feat_a"] = feat_a


def fix_seed(seed: int = 888):
    """Set seeds for all random number generators using 'seed' parameter (defaults to 888)"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def add_contrastive_label(adata):
    """Creates array with 1 and 0 labels for each spot- for contrastive learning"""
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm["label_CSL"] = label_CSL


class STGNN:
    """
    Graph neural network for representation learning of spatial transcriptomics data from only the gene expression
    matrix. Wraps preprocessing and training.

    adata : class `anndata.AnnData`
    spatial_key : str, default 'spatial'
        Key in .obsm where x- and y-coordinates are stored
    random_seed : int, default 50
        Sets seed for all random number generators
    add_regularization : bool, default True
        Set True to include weight-based penalty term in representation learning. This should be set True in
    device : str, default 'cpu'
        Options: 'cpu', 'cuda:_'. Perform computations on CPU or GPU. If GPU, provide the name of the device to run
        computations
    """

    def __init__(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        random_seed: int = 50,
        add_regularization: bool = True,
        device: str = "cpu",
    ):
        self.adata = adata.copy()
        self.random_seed = random_seed
        self.add_regularization = add_regularization
        self.device = torch.device(device)

        fix_seed(self.random_seed)
        construct_pairwise(self.adata, spatial_key=spatial_key)
        add_contrastive_label(self.adata)

        self.adata_output = self.adata.copy()

    def train_STGNN(self, clip: Union[None, float] = None):
        """

        Parameter
        ---------
        clip : optional float
            Threshold below which imputed feature values will be set to 0, as a percentile

        Returns
        -------
        adata_output : class `anndata.AnnData`

        """
        if self.add_regularization:
            # Compute two versions of embedding and store as separate entries in .obsm:
            adata = self.adata_output.copy()
            get_aug_feature(adata)
            model = Trainer(adata, device=self.device)
            emb = model.train()
            # Clipping constraint:
            if clip is not None:
                thresh = np.percentile(emb, clip, axis=0)
                mask = emb < thresh
                emb[mask] = 0
            # Non-negativity constraint:
            nz_mask = emb < 0
            emb[nz_mask] = 0

            # Save reconstruction to layers:
            self.adata_output.layers["X_smooth_gcn"] = emb

            # Reset random seed so that the model follows the same initialization procedures
            fix_seed(self.random_seed)
            adata = self.adata_output.copy()
            get_aug_feature(adata)
            model = Trainer(adata, add_regularization=True, device=self.device)
            emb_regularization = model.train()
            # Clipping constraint:
            if clip is not None:
                thresh = np.percentile(emb_regularization, clip, axis=0)
                mask = emb_regularization < thresh
                emb_regularization[mask] = 0
            # Non-negativity constraint:
            nz_mask = emb < 0
            emb[nz_mask] = 0
            self.adata_output.layers["X_smooth_gcn_reg"] = emb_regularization

        else:
            get_aug_feature(self.adata_output)
            model = Trainer(self.adata_output, device=self.device)
            emb = model.train()
            # Clipping constraint:
            if clip is not None:
                thresh = np.percentile(emb, clip, axis=0)
                mask = emb < thresh
                emb[mask] = 0
            # Non-negativity constraint:
            nz_mask = emb < 0
            emb[nz_mask] = 0
            self.adata_output.layers["X_smooth_gcn"] = emb

        return self.adata_output


class Trainer:
    """
    Graph neural network training module.

    Args:
        adata : class `anndata.AnnData`
        device : torch.device object
        learn_rate : float, default 0.001
            Controls magnitude of gradient for network learning
        weight_decay : float, default 0.0
            Controls degradation rate of parameters
        epochs : int, default 1000
            Number of iterations of training loop to perform
        dim_output : int, default 64
            Dimensionality of the output representation
        alpha : float, default 10
            Controls influence of reconstruction loss in representation learning
        beta : float, default 1
            Weight factor to control the influence of contrastive loss in representation learning
        theta: float, default 0.1
            Weight factor to control the influence of the regularization term in representation learning
        add_regularization : bool, default False
            Adds penalty term to representation learning
    """

    def __init__(
        self,
        adata: AnnData,
        device: "torch.device",
        learn_rate: float = 0.001,
        weight_decay: float = 0.00,
        epochs: int = 1000,
        dim_output: int = 64,
        alpha: float = 10,
        beta: float = 1,
        theta: float = 0.1,
        add_regularization: bool = False,
    ):
        self.adata = adata.copy()
        self.device = device
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.add_regularization = add_regularization

        self.features = torch.FloatTensor(adata.obsm["feat"].copy()).to(self.device)
        self.features_a = torch.FloatTensor(adata.obsm["feat_a"].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(adata.obsm["label_CSL"]).to(self.device)
        self.adj = adata.obsm["adj"]
        self.graph_neigh = torch.FloatTensor(adata.obsm["graph_neigh"].copy() + np.eye(self.adj.shape[0])).to(
            self.device
        )

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

        # Further preprocessing on the adjacency matrix:
        self.adj = normalize_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)

    def train(self):
        """
        Returns
        -------
        emb_rec : np.ndarray
            Reconstruction of the counts matrix
        """
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learn_rate, weight_decay=self.weight_decay)

        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            # Construct augmented graph (negative pair w/ the target graph) and then feed augmented graph and
            # original graph through the model:
            self.features_a = permutation(self.features)
            self.hidden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)

            if self.add_regularization:
                self.loss_norm = 0
                for name, parameters in self.model.named_parameters():
                    if name in ["weight1", "weight2"]:
                        self.loss_norm = self.loss_norm + torch.norm(parameters, p=2)

                loss = (
                    self.alpha * self.loss_feat
                    + self.beta * (self.loss_sl_1 + self.loss_sl_2)
                    + self.theta * self.loss_norm
                )
            else:
                loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            # Return reconstruction:
            self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()

        return self.emb_rec
