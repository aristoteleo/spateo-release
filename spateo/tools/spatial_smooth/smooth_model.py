from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class Discriminator(nn.Module):
    """Module that learns associations between graph embeddings and their positively-labeled augmentations

    Args:
        nf: Dimensionality (along the feature axis) of the input array
    """

    def __init__(self, nf: int):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(nf, nf, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g_repr: FloatTensor, g_pos: FloatTensor, g_neg: FloatTensor):
        """Feeds data forward through network and computes graph representations

        Args:
            g_repr: Representation of source graph, with aggregated neighborhood representations
            g_pos : Representation of augmentation of the source graph that can be considered a positive pairing,
                with aggregated neighborhood representations
            g_neg: Representation of augmentation of the source graph that can be considered a negative pairing,
                with aggregated neighborhood representations

        Returns:
             logits: Similarity score for the positive and negative paired graphs
        """
        c_x = g_repr.expand_as(g_pos)

        sc_1 = self.f_k(g_pos, c_x)
        sc_2 = self.f_k(g_neg, c_x)

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    """
    Aggregates graph embedding information over graph neighborhoods to obtain global representation of the graph
    """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb: FloatTensor, mask: FloatTensor):
        """
        Args:
            emb : float tensor
                Graph embedding
            mask : float tensor
                Selects elements to aggregate for each row
        """
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class Encoder(Module):
    """Representation learning for spatial transcriptomics data

    Args:
        in_features: Number of features in the dataset
        out_features: Size of the desired encoding
        graph_neigh: Pairwise adjacency matrix indicating which spots are neighbors of which other spots
        dropout: Proportion of weights in each layer to set to 0
        act: object of class `torch.nn.functional`, default `F.relu`. Activation function for each encoder layer
        clip: Threshold below which imputed feature values will be set to 0, as a percentile of the max value
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        graph_neigh: FloatTensor,
        dropout: float = 0.0,
        act=F.relu,
        clip: Union[None, float] = None,
    ):
        super(Encoder, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.clip = clip

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat: FloatTensor, feat_a: FloatTensor, adj: FloatTensor):
        """
        Args:
            feat: Counts matrix
            feat_a: Counts matrix following permutation and augmentation
            adj: Pairwise distance matrix
        """
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)

        hidden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        # Clipping constraint:
        if self.clip is not None:
            thresh = torch.quantile(h, self.clip, dim=0)
            mask = h < thresh
            h[mask] = 0
        # Non-negativity constraint:
        nz_mask = h < 0
        h[nz_mask] = 0

        emb = self.act(z)

        # Adversarial learning:
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hidden_emb, h, ret, ret_a
