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
        dim: Dimensionality (along the feature axis) of the input array (e.g. of the graph embedding)
    """

    def __init__(self, dim: int):
        super(Discriminator, self).__init__()
        self.disc = nn.Bilinear(dim, dim, 1)

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
        g_repr = g_repr.expand_as(g_pos)

        pos_score = self.disc(g_pos, g_repr)
        neg_score = self.disc(g_neg, g_repr)

        logits = torch.cat((pos_score, neg_score), 1)

        return logits


class GlobalGraphReadout(nn.Module):
    """
    Aggregates graph embedding information over graph neighborhoods to obtain global representation of the graph
    """

    def __init__(self):
        super(GlobalGraphReadout, self).__init__()

    def forward(self, emb: FloatTensor, mask: FloatTensor):
        """
        Args:
            emb: Graph embedding
            mask: Adjacency matrix, indicates which elements are neighbors for each sample

        Returns:
            global_emb: Global graph embedding
        """
        neighbor_agg = torch.mm(mask, emb)
        n_neighbors = torch.sum(mask, 1)
        n_neighbors = n_neighbors.expand((neighbor_agg.shape[1], n_neighbors.shape[0])).T
        global_emb = neighbor_agg / n_neighbors
        global_emb = F.normalize(global_emb, p=2, dim=1)

        return global_emb


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
        self.get_readout = GlobalGraphReadout()
        # Activation for the graph representation:
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat: FloatTensor, feat_permuted: FloatTensor, adj: FloatTensor):
        """
        Args:
            feat: Counts matrix
            feat_permuted: Counts matrix following permutation and augmentation
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

        ground_truth_embedding = self.act(z)

        # Adversarial learning:
        z_permuted = F.dropout(feat_permuted, self.dropout, self.training)
        z_permuted = torch.mm(z_permuted, self.weight1)
        z_permuted = torch.mm(adj, z_permuted)
        embedding_permuted = self.act(z_permuted)

        # Graph representations:
        g = self.get_readout(ground_truth_embedding, self.graph_neigh)
        g = self.sigmoid(g)

        g_permuted = self.get_readout(embedding_permuted, self.graph_neigh)
        g_permuted = self.sigmoid(g_permuted)

        # "Normal" graph: positive pair is the embedding for the input graph, negative pair is the permuted
        # representation, and vice versa.
        discriminator_out = self.disc(g, ground_truth_embedding, embedding_permuted)
        discriminator_out_permuted = self.disc(g_permuted, embedding_permuted, ground_truth_embedding)

        return hidden_emb, h, discriminator_out, discriminator_out_permuted
