"""
Modeling putative gene regulatory networks using a neural network.
"""
from typing import List

import anndata
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------------------------------
# Prepare AnnData
# ---------------------------------------------------------------------------------------------------
class AnnDataDataset(Dataset):
    def __init__(self, adata: anndata.AnnData, x_cols: List[str], y_col: str):
        """From AnnData object, prepare gene expression data for neural network modeling.

        Args:
            adata: AnnData object
            x_cols: List of genes to use as predictors
            y_col: Gene to serve as regression target
        """
        self.X = np.array(adata.X[:, x_cols])
        self.y = np.array(adata.X[:, y_col])

    def __getitem__(self, index: int):
        # Get one sample...
        x = self.X[index]
        y = self.y[index]

        # Convert to PyTorch tensor
        x = torch.FloatTensor(x)
        y = torch.FloatTensor([y])

        return x, y


# ---------------------------------------------------------------------------------------------------
# Gene regulatory network inference
# ---------------------------------------------------------------------------------------------------
class GRNM:
    """
    Data-driven construction of gene regulatory networks, informed by prior knowledge networks.

    Args:

    """

    # First search through the base GRN- if none of these reach the chosen threshold, then open search back up to all
    # TFs
