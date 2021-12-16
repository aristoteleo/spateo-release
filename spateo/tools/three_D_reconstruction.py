import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import math
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import anndata as ad
import ot
import torch
from scipy.spatial import distance_matrix
from scipy.sparse.csr import spmatrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def pairwise_align(slice1, slice2, alpha=0.1, numItermax=200, numItermaxEmd=100000, device='cpu'):
    '''

    Calculates and returns optimal alignment of two slices.

    Parameters
    ----------
    slice1: 'anndata.AnnData'
        An AnnData object.
    slice2: 'anndata.AnnData'
        An AnnData object.
    alpha: 'float' (default: 0.1)
        Trade-off parameter (0 < alpha < 1).
    numItermax: 'int' (default: 200)
        max number of iterations for cg.
    numItermaxEmd: 'int' (default: 100000)
        Max number of iterations for emd.
    device: 'str' or 'torch.device' (default: 'cpu')
        Equipment used to run the program. (torch.device(f'cuda:0'))

    Returns
    -------
    pi: 'np.array'
        alignment of spots.

    '''
    # subset for common genes
    common_genes = [value for value in slice1.var.index if value in set(slice2.var.index)]
    slice1, slice2 = slice1[:, common_genes], slice2[:, common_genes]

    to_dense_array = lambda X: np.array(X.todense()) if isinstance(X, spmatrix) else X
    X, Y = to_dense_array(slice1.X) + 0.01, to_dense_array(slice2.X) + 0.01
    X, Y = X / X.sum(axis=1, keepdims=True), Y / Y.sum(axis=1, keepdims=True)
    logX, logY = np.log(X), np.log(Y)
    X_log_X = np.matrix([np.dot(X[i], logX[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X, logY.T)

    M = torch.tensor(D, device=device, dtype=torch.float32)
    p = torch.tensor(np.ones((slice1.shape[0],)) / slice1.shape[0], device=device, dtype=torch.float32)
    q = torch.tensor(np.ones((slice2.shape[0],)) / slice2.shape[0], device=device, dtype=torch.float32)
    DA = torch.tensor(distance_matrix(slice1.obsm['spatial'], slice1.obsm['spatial']),
                      device=device, dtype=torch.float32)
    DB = torch.tensor(distance_matrix(slice2.obsm['spatial'], slice2.obsm['spatial']),
                      device=device, dtype=torch.float32)

    pi = ot.gromov.fused_gromov_wasserstein(M=M, C1=DA, C2=DB, p=p, q=q, loss_fun='square_loss', alpha=alpha,
                                            armijo=False, log=False, numItermax=numItermax, numItermaxEmd=numItermaxEmd)

    return pi.cpu().numpy()


def slice_alignment(slices=None, alpha=0.1, numItermax=200, numItermaxEmd=100000, device='cpu', verbose=True):
    """

    Align all slice coordinates.

    Parameters
    ----------
    slices: 'list'
        An AnnData list.
    alpha: 'float' (default: 0.1)
        Trade-off parameter (0 < alpha < 1).
    numItermax: 'int' (default: 200)
        max number of iterations for cg.
    numItermaxEmd: 'int' (default: 100000)
        Max number of iterations for emd.
    device: 'str' or 'torch.device' (default: 'cpu')
        Equipment used to run the program.
    verbose: 'bool' (default: True)
        Whether to print information along alignment.

    Returns
    -------
    slicesList: 'list'
        An AnnData list after alignment.

    """

    def _log(m):
        if verbose:
            print(m)

    warnings.filterwarnings('ignore')
    startTime = time.time()

    _log("\n************ Begin of alignment ************\n")
    piList = [
        pairwise_align(slices[i],
                       slices[i + 1],
                       alpha=alpha,
                       numItermax=numItermax,
                       numItermaxEmd=numItermaxEmd,
                       device=device)
        for i in tqdm(range(len(slices) - 1), desc=" Alignment ")
    ]

    slicesList = []
    for i in range(len(slices) - 1):
        slice1 = slices[i].copy()
        slice2 = slices[i + 1].copy()

        raw_slice1_coords, raw_slice2_coords = slice1.obsm['spatial'], slice2.obsm['spatial']
        pi = piList[i]
        slice1_coords = raw_slice1_coords - pi.sum(axis=1).dot(raw_slice1_coords)
        slice2_coords = raw_slice2_coords - pi.sum(axis=0).dot(raw_slice2_coords)
        H = slice2_coords.T.dot(pi.T.dot(slice1_coords))
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)
        slice2_coords = R.dot(slice2_coords.T).T

        slice1.obsm['spatial'] = slice1_coords
        slice2.obsm['spatial'] = slice2_coords
        if i == 0: slicesList.append(slice1)
        slicesList.append(slice2)

    for i, slice in enumerate(slicesList):
        slice.obs["x"] = slice.obsm['spatial'][:, 0]
        slice.obs["y"] = slice.obsm['spatial'][:, 1]

    _log(f'\n************ End of alignment (It takes {round(time.time() - startTime, 2)} seconds) ************\n')

    return slicesList



