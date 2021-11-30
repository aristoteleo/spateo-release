import re
import os
import time
import warnings
import numpy as np
import torch
import pandas as pd
import scanpy as sc
from scipy.spatial import distance_matrix
from scipy.sparse.csr import spmatrix
from scipy.sparse import csr_matrix
from tqdm import tqdm
import ot

def pairwise_align1(slice1, slice2, alpha=0.1, numItermax=200, numItermaxEmd=100000):
    '''
    Calculates and returns optimal alignment of two slices via CPU.

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

    Returns
    -------
        pi: 'np.array'
            alignment of spots.

    '''

    # subset for common genes
    common_genes = [value for value in slice1.var.index if value in set(slice2.var.index)]
    slice1, slice2 = slice1[:, common_genes], slice2[:, common_genes]

    # Calculate spatial distances
    DA = distance_matrix(slice1.obsm['spatial'], slice1.obsm['spatial'])
    DB = distance_matrix(slice2.obsm['spatial'], slice2.obsm['spatial'])

    # Calculate expression dissimilarity
    to_dense_array = lambda X: np.array(X.todense()) if isinstance(X, spmatrix) else X
    AX, BX = to_dense_array(slice1.X), to_dense_array(slice2.X)
    X, Y = AX + 0.01, BX + 0.01
    X ,Y = X / X.sum(axis=1, keepdims=True), Y / Y.sum(axis=1, keepdims=True)
    logX, logY = np.log(X), np.log(Y)
    X_log_X = np.matrix([np.dot(X[i], logX[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X, logY.T)
    M = np.asarray(D)

    # init distributions
    a = np.ones((slice1.shape[0],)) / slice1.shape[0]
    b = np.ones((slice2.shape[0],)) / slice2.shape[0]

    # Run OT
    pi = ot.gromov.fused_gromov_wasserstein(M=M, C1=DA, C2=DB, p=a, q=b, loss_fun='square_loss',alpha=alpha,
                                            armijo=False, log=False,numItermax = numItermax, numItermaxEmd=numItermaxEmd)
    return pi

def pairwise_align2(slice1, slice2, alpha=0.1, numItermax=200, numItermaxEmd=100000, device=0):
    '''
    Calculates and returns optimal alignment of two slices via GPU.

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
        device: 'int'  (default: 0)
            The index of GPU selected.

    Returns
    -------
        pi: 'np.array'
            alignment of spots.

    '''

    # subset for common genes
    common_genes = [value for value in slice1.var.index if value in set(slice2.var.index)]
    slice1, slice2 = slice1[:, common_genes], slice2[:, common_genes]

    # Calculate spatial distances
    DA = distance_matrix(slice1.obsm['spatial'], slice1.obsm['spatial'])
    DB = distance_matrix(slice2.obsm['spatial'], slice2.obsm['spatial'])

    # Calculate expression dissimilarity
    to_dense_array = lambda X: np.array(X.todense()) if isinstance(X, spmatrix) else X
    AX, BX = to_dense_array(slice1.X), to_dense_array(slice2.X)

    X, Y = AX + 0.01, BX + 0.01
    X ,Y = X / X.sum(axis=1, keepdims=True), Y / Y.sum(axis=1, keepdims=True)
    logX, logY = np.log(X), np.log(Y)
    XlogX = np.matrix([np.dot(X[i], logX[i].T) for i in range(X.shape[0])])
    D = XlogX.T - np.dot(X, logY.T)

    # init distributions
    p = np.ones((slice1.shape[0],)) / slice1.shape[0]
    q = np.ones((slice2.shape[0],)) / slice2.shape[0]

    # Run OT via GPU
    cuda = torch.device(f'cuda:{device}')
    constC, hC1, hC2 = ot.gromov.init_matrix(DA, DB, p, q, loss_fun= 'square_loss')
    constC = torch.from_numpy(constC).to(device=cuda)
    hC1, hC2 = torch.from_numpy(hC1).to(device=cuda), torch.from_numpy(hC2).to(device=cuda)
    DA, DB = torch.from_numpy(DA).to(device=cuda), torch.from_numpy(DB).to(device=cuda)
    M = torch.from_numpy(np.asarray(D)).to(device=cuda)
    p, q = torch.from_numpy(p).to(device=cuda), torch.from_numpy(q).to(device=cuda)
    G0 = p[:, None] * q[None, :]

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    torch.cuda.empty_cache()
    pi = ot.optim.cg(p, q, (1 - alpha) * M, alpha, f, df, G0=G0, log=False,numItermax=numItermax,
                     numItermaxEmd=numItermaxEmd, C1=DA, C2=DB, constC=constC, armijo=False)
    torch.cuda.empty_cache()

    return pi.cpu().numpy()

def slice_alignment(slicesList=None, alpha=0.1, numItermax=200, numItermaxEmd=100000,
                    useGpu=False, device=None, save=None, verbose=True):
    '''
    Align all slice coordinates.

    Parameters
    ----------
        slicesList: 'list'
            An AnnData list.
        alpha: 'float' (default: 0.1)
            Trade-off parameter (0 < alpha < 1).
        numItermax: 'int' (default: 200)
            max number of iterations for cg.
        numItermaxEmd: 'int' (default: 100000)
            Max number of iterations for emd.
        useGpu: 'bool' (default: False)
            Whether to use GPU.
        device: 'int'  (default: None)
            The index of GPU selected.
        save: 'str' (default: None)
            Whether to save the data after alignment.
        verbose: 'bool' (default: True)
            Whether to print information along alignment.

    Returns
    -------
        slicesList: 'list'
            An AnnData list after alignment.

    '''

    def _log(m):
        if verbose:
            print(m)

    sc.settings.verbosity = 0
    warnings.filterwarnings('ignore')
    startTime = time.time()

    if not useGpu:
        _log("************ Begin of Registration via CPU ************")
        piList = [
            pairwise_align1(slicesList[i],
                            slicesList[i + 1],
                            alpha=alpha,
                            numItermax=numItermax,
                            numItermaxEmd=numItermaxEmd)
            for i in tqdm(range(len(slicesList) - 1), desc=" Registration")
        ]
    else:
        _log("************ Begin of Registration via GPU ************")
        piList = [
            pairwise_align2(slicesList[i],
                            slicesList[i + 1],
                            alpha=alpha,
                            numItermax=numItermax,
                            numItermaxEmd=numItermaxEmd,
                            device=device)
            for i in tqdm(range(len(slicesList) - 1), desc=" Registration")
        ]

    for i in range(len(slicesList)-1):
        slice1 = slicesList[i].copy()
        slice2 = slicesList[i+1].copy()

        rawSlice1Coor, rawSlice2Coor = slice1.obsm['spatial'], slice2.obsm['spatial']
        pi = piList[i]
        slice1Coor = rawSlice1Coor - pi.sum(axis=1).dot(rawSlice1Coor)
        slice2Coor = rawSlice2Coor - pi.sum(axis=0).dot(rawSlice2Coor)
        H = slice2Coor.T.dot(pi.T.dot(slice1Coor))
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)
        slice2Coor = R.dot(slice2Coor.T).T

        slice1.obsm['spatial'] = pd.DataFrame(slice1Coor, columns=['x', 'y'], index=rawSlice1Coor.index, dtype=float)
        slice2.obsm['spatial'] = pd.DataFrame(slice2Coor, columns=['x', 'y'], index=rawSlice2Coor.index, dtype=float)
        slicesList[i] = slice1
        slicesList[i+1] = slice2

    for i, slice in enumerate(slicesList):
        z = re.findall(r"_S(\d+)", slice.obs['slice_ID'][0])[0]
        slice.obs["new_x"] = slice.obsm['spatial']['x']
        slice.obs["new_y"] = slice.obsm['spatial']['y']
        slice.obs["new_z"] = slice.obsm['spatial']['z'] = float(z)
        slicesList[i] = slice

    if save != None:
        if not os.path.exists(save):
            os.mkdir(save)
        for slice in slicesList:
            subSave = os.path.join(save, f"{slice.obs['slice_ID'][0]}.h5ad")
            slice.write_h5ad(subSave)

    _log(f'************ End of registration (It takes {round(time.time() - startTime, 2)} seconds) ************')

    return slicesList

def slice_alignment_hvg(slicesList=None, n_top_genes=2000, numItermax=200, numItermaxEmd=100000,
                        useGpu=False, device=None, save=None, verbose=True, **kwargs):
    '''
    Align the slices after selecting highly variable genes.

    Parameters
    ----------
        slicesList: 'list'
            An AnnData list.
        n_top_genes: 'int' (default: 2000)
            Number of highly-variable genes to keep.
        numItermax: 'int' (default: 200)
            max number of iterations for cg.
        numItermaxEmd: 'int' (default: 100000)
            Max number of iterations for emd.
        useGpu: 'bool' (default: False)
            Whether to use GPU.
        device: 'int'  (default: None)
            The index of GPU selected.
        save: 'str' (default: None)
            Whether to save the data after alignment.
        verbose: 'bool' (default: True)
            Whether to print information along alignment.
        **kwargs : dict
             Parameters for slice_alignment.
    Returns
    -------
        slicesList: 'list'
            An AnnData list after alignment.
    '''

    def _filter_hvg(adata=None, n_top_genes=2000):
        hvgAdata = adata.copy()
        hvgAdata.raw = hvgAdata
        sc.pp.normalize_total(hvgAdata)
        sc.pp.log1p(hvgAdata)
        sc.pp.highly_variable_genes(hvgAdata, n_top_genes=n_top_genes)
        adata.raw = adata
        return adata[:, hvgAdata.var.highly_variable]

    hvgAdataList = [
        _filter_hvg(adata=adata, n_top_genes=n_top_genes)
        for adata in slicesList
    ]

    regAdataList = slice_alignment(slicesList=hvgAdataList,
                            numItermax=numItermax,
                            numItermaxEmd=numItermaxEmd,
                            useGpu=useGpu,
                            device=device,
                            save=save,
                            verbose=verbose,
                            **kwargs)

    slicesList = []
    for adata in regAdataList:
        newAdata = adata.raw.copy()
        newAdata.obs = adata.obs
        newAdata.obsm = adata.obsm
        slicesList.append(newAdata)

    return slicesList

def slice_alignment_sample(slicesList=None, frac=0.5, numItermax=200, numItermaxEmd=100000,
                           useGpu=False, device=None, save=None, verbose=True, **kwargs):
    '''
    Align the slices after selecting frac*100 percent of genes.

    Parameters
    ----------
        slicesList: 'list'
            An AnnData list.
        frac: 'float' (default: 0.5)
            Fraction of gene items to return.
        numItermax: 'int' (default: 200)
            max number of iterations for cg.
        numItermaxEmd: 'int' (default: 100000)
            Max number of iterations for emd.
        useGpu: 'bool' (default: False)
            Whether to use GPU.
        device: 'int'  (default: None)
            The index of GPU selected.
        save: 'str' (default: None)
            Whether to save the data after alignment.
        verbose: 'bool' (default: True)
            Whether to print information along alignment.
        **kwargs : dict
             Parameters for slice_alignment.

    Returns
    -------
        slicesList: 'list'
            An AnnData list after alignment.
    '''

    def _select_sample(adata=None, frac=0.5):
        adata.raw = adata
        to_dense_array = lambda X: np.array(X.todense()) if isinstance(X, spmatrix) else X
        data = pd.DataFrame(to_dense_array(adata.X), columns=adata.var_names,index=adata.obs_names)
        sample_data = data.sample(frac=frac, axis=1)
        sample_data_m = csr_matrix(sample_data.values)
        newAdata = sc.AnnData(sample_data_m)
        newAdata.var_names = sample_data.columns
        newAdata.obs = adata.obs
        newAdata.obsm = adata.obsm
        newAdata.raw = adata
        return newAdata

    sampleAdataList = [
        _select_sample(adata=adata, frac=frac)
        for adata in slicesList
    ]

    regAdataList = slice_alignment(slicesList=sampleAdataList,
                                   numItermax=numItermax,
                                   numItermaxEmd=numItermaxEmd,
                                   useGpu=useGpu,
                                   device=device,
                                   save=save,
                                   verbose=verbose,
                                   **kwargs)

    slicesList = []
    for adata in regAdataList:
        newAdata = adata.raw.copy()
        newAdata.obs = adata.obs
        newAdata.obsm = adata.obsm
        slicesList.append(newAdata)

    return slicesList

