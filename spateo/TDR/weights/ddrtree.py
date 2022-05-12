from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import inv

#####################################################################
# DDRTree algorithm                                                 #
# ================================================================= #
# Original Code Repository Author: Xiaojie Qiu.                     #
# Adapted to Spateo by: spateo authors                              #
# Created Date: 4/30/2022                                           #
# Description:                                                      #
# Reference:                                                        #
# ================================================================= #
#####################################################################


def cal_ncenter(ncells, ncells_limit=100):
    if ncells >= ncells_limit:
        return np.round(2 * ncells_limit * np.log(ncells) / (np.log(ncells) + np.log(ncells_limit)))
    else:
        return None


def pca_projection(C: np.ndarray, L: int):
    """
    solve the problem size(C) = NxN, size(W) = NxL. max_W trace( W' C W ) : W' W = I.

    Args:
        C: An array like matrix.
        L: The number of Eigenvalues.

    Returns:
        W: The L largest Eigenvalues.
    """

    V, U = eig(C)
    eig_idx = np.argsort(V).tolist()
    eig_idx.reverse()

    W = U.T[eig_idx[0:L]].T
    return W


def repmat(X: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    This function returns an array containing m (n) copies of A in the row (column) dimensions. The size of B is
    size(A)*n when A is a matrix.For example, repmat(np.matrix(1:4), 2, 3) returns a 4-by-6 matrix.

    Args:
        X: An array like matrix.
        m: Number of copies on row dimension
        n: Number of copies on column dimension

    Returns:
        xy_rep: A matrix of repmat.
    """

    a = np.asanyarray(X)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape

    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    xy_rep = c.reshape(rows, cols)

    return xy_rep


def sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    calculate the square distance between a, b.

    Args:
        a: A matrix with :math:`D \times N` dimension
        b: A matrix with :math:`D \times N` dimension

    Returns:
        dist: A numeric value for the different between a and b
    """
    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = a.T.dot(b)

    aa_repmat = repmat(aa[:, None], 1, b.shape[1])
    bb_repmat = repmat(bb[None, :], a.shape[1], 1)
    dist = abs(aa_repmat + bb_repmat - 2 * ab)

    return dist


def DDRTree(
    X: np.ndarray,
    maxIter: int = 10,
    sigma: Union[int, float] = 0.001,
    gamma: Union[int, float] = 10,
    eps: int = 0,
    dim: int = 2,
    Lambda: Union[int, float] = 1.0,
    ncenter: Optional[int] = None,
):
    """
    This function is a pure Python implementation of the DDRTree algorithm.

    Args:
        X: DxN, data matrix list.
        maxIter: Maximum iterations.
        eps: Relative objective difference.
        dim: Reduced dimension.
        Lambda: Regularization parameter for inverse graph embedding.
        sigma: Bandwidth parameter.
        gamma: Regularization parameter for k-means.
        ncenter :(int)

    Returns:
        A tuple of Z, Y, stree, R, W, Q, C, objs
            W is the orthogonal set of d (dimensions) linear basis vector
            Z is the reduced dimension space
            stree is the smooth tree graph embedded in the low dimension space
            Y represents latent points as the center of
    """

    X = np.array(X)
    (D, N) = X.shape

    # initialization
    W = pca_projection(np.dot(X, X.T), dim)
    Z = np.dot(W.T, X)

    if ncenter is None:
        K = N
        Y = Z.T[0:K].T
    else:
        K = ncenter

        Y, _ = kmeans2(Z.T, K)
        Y = Y.T

    # main loop
    objs = []
    for iter in range(maxIter):

        # Kruskal method to find optimal B
        distsqMU = csr_matrix(sqdist(Y, Y)).toarray()
        stree = minimum_spanning_tree(np.tril(distsqMU)).toarray()
        stree = stree + stree.T
        B = stree != 0
        L = np.diag(sum(B.T)) - B

        # compute R using mean-shift update rule
        distZY = sqdist(Z, Y)
        tem_min_dist = np.array(np.min(distZY, 1)).reshape(-1, 1)
        min_dist = repmat(tem_min_dist, 1, K)
        tmp_distZY = distZY - min_dist
        tmp_R = np.exp(-tmp_distZY / sigma)
        R = tmp_R / repmat(np.sum(tmp_R, 1).reshape(-1, 1), 1, K)
        Gamma = np.diag(sum(R))

        # termination condition
        obj1 = -sigma * sum(np.log(np.sum(np.exp(-tmp_distZY / sigma), 1)) - tem_min_dist.T[0] / sigma)
        xwz = np.linalg.norm(X - np.dot(W, Z), 2)
        objs.append((np.dot(xwz, xwz)) + Lambda * np.trace(np.dot(Y, np.dot(L, Y.T))) + gamma * obj1)
        print("iter = ", iter, "obj = ", objs[iter])

        if iter > 0:
            if abs((objs[iter] - objs[iter - 1]) / abs(objs[iter - 1])) < eps:
                break

        # compute low dimension projection matrix
        tmp = np.dot(
            R,
            inv(csr_matrix(((gamma + 1) / gamma) * ((Lambda / gamma) * L + Gamma) - np.dot(R.T, R))).toarray(),
        )
        Q = (1 / (gamma + 1)) * (np.eye(N, N) + np.dot(tmp, R.T))
        C = np.dot(X, Q)

        tmp1 = np.dot(C, X.T)
        W = pca_projection((tmp1 + tmp1.T) / 2, dim)
        Z = np.dot(W.T, C)
        Y = np.dot(np.dot(Z, R), inv(csr_matrix((Lambda / gamma) * L + Gamma)).toarray())

    return Z, Y, stree, R, W, Q, C, objs
