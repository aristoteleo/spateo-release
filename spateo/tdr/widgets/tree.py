from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from scipy.cluster.vq import kmeans2
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import inv
from tensorflow.keras import optimizers

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
    maxIter: int = 20,
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
        ncenter: number of nodes allowed in the regularization graph

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
    Z = np.real(Z).astype(np.float64)

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


#####################################################################
# Principal curves algorithm                                        #
# ================================================================= #
# Original Code Repository Author: Matthew Artuso.                  #
# Adapted to Spateo by: spateo authors                              #
# Created Date: 6/11/2022                                           #
# Description: A principal curve is a smooth n-dimensional curve    #
# that passes through the middle of a dataset.                      #
# Reference: https://doi.org/10.1016/j.cam.2015.11.041              #
# ================================================================= #
#####################################################################


def orth_dist(y_true, y_pred):
    """
    Loss function for the NLPCA NN. Returns the sum of the orthogonal
    distance from the output tensor to the real tensor.
    """
    loss = tf.math.reduce_sum((y_true - y_pred) ** 2)
    return loss


class NLPCA(object):
    """This is a global solver for principal curves that uses neural networks.
    Attributes:
        None
    """

    def __init__(self):
        self.fit_points = None
        self.model = None
        self.intermediate_layer_model = None

    def fit(self, data: np.ndarray, epochs: int = 500, nodes: int = 25, lr: float = 0.01, verbose: int = 0):
        """
        This method creates a model and will fit it to the given m x n dimensional data.

        Args:
            data: A numpy array of shape (m,n), where m is the number of points and n is the number of dimensions.
            epochs: Number of epochs to train neural network, defaults to 500.
            nodes: Number of nodes for the construction layers. Defaults to 25. The more complex the curve, the higher
                   this number should be.
            lr: Learning rate for backprop. Defaults to .01
            verbose: Verbose = 0 mutes the training text from Keras. Defaults to 0.
        """
        num_dim = data.shape[1]  # get number of dimensions for pts

        # create models, base and intermediate
        model = self.create_model(num_dim, nodes=nodes, lr=lr)
        bname = model.layers[2].name  # bottle-neck layer name

        # The itermediate model gets the output of the bottleneck layer,
        # which acts as the projection layer.
        self.intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(bname).output)

        # Fit the model and set the instances self.model to model
        model.fit(data, data, epochs=epochs, verbose=verbose)
        self.model = model

        return

    def project(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The project function will project the points to the curve generated by the fit function. Given back is the
        projection index of the original data and a sorted version of the original data.

        Args:
            data: m x n array to project to the curve

        Returns:
            proj: A one-dimension array that contains the projection index for each point in data.
            all_sorted: A m x n+1 array that contains data sorted by its projection index, along with the index.
        """
        pts = self.model.predict(data)
        proj = self.intermediate_layer_model.predict(data)

        self.fit_points = pts

        all = np.concatenate([pts, proj], axis=1)
        all_sorted = all[all[:, 2].argsort()]

        return proj, all_sorted

    def create_model(self, num_dim: int, nodes: int, lr: float):
        """
        Creates a tf model.

        Args:
            num_dim: How many dimensions the input space is
            nodes: How many nodes for the construction layers
            lr: Learning rate of backpropigation

        Returns:
            model (object): Keras Model
        """
        # Create layers:
        # Function G
        input = Input(shape=(num_dim,))  # input layer
        mapping = Dense(nodes, activation="sigmoid")(input)  # mapping layer
        bottle = Dense(1, activation="sigmoid")(mapping)  # bottle-neck layer

        # Function H
        demapping = Dense(nodes, activation="sigmoid")(bottle)  # mapping layer
        output = Dense(num_dim)(demapping)  # output layer

        # Connect and compile model:
        model = Model(inputs=input, outputs=output)
        gradient_descent = optimizers.Adam(learning_rate=lr)
        model.compile(loss=orth_dist, optimizer=gradient_descent)

        return model
