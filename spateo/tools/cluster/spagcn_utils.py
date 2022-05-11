import math
import random

import anndata as ad
import lack
import numba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ...logging import logger_manager as lm


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    """(Part of spagcn algorithm) Function to calculate adjacent matrix according to spatial coordinate and image pixels.

    Args:
        x (list): a list which contains corresponding x-coordinates for the spots, spatialy.
        y (list): a list which contains corresponding y-coordinates for the spots, spatialy.
        x_pixel (list, optional): a list which contains corresponding x-pixels for the spots, in histology image. Defaults to None.
        y_pixel (list, optional): a list which contains corresponding y-pixels for the spots, in histology image. Defaults to None.
        image (class: `numpy.ndarray`, optional): the image(typically histology image) in `numpy.ndarray` format(can be obtained by cv2.imread). Defaults to None.
        beta (int, optional): to control the range of neighbourhood when calculate grey value for one spot. Defaults to 49.
        alpha (int, optional): to control the color scale. Defaults to 1.
        histology (bool, optional): if the image is histological. Defaults to True.

    Returns:
        class: `numpy.ndarray`: the calculated adjacent matrix.
    """

    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))
        lm.main_info("Calculateing adj matrix using histology image...")
        beta_half = round(beta / 2)
        g = []
        max_x = image.shape[0]
        max_y = image.shape[1]
        for i in range(len(x_pixel)):
            nbs = image[
                max(0, x_pixel[i] - beta_half) : min(max_x, x_pixel[i] + beta_half + 1),
                max(0, y_pixel[i] - beta_half) : min(max_y, y_pixel[i] + beta_half + 1),
            ]
            g.append(np.mean(np.mean(nbs, axis=0), axis=0))
        c0, c1, c2 = [], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        lm.main_info(f"Var of c0,c1,c2 = {np.var(c0)}, {np.var(c1)}, {np.var(c2)}")
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        lm.main_info(f"Var of x,y,z = {np.var(x)}, {np.var(y)}, {np.var(z)}")
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        lm.main_info("Calculateing adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return adj


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj**2) / (2 * (l**2)))
    return np.mean(np.sum(adj_exp, 1)) - 1


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    """Function to search proper `l` value for spagcn algorithm.

    Args:
        p (float, optional): parameter `p` in spagcn algorithm. See `SpaGCN` for details.
        adj (class: `numpy.ndarray`): the calculated adjacent matrix in spagcn algorithm.
        start (float, optional): lower boundary of search. Defaults to 0.01.
        end (int, optional): upper boundary of search. Defaults to 1000.
        tol (float, optional): step length for search. Defaults to 0.01.
        max_run (int, optional): maximum number of searching iteration. Defaults to 100.

    Returns:
        float: the `l` value
    """
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        lm.main_info("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        lm.main_info("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        lm.main_info(f"recommended l = {str(start)}.")
        return start
    elif np.abs(p_high - p) <= tol:
        lm.main_info(f"recommended l = {str(end)}.")
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        lm.main_info(
            "Run "
            + str(run)
            + ": l ["
            + str(start)
            + ", "
            + str(end)
            + "], p ["
            + str(p_low)
            + ", "
            + str(p_high)
            + "]"
        )
        if run > max_run:
            lm.main_info(
                "Exact l not found, closest values are:\n"
                + "l="
                + str(start)
                + ": "
                + "p="
                + str(p_low)
                + "\nl="
                + str(end)
                + ": "
                + "p="
                + str(p_high)
            )
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            lm.main_info(f"recommended l = {str(mid)}")
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid


def get_cluster_num(
    adata,
    adj,
    res,
    tol,
    lr,
    max_epochs,
    l,
    r_seed=100,
    t_seed=100,
    n_seed=100,
):
    """get the initial number of clusters corresponding to given louvain resolution.

    Args:
        adata, adj, res, tol, lr, max_epochs: further passed to SpaGCN.train(), see `SpaGCN.train`.
        l (float): parameter `l` in spagcn algorithm, see `SpaGCN` for details.
        r_seed, t_seed, n_seed (int, optional): Global seed for `random`, `torch`, `numpy`. Defaults to 100.

    Returns:
        int: number of clusters
    """
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    clf = SpaGCN()
    clf.set_l(l)
    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=tol,
        lr=lr,
        max_epochs=max_epochs,
    )
    y_pred, _ = clf.predict()
    return len(set(y_pred))


def search_res(
    adata,
    adj,
    l,
    target_num,
    start=0.4,
    step=0.1,
    tol=5e-3,
    lr=0.05,
    max_epochs=10,
    r_seed=100,
    t_seed=100,
    n_seed=100,
    max_run=10,
):
    """Function to search a proper initial louvain resolution to get desired number of clusters in spagcn algorithm.

    Args:
        adata (class:`~anndata.AnnData`): an Annadata object.
        adj (class: `numpy.ndarray`): the calculated adjacent matrix in spagcn algorithm.
        l (float): parameter `l` in spagcn algorithm, see `SpaGCN` for details.
        target_num (int): desired number of clusters.
        start (float, optional): the lower boundary of search for resolution. Defaults to 0.4.
        step (float, optional): search step length. Defaults to 0.1.
        tol, lr, max_epochs: further passed to SpaGCN.train(), see `SpaGCN.train`.
        r_seed, t_seed, n_seed (int, optional): Global seed for `random`, `torch`, `numpy`. Defaults to 100.
        max_run (int, optional): max number of iteration. Defaults to 10.

    Returns:
        float: calculated initial louvain resolution.
    """
    res = start
    lm.main_info(f"Start at res = {res} step = {step}")
    old_num = get_cluster_num(adata, adj, res, tol, lr, max_epochs, l, r_seed, t_seed, n_seed)
    lm.main_info(f"Res = {res} Num of clusters = {old_num}")
    run = 0
    while old_num != target_num:
        old_sign = -1 if (old_num < target_num) else 1
        new_num = get_cluster_num(
            adata,
            adj,
            res + step * old_sign,
            tol,
            lr,
            max_epochs,
            l,
            r_seed,
            t_seed,
            n_seed,
        )
        lm.main_info(f"Res = {res + step * old_sign} Num of clusters = {new_num}")
        if new_num == target_num:
            res = res + step * old_sign
            lm.main_info(f"recommended res = {res}")
            return res
        new_sign = -1 if (new_num < target_num) else 1
        if new_sign == old_sign:
            res = res + step * old_sign
            lm.main_info(f"Res changed to res")
            old_num = new_num
        else:
            step = step / 2
            lm.main_info(f"Step changed to {step}")
        if run > max_run:
            lm.main_info("Exact resolution not found")
            lm.main_info(f"Recommended res = {res}")
            return res
        run += 1
    lm.main_info(f"recommended res = {res}")
    return res


def refine(sample_id, pred, dis, shape="square"):
    """To refine(smooth) the boundary of spatial domains(clusters).

    Args:
        sample_id (list): list of sample(cell, spot or bin) names.
        pred (list): list of spatial domains corresponding to the sample_id list.
        dis (class: `numpy.ndarray`): the calculated adjacent matrix in spagcn algorithm.
        shape (str, optional): Smooth the spatial domains with given spatial topology, "hexagon" for Visium data, "square" for ST data. Defaults to "square".

    Returns:
        [list]: list of refined spatial domains corresponding to the sample_id list.
    """
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        lm.main_info("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0 : num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class simple_GC_DEC(nn.Module):
    """
    Simple NN model constructed with a GraphConvolution layer followed by a DeepEmbeddingClustering layer.
    For DEC, see https://arxiv.org/abs/1511.06335v2
    """

    def __init__(self, nfeat, nhid, alpha=0.2):
        super(simple_GC_DEC, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        # self.mu determined by the init method
        self.alpha = alpha

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        # weight = q ** 2 / q.sum(0)
        # return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(
        self,
        X,
        adj,
        lr=0.001,
        max_epochs=5000,
        update_interval=3,
        trajectory_interval=50,
        weight_decay=5e-4,
        opt="sgd",
        init="louvain",
        n_neighbors=10,
        res=0.4,
        n_clusters=10,
        init_spa=True,
        tol=1e-3,
    ):
        self.trajectory = []
        if opt == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features = self.gc(torch.FloatTensor(X), torch.FloatTensor(adj))
        # ----------------------------------------------------------------
        if init == "kmeans":
            lm.main_info("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                # ------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                # ------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  # Here we use X as numpy
        elif init == "louvain":
            lm.main_info(f"Initializing cluster centers with louvain, resolution = {res}")
            if init_spa:
                adata = ad.AnnData(features.detach().numpy())
            else:
                adata = ad.AnnData(X)

            import dynamo as dyn

            dyn.tl.neighbors(adata, n_neighbors=n_neighbors, X_data=adata.X)
            dyn.tl.louvain(adata, resolution=res)
            y_pred = adata.obs["louvain"].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        # ----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = nn.parameter.Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            if epoch % 10 == 0:
                lm.main_info(f"Epoch {epoch}")
            optimizer.zero_grad()
            z, q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                lm.main_info(f"delta_label {delta_label} < tol {tol}")
                lm.main_info("Reach tolerance threshold. Stopping training.")
                lm.main_info(f"Total epoch: {epoch}")
                break

    def predict(self, X, adj):
        z, q = self(torch.FloatTensor(X), torch.FloatTensor(adj))
        return z, q


class simple_GC_DEC_PyG(simple_GC_DEC):
    """
    NN model like simple_GC_DEC, but employed torch_geometric.GCNConv as the GCN layer.
    """

    def __init__(self, nfeat, nhid, alpha=0.2):
        super(simple_GC_DEC_PyG, self).__init__()

        # torch geometric
        try:
            from torch_geometric.nn import GCNConv
        except ModuleNotFoundError:
            # Installing torch geometric packages with specific CUDA+PyTorch version.
            # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details
            ImportError(
                """
            TORCH = torch.__version__.split('+')[0]
            CUDA = 'cu' + torch.version.cuda.replace('.','')

            !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
            !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
            !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
            !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
            !pip install torch-geometric"
            """
            )

            from torch_geometric.nn import GCNConv

        self.gc = GCNConv(nfeat, nhid)
        self.nhid = nhid
        # self.mu determined by the init method
        self.alpha = alpha

    def forward(self, x, edge_index, edge_attr):
        x = self.gc(x, edge_index, edge_attr)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def fit(
        self,
        X,
        adj,
        lr=0.001,
        max_epochs=5000,
        update_interval=3,
        trajectory_interval=50,
        weight_decay=5e-4,
        opt="sgd",
        init="louvain",
        n_neighbors=10,
        res=0.4,
        n_clusters=10,
        init_spa=True,
        tol=1e-3,
    ):
        self.trajectory = []
        if opt == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        adj_mlt = pd.DataFrame(adj).reset_index().melt("index")
        edge_index = torch.tensor([adj_mlt.loc[:, "index"], adj_mlt.loc[:, "variable"]], dtype=torch.long)
        edge_attr = torch.tensor(adj_mlt.loc[:, "value"], dtype=torch.float)

        features = self.gc(torch.FloatTensor(X), edge_index, edge_attr)
        # ----------------------------------------------------------------
        if init == "kmeans":
            lm.main_info("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                # ------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.detach().numpy())
            else:
                # ------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  # Here we use X as numpy
        elif init == "louvain":
            lm.main_info(f"Initializing cluster centers with louvain, resolution = {res}")
            if init_spa:
                adata = ad.AnnData(features.detach().numpy())
            else:
                adata = ad.AnnData(X)

            import dynamo as dyn

            dyn.tl.neighbors(adata, n_neighbors=n_neighbors)
            dyn.tl.louvain(adata, resolution=res)
            y_pred = adata.obs["louvain"].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        # ----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = nn.parameter.Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X = torch.FloatTensor(X)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(X, edge_index, edge_attr)
                p = self.target_distribution(q).data
            if epoch % 10 == 0:
                lm.main_info(f"Epoch {epoch}")
            optimizer.zero_grad()
            z, q = self(X, edge_index, edge_attr)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                lm.main_info(f"delta_label {delta_label} < tol {tol}")
                lm.main_info("Reach tolerance threshold. Stopping training.")
                lm.main_info(f"Total epoch: {epoch}")
                break

    def predict(self, X, adj):
        adj_mlt = pd.DataFrame(adj).reset_index().melt("index")
        edge_index = torch.tensor([adj_mlt.loc[:, "index"], adj_mlt.loc[:, "variable"]], dtype=torch.long)
        edge_attr = torch.tensor(adj_mlt.loc[:, "value"], dtype=torch.float)
        z, q = self(torch.FloatTensor(X), edge_index, edge_attr)
        return z, q


class SpaGCN(object):
    """
    Implementation for spagcn algorithm, see https://doi.org/10.1038/s41592-021-01255-8
    """

    def __init__(self):
        super(SpaGCN, self).__init__()
        self.l = None

    def set_l(self, l):
        self.l = l

    def train(
        self,
        adata,
        adj,
        num_pcs=50,
        lr=0.005,
        max_epochs=2000,
        weight_decay=0,
        opt="adam",
        init_spa=True,
        init="louvain",  # louvain or kmeans
        n_neighbors=10,  # for louvain
        n_clusters=None,  # for kmeans
        res=0.4,  # for louvain
        tol=1e-3,
    ):
        """train model for spagcn

        Args:
            adata (class:`~anndata.AnnData`): an Annadata object.
            adj (class: `numpy.ndarray`): the calculated adjacent matrix in spagcn algorithm.
            num_pcs (int, optional): number of pcs(out dimension of PCA) to use. Defaults to 50.
            lr (float, optional): learning rate in neural network. Defaults to 0.005.
            max_epochs (int, optional): max epochs to train in neural network. Defaults to 2000.
            weight_decay (int, optional): make learning rate decay while training. Defaults to 0.
            opt (str, optional): the optimizer to use. Defaults to "adam".
            init_spa (bool, optional): make initial clusters with louvain or kmeans. Defaults to True.
            init (str, optional): algorithm to use in inital clustering. Supports "louvain", "kmeans". Defaults to "louvain".
        """
        self.num_pcs = num_pcs
        self.res = res
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.opt = opt
        self.init_spa = init_spa
        self.init = init
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.res = res
        self.tol = tol
        assert adata.shape[0] == adj.shape[0] == adj.shape[1]
        pca = PCA(n_components=self.num_pcs)
        if issparse(adata.X):
            pca.fit(adata.X.A)
            embed = pca.transform(adata.X.A)
        else:
            pca.fit(adata.X)
            embed = pca.transform(adata.X)
        ###------------------------------------------###
        if self.l is None:
            raise ValueError("l should be set before fitting the model!")
        adj_exp = np.exp(-1 * (adj**2) / (2 * (self.l**2)))
        # ----------Train model----------
        self.model = simple_GC_DEC(embed.shape[1], embed.shape[1])
        self.model.fit(
            embed,
            adj_exp,
            lr=self.lr,
            max_epochs=self.max_epochs,
            weight_decay=self.weight_decay,
            opt=self.opt,
            init_spa=self.init_spa,
            init=self.init,
            n_neighbors=self.n_neighbors,
            n_clusters=self.n_clusters,
            res=self.res,
            tol=self.tol,
        )
        self.embed = embed
        self.adj_exp = adj_exp

    def predict(self):
        z, q = self.model.predict(self.embed, self.adj_exp)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # Max probability plot
        prob = q.detach().numpy()
        return y_pred, prob
