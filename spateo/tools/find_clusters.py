from .find_clusters_utils import *
import numpy as np
import random, torch
import cv2


def find_cluster_spagcn(
    adata,
    n_clusters,
    p=0.5,
    st_shape="hexagon",
    his_img_path=None,
    x_pixel="x_pixel",
    y_pixel="y_pixel",
    x_array="x_array",
    y_array="y_array",
    adj_save=None
):
    '''Function to find clusters with spagcn.
    Parameters
    ----------
        adata: class:`~anndata.AnnData`
            an Annodata object, after normalization.
        n_clusters: 'int'
            Desired number of clusters.
        p:'float'(defult=0.5)
            parameter `p` in spagcn algorithm. See `SpaGCN` for details.
        st_shape: `str`(default: "hexagon")
            "hexagon" for Visium data, "square" for ST data.
        his_img_path: `str` or None (default: `None`)
            The file path of histology image used to calculate adjacent matrix in spagcn algorithm.
        x_pixel: `str` or None (default: `None`)
            The key for x_pixel in `adata.obs`.
        y_pixel: `str` or None (default: `None`)
            The key for y_pixel in `adata.obs`.
        x_array: `str` or None (default: `None`)
            The key for x_array in `adata.obs`.
        y_array: `str` or None (default: `None`)
            The key for y_array in `adata.obs`.
        adj_save: `str` or None (default: `None`)
            The file path to save adjacent matrix.
    Returns
    -------
        An `~anndata.AnnData` object with cluster info.

    '''

    img=cv2.imread(his_img_path)
    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()
    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()

    s=1
    b=49

    if (his_img_path is None):
        adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
    else:
        adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
    
    if (adj_save is not None):
        np.savetxt(adj_save, adj, delimiter=',')
    
    l=search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    #Set seed
    r_seed=t_seed=n_seed=100
    
    #Seaech for suitable resolution
    res=search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

    clf=SpaGCN()
    clf.set_l(l)

    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs["spagcn_pred"]= y_pred
    adata.obs["spagcn_pred"]=adata.obs["spagcn_pred"].astype('category')

    #Do cluster refinement(optional)
    adj_2d=calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["spagcn_pred"].tolist(), dis=adj_2d, shape=st_shape)
    adata.obs["spagcn_pred_refined"]=refined_pred
    adata.obs["spagcn_pred_refined"]=adata.obs["spagcn_pred_refined"].astype('category')

    return adata
