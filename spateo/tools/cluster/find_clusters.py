from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import os

import anndata
import cv2
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix
from scipy.spatial import distance
from tqdm import tqdm

from ...configuration import SKM
from .leiden import calculate_leiden_partition, calculate_louvain_partition
from .spagcn_utils import *
from .utils import spatial_adj

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if isspmatrix(X) else X


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def spagcn_pyg(
    adata: anndata.AnnData,
    n_clusters: int,
    p: float = 0.5,
    s: int = 1,
    b: int = 49,
    refine_shape: Optional[str] = None,
    his_img_path: Optional[str] = None,
    total_umi: Optional[str] = None,
    x_pixel: str = None,
    y_pixel: str = None,
    x_array: str = None,
    y_array: str = None,
    seed: int = 100,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """Function to find clusters with spagcn.

    Reference:
        Jian Hu, Xiangjie Li, Kyle Coleman, Amelia Schroeder, Nan Ma, David J. Irwin, Edward B. Lee,
        Russell T. Shinohara & Mingyao Li. SpaGCN: Integrating gene expression, spatial location and histology to
        identify spatial domains and spatially variable genes by graph convolutional network. Nature Methods volume 18,
        pages1342–1351 (2021)

    Args:
        adata: an Anndata object, after normalization.
        n_clusters: Desired number of clusters.
        p: parameter `p` in spagcn algorithm. See `SpaGCN` for details. Defaults to 0.5.
        s: alpha to control the color scale in calculating adjacent matrix. Defaults to 1.
        b: beta to control the range of neighbourhood when calculate grey value for one spot in calculating adjacent matrix. Defaults to 49.
        refine_shape: Smooth the spatial domains with given spatial topology, "hexagon" for Visium data, "square" for ST data. Defaults to None.
        his_img_path: The file path of histology image used to calculate adjacent matrix in spagcn algorithm. Defaults to None.
        total_umi: By providing the key(colname) in `adata.obs` which contains total UMIs(counts) for each spot, the function use the total counts as
                                a grayscale image when histology image is not provided. Ignored if his_img_path is not `None`. Defaults to "total_umi".
        x_pixel: The key(colname) in `adata.obs` which contains corresponding x-pixels in histology image. Defaults to None.
        y_pixel: The key(colname) in `adata.obs` which contains corresponding y-pixels in histology image. Defaults to None.
        x_array: The key(colname) in `adata.obs` which contains corresponding x-coordinates. Defaults to None.
        y_array: The key(colname) in `adata.obs` which contains corresponding y-coordinates. Defaults to None.
        seed: Global seed for `random`, `torch`, `numpy`. Defaults to 100.
        copy: Whether to return a new deep copy of `adata` instead of updating `adata` object passed in arguments. Defaults to False.

    Returns:
        class:`~anndata.AnnData`: An `~anndata.AnnData` object with cluster info in "spagcn_pred", and in "spagcn_pred_refined" if `refine_shape` is set.
                                The adjacent matrix used in spagcn algorithm is saved in `adata.uns["adj_spagcn"]`.
    """

    if x_array is None:
        x_array = [i[0] for i in adata.obsm["X_spatial"]]
    else:
        x_array = adata.obs[x_array].tolist()

    if y_array is None:
        y_array = [i[1] for i in adata.obsm["X_spatial"]]
    else:
        y_array = adata.obs[y_array].tolist()

    if x_pixel is None:
        x_pixel = [int(i) for i in x_array]
    else:
        x_pixel = adata.obs[x_pixel].tolist()

    if y_pixel is None:
        y_pixel = [int(i) for i in y_array]
    else:
        y_pixel = adata.obs[y_pixel].tolist()

    s = 1
    b = 49

    if his_img_path is None:
        if total_umi is None:
            adj = calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        else:
            total_umi = adata.obs[total_umi].tolist()
            total_umi = [int(x / max(total_umi) * 254 + 1) for x in total_umi]
            total_umi_mtx = pd.DataFrame({"x_pos": x_pixel, "y_pos": y_pixel, "n_umis": total_umi})
            total_umi_mtx = total_umi_mtx.pivot(index="x_pos", columns="y_pos", values="n_umis").fillna(1).to_numpy()
            umi_gs_img = np.dstack((total_umi_mtx, total_umi_mtx, total_umi_mtx)).astype(int)
            adj = calculate_adj_matrix(
                x=x_array,
                y=y_array,
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                image=umi_gs_img,
                beta=b,
                alpha=s,
                histology=True,
            )
    else:
        img = cv2.imread(his_img_path)
        adj = calculate_adj_matrix(
            x=x_array,
            y=y_array,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            image=img,
            beta=b,
            alpha=s,
            histology=True,
        )

    adata.uns["adj_spagcn"] = adj

    l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # Set seed
    r_seed = t_seed = n_seed = seed

    # Seaech for suitable resolution
    res = search_res(
        adata,
        adj,
        l,
        n_clusters,
        start=0.7,
        step=0.1,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=r_seed,
        t_seed=t_seed,
        n_seed=n_seed,
    )

    clf = SpaGCN()
    clf.set_l(l)

    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    # Run
    clf.train(
        adata,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=200,
    )
    y_pred, prob = clf.predict()
    adata.obs["spagcn_pred"] = y_pred
    adata.obs["spagcn_pred"] = adata.obs["spagcn_pred"].astype("category")
    adata.obs["spagcn_pred"] = [str(i) for i in adata.obs["spagcn_pred"]]

    if refine_shape is not None:
        # Do cluster refinement(optional)
        adj_2d = calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        refined_pred = refine(
            sample_id=adata.obs.index.tolist(),
            pred=adata.obs["spagcn_pred"].tolist(),
            dis=adj_2d,
            shape=refine_shape,
        )
        adata.obs["spagcn_pred_refined"] = refined_pred
        adata.obs["spagcn_pred_refined"] = adata.obs["spagcn_pred_refined"].astype("category")

    if copy:
        return adata
    return None


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def scc(
    adata: anndata.AnnData,
    spatial_key: str = "spatial",
    key_added: Optional[str] = "scc",
    pca_key: str = "pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
    resolution: Optional[float] = None,
    cluster_method: str = "louvain",
) -> Optional[anndata.AnnData]:
    """Spatially constrained clustering (scc) to identify continuous tissue domains.

    Reference:
        Ao Chen, Sha Liao, Mengnan Cheng, Kailong Ma, Liang Wu, Yiwei Lai, Xiaojie Qiu, Jin Yang, Wenjiao Li, Jiangshan
        Xu, Shijie Hao, Xin Wang, Huifang Lu, Xi Chen, Xing Liu, Xin Huang, Feng Lin, Zhao Li, Yan Hong, Defeng Fu,
        Yujia Jiang, Jian Peng, Shuai Liu, Mengzhe Shen, Chuanyu Liu, Quanshui Li, Yue Yuan, Huiwen Zheng, Zhifeng Wang,
        H Xiang, L Han, B Qin, P Guo, PM Cánoves, JP Thiery, Q Wu, F Zhao, M Li, H Kuang, J Hui, O Wang, B Wang, M Ni, W
        Zhang, F Mu, Y Yin, H Yang, M Lisby, RJ Cornall, J Mulder, M Uhlen, MA Esteban, Y Li, L Liu, X Xu, J Wang.
        Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. Cell, 2022.

    Args:
        adata: an Anndata object, after normalization.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        key_added: adata.obs key under which to add the cluster labels.
        pca_key: label for the .obsm key containing PCA information (without the potential prefix "X_")
        e_neigh: the number of nearest neighbor in gene expression space.
        s_neigh: the number of nearest neighbor in physical space.
        resolution: the resolution parameter of the leiden clustering algorithm.

    Returns:
        adata: An `~anndata.AnnData` object with cluster info in .obs.
    """

    # Calculate the adjacent matrix.
    adj = spatial_adj(
        adata=adata,
        spatial_key=spatial_key,
        pca_key=pca_key,
        e_neigh=e_neigh,
        s_neigh=s_neigh,
    )

    # Perform Leiden clustering:
    if cluster_method == "louvain":
        clusters = calculate_louvain_partition(
            adj=adj,
            resolution=resolution,
        )
    else:
        clusters = calculate_leiden_partition(
            adj=adj,
            resolution=resolution,
        )

    adata.obs[key_added] = clusters
    adata.obs[key_added] = adata.obs[key_added].astype(str)

    return adata


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def smooth(adata: anndata.AnnData, radius: int = 50, key: str = "label") -> list:
    """
    Optimize the label by majority voting in the neighborhood.

    Args:
        adata: an Anndata object, after normalization.
        radius: the radius of the neighborhood.
        key: the key in `.obs` that corresponds to the cluster labels.
    """

    from ...logging import logger_manager as lm

    logger = lm.get_main_logger()

    logger.info("Optimizing the label by majority voting in the neighborhood.")
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm["spatial"]
    dist_matrix = distance.cdist(position, position, metric="euclidean")

    n_cell = dist_matrix.shape[0]

    from tqdm import tqdm

    for i in tqdm(range(n_cell)):
        vec = dist_matrix[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    adata.obs[key + "_smooth"] = new_type

    logger.info(f"Finish smoothing the label. The new label is stored in adata.obs[{key}+'_smooth']")
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def mclust_py(adata, n_components=None, use_rep: str = "X_pca", modelNames="EEE", random_seed=42):
    """Clustering using Gaussian Mixture Model (GMM), similar to mclust in R.

    Args:
        adata: an Anndata object, after normalization.
        n_components: int, optional, default: None
            The number of mixture components.
        use_rep: str, optional, default: 'X_pca'
            The representation to be used for clustering.
        modelNames: str, optional, default: 'EEE'
            The model name to be used for clustering.
                - EEE: represents Equal volume, shape, and orientation (spherical).
                - VVV: represents Variable volume, shape, and orientation.
                - EEV: represents Equal volume and shape, variable orientation (tied).
                - VVI: represents Variable volume and shape, equal orientation (diag).
        random_seed: int, optional, default: 42
            Random seed for reproducibility.

    """
    from ...logging import logger_manager as lm

    logger = lm.get_main_logger()
    if n_components is None:
        logger.info("You need to input the `n_components` when methods is `GMM`")
        return
    logger.info(f"""running GaussianMixture clustering""")
    # Extract the data to be clustered
    data = adata.obsm[use_rep]

    import numpy as np
    from sklearn.mixture import GaussianMixture

    np.random.seed(random_seed)

    # Extract the data to be clustered
    data = adata.obsm[use_rep]

    # Map modelNames to scikit-learn covariance_type
    covariance_type_map = {
        "EEE": "spherical",  # Equal volume, shape, and orientation (spherical)
        "VVV": "full",  # Variable volume, shape, and orientation
        "EEV": "tied",  # Equal volume and shape, variable orientation (tied)
        "VVI": "diag",  # Variable volume and shape, equal orientation (diag)
        # Add more mappings as needed
    }

    covariance_type = covariance_type_map.get(modelNames, "full")

    # Initialize and fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_seed)
    gmm.fit(data)

    # Get the cluster labels
    mclust_res = gmm.predict(data)
    logger.finish_progress(progress_name="GaussianMixture clustering")

    # Add the cluster labels to adata.obs
    logger.info(f"""Adding the cluster labels to adata.obs['mclust']""")
    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int")
    adata.obs["mclust"] = adata.obs["mclust"].astype("str")
    # adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    adata.obs["gmm_cluster"] = adata.obs["mclust"]

    return adata


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def CAST(
    adata,
    sample_key=None,
    basis="spatial",
    layer="norm_1e4",
    n_components=10,
    output_path="output/CAST_Mark",
    gpu_t=0,
    device="cuda:0",
    **kwargs,
):
    """
    CAST is a Python library for physically aligning different spatial transcriptome regardless of technologies, magnification, individual variation, and experimental batch effects. CAST is composed of three modules: CAST Mark, CAST Stack, and CAST Projection.

    Args:
        adata: an Anndata object, after normalization.
        sample_key: str, optional, default: None
            The key in `.obs` that corresponds to the sample labels.
        basis: str, optional, default: 'spatial'
            The basis used for CAST.
        layer: str, optional, default: 'norm_1e4'
            The layer used for CAST.
        output_path: str, optional, default: 'output/CAST_Mark'
            The path to save the CAST results.
        gpu_t: int, optional, default: 0
            The GPU index to be used.
        device: str, optional, default: 'cuda:0'
            The device to be used.
        kwargs: additional parameters for CAST.
    """
    from ...logging import logger_manager as lm

    logger = lm.get_main_logger()
    if issparse(adata.obsm[basis]):
        adata.obsm[basis] = adata.obsm[basis].toarray()
    adata.obs["x"] = adata.obsm[basis][:, 0]
    adata.obs["y"] = adata.obsm[basis][:, 1]

    logger.info(f"""running CAST""")
    logger.info(f"Get the coordinates and expression data for each sample", indent_level=2)
    # Get the coordinates and expression data for each sample
    samples = np.unique(adata.obs[sample_key])  # used samples in adata
    coords_raw = {sample_t: np.array(adata.obs[["x", "y"]])[adata.obs[sample_key] == sample_t] for sample_t in samples}
    exp_dict = {sample_t: adata[adata.obs[sample_key] == sample_t].layers[layer] for sample_t in samples}

    os.makedirs(output_path, exist_ok=True)
    from ...external.CAST import CAST_MARK

    embed_dict = CAST_MARK(coords_raw, exp_dict, output_path, gpu_t=gpu_t, device=device, **kwargs)
    logger.finish_progress(progress_name="CAST")

    adata.obsm["X_cast"] = np.zeros((adata.shape[0], 512))

    adata.obsm["X_cast"] = pd.DataFrame(adata.obsm["X_cast"], index=adata.obs.index)
    for key in tqdm(embed_dict.keys()):
        adata.obsm["X_cast"].loc[adata.obs[sample_key] == key] += embed_dict[key].cpu().numpy()
    adata.obsm["X_cast"] = adata.obsm["X_cast"].values
    logger.info(f"""CAST embedding is saved in adata.obsm['X_cast']""", indent_level=2)
    # print('CAST embedding is saved in adata.obsm[\'X_cast\']')

    from sklearn.cluster import KMeans, MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=42).fit(adata.obsm["X_cast"])
    adata.obs["CAST_clusters"] = kmeans.labels_
    adata.obs["CAST_clusters"] = adata.obs["CAST_clusters"].astype("str")
    logger.info(f"""CAST clusters using kmeans are saved in adata.obs['CAST_clusters']""", indent_level=2)
    # adata.obs['cast_clusters']=adata.obs['cast_clusters'].astype('category')


def kmeans_clustering(adata, n_clusters=10, use_rep="X_cast", random_state=42, cluster_key="kmeans_clusters"):
    """
    KMeans clustering for spatial transcriptomics data.

    Args:
        adata: an Anndata object, after normalization.
        n_clusters: int, optional, default: 10
            The number of clusters.
        use_rep: str, optional, default: 'X_cast'
            The representation to be used for clustering.
        random_state: int, optional, default: 42
            Random seed for reproducibility.
        cluster_key: str, optional, default: 'kmeans_clusters'
            The key in `.obs` that corresponds to the cluster labels
    """
    from ...logging import logger_manager as lm

    logger = lm.get_main_logger()

    logger.info(f"""running KMeans clustering""")
    logger.info(f"Get the coordinates and expression data", indent_level=2)
    # Get the coordinates and expression data
    data = adata.obsm[use_rep]
    from sklearn.cluster import KMeans, MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
    adata.obs[cluster_key] = kmeans.labels_
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
    logger.info(f"""KMeans clusters are saved in adata.obs['{cluster_key}']""", indent_level=2)
