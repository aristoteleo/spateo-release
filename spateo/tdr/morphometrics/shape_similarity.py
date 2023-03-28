import math
from typing import Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from numpy.linalg import norm
from scipy.linalg import lstsq
from scipy.spatial import cKDTree


def rough_subspace(pcs: np.ndarray, n: int = 20) -> list:
    """Rough subspace segmentation.
    Using the minimal spatial segmentation algorithm to divide points cloud into several subspace
    """
    cuboid_startpoint = np.min(pcs, axis=0)
    cuboid_l, cuboid_w, cuboid_h = np.ptp(pcs, axis=0)
    cuboid_l, cuboid_w, cuboid_h = (
        math.ceil(cuboid_l),
        math.ceil(cuboid_w),
        math.ceil(cuboid_h),
    )

    n_layer_subspace = int(math.pow(n, 2))
    n_cuboid_subspace = int(math.pow(n, 3))
    subspace_l, subspace_w, subspace_h = (
        cuboid_l / n,
        cuboid_w / n,
        cuboid_h / n,
    )

    rough_subspaces = []
    for i in range(n_cuboid_subspace):
        i_layer = math.floor(i / n_layer_subspace)
        i_line = math.floor((i - n_layer_subspace * i_layer) / n)
        i_piece = i - n_layer_subspace * i_layer - n * i_line

        start_l = cuboid_startpoint[0] + i_piece * subspace_l
        start_w = cuboid_startpoint[1] + i_line * subspace_w
        start_h = cuboid_startpoint[2] + i_layer * subspace_h
        end_l, end_w, end_h = start_l + subspace_l, start_w + subspace_w, start_h + subspace_h

        subspace_pc_coords = pcs.copy()[start_l <= pcs[:, 0]]
        subspace_pc_coords = subspace_pc_coords[subspace_pc_coords[:, 0] < end_l]
        subspace_pc_coords = subspace_pc_coords[start_w <= subspace_pc_coords[:, 1]]
        subspace_pc_coords = subspace_pc_coords[subspace_pc_coords[:, 1] < end_w]
        subspace_pc_coords = subspace_pc_coords[start_h <= subspace_pc_coords[:, 2]]
        subspace_pc_coords = subspace_pc_coords[subspace_pc_coords[:, 2] < end_h]
        if subspace_pc_coords.shape[0] > 1:
            subspace_pc_coords = subspace_pc_coords[subspace_pc_coords[:, 1].argsort()]
            rough_subspaces.append(subspace_pc_coords)
    print(f"Amount of rough clusters: {len(rough_subspaces)}.")
    return rough_subspaces


def subspace_surface_fitting(pcs: np.ndarray, order: Literal["linear", "quadratic", "cubic"] = "linear") -> np.ndarray:
    """
    Determines the best fitting plane/surface over a set of 3D points based on ordinary least squares regression.\
    Reference: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    """
    # regular grid covering the domain of the data
    control_pcs = pcs.copy()
    mn = np.min(control_pcs, axis=0)
    mx = np.max(control_pcs, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    n_control_pcs = control_pcs.shape[0]
    control_x, control_y, control_z, control_xy = (
        control_pcs[:, 0],
        control_pcs[:, 1],
        control_pcs[:, 2],
        control_pcs[:, :2],
    )
    if order == "linear":
        # best-fit linear plane
        A = np.c_[control_x, control_y, np.ones(n_control_pcs)]
        C, _, _, _ = lstsq(A, control_z)  # coefficients
        Z = C[0] * X + C[1] * Y + C[2]
    elif order == "quadratic":
        # best-fit quadratic curve
        A = np.c_[np.ones(n_control_pcs), control_xy, np.prod(control_xy, axis=1), control_xy**2]
        C, _, _, _ = lstsq(A, control_z)
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C).reshape(X.shape)
    elif order == "cubic":
        # best-fit cubic surface
        A = np.c_[
            np.ones(n_control_pcs),
            control_pcs[:, :2],
            control_x**2,
            np.prod(control_xy, axis=1),
            control_y**2,
            control_x**3,
            np.prod(np.c_[control_x**2, control_y], axis=1),
            np.prod(np.c_[control_x, control_y**2], axis=1),
            control_y**3,
        ]
        C, _, _, _ = lstsq(A, control_z)
        Z = np.dot(
            np.c_[np.ones(XX.shape), XX, YY, XX**2, XX * YY, YY**2, XX**3, XX**2 * YY, XX * YY**2, YY**3], C
        ).reshape(X.shape)
    else:
        raise ValueError("``order`` value is wrong.")

    surface_pcs = np.asarray([i.flatten() for i in [X, Y, Z]]).T
    return surface_pcs


def dist_global_centroid_to_subspace(
    centroid: Union[tuple, list, np.ndarray], subspace_surface: np.ndarray, **kwargs
) -> np.ndarray:
    """Calculate the average distance from the centroid to the surface of subspace based on KDtree."""
    n_sc = len(np.asarray(subspace_surface))
    surface_kdtree = cKDTree(np.asarray(subspace_surface), **kwargs)
    d, _ = surface_kdtree.query(np.asarray(centroid), k=[i for i in range(1, n_sc + 1)])
    return np.mean(d)


def cos_global_centroid_to_subspace(
    global_centroid: Union[tuple, list, np.ndarray],
    subspace_pcs: np.ndarray,
) -> np.ndarray:
    """Calculate the cosine of the included angle from the centroid to the pcs of subspace."""
    subspace_centroid = np.mean(subspace_pcs, axis=0)
    # v_i = global_centroid - subspace_centroid
    # v_z = global_centroid - np.asarray([global_centroid[0], global_centroid[1], subspace_centroid[2]])
    # cosine = np.dot(v_z, v_i) / np.dot(np.abs(v_z), np.abs(v_i))
    cosine = (subspace_centroid[2] - global_centroid[2]) / norm(subspace_centroid - global_centroid)
    return np.abs(cosine)


def calculate_eigenvector(vetorspaces: np.ndarray, m: int = 10, s: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the subspace eigenvectors."""
    eigenvector, weightvector = [], []
    for i in range(1, m + 1):
        w_i1, w_i2 = (i - 1) / m, i / m
        i_vetorspaces = vetorspaces[vetorspaces[:, 0] >= w_i1, :]
        i_vetorspaces = i_vetorspaces[i_vetorspaces[:, 0] < w_i2, :]
        if i_vetorspaces.shape[0] == 0:
            eigenvector.extend([0] * s)
            weightvector.extend([0] * s)
        else:
            max_dist_i, ptp_dist_i = np.max(i_vetorspaces[:, 1]), np.ptp(i_vetorspaces[:, 1])
            for j in range(1, s + 1):
                w_ptp_i1, w_ptp_i2 = ptp_dist_i * (j - 1) / s, ptp_dist_i * j / s

                j_vetorspaces = i_vetorspaces[i_vetorspaces[:, 1] >= w_ptp_i1, :]
                j_vetorspaces = j_vetorspaces[j_vetorspaces[:, 1] < w_ptp_i2, :]
                if j_vetorspaces.shape[0] == 0:
                    dist_rat_j, segma_i = 0, 0
                else:
                    segma_i = j_vetorspaces.shape[0]
                    dist_ave_j = np.mean(j_vetorspaces[:, 1])
                    dist_rat_j = dist_ave_j / max_dist_i
                eigenvector.append(dist_rat_j)
                weightvector.append(segma_i)
    return np.asarray(eigenvector), np.asarray(weightvector) / np.sum(weightvector)


def model_eigenvector(
    model_pcs: np.ndarray, n_subspace: int = 20, m: int = 10, s: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the subspace eigenvectors of the 3D point cloud model."""
    rough_subspaces = rough_subspace(pcs=np.asarray(model_pcs).copy(), n=n_subspace)
    subspace_vetorspaces = []
    for subspace_pcs in rough_subspaces:
        surface_pcs = subspace_surface_fitting(pcs=subspace_pcs, order="cubic")
        global_centroid = np.mean(np.asarray(model_pcs).copy(), axis=0)
        global_mean_dist = dist_global_centroid_to_subspace(centroid=global_centroid, subspace_surface=surface_pcs)
        cosine = cos_global_centroid_to_subspace(global_centroid=global_centroid, subspace_pcs=subspace_pcs)
        subspace_vetorspaces.append([cosine, global_mean_dist])
    eigenvector, weightvector = calculate_eigenvector(vetorspaces=np.asarray(subspace_vetorspaces), m=m, s=s)
    return eigenvector, weightvector


def pairwise_shape_similarity(
    model1_pcs: np.ndarray, model2_pcs: np.ndarray, n_subspace: int = 20, m: int = 10, s: int = 5
) -> float:
    """
    Calculate the shape similarity of pairwise 3D point cloud models based on the eigenvectors of the 3D point cloud model subspace.
    References: Hu Xiaotong, Wang Jiandong. Similarity analysis of three-dimensional point cloud based on eigenvector of subspace.

    Args:
        model1_pcs: The coordinates of the 3D point cloud model1.
        model2_pcs: The coordinates of the 3D point cloud model2.
        n_subspace: The number of subspaces initially divided is ``n_subspace``**3.
        m: The number of eigenvalues contained in the eigenvector is m*s.
        s: The number of eigenvalues contained in the eigenvector is m*s.

    Returns:
        similarity_score: Shape similarity score.
    """
    e1, w1 = model_eigenvector(model_pcs=model1_pcs, n_subspace=n_subspace, m=m, s=s)
    e2, w2 = model_eigenvector(model_pcs=model2_pcs, n_subspace=n_subspace, m=m, s=s)
    w = np.max(np.c_[w1, w2], axis=1)
    similarity_score = np.sum(w * e1 * e2) / (norm((np.sqrt(w) * e1)) * norm((np.sqrt(w) * e2)))
    return round(similarity_score, 4)
