import numpy as np
from shapely.geometry import MultiPolygon, Polygon

try:
    from typing import Any, List, Literal, Optional, Tuple, Union
except ImportError:
    from typing_extensions import Literal

import itertools

import cv2
from scipy.spatial import KDTree

from ..utils import _iteration

try:
    import alphashape
except ImportError:
    # print("alphashape is not installed. Please install it using 'pip install alphashape'.")
    pass

##################
# Transformation #
##################


def _transform_points(
    points: np.ndarray,
    rotation: Union[np.ndarray, list],
    translation: Union[np.ndarray, list],
    scaling: float,
) -> np.ndarray:
    """
    Transforms the given points by applying rotation, translation, and scaling.

    Args:
        points (np.ndarray): The points to be transformed, with shape (N, 3).
        rotation (Union[np.ndarray, list]): The rotation angles (in degrees) around x, y, and z axes.
        translation (Union[np.ndarray, list]): The translation vector only for z axes.
        scaling (Union[float, np.ndarray]): The scaling factor with single float.

    Returns:
        np.ndarray: The transformed points with shape (N, 3).
    """

    # Center the points around the mean
    mean_points = np.mean(points, axis=0)
    centered_points = points - mean_points

    # Convert rotation angles from degrees to radians
    euler_angles = np.deg2rad(rotation)

    # Rotation matrices for x, y, z axes
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0])],
            [0, np.sin(euler_angles[0]), np.cos(euler_angles[0])],
        ]
    )

    R_y = np.array(
        [
            [np.cos(euler_angles[1]), 0, np.sin(euler_angles[1])],
            [0, 1, 0],
            [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1])],
        ]
    )

    R_z = np.array(
        [
            [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
            [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Apply rotation, scaling, and translation
    transformed_points = scaling * np.dot(centered_points, R.T) + mean_points
    transformed_points[:2] += translation

    return transformed_points


####################
# Extract contours #
####################

# TODO: using rasterization to convert point cloud to image. From STAlign method


def _extract_contour_opencv(
    points: np.ndarray,
    average_n: float = 0.2,
    kernel_size: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extracts contours from a point cloud using OpenCV.

    Args:
        points (np.ndarray): A numpy array of shape (N, 2) representing the point cloud in a slice.
        average_n (float, optional): Average number of points per unit area. Defaults to 0.2.
        kernel_size (Optional[int], optional): Size of the structuring element for morphological operations.
                                               If None, it will be computed based on the image size. Defaults to None.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each containing the contour of the points of slice.
    """

    # Determine the bounding box of the point cloud
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    height, width = max_y - min_y, max_x - min_x
    area = height * width
    scaling = 1 / np.sqrt(points.shape[0] / (average_n * area))
    points = points / scaling
    min_x, max_x = min_x / scaling, max_x / scaling
    min_y, max_y = min_y / scaling, max_y / scaling
    height, width = height / scaling, width / scaling

    min_y_i = int(min_y - 0.1 * height)
    max_y_i = int(max_y + 0.1 * height)
    max_x_i = int(max_x + 0.1 * width)
    min_x_i = int(min_x - 0.1 * width)

    # Create a grayscale image with the point cloud
    image = np.zeros((max_y_i - min_y_i + 1, max_x_i - min_x_i + 1), dtype=np.uint8)
    for point in points:
        x = int(point[0] - min_x_i)
        y = int(point[1] - min_y_i)
        image[y, x] = 255

    # Create a structuring element for the erosion and dilation operations
    # Determine kernel size if not provided
    if kernel_size is None:
        kernel_size = int(image.shape[0] * image.shape[1] / 9e4)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation
    image = cv2.dilate(image, kernel, iterations=3)

    # Perform erosion
    image = cv2.erode(image, kernel, iterations=2)

    # Apply edge detection
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract individual edges from the contours
    edge_points_list = []
    for contour in contours:
        edge_points = (contour.squeeze(axis=1) + np.array([[min_x_i, min_y_i]])) * scaling
        edge_points_list.append(edge_points)
    return edge_points_list


def _extract_contour_alpha_shape(points: np.ndarray, alpha: float = 0.5) -> List[np.ndarray]:
    """
    Extracts contours from a point cloud in a slice using the alpha shape method.

    Args:
        points (np.ndarray): A numpy array of shape (N, 2) representing the point cloud.
        alpha (float, optional): Alpha parameter for the alpha shape algorithm. Defaults to 0.5.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each containing the vertices of a contour.
    """

    # Calculate the alpha shape (concave hull)
    alpha_shape = alphashape.alphashape(points, alpha)
    vertex = []
    alpha_shapes = []

    # Handle single Polygon and MultiPolygon cases
    if isinstance(alpha_shape, Polygon):
        vertex = [np.array(alpha_shape.exterior.coords)]
        alpha_shapes = [alpha_shape]
    elif isinstance(alpha_shape, MultiPolygon):
        polys = alpha_shape.geoms
        for poly in polys:
            # Only consider polygons with more than 20 vertices
            # if np.array(poly.exterior.coords).shape[0] >= 20:
            vertex.append(np.array(poly.exterior.coords))
            alpha_shapes.append(poly)
    return vertex


def _smooth_contours(vertex: List[np.ndarray], window_size: int = 5, iterations: int = 1):
    """
    Smooths the contours using a moving average filter.

    Args:
        vertex (List[np.ndarray]): List of contour vertices to be smoothed.
        window_size (int, optional): Size of the smoothing window. Defaults to 5.
        iterations (int, optional): Number of smoothing iterations. Defaults to 1.

    Returns:
        List[np.ndarray]: List of smoothed contour vertices.
    """

    new_veretx = []

    for v in vertex:
        N, D = v.shape[0], v.shape[1]
        new_v = v.copy()

        for it in range(iterations):
            # Create a padded version of the contour for smoothing
            full_v = np.zeros((N + 2 * window_size, 2))
            full_v[window_size:-window_size, :] = new_v
            full_v[:window_size, :] = new_v[-window_size:, :]
            full_v[-window_size:, :] = new_v[:window_size, :]

            for i in range(N):
                new_v[i] = np.mean(full_v[i : i + 2 * window_size, :], axis=0)

        new_veretx.append(new_v)

    return new_veretx


def _extract_contours_from_mesh(
    mesh,
    z_values,
):
    contours_Z = mesh.contour(isosurfaces=z_values, scalars=mesh.points[:, 2])
    transform_slices = [np.array(contours_Z.points[contours_Z.points[:, 2] == z, :2]) for z in z_values]
    # print(f'z_values: {z_values}')
    # print('s shape: ',[s.shape[0] for s in transform_slices])
    # print(f'mesh_height: {mesh.points[:,2].max()} - {mesh.points[:,2].min()}')
    flag = True
    for s in transform_slices:
        if s.shape[0] == 0:
            flag = False
            break
    return transform_slices, flag


#########################
# Discrete optimization #
#########################

# generates a labeling like [-'maxValue';'maxValue'] with 'numberOfSteps' values. The first position is always 0 (zero)
def _generate_labeling(max_value: float, number_of_steps: int, scale_type: str = "linear") -> List[float]:
    """
    Generates labeling values for a given parameter.

    Args:
        max_value (float): The maximum value for the parameter.
        number_of_steps (int): The number of labels to generate.
        scale_type (str, optional): The scale type ('linear' or 'log'). Defaults to 'linear'.

    Returns:
        List[float]: A list of generated labels.
    """

    if scale_type == "linear":
        step_size = max_value * 2.0 / (number_of_steps - 1)
    elif scale_type == "log":
        step_size = np.log(max_value) * 2.0 / (number_of_steps - 1)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    labels = []
    for n_step in range(number_of_steps):
        if scale_type == "linear":
            current_value = -max_value + n_step * step_size
            if current_value == 0:
                labels.insert(0, current_value)
            else:
                labels.append(current_value)
        elif scale_type == "log":
            current_value = np.exp(-np.log(max_value) + n_step * step_size)
            if current_value == 1:
                labels.insert(0, current_value)
            else:
                labels.append(current_value)

    if number_of_steps % 2 == 0:
        labels[0] = 0 if scale_type == "linear" else 1

    return labels


def _update_parameter(transformation_labels, parameters):
    transformation_labels[:, :3] += parameters["rotation"]
    transformation_labels[:, 3] += parameters["translation"]
    transformation_labels[:, 4] *= parameters["scaling"]
    # transformation_labels[:,:-1] = transformation_labels[:,:-1] + parameters[:-1]
    # transformation_labels[:,-1] = transformation_labels[:,-1] * parameters[-1]
    return transformation_labels


# generates all the unique pair combinations among the variables
def _make_pairs(nVars=5):
    listvars = np.arange(nVars)
    pairs = list(itertools.combinations(listvars, 2))
    return np.array(pairs, dtype=np.int32)


def _getUnaries(L, N=5):
    return np.ones((L, N), dtype=np.float32)


def _get_parameters_from_pair(pair, transformation_labels):
    parameter_index_1 = transformation_labels[:, pair[0]]
    parameter_index_2 = transformation_labels[:, pair[1]]
    default_parameters = transformation_labels[0, :]
    parameters_set = []
    for parameter_1 in parameter_index_1:
        parameters_1_set = []
        for parameter_2 in parameter_index_2:
            cur_parameters = default_parameters.copy()
            cur_parameters[pair[0]] = parameter_1
            cur_parameters[pair[1]] = parameter_2
            parameters_1_set.append(cur_parameters)
        parameters_set.append(parameters_1_set)
    return np.array(parameters_set)


def _getBinary(
    contours,
    mesh,
    z_values,
    pairs,
    transformation_labels,
    verbose: bool = False,
):
    # print(transformation_labels)
    binaries = []
    # print("evaluate {} pairs".format(len(pairs)))
    progress_name = "Evaluate binary term."
    for npair in _iteration(len(pairs), progress_name=progress_name, verbose=verbose, indent_level=2):
        # for npair in range(len(pairs)):
        # print("Current pair: {}/{}".format(npair, len(pairs)))
        pair = pairs[npair]
        b = _get_binary_values(
            contours,
            mesh,
            z_values,
            pair,
            transformation_labels,
        )
        binaries.append(b)
    return np.array(binaries)


def _get_binary_values(
    contours,
    mesh,
    z_values,
    pair,
    transformation_labels,
):
    L = transformation_labels.shape[0]
    parameter_sets = _get_parameters_from_pair(pair, transformation_labels)
    binary_values = []
    for parameter_set_1 in parameter_sets:
        for parameter_set_2 in parameter_set_1:
            # print(parameter_set_2)
            label_cost = _calculate_loss(contours, mesh, parameter_set_2, z_values, "ICP")
            binary_values.append(label_cost)
    binary_values = np.array(binary_values, dtype=np.float32)
    binary_values = np.reshape(binary_values, (L, L))
    return binary_values


# calculate the loss based on contour similarity
def _calculate_loss(
    contours,
    mesh,
    transformation,
    z_values,
    method: Literal["CPD", "ICP"] = "ICP",
):
    assert len(contours) == len(z_values)
    # print('transformation: ', transformation)
    transformed_mesh = mesh.copy()
    transformed_mesh.points = _transform_points(
        np.array(transformed_mesh.points), transformation[:3], transformation[3], transformation[4]
    )
    mesh_contours, cut_flag = _extract_contours_from_mesh(transformed_mesh, z_values)
    # print(cut_flag)

    if cut_flag == True:
        label_cost = 0
        for c, m_c in zip(contours, mesh_contours):
            if method == "CPD":
                raise NotImplementedError
            elif method == "ICP":
                # TODO: precalculate the KD-trees
                # print(c.shape, m_c.shape)
                gamma, _, _, _, _, _ = ICP(c, m_c, allow_rotation=True)
            label_cost += 1 - gamma
    else:
        label_cost = 1e6
    label_cost = label_cost / len(contours)

    return label_cost


def _eliminate_shift(
    contours,
    mesh,
    z_values,
    allow_rotation: bool = True,
):
    assert len(contours) == len(z_values)

    contours_Z = mesh.contour(isosurfaces=z_values, scalars=mesh.points[:, 2])
    tol = np.abs(z_values[:-1] - z_values[1:]).min()
    transform_contours = []
    for z in z_values:
        idx = (contours_Z.points[:, 2] > (z - 0.1 * tol)) & (contours_Z.points[:, 2] < (z + 0.1 * tol))
        transform_contours.append(np.array(contours_Z.points[idx, :]))
    flag = True
    for s in transform_contours:
        if s.shape[0] == 0:
            flag = False
            break
    ts = []
    Rs = []
    if flag == True:
        for i in range(len(contours)):
            gamma1, _, t, _, _, R = ICP(
                contours[i], transform_contours[i][:, :2], max_iter=100, allow_rotation=allow_rotation
            )
            gamma2, _, t2, _, _, R2 = ICP(
                transform_contours[i][:, :2], contours[i], max_iter=100, allow_rotation=allow_rotation
            )
            if i == 0:
                print(gamma1, gamma2)
                print(t, t2)
                print(R, R2)
            if gamma2 > gamma1:
                t = -t2 @ R2
                R = R2.T
            # contours[i] = (contours[i] - t) @ R.T
            ts.append(t)
            Rs.append(R)

    return ts, Rs


def ICP(
    contour_1: np.ndarray,
    contour_2: np.ndarray,
    max_iter: int = 20,
    error_threshold: float = 1e-6,
    inlier_threshold: float = 0.1,
    subsample: int = 500,
    allow_rotation: bool = False,
) -> Tuple[float, float, Union[float, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative Closest Point (ICP) algorithm for aligning two sets of points.

    Args:
        contour_1 (np.ndarray): Data points.
        contour_2 (np.ndarray): Model points.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.
        error_threshold (float, optional): Error threshold for convergence. Defaults to 1e-6.
        inlier_threshold (float, optional): Inlier threshold distance. Defaults to 0.1.
        subsample (int, optional): Number of points to subsample. Defaults to 500.
        allow_rotation (bool, optional): Whether to allow estimate rotation. Defaults to False.

    Returns:
        Tuple[float, float, Union[float, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        Convergence ratio, sigma2 (placeholder), translation vector, original contour, transformed contour, rotation matrix.
    """

    # contour_1 is data point and contour_2 is model point
    if contour_1.shape[0] > subsample and subsample > 0:
        contour_1 = contour_1[np.random.choice(contour_1.shape[0], subsample, replace=False), :]
    if contour_2.shape[0] > subsample and subsample > 0:
        contour_2 = contour_2[np.random.choice(contour_2.shape[0], subsample, replace=False), :]
    assert contour_1.shape[1] == contour_2.shape[1]
    N, M, D = contour_1.shape[0], contour_2.shape[0], contour_1.shape[1]

    # normalize
    mean_contour_1 = (np.max(contour_1, axis=0) + np.min(contour_1, axis=0)) / 2
    mean_contour_2 = (np.max(contour_2, axis=0) + np.min(contour_2, axis=0)) / 2
    contour_1_demean = contour_1 - mean_contour_1
    contour_2_demean = contour_2 - mean_contour_2

    scale = (np.sqrt(np.sum(contour_1_demean**2) / N) + np.sqrt(np.sum(contour_2_demean**2) / M)) / 2
    contour_1_demean, contour_2_demean = contour_1_demean / scale, contour_2_demean / scale

    # initialize
    T_contour_2 = contour_2_demean
    prev_error = np.inf
    tree = KDTree(contour_1_demean)

    for iter in range(max_iter):
        distances, indices = tree.query(T_contour_2)

        inliers = distances < inlier_threshold

        if np.sum(inliers) < 3:
            translation = 0
            Rotation = np.eye(D)
            break
        else:

            T_contour_2_corr = T_contour_2[inliers.flatten()]
            contour_1_corr = contour_1_demean[indices[inliers.flatten()]]

            T_contour_2_corr_mean = np.mean(T_contour_2_corr, axis=0)
            contour_1_corr_mean = np.mean(contour_1_corr, axis=0)

            T_contour_2_corr_demean = T_contour_2_corr - T_contour_2_corr_mean
            contour_1_corr_demean = contour_1_corr - contour_1_corr_mean

            if not allow_rotation:
                translation = contour_1_corr_mean - T_contour_2_corr_mean
                Rotation = np.eye(D)
            else:
                covariance_matrix = np.dot(T_contour_2_corr_demean.T, contour_1_corr_demean)
                U, _, Vt = np.linalg.svd(covariance_matrix)
                Rotation = np.dot(Vt.T, U.T)
                translation = contour_1_corr_mean - np.dot(Rotation, T_contour_2_corr_mean)

        T_contour_2 = np.dot(Rotation, T_contour_2.T).T + translation

        error = np.mean(distances[inliers])

        if np.abs(prev_error - error) < error_threshold:
            break

        prev_error = error

    gamma = np.sum(distances < 0.05) / M  # adopt a stricter threshold for gamma

    T_contour_2 = scale * T_contour_2 + mean_contour_1
    if allow_rotation == False:
        translation = scale * translation + mean_contour_1 - mean_contour_2
    else:
        Rotation, translation = solve_RT_by_correspondence(T_contour_2, contour_2)

    return gamma, 0, translation, contour_1, T_contour_2, Rotation


def solve_RT_by_correspondence(
    X: np.ndarray, Y: np.ndarray, return_scale: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
    """
    Solve for the rotation matrix R and translation vector t that best align the points in X to the points in Y.

    Args:
        X (np.ndarray): Source points, shape (N, D).
        Y (np.ndarray): Target points, shape (N, D).
        return_scale (bool, optional): Whether to return the scale factor. Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
        If return_scale is False, returns the rotation matrix R and translation vector t.
        If return_scale is True, also returns the scale factor s.
    """

    D = X.shape[1]
    N = X.shape[0]

    # Calculate centroids of X and Y
    tX = np.mean(X, axis=0)
    tY = np.mean(Y, axis=0)

    # Demean the points
    X_demean = X - tX
    Y_demean = Y - tY

    # Compute the covariance matrix
    H = np.dot(Y_demean.T, X_demean)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure the rotation matrix is proper
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    # t = tY - np.dot(tX, R.T)
    t = tX - np.dot(tY, R.T)

    if return_scale:
        # Compute the scale factor
        s = np.trace(np.dot(X_demean.T, X_demean) - np.dot(R.T, np.dot(Y_demean.T, X_demean))) / np.trace(
            np.dot(Y_demean.T, Y_demean)
        )
        return R, t, s
    else:
        return R, t


# def solve_RT_by_correspondence(
#     X,
#     Y,
#     return_s = False,
# ):
#     # if len(X.shape) == 3:
#     #     X = X[:, :2]
#     # if len(Y.shape) == 3:
#     #     Y = Y[:, :2]
#     D = X.shape[1]
#     N = X.shape[0]
#     # find R and t that minimize the distance between spatial1 and spatial2

#     tX = np.mean(X, axis=0)
#     tY = np.mean(Y, axis=0)
#     X = X - tX
#     Y = Y - tY
#     H = np.dot(Y.T, X)
#     U, S, Vt = np.linalg.svd(H)
#     R = Vt.T.dot(U.T)
#     t = np.mean(X, axis=0) - np.mean(Y, axis=0) + tX - np.dot(tY, R.T)
#     s = np.trace(np.dot(X.T, X) - np.dot(R.T, np.dot(Y.T, X))) / np.trace(np.dot(Y.T, Y))
#     if return_s:
#         return R, t, s
#     else:
#         return R, t
