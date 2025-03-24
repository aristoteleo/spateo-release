import random

import numpy as np
from anndata import AnnData  # type: ignore

try:
    from typing import Any, Dict, List, Literal, Optional, Tuple, Union
except ImportError:
    from typing_extensions import Literal

from typing import List, Optional, Tuple, Union

from pyvista import PolyData  # type: ignore

from spateo.logging import logger_manager as lm

from ..utils import _iteration
from .mesh_correction_utils import (
    _calculate_loss,
    _extract_contour_alpha_shape,
    _extract_contour_opencv,
    _generate_labeling,
    _getBinary,
    _getUnaries,
    _make_pairs,
    _smooth_contours,
    _transform_points,
    _update_parameter,
)

try:
    from .libfastpd import fastpd
except ImportError:
    # print("fastpd is not installed. If you need mesh correction, please compile the fastpd library.")
    pass


# TODO: add str as the input type for the models
class Mesh_correction:
    """
    A class to perform 3D reconstruction correction from slices using a mesh.

    Attributes:
        slices (List[AnnData]): A list of aligned slices by Spateo or other methods.
        z_heights (Union[List, np.ndarray]): The z-coordinates for each slice.
        mesh (PolyData): The mesh used for correction.
        spatial_key (str): The key to access spatial coordinates in the AnnData objects. The spatial coordinates under this key should be pre-aligned.
        key_added (str): The key under which the corrected spatial coordinates will be added.
        normalize_spatial (bool): Flag indicating if spatial coordinates will be normalized.
        init_rotation (np.ndarray): Initial rotation angles (in degrees) for the correction.
        init_translation (np.ndarray): Initial translation vector for the correction.
        init_scaling (np.ndarray): Initial scaling factor for the correction.
        max_rotation_angle (float): Maximum rotation angle allowed during the correction process.
        max_translation_scale (float): Maximum translation scale allowed during the correction process.
        max_scaling (float): Maximum scaling factor allowed during the correction process.
        label_num (int): Number of labels used for optimization.
        fastpd_iter (int): Number of iterations for the fastPD algorithm.
        max_iter (int): Maximum number of iterations for the correction process.
        anneal_rate (float): Annealing rate for the optimization process.
        multi_processing (bool): Flag indicating if multiprocessing will be used for the correction process.
    """

    def __init__(
        self,
        slices: List[AnnData],
        z_heights: Union[List, np.ndarray],
        mesh: PolyData,
        spatial_key: str = "spatial",
        key_added: str = "align_spatial",
        normalize_spatial: bool = False,
        init_rotation: Optional[np.ndarray] = np.array([0.0, 0.0, 0.0]),
        init_translation: Optional[np.ndarray] = 0.0,
        init_scaling: Optional[np.ndarray] = 1.0,
        max_rotation_angle: float = 180,
        max_translation_scale: float = 0.5,
        max_scaling: float = 1.5,
        min_rotation_angle: float = 10,
        min_translation_scale: float = 1,
        min_scaling: float = 1.1,
        label_num: int = 15,
        fastpd_iter: int = 100,
        max_iter: int = 10,
        anneal_rate: float = 0.7,
        multi_processing: bool = False,
        subsample_slices: Optional[int] = None,
        verbose: bool = False,
    ) -> None:

        self.n_slices = len(slices)

        # check if all slices have the same spatial key in the ".obsm" attribute
        if not all([spatial_key in s.obsm.keys() for s in slices]):
            raise ValueError("All slices must have the same spatial key in the '.obsm' attribute.")
        self.slices_spatial = [s.obsm[spatial_key] for s in slices]

        # check z_heights are unique and convert to numpy array
        if z_heights is None:
            raise ValueError("z_heights must be provided.")
        self.z_heights = z_heights if isinstance(z_heights, np.ndarray) else np.array(z_heights)
        # z_height must be unique
        if len(np.unique(self.z_heights)) != len(self.z_heights):
            raise ValueError("z_heights must be unique value.")
        # z_height must have the same length as the number of slices
        if len(self.z_heights) != self.n_slices:
            raise ValueError("z_heights must have the same length as the number of slices.")

        self.mesh = mesh.copy()
        self.key_added = key_added
        self.normalize_spatial = normalize_spatial
        self.set_init_parameters(init_rotation, init_translation, init_scaling)

        # scale the mesh to the same scale as the slices
        self.normalize_mesh_spatial_coordinates()

        self.max_rotation_angle = max_rotation_angle
        self.max_translation_scale = max_translation_scale
        self.max_scaling = max_scaling
        self.min_rotation_angle = min_rotation_angle
        self.min_translation_scale = min_translation_scale
        self.min_scaling = min_scaling
        self.label_num = label_num
        self.fastpd_iter = fastpd_iter
        self.max_iter = max_iter
        self.anneal_rate = anneal_rate
        self.multi_processing = multi_processing
        self.subsample_slices = subsample_slices
        self.verbose = verbose
        self.contours = [None] * self.n_slices

    def set_init_parameters(
        self,
        init_rotation: Optional[np.ndarray] = np.array([0.0, 0.0, 0.0]),
        init_translation: Optional[np.ndarray] = 0.0,
        init_scaling: Optional[np.ndarray] = 1.0,
    ):
        """
        Sets the initial transformation parameters for the mesh.

        Args:
            init_rotation (Optional[np.ndarray], optional): Initial rotation angles (in degrees) for the mesh. Defaults to np.array([0., 0., 0.]).
            init_translation (Optional[np.ndarray], optional): Initial translation vector for the mesh. Defaults to 0.
            init_scaling (Optional[np.ndarray], optional): Initial scaling factor for the mesh. Defaults to 1.
        """

        self.mesh.points = _transform_points(np.array(self.mesh.points), init_rotation, init_translation, init_scaling)

    def normalize_mesh_spatial_coordinates(
        self,
    ):
        """
        Normalizes the spatial coordinates of the mesh to align with the slices' spatial scale.

        This method calculates the scaling factor based on the maximum spatial extent of the slices
        and the z-height range, then applies this scaling to the mesh points. It also centers the
        mesh points along the z-axis to match the mean z-height of the slices.
        """

        # Calculate the scaling factor based on the slices' spatial extent and z-height range
        # self.slices_scale = np.max(
        #     [np.linalg.norm(np.max(spatial, axis=0) - np.min(spatial, axis=0)) for spatial in self.slices_spatial]
        #       + [self.z_heights.max() - self.z_heights.min()]

        # )
        self.slices_scale = self.z_heights.max() - self.z_heights.min()

        if self.normalize_spatial:
            # Calculate the mesh scaling factor
            # mesh_scale = np.max(np.max(self.mesh.points, axis=0) - np.min(self.mesh.points, axis=0))
            mesh_scale = self.mesh.points[:, 2].max() - self.mesh.points[:, 2].min()

            # Calculate the mean of the z-heights of the slices
            slices_mean_z = (self.z_heights.max() + self.z_heights.min()) / 2
            slices_mean_xy = np.concatenate(self.slices_spatial, axis=0)
            slices_mean_xy = (slices_mean_xy.max(axis=0) + slices_mean_xy.min(axis=0)) / 2

            # Center the mesh points and apply the scaling factor
            # mesh_mean = np.mean(self.mesh.points, axis=0)
            mesh_mean = (self.mesh.points.max(axis=0) + self.mesh.points.min(axis=0)) / 2
            self.mesh.points = (self.mesh.points - mesh_mean) * self.slices_scale / mesh_scale

            # Adjust the z-coordinates to match the mean z-height of the slices
            self.mesh.points[:, :2] += slices_mean_xy
            self.mesh.points[:, 2] += slices_mean_z

    def extract_contours(
        self,
        method: Literal["opencv", "alpha_shape"] = "alpha_shape",
        n_sampling: Optional[int] = None,
        smoothing: bool = True,
        window_size: int = 5,
        filter_contours: bool = True,
        contour_filter_threshold: int = 20,
        opencv_kwargs: Optional[Dict] = None,
        alpha_shape_kwargs: Optional[Dict] = None,
    ):
        """
        Extracts contours of slices using the specified method.

        Args:
            method (Literal["opencv", "alpha_shape"], optional): Method to extract contours. Defaults to "alpha_shape".
            n_sampling (Optional[int], optional): Number of points to sample from each slice. Defaults to None.
            smoothing (bool, optional): Whether to smooth the contours. Defaults to True.
            window_size (int, optional): Window size for contour smoothing. Defaults to 5.
            filter_contours (bool, optional): Whether to filter the contours based on the threshold. Defaults to True.
            contour_filter_threshold (int, optional): Threshold for filtering contours based on the number of points. Defaults to 20.
            opencv_kwargs (Optional[Dict], optional): Additional arguments for the OpenCV method. Defaults to None.
            alpha_shape_kwargs (Optional[Dict], optional): Additional arguments for the alpha shape method. Defaults to None.
        """

        if opencv_kwargs is None:
            opencv_kwargs = {}
        if alpha_shape_kwargs is None:
            alpha_shape_kwargs = {}

        progress_name = f"Extract contours of slices, method: {method}."
        for model_index in _iteration(n=self.n_slices, progress_name=progress_name, verbose=self.verbose):
            points = self.slices_spatial[model_index].copy()
            sampling_idx = (
                np.random.choice(points.shape[0], n_sampling, replace=False)
                if (n_sampling is not None) and (n_sampling > 0) and (n_sampling < points.shape[0])
                else np.arange(points.shape[0])
            )
            points = points[sampling_idx]
            if method == "opencv":
                cur_contours = _extract_contour_opencv(points, **opencv_kwargs)
            elif method == "alpha_shape":
                cur_contours = _extract_contour_alpha_shape(points=points, **alpha_shape_kwargs)
            else:
                raise NotImplementedError(f"Method {method} is not implemented.")

            # filter the contours
            if filter_contours:
                cur_contours = [c for c in cur_contours if c.shape[0] >= contour_filter_threshold]

            # smoothing the contours
            if smoothing:
                cur_contours = _smooth_contours(cur_contours, window_size)

            self.contours[model_index] = np.concatenate(cur_contours, axis=0) if cur_contours else np.array([])

    def run_discrete_optimization(
        self,
    ) -> None:
        """
        Runs the discrete optimization process to find the best transformation parameters.
        """

        self.max_translation = self.max_translation_scale * self.slices_scale

        # subsample the slices for the discrete optimization
        if self.subsample_slices is not None and self.subsample_slices < self.n_slices and self.subsample_slices > 0:
            self.subsample_slices = np.random.choice(self.n_slices, self.subsample_slices, replace=False)
            self.contours_subsample = [self.contours[i] for i in self.subsample_slices]
            self.z_heights_subsample = self.z_heights[self.subsample_slices]

        else:
            self.contours_subsample = self.contours
            self.z_heights_subsample = self.z_heights

        # run the discrete optimization
        self.losses = []
        self.transformations = []
        self.best_loss = 1e8
        self.best_transformation = {"rotation": np.array([0.0, 0.0, 0.0]), "translation": 0.0, "scaling": 1.0}
        self.losses.append(self.best_loss)
        self.transformations.append(self.best_transformation)
        progress_name = "Run discrete optimization."
        lm.main_info(message=f"Run discrete optimization on {len(self.contours_subsample)} contours", indent_level=1)
        for i in _iteration(n=self.max_iter, progress_name=progress_name, verbose=self.verbose, indent_level=1):
            cur_loss, cur_transformation = self.discrete_optimization_step()

            if self.verbose:
                lm.main_info(message=f"Iteration {i+1}/{self.max_iter}, current loss: {cur_loss}", indent_level=2)

            if cur_loss < self.best_loss:
                self.best_loss = cur_loss
                self.best_transformation = cur_transformation
            self.losses.append(cur_loss)
            self.transformations.append(cur_transformation)

            # anneling
            self.max_rotation_angle = max(self.max_rotation_angle * self.anneal_rate, self.min_rotation_angle)
            self.max_translation = max(
                self.max_translation * self.anneal_rate, self.min_translation_scale * self.slices_scale
            )
            self.max_scaling = max(self.max_scaling * self.anneal_rate, self.min_scaling)

        # finish the optimization
        lm.main_info(message=f"Optimization finished. Best loss: {self.best_loss}", indent_level=1)

    def discrete_optimization_step(
        self,
    ) -> Tuple[float, np.ndarray]:
        """
        Performs a discrete optimization step to find the best transformation.

        Returns:
            Tuple[float, np.ndarray]: The loss and the best transformation found.
        """

        # generate the transformation labels
        transformation_labels = self.generate_labels()

        # initialize the evaluation pairs
        pairs = _make_pairs()
        u = _getUnaries(self.label_num)

        barray = _getBinary(
            self.contours_subsample,
            self.mesh,
            self.z_heights_subsample,
            pairs,
            transformation_labels,
            verbose=self.verbose,
        )
        blist = [b for b in barray]
        # print(blist)
        # run the fastpd algorithm
        labels = fastpd(u, blist, pairs, self.fastpd_iter)

        # get the best transformation
        parameters = np.array([transformation_labels[labels[i], i] for i in range(len(labels))])

        # evaluate the loss
        loss = _calculate_loss(self.contours_subsample, self.mesh, parameters, self.z_heights_subsample, "ICP")

        transformation = {"rotation": parameters[:3], "translation": parameters[3], "scaling": parameters[4]}
        return loss, transformation

    def generate_labels(
        self,
    ) -> np.ndarray:
        """
        Generates discrete labels for rotation, translation, and scaling.

        Returns:
            np.ndarray: An array of discrete transformation labels.
        """
        rotation_labels = _generate_labeling(self.max_rotation_angle, self.label_num)
        translation_labels = _generate_labeling(self.max_translation, self.label_num)
        scaling_labels = _generate_labeling(self.max_scaling, self.label_num, "log")

        transformation_labels = np.array(
            [rotation_labels, rotation_labels, rotation_labels, translation_labels, scaling_labels]
        ).T
        transformation_labels = _update_parameter(transformation_labels, self.best_transformation)
        return transformation_labels

    def perform_correction(
        self,
    ):
        """
        Performs the correction using the best transformation found.
        """

        # apply the best transformation to the mesh
        self.mesh.points = _transform_points(
            self.mesh.points,
            self.best_transformation["rotation"],
            self.best_transformation["translation"],
            self.best_transformation["scaling"],
        )

        # get rotation and translation for each slice
        rotations, translations = _eliminate_shift(self.contours, self.mesh, self.z_heights)
