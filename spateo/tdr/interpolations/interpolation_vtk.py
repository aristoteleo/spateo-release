from typing import Optional, Union

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from anndata import AnnData
from numpy import ndarray

from ...logging import logger_manager as lm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def vtk_interpolation(
    source_adata: AnnData,
    target_points: Optional[ndarray] = None,
    keys: Union[str, list] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    radius: Optional[float] = None,
    n_points: Optional[int] = None,
    kernel: Literal["shepard", "gaussian", "linear"] = "shepard",
    null_strategy: Literal[0, 1, 2] = 1,
    null_value: Union[int, float] = 0,
) -> AnnData:
    """
    Learn a continuous mapping from space to gene expression pattern with the method contained in VTK.

    Args:
        source_adata: AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        target_points: The spatial coordinates of new data point. If target_coords is None, generate new points based on grid_num.
        keys: Gene list or info list in the `obs` attribute whose interpolate expression across space needs to learned.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each bucket.
        layer: If ``'X'``, uses ``.X``, otherwise uses the representation given by ``.layers[layer]``.
        radius: Set the radius of the point cloud. If you are generating a Gaussian distribution, then this is the
                standard deviation for each of x, y, and z.
        n_points: Specify the number of points for the source object to hold.
                  If n_points (number of the closest points to use) is set then radius value is ignored.
        kernel: The kernel of interpolations kernel. Available `kernels` are:
                * `shepard`: vtkShepardKernel is an interpolations kernel that uses the method of Shepard to perform
                             interpolations. The weights are computed as 1/r^p, where r is the distance to a neighbor
                             point within the kernel radius R; and p (the power parameter) is a positive exponent
                             (typically p=2).
                * `gaussian`: vtkGaussianKernel is an interpolations kernel that simply returns the weights for all
                              points found in the sphere defined by radius R. The weights are computed as:
                              exp(-(s*r/R)^2) where r is the distance from the point to be interpolated to a neighboring
                              point within R. The sharpness s simply affects the rate of fall off of the Gaussian.
                * `linear`: vtkLinearKernel is an interpolations kernel that averages the contributions of all points in
                            the basis.
        null_strategy: Specify a strategy to use when encountering a "null" point during the interpolations process.
                      Null points occur when the local neighborhood(of nearby points to interpolate from) is empty.
                * Case 0: an output array is created that marks points as being valid (=1) or null (invalid =0), and
                          the nullValue is set as well
                * Case 1: the output data value(s) are set to the provided nullValue
                * Case 2: simply use the closest point to perform the interpolations.
        null_value: see above.

    Returns:
        interp_adata: an anndata object that has interpolated expression.
    """

    # Inference
    source_adata = source_adata.copy()
    source_adata.X = source_adata.X if layer == "X" else source_adata.layers[layer]

    source_model = pv.PolyData(np.asarray(source_adata.obsm[spatial_key]))
    assert not (keys is None), "`keys` cannot be None."
    keys = [keys] if isinstance(keys, str) else keys
    obs_keys, var_keys = [], []
    for key in keys:
        if key in source_adata.obs.keys():
            source_model.point_data[key] = np.asarray(source_adata.obs[key]).flatten()
            obs_keys.append(key)
        elif key in source_adata.var_names.tolist():
            source_model.point_data[key] = np.asarray(source_adata[:, key].X.flatten())
            var_keys.append(key)
        else:
            raise ValueError(f"`{key}` is not in source_adata.")

    # Interpolation
    target_model = pv.PolyData(np.asarray(target_points))

    # Kernel
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(source_model)
    locator.BuildLocator()

    if kernel.lower() == "shepard":
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(2)
    elif kernel.lower() == "gaussian":
        kern = vtk.vtkGaussianKernel()
        kern.SetSharpness(2)
    elif kernel.lower() == "linear":
        kern = vtk.vtkLinearKernel()
    else:
        raise ValueError(
            "`kernels` value is wrong." "\nAvailable `kernels` are: `'shepard'`, `'gaussian'`, `'linear'`."
        )

    if radius is None and not n_points:
        raise ValueError("Please set either radius or n_points")
    if n_points:
        kern.SetNumberOfPoints(n_points)
        kern.SetKernelFootprintToNClosest()
    else:
        kern.SetRadius(radius)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(target_model)
    interpolator.SetSourceData(source_model)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    interpolator.PassFieldArraysOff()
    interpolator.SetNullPointsStrategy(null_strategy)
    interpolator.SetNullValue(null_value)
    interpolator.SetValidPointsMaskArrayName("ValidPointMask")
    interpolator.Update()
    cpoly = interpolator.GetOutput()
    interpolated_model = pv.wrap(cpoly)

    # Output interpolated anndata
    lm.main_info("Creating an adata object with the interpolated expression...")

    if len(obs_keys) != 0:
        obs_data = np.asarray([np.asarray(interpolated_model.point_data[key]) for key in obs_keys]).T
        obs_data = pd.DataFrame(obs_data, columns=obs_keys)
    if len(var_keys) != 0:
        X = np.asarray(interpolated_model.point_data[var_keys[0]]).reshape(-1, 1)
        for key in var_keys[1:]:
            X = np.c_[X, np.asarray(interpolated_model.point_data[key])]
        var_data = pd.DataFrame(index=var_keys)

    interp_adata = AnnData(
        X=X if len(var_keys) != 0 else None,
        obs=obs_data if len(obs_keys) != 0 else None,
        obsm={spatial_key: np.asarray(interpolated_model.points)},
        var=var_data if len(var_keys) != 0 else None,
    )

    lm.main_finish_progress(progress_name="VTKInterpolation")
    return interp_adata
