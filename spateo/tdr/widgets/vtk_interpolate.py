from typing import Optional, Union

import pyvista as pv
import vtk

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def interpolate_model(
    model,
    source,
    source_key: Union[str, list] = None,
    radius: Optional[float] = None,
    N: Optional[int] = None,
    kernel: Literal["shepard", "gaussian", "linear"] = "shepard",
    where: Literal["point_data", "cell_data"] = "cell_data",
    nullStrategy: Literal[0, 1, 2] = 1,
    nullValue: Union[int, float] = 0,
):
    """
    Interpolate over source to port its data onto the current object using various kernels.

    Args:
        model: Model that require interpolation.
        source: A data source that provides coordinates and data. Usually a point cloud object.
        source_key: Which data to migrate to the model object. If `source_key` is None, migrate all to the model object.
        radius: Set the radius of the point cloud. If you are generating a Gaussian distribution, then this is the
                standard deviation for each of x, y, and z.
        N: Specify the number of points for the source object to hold.
           If N (number of the closest points to use) is set then radius value is ignored.
        kernel: The kernel of interpolation kernel. Available `kernels` are:
                * `shepard`: vtkShepardKernel is an interpolation kernel that uses the method of Shepard to perform
                             interpolation. The weights are computed as 1/r^p, where r is the distance to a neighbor
                             point within the kernel radius R; and p (the power parameter) is a positive exponent
                             (typically p=2).
                * `gaussian`: vtkGaussianKernel is an interpolation kernel that simply returns the weights for all
                              points found in the sphere defined by radius R. The weights are computed as:
                              exp(-(s*r/R)^2) where r is the distance from the point to be interpolated to a neighboring
                              point within R. The sharpness s simply affects the rate of fall off of the Gaussian.
                * `linear`: vtkLinearKernel is an interpolation kernel that averages the contributions of all points in
                            the basis.
        where: The location where the data is stored in the model.
        nullStrategy: Specify a strategy to use when encountering a "null" point during the interpolation process.
                      Null points occur when the local neighborhood(of nearby points to interpolate from) is empty.
                * Case 0: an output array is created that marks points as being valid (=1) or null (invalid =0), and
                          the nullValue is set as well
                * Case 1: the output data value(s) are set to the provided nullValue
                * Case 2: simply use the closest point to perform the interpolation.
        nullValue: see above.

    Returns:
        interpolated_model: Interpolated model.
    """

    _model = model.copy()
    source = source.cell_data_to_point_data()
    _source = source.copy()
    if not (source_key is None):
        _source.clear_data()
        source_key = [source_key] if isinstance(source_key, str) else source_key
        for key in source_key:
            _source[key] = source[key]

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(_source)
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
            "\n`kernels` value is wrong." "\nAvailable `kernels` are: `'shepard'`, `'gaussian'`, `'linear'`."
        )

    if radius is None and not N:
        raise ValueError("\nPlease set either radius or N")
    if N:
        kern.SetNumberOfPoints(N)
        kern.SetKernelFootprintToNClosest()
    else:
        kern.SetRadius(radius)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(_model)
    interpolator.SetSourceData(_source)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    interpolator.PassFieldArraysOff()
    interpolator.SetNullPointsStrategy(nullStrategy)
    interpolator.SetNullValue(nullValue)
    interpolator.SetValidPointsMaskArrayName("ValidPointMask")
    interpolator.Update()

    if where == "point_data":
        cpoly = interpolator.GetOutput()
    else:
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(interpolator.GetOutput())
        p2c.Update()
        cpoly = p2c.GetOutput()

    interpolated_model = pv.wrap(cpoly)

    raw_pdk = model.point_data.keys()
    raw_cdk = model.cell_data.keys()
    if not (raw_pdk is None):
        for key in raw_pdk:
            interpolated_model.point_data[key] = model.point_data[key]
    if not (raw_cdk is None):
        for key in raw_cdk:
            interpolated_model.cell_data[key] = model.cell_data[key]

    return interpolated_model
