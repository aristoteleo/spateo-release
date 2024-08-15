# Original code from PythonOT: ot.backend
# Source: https://github.com/PythonOT/POT/blob/master/ot/backend.py
# Copyright (c) 2016-2023 POT contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import time
import warnings

import numpy as np
import scipy
import scipy.linalg
import scipy.special as special
from scipy.sparse import coo_matrix, csr_matrix, issparse

DISABLE_TORCH_KEY = "SPATEO_BACKEND_DISABLE_PYTORCH"


if not os.environ.get(DISABLE_TORCH_KEY, False):
    try:
        import torch

        torch_type = torch.Tensor
    except ImportError:
        torch = False
        torch_type = float
else:
    torch = False
    torch_type = float


str_type_error = "All array should be from the same type/backend. Current types are : {}"


# Mapping between argument types and the existing backend
_BACKEND_IMPLEMENTATIONS = []
_BACKENDS = {}


def _register_backend_implementation(backend_impl):
    _BACKEND_IMPLEMENTATIONS.append(backend_impl)


def _get_backend_instance(backend_impl):
    if backend_impl.__name__ not in _BACKENDS:
        _BACKENDS[backend_impl.__name__] = backend_impl()
    return _BACKENDS[backend_impl.__name__]


def _check_args_backend(backend_impl, args):
    is_instance = set(isinstance(arg, backend_impl.__type__) for arg in args)
    # check that all arguments matched or not the type
    if len(is_instance) == 1:
        return is_instance.pop()

    # Otherwise return an error
    raise ValueError(str_type_error.format([type(arg) for arg in args]))


def get_backend_list():
    """Returns instances of all available backends.

    Note that the function forces all detected implementations
    to be instantiated even if specific backend was not use before.
    Be careful as instantiation of the backend might lead to side effects,
    like GPU memory pre-allocation. See the documentation for more details.
    If you only need to know which implementations are available,
    use `:py:func:`ot.backend.get_available_backend_implementations`,
    which does not force instance of the backend object to be created.
    """
    return [_get_backend_instance(backend_impl) for backend_impl in get_available_backend_implementations()]


def get_available_backend_implementations():
    """Returns the list of available backend implementations."""
    return _BACKEND_IMPLEMENTATIONS


def get_backend(*args):
    """Returns the proper backend for a list of input arrays

    Accepts None entries in the arguments, and ignores them

    Also raises TypeError if all arrays are not from the same backend
    """
    args = [arg for arg in args if arg is not None]  # exclude None entries

    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one (non-None) parameter")

    for backend_impl in _BACKEND_IMPLEMENTATIONS:
        if _check_args_backend(backend_impl, args):
            return _get_backend_instance(backend_impl)

    raise ValueError("Unknown type of non implemented backend.")


def to_numpy(*args):
    """Returns numpy arrays from any compatible backend"""

    if len(args) == 1:
        return get_backend(args[0]).to_numpy(args[0])
    else:
        return [get_backend(a).to_numpy(a) for a in args]


class Backend:
    """
    Backend abstract class.
    Implementations: :py:class:`JaxBackend`, :py:class:`NumpyBackend`, :py:class:`TorchBackend`,
    :py:class:`CupyBackend`, :py:class:`TensorflowBackend`

    - The `__name__` class attribute refers to the name of the backend.
    - The `__type__` class attribute refers to the data structure used by the backend.
    """

    __name__ = None
    __type__ = None
    __type_list__ = None

    rng_ = None

    def __str__(self):
        return self.__name__

    # convert batch of tensors to numpy
    def to_numpy(self, *arrays):
        """Returns the numpy version of tensors"""
        if len(arrays) == 1:
            return self._to_numpy(arrays[0])
        else:
            return [self._to_numpy(array) for array in arrays]

    # convert a tensor to numpy
    def _to_numpy(self, a):
        """Returns the numpy version of a tensor"""
        raise NotImplementedError()

    # convert batch of arrays from numpy
    def from_numpy(self, *arrays, type_as=None):
        """Creates tensors cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        if len(arrays) == 1:
            return self._from_numpy(arrays[0], type_as=type_as)
        else:
            return [self._from_numpy(array, type_as=type_as) for array in arrays]

    # convert an array from numpy
    def _from_numpy(self, a, type_as=None):
        """Creates a tensor cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        raise NotImplementedError()

    def zeros(self, shape, type_as=None):
        r"""
        Creates a tensor full of zeros.

        This function follows the api from :any:`numpy.zeros`

        See: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
        """
        raise NotImplementedError()

    def ones(self, shape, type_as=None):
        r"""
        Creates a tensor full of ones.

        This function follows the api from :any:`numpy.ones`

        See: https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        """
        raise NotImplementedError()

    def full(self, shape, fill_value, type_as=None):
        r"""
        Creates a tensor with given shape, filled with given value.

        This function follows the api from :any:`numpy.full`

        See: https://numpy.org/doc/stable/reference/generated/numpy.full.html
        """
        raise NotImplementedError()

    def eye(self, N, M=None, type_as=None):
        r"""
        Creates the identity matrix of given size.

        This function follows the api from :any:`numpy.eye`

        See: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
        """
        raise NotImplementedError()

    def sum(self, a, axis=None, keepdims=False):
        r"""
        Sums tensor elements over given dimensions.

        This function follows the api from :any:`numpy.sum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        raise NotImplementedError()

    def arange(self, stop, start=0, step=1, type_as=None):
        r"""
        Returns evenly spaced values within a given interval.

        This function follows the api from :any:`numpy.arange`

        See: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        """
        raise NotImplementedError()

    def max(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        """
        raise NotImplementedError()

    def min(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amin.html
        """
        raise NotImplementedError()

    def maximum(self, a, b):
        r"""
        Returns element-wise maximum of array elements.

        This function follows the api from :any:`numpy.maximum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        """
        raise NotImplementedError()

    def minimum(self, a, b):
        r"""
        Returns element-wise minimum of array elements.

        This function follows the api from :any:`numpy.minimum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        """
        raise NotImplementedError()

    def dot(self, a, b):
        r"""
        Returns the dot product of two tensors.

        This function follows the api from :any:`numpy.dot`

        See: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        """
        raise NotImplementedError()

    def log(self, a):
        r"""
        Computes the natural logarithm, element-wise.

        This function follows the api from :any:`numpy.log`

        See: https://numpy.org/doc/stable/reference/generated/numpy.log.html
        """
        raise NotImplementedError()

    def sqrt(self, a):
        r"""
        Returns the non-ngeative square root of a tensor, element-wise.

        This function follows the api from :any:`numpy.sqrt`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        raise NotImplementedError()

    def power(self, a, exponents):
        r"""
        First tensor elements raised to powers from second tensor, element-wise.

        This function follows the api from :any:`numpy.power`

        See: https://numpy.org/doc/stable/reference/generated/numpy.power.html
        """
        raise NotImplementedError()

    def norm(self, a, axis=None, keepdims=False):
        r"""
        Computes the matrix frobenius norm.

        This function follows the api from :any:`numpy.linalg.norm`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        raise NotImplementedError()

    def any(self, a):
        r"""
        Tests whether any tensor element along given dimensions evaluates to True.

        This function follows the api from :any:`numpy.any`

        See: https://numpy.org/doc/stable/reference/generated/numpy.any.html
        """
        raise NotImplementedError()

    def isnan(self, a):
        r"""
        Tests element-wise for NaN and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isnan`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        """
        raise NotImplementedError()

    def isinf(self, a):
        r"""
        Tests element-wise for positive or negative infinity and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isinf`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
        """
        raise NotImplementedError()

    def einsum(self, subscripts, *operands):
        r"""
        Evaluates the Einstein summation convention on the operands.

        This function follows the api from :any:`numpy.einsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        raise NotImplementedError()

    def sort(self, a, axis=-1):
        r"""
        Returns a sorted copy of a tensor.

        This function follows the api from :any:`numpy.sort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        raise NotImplementedError()

    def argsort(self, a, axis=None):
        r"""
        Returns the indices that would sort a tensor.

        This function follows the api from :any:`numpy.argsort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        """
        raise NotImplementedError()

    def searchsorted(self, a, v, side="left"):
        r"""
        Finds indices where elements should be inserted to maintain order in given tensor.

        This function follows the api from :any:`numpy.searchsorted`

        See: https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError()

    def flip(self, a, axis=None):
        r"""
        Reverses the order of elements in a tensor along given dimensions.

        This function follows the api from :any:`numpy.flip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
        """
        raise NotImplementedError()

    def clip(self, a, a_min, a_max):
        """
        Limits the values in a tensor.

        This function follows the api from :any:`numpy.clip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        """
        raise NotImplementedError()

    def repeat(self, a, repeats, axis=None):
        r"""
        Repeats elements of a tensor.

        This function follows the api from :any:`numpy.repeat`

        See: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        """
        raise NotImplementedError()

    def take_along_axis(self, arr, indices, axis):
        r"""
        Gathers elements of a tensor along given dimensions.

        This function follows the api from :any:`numpy.take_along_axis`

        See: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
        """
        raise NotImplementedError()

    def concatenate(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along an existing dimension.

        This function follows the api from :any:`numpy.concatenate`

        See: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        """
        raise NotImplementedError()

    def zero_pad(self, a, pad_width, value=0):
        r"""
        Pads a tensor with a given value (0 by default).

        This function follows the api from :any:`numpy.pad`

        See: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        raise NotImplementedError()

    def argmax(self, a, axis=None):
        r"""
        Returns the indices of the maximum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        raise NotImplementedError()

    def argmin(self, a, axis=None):
        r"""
        Returns the indices of the minimum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        """
        raise NotImplementedError()

    def mean(self, a, axis=None):
        r"""
        Computes the arithmetic mean of a tensor along given dimensions.

        This function follows the api from :any:`numpy.mean`

        See: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        raise NotImplementedError()

    def median(self, a, axis=None):
        r"""
        Computes the median of a tensor along given dimensions.

        This function follows the api from :any:`numpy.median`

        See: https://numpy.org/doc/stable/reference/generated/numpy.median.html
        """
        raise NotImplementedError()

    def std(self, a, axis=None):
        r"""
        Computes the standard deviation of a tensor along given dimensions.

        This function follows the api from :any:`numpy.std`

        See: https://numpy.org/doc/stable/reference/generated/numpy.std.html
        """
        raise NotImplementedError()

    def linspace(self, start, stop, num, type_as=None):
        r"""
        Returns a specified number of evenly spaced values over a given interval.

        This function follows the api from :any:`numpy.linspace`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        """
        raise NotImplementedError()

    def meshgrid(self, a, b):
        r"""
        Returns coordinate matrices from coordinate vectors (Numpy convention).

        This function follows the api from :any:`numpy.meshgrid`

        See: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        """
        raise NotImplementedError()

    def diag(self, a, k=0):
        r"""
        Extracts or constructs a diagonal tensor.

        This function follows the api from :any:`numpy.diag`

        See: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        """
        raise NotImplementedError()

    def unique(self, a, return_inverse=False):
        r"""
        Finds unique elements of given tensor.

        This function follows the api from :any:`numpy.unique`

        See: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        """
        raise NotImplementedError()

    def logsumexp(self, a, axis=None):
        r"""
        Computes the log of the sum of exponentials of input elements.

        This function follows the api from :any:`scipy.special.logsumexp`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        raise NotImplementedError()

    def stack(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along a new dimension.

        This function follows the api from :any:`numpy.stack`

        See: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        """
        raise NotImplementedError()

    def outer(self, a, b):
        r"""
        Computes the outer product between two vectors.

        This function follows the api from :any:`numpy.outer`

        See: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        """
        raise NotImplementedError()

    def reshape(self, a, shape):
        r"""
        Gives a new shape to a tensor without changing its data.

        This function follows the api from :any:`numpy.reshape`

        See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        r"""
        Sets the seed for the random generator.

        This function follows the api from :any:`numpy.random.seed`

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
        """
        raise NotImplementedError()

    def rand(self, *size, type_as=None):
        r"""
        Generate uniform random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def randn(self, *size, type_as=None):
        r"""
        Generate normal Gaussian random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        r"""
        Creates a sparse tensor in COOrdinate format.

        This function follows the api from :any:`scipy.sparse.coo_matrix`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        """
        raise NotImplementedError()

    def issparse(self, a):
        r"""
        Checks whether or not the input tensor is a sparse tensor.

        This function follows the api from :any:`scipy.sparse.issparse`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html
        """
        raise NotImplementedError()

    def tocsr(self, a):
        r"""
        Converts this matrix to Compressed Sparse Row format.

        This function follows the api from :any:`scipy.sparse.coo_matrix.tocsr`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html
        """
        raise NotImplementedError()

    def eliminate_zeros(self, a, threshold=0.0):
        r"""
        Removes entries smaller than the given threshold from the sparse tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.eliminate_zeros`

        See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html
        """
        raise NotImplementedError()

    def todense(self, a):
        r"""
        Converts a sparse tensor to a dense tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.toarray`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html
        """
        raise NotImplementedError()

    def where(self, condition, x, y):
        r"""
        Returns elements chosen from x or y depending on condition.

        This function follows the api from :any:`numpy.where`

        See: https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """
        raise NotImplementedError()

    def copy(self, a):
        r"""
        Returns a copy of the given tensor.

        This function follows the api from :any:`numpy.copy`

        See: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
        """
        raise NotImplementedError()

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        r"""
        Returns True if two arrays are element-wise equal within a tolerance.

        This function follows the api from :any:`numpy.allclose`

        See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        raise NotImplementedError()

    def dtype_device(self, a):
        r"""
        Returns the dtype and the device of the given tensor.
        """
        raise NotImplementedError()

    def assert_same_dtype_device(self, a, b):
        r"""
        Checks whether or not the two given inputs have the same dtype as well as the same device
        """
        raise NotImplementedError()

    def squeeze(self, a, axis=None):
        r"""
        Remove axes of length one from a.

        This function follows the api from :any:`numpy.squeeze`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        """
        raise NotImplementedError()

    def bitsize(self, type_as):
        r"""
        Gives the number of bits used by the data type of the given tensor.
        """
        raise NotImplementedError()

    def device_type(self, type_as):
        r"""
        Returns CPU or GPU depending on the device where the given tensor is located.
        """
        raise NotImplementedError()

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        r"""
        Executes a benchmark of the given callable with the given arguments.
        """
        raise NotImplementedError()

    def solve(self, a, b):
        r"""
        Solves a linear matrix equation, or system of linear scalar equations.

        This function follows the api from :any:`numpy.linalg.solve`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        """
        raise NotImplementedError()

    def trace(self, a):
        r"""
        Returns the sum along diagonals of the array.

        This function follows the api from :any:`numpy.trace`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.trace.html
        """
        raise NotImplementedError()

    def inv(self, a):
        r"""
        Computes the inverse of a matrix.

        This function follows the api from :any:`scipy.linalg.inv`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html
        """
        raise NotImplementedError()

    def sqrtm(self, a):
        r"""
        Computes the matrix square root. Requires input to be definite positive.

        This function follows the api from :any:`scipy.linalg.sqrtm`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
        """
        raise NotImplementedError()

    def eigh(self, a):
        r"""
        Computes the eigenvalues and eigenvectors of a symmetric tensor.

        This function follows the api from :any:`scipy.linalg.eigh`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
        """
        raise NotImplementedError()

    def kl_div(self, p, q, mass=False, eps=1e-16):
        r"""
        Computes the (Generalized) Kullback-Leibler divergence.

        This function follows the api from :any:`scipy.stats.entropy`.

        Parameter eps is used to avoid numerical errors and is added in the log.

        .. math::
             KL(p,q) = \langle \mathbf{p}, log(\mathbf{p} / \mathbf{q} + eps \rangle
             + \mathbb{1}_{mass=True} \langle \mathbf{q} - \mathbf{p}, \mathbf{1} \rangle

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
        """
        raise NotImplementedError()

    def isfinite(self, a):
        r"""
        Tests element-wise for finiteness (not infinity and not Not a Number).

        This function follows the api from :any:`numpy.isfinite`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
        """
        raise NotImplementedError()

    def array_equal(self, a, b):
        r"""
        True if two arrays have the same shape and elements, False otherwise.

        This function follows the api from :any:`numpy.array_equal`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        """
        raise NotImplementedError()

    def is_floating_point(self, a):
        r"""
        Returns whether or not the input consists of floats
        """
        raise NotImplementedError()

    def tile(self, a, reps):
        r"""
        Construct an array by repeating a the number of times given by reps

        See: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
        """
        raise NotImplementedError()

    def floor(self, a):
        r"""
        Return the floor of the input element-wise

        See: https://numpy.org/doc/stable/reference/generated/numpy.floor.html
        """
        raise NotImplementedError()

    def prod(self, a, axis=None):
        r"""
        Return the product of all elements.

        See: https://numpy.org/doc/stable/reference/generated/numpy.prod.html
        """
        raise NotImplementedError()

    def sort2(self, a, axis=None):
        r"""
        Return the sorted array and the indices to sort the array

        See: https://pytorch.org/docs/stable/generated/torch.sort.html
        """
        raise NotImplementedError()

    def qr(self, a):
        r"""
        Return the QR factorization

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html
        """
        raise NotImplementedError()

    def atan2(self, a, b):
        r"""
        Element wise arctangent

        See: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
        """
        raise NotImplementedError()

    def transpose(self, a, axes=None):
        r"""
        Returns a tensor that is a transposed version of a. The given dimensions dim0 and dim1 are swapped.

        See: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        """
        raise NotImplementedError()

    def matmul(self, a, b):
        r"""
        Matrix product of two arrays.

        See: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul
        """
        raise NotImplementedError()

    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        r"""
        Replace NaN with zero and infinity with large finite numbers or with the numbers defined by the user.

        See: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html#numpy.nan_to_num
        """
        raise NotImplementedError()

    def randperm(self, length):
        r"""
        Returns a random permutation of integers from 0 to length - 1.

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        """
        raise NotImplementedError()

    def choice(self, a, size, replace=False):
        r"""
        Generates a random sample from a given 1-D array.

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        """
        raise NotImplementedError()

    def topk(self, a, topk, axis=-1):
        r"""
        Returns the indices of the topk elements along a given axis.

        See: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
        """
        raise NotImplementedError()

    def dstack(self, a):
        r"""
        Stack arrays in sequence along the third axis.

        See: https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
        """
        raise NotImplementedError()

    def vstack(self, a):
        r"""
        Stack arrays in sequence vertically (row wise).

        See: https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
        """
        raise NotImplementedError()

    def hstack(self, a):
        r"""
        Stack arrays in sequence horizontally (column wise).

        See: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
        """
        raise NotImplementedError()

    def chunk(self, a, chunk_num, axis=0):
        r"""
        Split the tensor into a list of sub-tensors.

        See: https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        """
        raise NotImplementedError()

    def roll(self, a, shift, axis=None):
        r"""
        Roll array elements along a given axis.

        See: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
        """
        raise NotImplementedError()

    def pi(self, type_as=None):
        r"""
        Return the value of pi.

        See: https://numpy.org/doc/stable/reference/constants.html
        """
        raise NotImplementedError()


class NumpyBackend(Backend):
    """
    NumPy implementation of the backend.

    - `__name__` is "numpy"
    - `__type__` is np.ndarray
    """

    __name__ = "numpy"
    __type__ = np.ndarray
    __type_list__ = [np.array(1, dtype=np.float32), np.array(1, dtype=np.float64)]
    rng_ = np.random.RandomState()

    def _to_numpy(self, a):
        return a

    def _from_numpy(self, a, type_as=None):
        if type_as is None:
            return a
        elif isinstance(a, float):
            return a
        else:
            return a.astype(type_as.dtype)

    def zeros(self, shape, type_as=None):
        if type_as is None:
            return np.zeros(shape)
        else:
            return np.zeros(shape, dtype=type_as.dtype)

    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, *operands)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return np.full(shape, fill_value)
        else:
            return np.full(shape, fill_value, dtype=type_as.dtype)

    def sqrt(self, a):
        return np.sqrt(a)

    def ones(self, shape, type_as=None):
        if type_as is None:
            return np.ones(shape)
        else:
            return np.ones(shape, dtype=type_as.dtype)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def max(self, a, axis=None, keepdims=False):
        return np.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return np.min(a, axis, keepdims=keepdims)

    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return np.eye(N, M)
        else:
            return np.eye(N, M, dtype=type_as.dtype)

    def argsort(self, a, axis=-1):
        return np.argsort(a, axis)

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis)

    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis, keepdims=keepdims)

    def arange(self, stop, start=0, step=1, type_as=None):
        return np.arange(start, stop, step)

    def data(self, a, type_as=None):
        if type_as is None:
            return np.asarray(a)
        else:
            return np.asarray(a, dtype=type_as.dtype)

    def unique(self, a, return_inverse=False, axis=None):
        return np.unique(a, return_inverse=return_inverse, axis=axis)

    def unsqueeze(self, a, axis=-1):
        return np.expand_dims(a, axis)

    def multiply(self, a, b):
        return np.multiply(a, b)

    def power(self, a, exponents):
        return np.power(a, exponents)

    def dot(self, a, b):
        return np.dot(a, b)

    def prod(self, a, axis=0):
        return np.prod(a, axis=axis)

    def pi(self, type_as=None):
        return np.pi

    def chunk(self, a, chunk_num, axis=0):
        return np.array_split(a, chunk_num, axis=axis)

    def randperm(self, length):
        return np.random.permutation(length)

    def roll(self, a, shift, axis=None):
        return np.roll(a, shift, axis=axis)

    def choice(self, a, size, replace=False):
        return np.random.choice(a, size, replace=replace)

    def topk(self, a, topk, axis=-1):
        return np.argpartition(a, topk, axis=axis)

    def dstack(self, a):
        return np.dstack(a)

    def vstack(self, a):
        return np.vstack(a)

    def hstack(self, a):
        return np.hstack(a)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis)

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return np.where(condition)
        else:
            return np.where(condition, x, y)

    def copy(self, a):
        return a.copy()

    def repeat(self, a, repeats, axis=None):
        return np.repeat(a, repeats, axis)

    def sort2(self, a, axis=-1, descending=False):
        if not descending:
            return np.sort(a, axis=axis), np.argsort(a, axis=axis)
        else:
            return np.sort(-a, axis=axis), np.argsort(-a, axis=axis)

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if type_as is None:
            return coo_matrix((data, (rows, cols)), shape=shape)
        else:
            return coo_matrix((data, (rows, cols)), shape=shape, dtype=type_as.dtype)

    def issparse(self, a):
        return issparse(a)

    def eliminate_zeros(self, a, threshold=0.0):
        if threshold > 0:
            if self.issparse(a):
                a.data[self.abs(a.data) <= threshold] = 0
            else:
                a[self.abs(a) <= threshold] = 0
        if self.issparse(a):
            a.eliminate_zeros()
        return a

    def todense(self, a):
        if self.issparse(a):
            return a.toarray()
        else:
            return a

    def dtype_device(self, a):
        if hasattr(a, "dtype"):
            return a.dtype, "cpu"
        else:
            return type(a), "cpu"


_register_backend_implementation(NumpyBackend)


class TorchBackend(Backend):
    """
    PyTorch implementation of the backend

    - `__name__` is "torch"
    - `__type__` is torch.Tensor
    """

    __name__ = "torch"
    __type__ = torch_type
    __type_list__ = None

    rng_ = None

    def __init__(self):

        self.rng_ = torch.Generator("cpu")
        self.rng_.seed()

        self.__type_list__ = [torch.tensor(1, dtype=torch.float32), torch.tensor(1, dtype=torch.float64)]

        if torch.cuda.is_available():
            self.rng_cuda_ = torch.Generator("cuda")
            self.rng_cuda_.seed()
            self.__type_list__.append(torch.tensor(1, dtype=torch.float32, device="cuda"))
            self.__type_list__.append(torch.tensor(1, dtype=torch.float64, device="cuda"))
        else:
            self.rng_cuda_ = torch.Generator("cpu")

        from torch.autograd import Function

        # define a function that takes inputs val and grads
        # ad returns a val tensor with proper gradients
        class ValFunction(Function):
            @staticmethod
            def forward(ctx, val, grads, *inputs):
                ctx.grads = grads
                return val

            @staticmethod
            def backward(ctx, grad_output):
                # the gradients are grad
                return (None, None) + tuple(g * grad_output for g in ctx.grads)

        self.ValFunction = ValFunction

    def _to_numpy(self, a):
        if isinstance(a, float) or isinstance(a, int) or isinstance(a, np.ndarray):
            return np.array(a)
        return a.cpu().detach().numpy()

    def _from_numpy(self, a, type_as=None):
        if isinstance(a, float) or isinstance(a, int):
            a = np.array(a)
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def zeros(self, shape, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.zeros(shape)
        else:
            return torch.zeros(shape, dtype=type_as.dtype, device=type_as.device)

    def einsum(self, subscripts, *operands):
        return torch.einsum(subscripts, *operands)

    def mean(self, a, axis=None):
        if axis is not None:
            return torch.mean(a, dim=axis)
        else:
            return torch.mean(a)

    def full(self, shape, fill_value, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.full(shape, fill_value)
        else:
            return torch.full(shape, fill_value, dtype=type_as.dtype, device=type_as.device)

    def sqrt(self, a):
        return torch.sqrt(a)

    def ones(self, shape, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.ones(shape)
        else:
            return torch.ones(shape, dtype=type_as.dtype, device=type_as.device)

    def arange(self, stop, start=0, step=1, type_as=None):
        if type_as is None:
            return torch.arange(start, stop, step)
        else:
            return torch.arange(start, stop, step, device=type_as.device)

    def maximum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if hasattr(torch, "maximum"):
            return torch.maximum(a, b)
        else:
            return torch.max(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def minimum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if hasattr(torch, "minimum"):
            return torch.minimum(a, b)
        else:
            return torch.min(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def max(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.max(a)
        else:
            return torch.max(a, axis, keepdim=keepdims)[0]

    def min(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.min(a)
        else:
            return torch.min(a, axis, keepdim=keepdims)[0]

    def eye(self, N, M=None, type_as=None):
        if M is None:
            M = N
        if type_as is None:
            return torch.eye(N, m=M)
        else:
            return torch.eye(N, m=M, dtype=type_as.dtype, device=type_as.device)

    def argsort(self, a, axis=-1):
        sorted, indices = torch.sort(a, dim=axis)
        return indices

    def exp(self, a):
        return torch.exp(a)

    def log(self, a):
        return torch.log(a)

    def concatenate(self, arrays, axis=0):
        return torch.cat(arrays, dim=axis)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.sum(a)
        else:
            return torch.sum(a, axis, keepdim=keepdims)

    def data(self, a, type_as=None):
        if type_as is None:
            return torch.Tensor(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def unique(self, a, return_inverse=False, axis=None):
        return torch.unique(a, return_inverse=return_inverse, dim=axis)

    def unsqueeze(self, a, axis=-1):
        return torch.unsqueeze(a, axis)

    def multiply(self, a, b):
        return torch.mul(a, b)

    def power(self, a, exponents):
        return torch.pow(a, exponents)

    def dot(self, a, b):
        return torch.matmul(a, b)

    def prod(self, a, axis=0):
        return torch.prod(a, dim=axis)

    def pi(self, type_as=None):
        if type_as is None:
            return torch.tensor([np.pi])
        else:
            return torch.tensor([np.pi], dtype=type_as.dtype, device=type_as.device)

    def chunk(self, a, chunk_num, axis=0):
        return torch.chunk(a, chunks=chunk_num, dim=axis)

    def randperm(self, length):
        return torch.randperm(length)

    def roll(self, a, shift, axis=None):
        return torch.roll(a, shift, dims=axis)

    def choice(self, a, size, replace=False):
        return torch.randint(0, len(a), size=size, device=a.device)

    def topk(self, a, topk, axis=-1):
        return torch.topk(a, topk, dim=axis).indices

    def dstack(self, a):
        return torch.stack(a, dim=2)

    def vstack(self, a):
        return torch.cat(a, dim=0)

    def hstack(self, a):
        return torch.cat(a, dim=1)

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return torch.where(condition)
        else:
            return torch.where(condition, x, y)

    def copy(self, a):
        return a.clone()

    def repeat(self, a, repeats, axis=None):
        return torch.repeat_interleave(a, repeats, dim=axis)

    def sort2(self, a, axis=-1, descending=False):
        sorted, indices = torch.sort(a, dim=axis, descending=descending)
        return sorted, indices

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if type_as is None:
            return torch.sparse_coo_tensor(torch.stack([rows, cols]), data, size=shape)
        else:
            return torch.sparse_coo_tensor(
                torch.stack([rows, cols]), data, size=shape, dtype=type_as.dtype, device=type_as.device
            )

    def issparse(self, a):
        return getattr(a, "is_sparse", False) or getattr(a, "is_sparse_csr", False)

    def eliminate_zeros(self, a, threshold=0.0):
        if self.issparse(a):
            if threshold > 0:
                mask = self.abs(a) <= threshold
                mask = ~mask
                mask = mask.nonzero()
            else:
                mask = a._values().nonzero()
            nv = a._values().index_select(0, mask.view(-1))
            ni = a._indices().index_select(1, mask.view(-1))
            return self.coo_matrix(nv, ni[0], ni[1], shape=a.shape, type_as=a)
        else:
            if threshold > 0:
                a[self.abs(a) <= threshold] = 0
            return a

    def todense(self, a):
        if self.issparse(a):
            return a.to_dense()
        else:
            return a

    def dtype_device(self, a):
        return a.dtype, a.device


if torch:
    # Only register torch backend if it is installed
    _register_backend_implementation(TorchBackend)


# # Empty cache
# def empty_cache(device: str = "cpu"):
#     if device != "cpu":
#         torch.cuda.empty_cache()


# # Check if nx is a torch backend
# nx_torch = lambda nx: True if isinstance(nx, ot.backend.TorchBackend) else False

# # Concatenate expression matrices
# _cat = lambda nx, x, dim: torch.cat(x, dim=dim) if nx_torch(nx) else np.concatenate(x, axis=dim)  # np
# _unique = lambda nx, x, dim: torch.unique(x, dim=dim) if nx_torch(nx) else np.unique(x, axis=dim)  # np
# _var = lambda nx, x, dim: torch.var(x, dim=dim) if nx_torch(nx) else np.var(x, axis=dim)

# _data = (
#     lambda nx, data, type_as: torch.tensor(data, device=type_as.device, dtype=type_as.dtype)
#     if nx_torch(nx)
#     else np.asarray(data, dtype=type_as.dtype)
# )  # np
# _unsqueeze = lambda nx: torch.unsqueeze if nx_torch(nx) else np.expand_dims  # np
# _mul = lambda nx: torch.multiply if nx_torch(nx) else np.multiply  # np
# _power = lambda nx: torch.pow if nx_torch(nx) else np.power  # np
# _psi = lambda nx: torch.special.psi if nx_torch(nx) else psi
# _pinv = lambda nx: torch.linalg.pinv if nx_torch(nx) else pinv
# _dot = lambda nx: torch.matmul if nx_torch(nx) else np.dot  # np
# _identity = (
#     lambda nx, N, type_as: torch.eye(N, dtype=type_as.dtype, device=type_as.device)
#     if nx_torch(nx)
#     else np.identity(N, dtype=type_as.dtype)
# )  # should be eye
# _linalg = lambda nx: torch.linalg if nx_torch(nx) else np.linalg
# _prod = lambda nx: torch.prod if nx_torch(nx) else np.prod
# _pi = lambda nx: torch.pi if nx_torch(nx) else np.pi  #$ np
# _chunk = (
#     lambda nx, x, chunk_num, dim: torch.chunk(x, chunk_num, dim=dim)
#     if nx_torch(nx)
#     else np.array_split(x, chunk_num, axis=dim)
# )  # np
# _randperm = lambda nx: torch.randperm if nx_torch(nx) else np.random.permutation  # np
# _roll = lambda nx: torch.roll if nx_torch(nx) else np.roll  # np
# _choice = (
#     lambda nx, length, size: torch.randperm(length)[:size]
#     if nx_torch(nx)
#     else np.random.choice(length, size, replace=False)
# )  # np
# _topk = (
#     lambda nx, x, topk, axis: torch.topk(x, topk, dim=axis)[1] if nx_torch(nx) else np.argpartition(x, topk, axis=axis)
# )  # np
# _dstack = lambda nx: torch.dstack if nx_torch(nx) else np.dstack
# _vstack = lambda nx: torch.vstack if nx_torch(nx) else np.vstack
# _hstack = lambda nx: torch.hstack if nx_torch(nx) else np.hstack

# # _split = (
# #     lambda nx, x, chunk_size, dim: torch.split(x, chunk_size, dim)
# #     if nx_torch(nx)
# #     else np.array_split(x, chunk_size, axis=dim)
# # )


# # def torch_like_split(arr, size, dim=0):
# #     if dim < 0:
# #         dim += arr.ndim
# #     shape = arr.shape
# #     arr = np.swapaxes(arr, dim, -1)
# #     flat_arr = arr.reshape(-1, shape[dim])
# #     num_splits = flat_arr.shape[-1] // size
# #     remainder = flat_arr.shape[-1] % size
# #     splits = np.array_split(flat_arr[:, : num_splits * size], num_splits, axis=-1)
# #     if remainder:
# #         splits.append(flat_arr[:, num_splits * size :])
# #     splits = [np.swapaxes(split.reshape(*shape[:dim], -1, *shape[dim + 1 :]), dim, -1) for split in splits]

# #     return splits


# _where = lambda nx, condition: torch.where(condition) if nx_torch(nx) else np.where(condition)
# _repeat_interleave = (
#     lambda nx, x, repeats, axis: torch.repeat_interleave(x, repeats, dim=axis)
#     if nx_torch(nx)
#     else np.repeat(x, repeats, axis)
# )


# def _sort(nx, arr, axis=-1, descending=False):
#     if not descending:
#         sorted_arr, sorted_idx = nx.sort2(arr, axis=axis)
#     else:
#         sorted_arr, sorted_idx = nx.sort2(-arr, axis=axis)
#         sorted_arr = -sorted_arr
#     return sorted_arr, sorted_idx  # np


# def _SparseTensor(nx, row, col, value, sparse_sizes):

#     return SparseTensor(indices=torch.vstack((row, col)), values=value, size=sparse_sizes)


# def sparse_tensor_to_scipy(sparse_tensor):
#     from scipy.sparse import coo_matrix

#     """
#     Convert a PyTorch SparseTensor to a SciPy sparse matrix (COO format).

#     Args:
#         sparse_tensor (torch.sparse.Tensor): The input PyTorch sparse tensor.

#     Returns:
#         scipy.sparse.coo_matrix: The output SciPy sparse matrix.
#     """
#     if not sparse_tensor.is_sparse:
#         raise ValueError("Input tensor is not a sparse tensor")

#     sparse_tensor = sparse_tensor.coalesce()  # Ensure the sparse tensor is in coalesced format
#     values = sparse_tensor.values().cpu().numpy()
#     indices = sparse_tensor.indices().cpu().numpy()

#     shape = sparse_tensor.shape
#     coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)

#     return coo
