import numpy as np
import torch


class Backend:
    pass


class Backend_Torch:
    pass


class Backend_Numpy:
    __name__ = "numpy"
    __type__ = np.ndarray
    __type_list__ = [np.array(1, dtype=np.float32), np.array(1, dtype=np.float64)]

    def to_numpy(
        self,
    ):
        pass

    def from_numpy(
        self,
    ):
        pass

    def zeros(
        self,
    ):
        pass

    def einsum(
        self,
    ):
        pass

    def mean(
        self,
    ):
        pass

    def full(
        self,
    ):
        pass

    def sqrt(
        self,
    ):
        pass

    def ones(
        self,
    ):
        pass

    def maximum(
        self,
    ):
        pass

    def minimum(
        self,
    ):
        pass

    def max(
        self,
    ):
        pass

    def min(
        self,
    ):
        pass

    def argsort(
        self,
    ):
        pass

    def exp(
        self,
    ):
        pass

    def concatenate(
        self,
    ):
        pass

    def sum(
        self,
    ):
        pass

    def to_numpy(
        self,
    ):
        pass

    def to_numpy(
        self,
    ):
        pass

    def to_numpy(
        self,
    ):
        pass


# Empty cache
def empty_cache(device: str = "cpu"):
    if device != "cpu":
        torch.cuda.empty_cache()


# Check if nx is a torch backend
nx_torch = lambda nx: True if isinstance(nx, ot.backend.TorchBackend) else False

# Concatenate expression matrices
_cat = lambda nx, x, dim: torch.cat(x, dim=dim) if nx_torch(nx) else np.concatenate(x, axis=dim)
_unique = lambda nx, x, dim: torch.unique(x, dim=dim) if nx_torch(nx) else np.unique(x, axis=dim)
_var = lambda nx, x, dim: torch.var(x, dim=dim) if nx_torch(nx) else np.var(x, axis=dim)

_data = (
    lambda nx, data, type_as: torch.tensor(data, device=type_as.device, dtype=type_as.dtype)
    if nx_torch(nx)
    else np.asarray(data, dtype=type_as.dtype)
)
_unsqueeze = lambda nx: torch.unsqueeze if nx_torch(nx) else np.expand_dims
_mul = lambda nx: torch.multiply if nx_torch(nx) else np.multiply
_power = lambda nx: torch.pow if nx_torch(nx) else np.power
_psi = lambda nx: torch.special.psi if nx_torch(nx) else psi
_pinv = lambda nx: torch.linalg.pinv if nx_torch(nx) else pinv
_dot = lambda nx: torch.matmul if nx_torch(nx) else np.dot
_identity = (
    lambda nx, N, type_as: torch.eye(N, dtype=type_as.dtype, device=type_as.device)
    if nx_torch(nx)
    else np.identity(N, dtype=type_as.dtype)
)
_linalg = lambda nx: torch.linalg if nx_torch(nx) else np.linalg
_prod = lambda nx: torch.prod if nx_torch(nx) else np.prod
_pi = lambda nx: torch.pi if nx_torch(nx) else np.pi
_chunk = (
    lambda nx, x, chunk_num, dim: torch.chunk(x, chunk_num, dim=dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_num, axis=dim)
)
_randperm = lambda nx: torch.randperm if nx_torch(nx) else np.random.permutation
_roll = lambda nx: torch.roll if nx_torch(nx) else np.roll
_choice = (
    lambda nx, length, size: torch.randperm(length)[:size]
    if nx_torch(nx)
    else np.random.choice(length, size, replace=False)
)
_topk = (
    lambda nx, x, topk, axis: torch.topk(x, topk, dim=axis)[1] if nx_torch(nx) else np.argpartition(x, topk, axis=axis)
)
_dstack = lambda nx: torch.dstack if nx_torch(nx) else np.dstack
_vstack = lambda nx: torch.vstack if nx_torch(nx) else np.vstack
_hstack = lambda nx: torch.hstack if nx_torch(nx) else np.hstack

_split = (
    lambda nx, x, chunk_size, dim: torch.split(x, chunk_size, dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_size, axis=dim)
)


def torch_like_split(arr, size, dim=0):
    if dim < 0:
        dim += arr.ndim
    shape = arr.shape
    arr = np.swapaxes(arr, dim, -1)
    flat_arr = arr.reshape(-1, shape[dim])
    num_splits = flat_arr.shape[-1] // size
    remainder = flat_arr.shape[-1] % size
    splits = np.array_split(flat_arr[:, : num_splits * size], num_splits, axis=-1)
    if remainder:
        splits.append(flat_arr[:, num_splits * size :])
    splits = [np.swapaxes(split.reshape(*shape[:dim], -1, *shape[dim + 1 :]), dim, -1) for split in splits]

    return splits


_where = lambda nx, condition: torch.where(condition) if nx_torch(nx) else np.where(condition)
_repeat_interleave = (
    lambda nx, x, repeats, axis: torch.repeat_interleave(x, repeats, dim=axis)
    if nx_torch(nx)
    else np.repeat(x, repeats, axis)
)


def _sort(nx, arr, axis=-1, descending=False):
    if not descending:
        sorted_arr, sorted_idx = nx.sort2(arr, axis=axis)
    else:
        sorted_arr, sorted_idx = nx.sort2(-arr, axis=axis)
        sorted_arr = -sorted_arr
    return sorted_arr, sorted_idx


def _SparseTensor(nx, row, col, value, sparse_sizes):

    return SparseTensor(indices=torch.vstack((row, col)), values=value, size=sparse_sizes)


def sparse_tensor_to_scipy(sparse_tensor):
    from scipy.sparse import coo_matrix

    """
    Convert a PyTorch SparseTensor to a SciPy sparse matrix (COO format).

    Args:
        sparse_tensor (torch.sparse.Tensor): The input PyTorch sparse tensor.

    Returns:
        scipy.sparse.coo_matrix: The output SciPy sparse matrix.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input tensor is not a sparse tensor")

    sparse_tensor = sparse_tensor.coalesce()  # Ensure the sparse tensor is in coalesced format
    values = sparse_tensor.values().cpu().numpy()
    indices = sparse_tensor.indices().cpu().numpy()

    shape = sparse_tensor.shape
    coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)

    return coo
