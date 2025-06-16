import os
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData
from numpy import ndarray
from scipy.linalg import pinv
from scipy.sparse import issparse
from scipy.special import psi
from sklearn.neighbors import kneighbors_graph
from torch import sparse_coo_tensor as SparseTensor

from spateo.alignment.methods.backend import NumpyBackend, TorchBackend, get_backend
from spateo.logging import logger_manager as lm

# Get the intersection of lists
intersect_lsts = lambda *lsts: list(set(lsts[0]).intersection(*lsts[1:]))

# Covert a sparse matrix into a dense np array
to_dense_matrix = lambda X: X.toarray() if issparse(X) else np.array(X)

# Returns the data matrix or representation
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.layers[rep]


#########################
# Check data and device #
#########################

# Finished
def check_backend(device: str = "cpu", dtype: str = "float32", verbose: bool = True):
    """
    Check the proper backend for the device.

    Args:
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        dtype: The floating-point number type. Only float32 and float64.
        verbose: If ``True``, print progress updates.

    Returns:
        backend: The proper backend.
        type_as: The type_as.device is the device used to run the program and the type_as.dtype is the floating-point number type.
    """
    if device == "cpu":
        backend = NumpyBackend()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if torch.cuda.is_available():
            torch.cuda.init()
            backend = TorchBackend()
        else:
            backend = NumpyBackend()
            if verbose:
                lm.main_info(
                    message="GPU is not available, resorting to torch cpu.",
                    indent_level=1,
                )
    if nx_torch(backend):
        type_as = backend.__type_list__[-2] if dtype == "float32" else backend.__type_list__[-1]
    else:
        type_as = backend.__type_list__[0] if dtype == "float32" else backend.__type_list__[1]
    return backend, type_as


# Finished
def check_spatial_coords(sample: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Check and return the spatial coordinate information from an AnnData object.

    Args:
        sample (AnnData): An AnnData object containing the sample data.
        spatial_key (str, optional): The key in `.obsm` that corresponds to the raw spatial coordinates. Defaults to "spatial".

    Returns:
        np.ndarray: The spatial coordinates.

    Raises:
        KeyError: If the specified spatial_key is not found in `sample.obsm`.
    """

    if spatial_key not in sample.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in AnnData object.")

    coordinates = sample.obsm[spatial_key].copy()
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.values

    mask = []
    for i in range(coordinates.shape[1]):
        if len(np.unique(coordinates[:, i])) == 1:
            lm.main_info(
                message=f"The {i}-th dimension of the spatial coordinate has single value, which will be ignored.",
                indent_level=2,
            )
        else:
            mask.append(i)

    # Select only dimensions with more than one unique value
    coordinates = coordinates[:, mask]

    if coordinates.shape[1] > 3 or coordinates.shape[1] < 2:
        raise ValueError(f"The spatial coordinate '{spatial_key}' should only has 2 / 3 dimension")

    return np.asarray(coordinates)


# Finished
def check_exp(sample: AnnData, layer: str = "X") -> np.ndarray:
    """
    Check expression matrix.

    Args:
        sample (AnnData): An AnnData object containing the sample data.
        layer (str, optional): The key in `.layers` that corresponds to the expression matrix. Defaults to "X".

    Returns:
        The expression matrix.

    Raises:
        KeyError: If the specified layer is not found in `sample.layers`.
    """

    if layer == "X":
        exp_matrix = sample.X.copy()
    else:
        if layer not in sample.layers:
            raise KeyError(f"Layer '{layer}' not found in AnnData object.")
        exp_matrix = sample.layers[layer].copy()

    exp_matrix = to_dense_matrix(exp_matrix)
    return exp_matrix


# Finished
def check_obs(rep_layer: List[str], rep_field: List[str]) -> Optional[str]:
    """
    Check that the number of occurrences of 'obs' in the list of representation fields is no more than one.

    Args:
        rep_layer (List[str]): A list of representations to check.
        rep_field (List[str]): A list of representation types corresponding to the representations in `rep_layer`.

    Returns:
        Optional[str]: The representation key if 'obs' occurs exactly once, otherwise None.

    Raises:
        ValueError: If 'obs' occurs more than once in the list.
    """

    count = 0
    position = -1

    for i, s in enumerate(rep_field):
        if s == "obs":
            count += 1
            position = i
            if count > 1:
                raise ValueError(
                    f"'obs' occurs more than once in the list. Currently Spateo only support one label consistency."
                )

    # Return the 'obs' key if found exactly once
    if count == 1:
        return rep_layer[position]
    else:
        return None


# Finished
def check_rep_layer(
    samples: List[AnnData],
    rep_layer: Union[str, List[str]] = "X",
    rep_field: Union[str, List[str]] = "layer",
) -> bool:
    """
    Check if specified representations exist in the `.layers`, `.obsm`, or `.obs` attributes of AnnData objects.

    Args:
        samples (List[AnnData]):
            A list of AnnData objects containing the data samples.
        rep_layer (Union[str, List[str]], optional):
            The representation layer(s) to check. Defaults to "X".
        rep_field (Union[str, List[str]], optional):
            The field(s) indicating the type of representation. Acceptable values are "layer", "obsm", and "obs". Defaults to "layer".

    Returns:
        bool:
            True if all specified representations exist in the corresponding attributes of all AnnData objects, False otherwise.

    Raises:
        ValueError:
            If the specified representation is not found in the specified attribute or if the attribute type is invalid.
    """

    for sample in samples:
        for rep, rep_f in zip(rep_layer, rep_field):
            if rep_f == "layer":
                if (rep != "X") and (rep not in sample.layers):
                    raise ValueError(
                        f"The specified representation '{rep}' not found in the '{rep_f}' attribute of some of the AnnData objects."
                    )
            elif rep_f == "obsm":
                if rep not in sample.obsm:
                    raise ValueError(
                        f"The specified representation '{rep}' not found in the '{rep_f}' attribute of some of the AnnData objects."
                    )
            elif rep_f == "obs":
                if rep not in sample.obs:
                    raise ValueError(
                        f"The specified representation '{rep}' not found in the '{rep_f}' attribute of some of the AnnData objects."
                    )

                # judge if the sample.obs[rep] is categorical
                if not isinstance(sample.obs[rep].dtype, pd.CategoricalDtype):
                    raise ValueError(
                        f"The specified representation '{rep}' found in the '{rep_f}' attribute should be categorical."
                    )
            else:
                raise ValueError("rep_field must be either 'layer', 'obsm' or 'obs'")
    return True


# Finished
def check_label_transfer_dict(
    catA: List[str],
    catB: List[str],
    label_transfer_dict: Dict[str, Dict[str, float]],
):
    """
    Check the label transfer dictionary for consistency with given categories.

    Args:
        catA (List[str]):
            List of category labels from the first dataset.
        catB (List[str]):
            List of category labels from the second dataset.
        label_transfer_dict (Dict[str, Dict[str, float]]):
            Dictionary defining the transfer probabilities between categories.

    Raises:
        KeyError:
            If a category from `catA` is not found in `label_transfer_dict`.
        KeyError:
            If a category from `catB` is not found in the nested dictionary of `label_transfer_dict`.
    """

    for ca in catA:
        if ca in label_transfer_dict.keys():
            for cb in catB:
                if cb not in label_transfer_dict[ca].keys():
                    raise KeyError(
                        f"Category '{cb}' from catB not found in label_transfer_dict for category '{ca}' from catA."
                    )

        else:
            raise KeyError(f"Category '{ca}' from catA not found in label_transfer_dict.")


# Finished
def check_label_transfer(
    nx: Union[TorchBackend, NumpyBackend],
    type_as: Union[torch.Tensor, np.ndarray],
    sampleA: AnnData,
    sampleB: AnnData,
    obs_key: str,
    label_transfer_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Check and generate label transfer matrices for the given samples.

    Args:
        nx (module):
            Backend module (e.g., numpy or torch).
        type_as (type):
            Type to which the output should be cast.
        samples (List[AnnData]):
            List of AnnData objects containing the samples.
        obs_key (str):
            The key in `.obs` that corresponds to the labels.
        label_transfer_dict (Optional[List[Dict[str, Dict[str, float]]]], optional):
            List of dictionaries defining the label transfer cost between categories of each pair of samples. Defaults to None.

    Returns:
        List[Union[np.ndarray, torch.Tensor]]:
            List of label transfer matrices, each as either a NumPy array or torch Tensor.

    Raises:
        ValueError:
            If the length of `label_transfer_dict` does not match `len(samples) - 1`.
    """

    if label_transfer_dict is not None:
        if not isinstance(label_transfer_dict, dict):
            raise ValueError("label_transfer_dict should be a list or a dictionary.")

    catA = sampleA.obs[obs_key].cat.categories.tolist()
    catB = sampleB.obs[obs_key].cat.categories.tolist()
    label_transfer = np.zeros((len(catA), len(catB)), dtype=np.float32)

    if label_transfer_dict is None:
        label_transfer_dict = generate_label_transfer_dict(catA, catB)

    for j, ca in enumerate(catA):
        for k, cb in enumerate(catB):
            label_transfer[j, k] = label_transfer_dict[ca][cb]
    label_transfer = nx.from_numpy(label_transfer, type_as=type_as)

    return label_transfer


# def check_label_transfer(
#     nx: Union[TorchBackend, NumpyBackend],
#     type_as: Union[torch.Tensor, np.ndarray],
#     samples: List[AnnData],
#     obs_key: str,
#     label_transfer_dict: Optional[List[Dict[str, Dict[str, float]]]] = None,
# ) -> List[Union[np.ndarray, torch.Tensor]]:
#     """
#     Check and generate label transfer matrices for the given samples.

#     Args:
#         nx (module):
#             Backend module (e.g., numpy or torch).
#         type_as (type):
#             Type to which the output should be cast.
#         samples (List[AnnData]):
#             List of AnnData objects containing the samples.
#         obs_key (str):
#             The key in `.obs` that corresponds to the labels.
#         label_transfer_dict (Optional[List[Dict[str, Dict[str, float]]]], optional):
#             List of dictionaries defining the label transfer cost between categories of each pair of samples. Defaults to None.

#     Returns:
#         List[Union[np.ndarray, torch.Tensor]]:
#             List of label transfer matrices, each as either a NumPy array or torch Tensor.

#     Raises:
#         ValueError:
#             If the length of `label_transfer_dict` does not match `len(samples) - 1`.
#     """

#     if label_transfer_dict is not None:
#         if isinstance(label_transfer_dict, dict):
#             label_transfer_dict = [label_transfer_dict]
#         if isinstance(label_transfer_dict, list):
#             if len(label_transfer_dict) != (len(samples) - 1):
#                 raise ValueError("The length of label_transfer_dict must be equal to len(samples) - 1.")
#         else:
#             raise ValueError("label_transfer_dict should be a list or a dictionary.")

#     label_transfer = []
#     for i in range(len(samples) - 1):
#         catB = samples[i].obs[obs_key].cat.categories.tolist()
#         catA = samples[i + 1].obs[obs_key].cat.categories.tolist()
#         cur_label_transfer = np.zeros(len(catA), len(catB), dtype=np.float32)

#         if label_transfer_dict is not None:
#             cur_label_transfer_dict = label_transfer_dict[i]
#             check_label_transfer_dict(catA=catA, catB=catB, label_transfer_dict=cur_label_transfer_dict)
#         else:
#             cur_label_transfer_dict = generate_label_transfer_dict(catA, catB)

#         for j, ca in enumerate(catA):
#             for k, cb in enumerate(catB):
#                 cur_label_transfer[j, k] = cur_label_transfer_dict[ca][cb]
#         label_transfer.append(nx.from_numpy(cur_label_transfer, type_as=type_as))

#     return label_transfer


# Finished
def generate_label_transfer_dict(
    cat1: List[str],
    cat2: List[str],
    positive_pairs: Optional[List[Dict[str, Union[List[str], float]]]] = None,
    negative_pairs: Optional[List[Dict[str, Union[List[str], float]]]] = None,
    default_positive_value: float = 10.0,
    default_negative_value: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Generate a label transfer dictionary with normalized values.

    Args:
        cat1 (List[str]):
            List of categories from the first dataset.
        cat2 (List[str]):
            List of categories from the second dataset.
        positive_pairs (Optional[List[Dict[str, Union[List[str], float]]]], optional):
            List of positive pairs with transfer values. Each dictionary should have 'left', 'right', and 'value' keys. Defaults to None.
        negative_pairs (Optional[List[Dict[str, Union[List[str], float]]]], optional):
            List of negative pairs with transfer values. Each dictionary should have 'left', 'right', and 'value' keys. Defaults to None.
        default_positive_value (float, optional):
            Default value for positive pairs if none are provided. Defaults to 10.0.
        default_negative_value (float, optional):
            Default value for negative pairs if none are provided. Defaults to 1.0.

    Returns:
        Dict[str, Dict[str, float]]:
            A normalized label transfer dictionary.
    """

    # Initialize label transfer dictionary with default values
    label_transfer_dict = {c1: {c2: 1.0 for c2 in cat2} for c1 in cat1}

    # Generate default positive pairs if none provided
    if (positive_pairs is None) and (negative_pairs is None):
        label_transfer_dict = {c1: {c2: default_negative_value for c2 in cat2} for c1 in cat1}
        common_cat = np.union1d(cat1, cat2)
        positive_pairs = [{"left": [c], "right": [c], "value": default_positive_value} for c in common_cat]

    # Apply positive pairs to the dictionary
    if positive_pairs is not None:
        for p in positive_pairs:
            for l in p["left"]:
                for r in p["right"]:
                    if r in label_transfer_dict and l in label_transfer_dict[r]:
                        label_transfer_dict[r][l] = p["value"]

    # Apply negative pairs to the dictionary
    if negative_pairs is not None:
        for p in negative_pairs:
            for l in p["left"]:
                for r in p["right"]:
                    if r in label_transfer_dict and l in label_transfer_dict[r]:
                        label_transfer_dict[r][l] = p["value"]

    # Normalize the label transfer dictionary
    norm_label_transfer_dict = dict()
    for c1 in cat1:
        norm_c = np.array([label_transfer_dict[c1][c2] for c2 in cat2]).sum()
        norm_label_transfer_dict[c1] = {c2: label_transfer_dict[c1][c2] / (norm_c + 1e-8) for c2 in cat2}

    return norm_label_transfer_dict


# Finished
def get_rep(
    nx: Union[TorchBackend, NumpyBackend],
    type_as: Union[torch.Tensor, np.ndarray],
    sample: AnnData,
    rep: str = "X",
    rep_field: str = "layer",
    genes: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Get the specified representation from the AnnData object.

    Args:
        nx (module): Backend module (e.g., numpy or torch).
        type_as (type): Type to which the output should be cast.
        sample (AnnData): The AnnData object containing the sample data.
        rep (str, optional): The name of the representation to retrieve. Defaults to "X".
        rep_field (str, optional): The type of representation. Acceptable values are "layer", "obs" and "obsm". Defaults to "layer".
        genes (Optional[Union[list, np.ndarray]], optional): List of genes to filter if `rep_field` is "layer". Defaults to None.

    Returns:
        Union[np.ndarray, torch.Tensor]: The requested representation from the AnnData object, cast to the specified type.

    Raises:
        ValueError: If `rep_field` is not one of the expected values.
        KeyError: If the specified representation is not found in the AnnData object.
    """

    # gene expression stored in ".layer" field
    if rep_field == "layer":
        representation = nx.from_numpy(check_exp(sample=sample[:, genes], layer=rep), type_as=type_as)

    # label information stored in ".obs" field
    elif rep_field == "obs":
        # Sort categories and convert to integer codes
        representation = np.array(sample.obs[rep].cat.codes.values, dtype=np.int32)
        representation = nx.from_numpy(representation)
        if nx_torch(nx):
            representation = representation.to(type_as.device)

    # scalar values stored in ".obsm" field
    elif rep_field == "obsm":
        representation = nx.from_numpy(sample.obsm[rep], type_as=type_as)
    else:
        raise ValueError("rep_field must be either 'layer', 'obsm' or 'obs'")

    return representation


######################
# Data preprocessing #
######################

# Finished
def filter_common_genes(*genes, verbose: bool = True) -> list:
    """
    Filters for the intersection of genes between all samples.

    Args:
        genes: List of genes.
        verbose: If ``True``, print progress updates.
    """

    common_genes = intersect_lsts(*genes)
    if len(common_genes) == 0:
        raise ValueError("The number of common gene between all samples is 0.")
    else:
        if verbose:
            lm.main_info(
                message=f"Filtered all samples for common genes. There are {(len(common_genes))} common genes.",
                indent_level=1,
            )
        return common_genes


# Finished
def normalize_coords(
    nx: Union[TorchBackend, NumpyBackend],
    coords: List[Union[np.ndarray, torch.Tensor]],
    verbose: bool = True,
    separate_scale: bool = True,
    separate_mean: bool = True,
) -> Tuple[
    List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]]
]:
    """
    Normalize the spatial coordinate.

    Parameters
    ----------
    coords : List[Union[np.ndarray, torch.Tensor]]
        Spatial coordinates of the samples. Each element in the list can be a numpy array or a torch tensor.
    nx : Union[TorchBackend, NumpyBackend], optional
        The backend to use for computations. Default is `NumpyBackend`.
    verbose : bool, optional
        If `True`, print progress updates. Default is `True`.
    separate_scale : bool, optional
        If `True`, normalize each coordinate axis independently. When doing the global refinement, this weill be set to False. Default is `True`.
    separate_mean : bool, optional
        If `True`, normalize each coordinate axis to have zero mean independently. When doing the global refinement, this weill be set to False. Default is `True`.

    Returns
    -------
    Tuple[List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]]]
        A tuple containing:
        - coords: List of normalized spatial coordinates.
        - normalize_scales: List of normalization scale factors applied to each coordinate axis.
        - normalize_means: List of mean values used for normalization of each coordinate axis.
    """

    D = coords[0].shape[1]
    normalize_scales = nx.zeros((len(coords),), type_as=coords[0])
    normalize_means = nx.zeros((len(coords), D), type_as=coords[0])

    # get the means for each coords
    for i in range(len(coords)):
        normalize_mean = nx.einsum("ij->j", coords[i]) / coords[i].shape[0]
        normalize_means[i] = normalize_mean

    # get the global means for whole coords if "separate_mean" is True
    if not separate_mean:
        global_mean = nx.mean(normalize_means, axis=0)
        normalize_means = nx.full((len(coords), D), global_mean)

    # move each coords to zero center and calculate the normalization scale
    for i in range(len(coords)):
        coords[i] -= normalize_means[i]
        normalize_scale = nx.sqrt(nx.einsum("ij->", nx.einsum("ij,ij->ij", coords[i], coords[i])) / coords[i].shape[0])
        normalize_scales[i] = normalize_scale

    # get the global scale for whole coords if "separate_scale" is True
    if not separate_scale:
        global_scale = nx.mean(normalize_scales)
        normalize_scales = nx.full((len(coords),), global_scale)

    # normalize the scale of the coords
    for i in range(len(coords)):
        coords[i] /= normalize_scales[i]

    # show the normalization results if "verbose" is True
    if verbose:
        lm.main_info(message=f"Spatial coordinates normalization params:", indent_level=1)
        lm.main_info(message=f"Scale: {normalize_scales[:2]}...", indent_level=2)
        lm.main_info(message=f"Scale: {normalize_means[:2]}...", indent_level=2)
    return coords, normalize_scales, normalize_means


# Finished
def normalize_exps(
    nx: Union[TorchBackend, NumpyBackend],
    exp_layers: List[List[Union[np.ndarray, torch.Tensor]]],
    rep_field: Union[str, List[str]] = "layer",
    verbose: bool = True,
) -> List[List[Union[np.ndarray, torch.Tensor]]]:
    """
    Normalize the gene expression matrices.

    Args:
        nx (Union[TorchBackend, NumpyBackend], optional):
            The backend to use for computations. Defaults to `NumpyBackend`.
        exp_layers (List[List[Union[np.ndarray, torch.Tensor]]]):
            Gene expression and optionally the representation matrices of the samples.
            Each element in the list can be a numpy array or a torch tensor.
        rep_field (Union[str, List[str]], optional):
            Field(s) indicating the type of representation. If 'layer', normalization can be applied.
            Defaults to "layer".
        verbose (bool, optional):
            If `True`, print progress updates. Default is `True`.

    Returns:
        List[List[Union[np.ndarray, torch.Tensor]]]:
            A list of lists containing normalized gene expression matrices.
            Each matrix in the list is a numpy array or a torch tensor.
    """

    if isinstance(rep_field, str):
        rep_field = [rep_field] * len(exp_layers[0])

    for i, rep_f in enumerate(rep_field):
        if rep_f == "layer":
            normalize_scale = 0

            # Calculate the normalization scale
            for l in range(len(exp_layers)):
                normalize_scale += nx.sqrt(
                    nx.einsum("ij->", nx.einsum("ij,ij->ij", exp_layers[i][l], exp_layers[i][l]))
                    / exp_layers[i][l].shape[0]
                )

            normalize_scale /= len(exp_layers)

            # Apply the normalization scale
            for i in range(len(exp_layers)):
                exp_layers[i][l] /= normalize_scale

            if verbose:
                lm.main_info(message=f"Gene expression normalization params:", indent_level=1)
                lm.main_info(message=f"Scale: {normalize_scale}.", indent_level=2)

    return exp_layers


##############################################
# Calculate  dissimilarity / distance matrix #
##############################################


def _kl_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    probabilistic: bool = True,
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the pairwise KL divergence between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    probabilistic : bool, optional
        If True, normalize the rows of X and Y to sum to 1 (to interpret them as probabilities).
        Default is True.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray
        Pairwise KL divergence matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = get_backend(X, Y)
    X = X + 0.01
    Y = Y + 0.01
    # Normalize rows to sum to 1 if probabilistic is True
    if probabilistic:
        X = X / nx.sum(X, axis=1, keepdims=True)
        Y = Y / nx.sum(Y, axis=1, keepdims=True)

    # Compute log of X and Y
    log_X = nx.log(X + eps)  # Adding epsilon to avoid log(0)
    log_Y = nx.log(Y + eps)  # Adding epsilon to avoid log(0)

    # Compute X log X and the pairwise KL divergence
    X_log_X = nx.sum(X * log_X, axis=1, keepdims=True)

    D = X_log_X - nx.dot(X, log_Y.T)

    return D


def _cosine_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the pairwise cosine similarity between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise cosine similarity matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = get_backend(X, Y)

    # Normalize rows to unit vectors
    X_norm = nx.sqrt(nx.sum(X**2, axis=1, keepdims=True))
    Y_norm = nx.sqrt(nx.sum(Y**2, axis=1, keepdims=True))
    X = X / nx.maximum(X_norm, eps)
    Y = Y / nx.maximum(Y_norm, eps)

    # Compute cosine similarity
    D = -nx.dot(X, Y.T) * 0.5 + 0.5

    return D


def _euc_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    squared: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the pairwise Euclidean distance between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    squared : bool, optional
        If True, return squared Euclidean distances. Default is True.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise Euclidean distance matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = get_backend(X, Y)

    D = nx.sum(X**2, 1)[:, None] + nx.sum(Y**2, 1)[None, :] - 2 * nx.dot(X, Y.T)

    # Ensure non-negative distances (can arise due to floating point arithmetic)
    D = nx.maximum(D, 0.0)

    if not squared:
        D = nx.sqrt(D)

    return D


def _label_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    label_transfer: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a matrix of size (N, M) by indexing into the label_transfer matrix using the values in X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Array with shape (N, ) containing integer values ranging from 0 to K.
    Y : np.ndarray or torch.Tensor
        Array with shape (M, ) containing integer values ranging from 0 to L.
    label_transfer : np.ndarray or torch.Tensor
        Matrix with shape (K, L) containing the label transfer cost.

    Returns
    -------
    np.ndarray or torch.Tensor
        Matrix with shape (N, M) where each element is the value from label_transfer indexed by the corresponding values in X and Y.

    Raises
    ------
    AssertionError
        If the shape of X or Y is not one-dimensional or if they contain non-integer values.
    """
    assert X.ndim == 1, "X should be a 1-dimensional array."
    assert Y.ndim == 1, "Y should be a 1-dimensional array."

    nx = get_backend(X, Y, label_transfer)

    if nx_torch(nx):
        assert not (torch.is_floating_point(X) or torch.is_floating_point(Y)), "X and Y should contain integer values."
    else:
        assert np.issubdtype(X.dtype, np.integer) and np.issubdtype(
            X.dtype, np.integer
        ), "X should contain integer values."

    D = label_transfer[X, :][:, Y]

    return D


# TODO: finish these


def _correlation_distance_backend(X, Y):
    pass


def _jaccard_distance_backend(X, Y):
    pass


def _chebyshev_distance_backend(X, Y):
    pass


def _canberra_distance_backend(X, Y):
    pass


def _braycurtis_distance_backend(X, Y):
    pass


def _hamming_distance_backend(X, Y):
    pass


def _minkowski_distance_backend(X, Y):
    pass


def calc_distance(
    X: Union[List[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]],
    Y: Union[List[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]],
    metric: Union[List[str], str] = "euc",
    label_transfer: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the distance between all pairs of samples in matrices X and Y using the specified metric.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    metric : str, optional
        The metric to use for calculating distances. Options are 'euc', 'euclidean', 'square_euc', 'square_euclidean',
        'kl', 'sym_kl', 'cos', 'cosine', 'label'. Default is 'euc'.
    label_transfer : Optional[np.ndarray or torch.Tensor], optional
        Matrix with shape (K, L) containing the label transfer cost. Required if metric is 'label'. Default is None.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise distance matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
        If `metric` is not one of the supported metrics.
        If `label_transfer` is required but not provided.
    """

    if not isinstance(X, list):
        X = [X]
    if not isinstance(Y, list):
        Y = [Y]
    if not isinstance(metric, list):
        metric = [metric]
    dist_mats = []
    for (x, y, m) in zip(X, Y, metric):
        if m == "label":
            assert label_transfer is not None, "label_transfer must be provided for metric 'label'."
            dist_mats.append(_label_distance_backend(x, y, label_transfer))
        elif m in ["euc", "euclidean"]:
            dist_mats.append(_euc_distance_backend(x, y, squared=True))
        elif m in ["square_euc", "square_euclidean"]:
            dist_mats.append(_euc_distance_backend(x, y, squared=False))
        elif m == "kl":
            dist_mats.append(
                _kl_distance_backend(
                    x,
                    y,
                )
            )
        elif m == "sym_kl":
            dist_mats.append(
                (
                    _kl_distance_backend(
                        x,
                        y,
                    )
                    + _kl_distance_backend(y, x).T
                )
                / 2
            )
        elif m in ["cos", "cosine"]:
            dist_mats.append(
                _cosine_distance_backend(
                    x,
                    y,
                )
            )

    return dist_mats


def calc_probability(
    nx,
    distance_matrix: Union[np.ndarray, torch.Tensor],
    probability_type: str = "gauss",
    probability_parameter: Optional[float] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate probability based on the distance matrix and specified probability type.

    Parameters
    ----------
    distance_matrix : np.ndarray or torch.Tensor
        The distance matrix.
    probability_type : str, optional
        The type of probability to calculate. Options are 'Gauss', 'cos_prob', and 'prob'. Default is 'Gauss'.
    probability_parameter : Optional[float], optional
        The parameter for the probability calculation. Required for certain probability types. Default is None.

    Returns
    -------
    np.ndarray or torch.Tensor
        The calculated probability matrix.

    Raises
    ------
    ValueError
        If `probability_type` is not one of the supported types or if required parameters are missing.
    """

    # Get the appropriate backend (either NumPy or PyTorch)
    if probability_type.lower() in ["gauss", "gaussian"]:
        if probability_parameter is None:
            raise ValueError("probability_parameter must be provided for 'Gauss' probability type.")
        probability = nx.exp(-distance_matrix / (2 * probability_parameter))
    elif probability_type.lower() in ["cos", "cosine"]:
        probability = 1 - distance_matrix
    elif probability_type.lower() == "prob":
        probability = distance_matrix
    else:
        raise ValueError(f"Unsupported probability type: {probability_type}")

    return probability


###############################
# Calculate assignment matrix #
###############################


def get_P_core(
    nx: Union[TorchBackend, NumpyBackend],
    type_as: Union[torch.Tensor, np.ndarray],
    Dim: Union[torch.Tensor, np.ndarray],
    spatial_dist: Union[np.ndarray, torch.Tensor],
    exp_dist: List[Union[np.ndarray, torch.Tensor]],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    model_mul: Union[np.ndarray, torch.Tensor],
    gamma: Union[float, np.ndarray, torch.Tensor],
    samples_s: Optional[List[float]] = None,
    sigma2_variance: float = 1,
    probability_type: Union[str, List[str]] = "Gauss",
    probability_parameters: Optional[List] = None,
    eps: float = 1e-8,
    sparse_calculation_mode: bool = False,
    top_k: int = -1,
):
    """
    Compute assignment matrix P and additional results based on given distances and parameters.

    Parameters
    ----------
    nx : module
        Backend module (e.g., numpy or torch).
    type_as : type
        Type to which the output should be cast.
    spatial_dist : np.ndarray or torch.Tensor
        Spatial distance matrix.
    exp_dist : List[np.ndarray or torch.Tensor]
        List of expression distance matrices.
    sigma2 : int, float, np.ndarray or torch.Tensor
        Sigma squared value.
    alpha : np.ndarray or torch.Tensor
        Alpha values.
    gamma : float, np.ndarray or torch.Tensor
        Gamma value.
    Sigma : np.ndarray or torch.Tensor
        Sigma values.
    samples_s : Optional[List[float]], optional
        Samples. Default is None.
    sigma2_variance : float, optional
        Sigma squared variance. Default is 1.
    probability_type : Union[str, List[str]], optional
        Probability type. Default is 'Gauss'.
    probability_parameters : Optional[List[float]], optional
        Probability parameters. Default is None.

    Returns
    -------
    np.ndarray or torch.Tensor
        Assignment matrix P.
    dict
        Additional results.
    """

    # Calculate spatial probability with sigma2_variance
    spatial_prob = calc_probability(nx, spatial_dist, "gauss", probability_parameter=sigma2 / sigma2_variance)  # N x M
    # TODO: everytime this will generate D/2 on GPU, may influence the runtime
    outlier_s = samples_s * spatial_dist.shape[0]
    # outlier_s = samples_s
    spatial_outlier = _power(nx)((2 * _pi(nx) * sigma2), Dim / 2) * (1 - gamma) / (gamma * outlier_s)  # scalar
    # TODO: the position of the following is unclear
    spatial_inlier = 1 - spatial_outlier / (spatial_outlier + nx.sum(spatial_prob, axis=0, keepdims=True))  # 1 x M
    spatial_prob = spatial_prob * model_mul

    # spatial P
    P = spatial_prob / (spatial_outlier + nx.sum(spatial_prob, axis=0, keepdims=True))  # N x M
    K_NA_spatial = P.sum(1)

    # Calculate spatial probability without sigma2_variance
    spatial_prob = calc_probability(
        nx,
        spatial_dist,
        "gauss",
        probability_parameter=sigma2,
    )  # N x M

    spatial_prob = spatial_prob * model_mul

    # sigma2 P
    P = spatial_inlier * spatial_prob / (nx.sum(spatial_prob, axis=0, keepdims=True) + eps)
    K_NA_sigma2 = P.sum(1)
    sigma2_related = (P * spatial_dist).sum()

    # Calculate probabilities for expression distances
    if probability_parameters is None:
        probability_parameters = [None] * len(exp_dist)
    for e_d, p_t, p_p in zip(exp_dist, probability_type, probability_parameters):
        spatial_prob *= calc_probability(nx, e_d, p_t, p_p)

    P = spatial_inlier * spatial_prob / (nx.sum(spatial_prob, axis=0, keepdims=True) + eps)

    if sparse_calculation_mode:
        P = _dense_to_sparse(
            nx=nx,
            type_as=type_as,
            mat=P,
            sparse_method="topk",
            threshold=top_k,
            axis=0,
            descending=True,
        )
    # print(P.sum())
    return P, K_NA_spatial, K_NA_sigma2, sigma2_related


def solve_RT_by_correspondence(
    X: np.ndarray,
    Y: np.ndarray,
    return_s=False,
):
    # if len(X.shape) == 3:
    #     X = X[:, :2]
    # if len(Y.shape) == 3:
    #     Y = Y[:, :2]
    D = X.shape[1]
    N = X.shape[0]
    # find R and t that minimize the distance between spatial1 and spatial2

    tX = np.mean(X, axis=0)
    tY = np.mean(Y, axis=0)
    X = X - tX
    Y = Y - tY
    H = np.dot(Y.T, X)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    t = np.mean(X, axis=0) - np.mean(Y, axis=0) + tX - np.dot(tY, R.T)
    s = np.trace(np.dot(X.T, X) - np.dot(R.T, np.dot(Y.T, X))) / np.trace(np.dot(Y.T, Y))
    if return_s:
        return R, t, s
    else:
        return R, t


#################################
# Kernel construction functions #
#################################


def con_K(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    beta: Union[int, float] = 0.01,
) -> Union[np.ndarray, torch.Tensor]:
    """con_K constructs the Squared Exponential (SE) kernel, where K(i,j)=k(X_i,Y_j)=exp(-beta*||X_i-Y_j||^2).

    Args:
        X: The first vector X\in\mathbb{R}^{N\times d}
        Y: The second vector X\in\mathbb{R}^{M\times d}
        beta: The length-scale of the SE kernel.
        use_chunk (bool, optional): Whether to use chunk to reduce the GPU memory usage. Note that if set to ``True'' it will slow down the calculation. Defaults to False.

    Returns:
        K: The kernel K\in\mathbb{R}^{N\times M}
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = get_backend(X, Y)

    [K] = calc_distance(
        X=X,
        Y=Y,
        metric="euc",
    )
    K = nx.exp(-beta * K)
    return K


def construct_knn_graph(
    points: Union[np.ndarray, torch.Tensor],
    knn: int = 10,
):
    """
    Construct a k-nearest neighbor graph from the given points.

    Args:
        points: The points to construct the graph from.
        knn: The number of nearest neighbors to consider.

    Returns:
        The networks graph object.
    """
    nx = get_backend(points)
    if nx_torch(nx):
        points = points.cpu().numpy()
    A = kneighbors_graph(points, knn, mode="distance", include_self=False)
    A = A.toarray()

    graph = networkx.Graph()
    for i in range(points.shape[0]):
        for j, connected in enumerate(A[i]):
            if connected:
                graph.add_edge(i, j, weight=connected)

    return graph


def con_K_graph(
    graph: networkx.Graph,
    inducing_idx: Union[np.ndarray, torch.Tensor],
    beta: Union[int, float] = 0.01,
):
    """
    Construct the kernel matrix from the given graph and inducing points.

    Args:
        graph: The graph object.
        inducing_idx: The indices of the inducing points.

    Returns:
        The kernel matrix.
    """
    nx = get_backend(inducing_idx)
    D = 1e5 * nx.ones((graph.number_of_nodes(), inducing_idx.shape[0]), type_as=inducing_idx)
    inducing_idx = nx.to_numpy(inducing_idx)

    for i in range(inducing_idx.shape[0]):
        distance, path = networkx.single_source_dijkstra(graph, source=inducing_idx[i], weight="weight")
        for j in range(graph.number_of_nodes()):
            try:
                D[j, i] = distance[j]
            except KeyError:
                pass
    K = nx.exp(-beta * D**2)
    return K


def inlier_from_NN(
    train_x,
    train_y,
    distance,
):
    N, D = train_x.shape[0], train_x.shape[1]
    alpha = 1
    distance = np.maximum(0, distance)
    normalize = np.max(distance) / (np.log(10) * 2)
    distance = distance / (normalize)
    R = np.eye(D)
    t = np.ones((D, 1))
    y_hat = train_x
    sigma2 = np.sum((y_hat - train_y) ** 2) / (D * N)
    weight = np.exp(-distance * alpha)
    init_weight = weight
    P = np.multiply(np.ones((N, 1)), weight)
    max_iter = 100
    alpha_end = 0.1
    alpha_decrease = np.power(alpha_end / alpha, 1 / (max_iter - 20))
    gamma = 0.5
    a = np.maximum(
        np.prod(np.max(train_x, axis=0) - np.min(train_x, axis=0)),
        np.prod(np.max(train_y, axis=0) - np.min(train_y, axis=0)),
    )
    Sp = np.sum(P)
    for iter in range(max_iter):
        # solve rigid transformation
        mu_x = np.sum(np.multiply(train_x, P), 0) / (Sp)
        mu_y = np.sum(np.multiply(train_y, P), 0) / (Sp)

        X_mu, Y_mu = train_x - mu_x, train_y - mu_y
        A = np.dot(Y_mu.T, np.multiply(X_mu, P))
        svdU, svdS, svdV = np.linalg.svd(A)
        C = np.eye(D)
        C[-1, -1] = np.linalg.det(np.dot(svdU, svdV))
        R = np.dot(np.dot(svdU, C), svdV)
        t = mu_y - np.dot(mu_x, R.T)
        y_hat = np.dot(train_x, R.T) + t
        # get P
        term1 = np.multiply(np.exp(-(np.sum((train_y - y_hat) ** 2, 1, keepdims=True)) / (2 * sigma2)), weight)
        outlier_part = np.max(weight) * (1 - gamma) * np.power((2 * np.pi * sigma2), D / 2) / (gamma * a)
        P = term1 / (term1 + outlier_part)
        Sp = np.sum(P)
        gamma = np.minimum(np.maximum(Sp / N, 0.01), 0.99)
        P = np.maximum(P, 1e-6)

        # update sigma2
        sigma2 = np.sum(np.multiply((y_hat - train_y) ** 2, P)) / (D * Sp)
        if iter > 20:
            alpha = alpha * alpha_decrease
            weight = np.exp(-distance * alpha)
            weight = weight / np.max(weight)

    fix_sigma2 = 1e-2
    fix_gamma = 0.1
    term1 = np.multiply(np.exp(-(np.sum((train_y - y_hat) ** 2, 1, keepdims=True)) / (2 * fix_sigma2)), weight)
    outlier_part = np.max(weight) * (1 - fix_gamma) * np.power((2 * np.pi * fix_sigma2), D / 2) / (fix_gamma * a)
    P = term1 / (term1 + outlier_part)
    gamma = np.minimum(np.maximum(np.sum(P) / N, 0.01), 0.99)
    return P, R, t, init_weight, sigma2, gamma


def voxel_data(
    nx: Union[TorchBackend, NumpyBackend],
    coords: Union[np.ndarray, torch.Tensor],
    gene_exp: Union[np.ndarray, torch.Tensor],
    voxel_size: Optional[float] = None,
    voxel_num: Optional[int] = 10000,
):
    """
    Voxelization of the data.
    Parameters
    ----------
    coords: np.ndarray or torch.Tensor
        The coordinates of the data points.
    gene_exp: np.ndarray or torch.Tensor
        The gene expression of the data points.
    voxel_size: float
        The size of the voxel.
    voxel_num: int
        The number of voxels.
    Returns
    -------
    voxel_coords: np.ndarray or torch.Tensor
        The coordinates of the voxels.
    voxel_gene_exp: np.ndarray or torch.Tensor
        The gene expression of the voxels.
    """
    # nx = get_backend(coords, gene_exp)
    N, D = coords.shape[0], coords.shape[1]
    coords = nx.to_numpy(coords)
    gene_exp = nx.to_numpy(gene_exp)

    # create the voxel grid
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    if voxel_size is None:
        voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)
    voxel_steps = (max_coords - min_coords) / int(np.sqrt(voxel_num))
    voxel_coords = [
        np.arange(min_coord, max_coord, voxel_step)
        for min_coord, max_coord, voxel_step in zip(min_coords, max_coords, voxel_steps)
    ]
    voxel_coords = np.stack(np.meshgrid(*voxel_coords), axis=-1).reshape(-1, D)
    voxel_gene_exps = np.zeros((voxel_coords.shape[0], gene_exp.shape[1]))
    is_voxels = np.zeros((voxel_coords.shape[0],))
    # assign the data points to the voxels
    for i, voxel_coord in enumerate(voxel_coords):
        dists = np.sqrt(np.sum((coords - voxel_coord) ** 2, axis=1))
        mask = dists < voxel_size / 2
        if np.any(mask):
            voxel_gene_exps[i] = np.mean(gene_exp[mask], axis=0)
            is_voxels[i] = 1
    voxel_coords = voxel_coords[is_voxels == 1, :]
    voxel_gene_exps = voxel_gene_exps[is_voxels == 1, :]
    return voxel_coords, voxel_gene_exps


def _init_guess_sigma2(
    XA,
    XB,
    subsample=20000,
):
    NA, NB, D = XA.shape[0], XB.shape[0], XA.shape[1]
    sub_sample_A = np.random.choice(NA, subsample, replace=False) if NA > subsample else np.arange(NA)
    sub_sample_B = np.random.choice(NB, subsample, replace=False) if NB > subsample else np.arange(NB)
    [SpatialDistMat] = calc_distance(
        X=XA[sub_sample_A, :],
        Y=XB[sub_sample_B, :],
        metric="euc",
    )
    SpatialDistMat = SpatialDistMat**2
    sigma2 = SpatialDistMat.sum() / (D * sub_sample_A.shape[0] * sub_sample_A.shape[0])  # 2 for 3D
    return sigma2


def _get_anneling_factor(
    nx,
    type_as,
    start,
    end,
    iter,
):
    anneling_factor = _power(nx)(_data(nx, end / start, type_as=type_as), 1 / (iter))
    return anneling_factor


## Sparse operation
def _dense_to_sparse(
    nx,
    type_as,
    mat: Union[np.ndarray, torch.Tensor],
    sparse_method: str = "topk",
    threshold: Union[int, float] = 100,
    axis: int = 0,
    descending=False,
):
    assert sparse_method in [
        "topk",
        "threshold",
    ], "``sparse_method`` value is wrong. Available ``sparse_method`` are: ``'topk'`` and ``'threshold'``."
    threshold = int(threshold) if sparse_method == "topk" else threshold
    nx = get_backend(mat)
    NA, NB = mat.shape[0], mat.shape[1]
    if sparse_method == "topk":
        sorted_mat, sorted_idx = _sort(nx, mat, axis=axis, descending=descending)
        if axis == 0:
            if threshold > NA:
                threshold = NA
            col = _repeat_interleave(nx, nx.arange(NB, type_as=mat), threshold, axis=0)
            row = sorted_idx[:threshold, :].T.reshape(-1)
            val = sorted_mat[:threshold, :].T.reshape(-1)

        elif axis == 1:
            if threshold > NB:
                threshold = NB
            col = sorted_idx[:, :threshold].reshape(-1)
            row = _repeat_interleave(nx, nx.arange(NA, type_as=mat), threshold, axis=0)
            val = sorted_mat[:, :threshold].reshape(-1)
    elif sparse_method == "threshold":
        row, col = _where(nx, mat < threshold)
        val = mat[row, col]
    results = _SparseTensor(nx=nx, row=row, col=col, value=val, sparse_sizes=(NA, NB))
    return results


#################################
# Funcs between Numpy and Torch #
#################################


# Empty cache
def empty_cache(device: str = "cpu"):
    if device != "cpu":
        torch.cuda.empty_cache()


# Check if nx is a torch backend
nx_torch = lambda nx: True if isinstance(nx, TorchBackend) else False

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

_copy = lambda nx, data: data.clone() if nx_torch(nx) else data.copy()


def _sort(nx, arr, axis=-1, descending=False):
    if not descending:
        sorted_arr, sorted_idx = nx.sort2(arr, axis=axis)
    else:
        sorted_arr, sorted_idx = nx.sort2(-arr, axis=axis)
        sorted_arr = -sorted_arr
    return sorted_arr, sorted_idx


def _SparseTensor(nx, row, col, value, sparse_sizes):
    if nx_torch(nx):
        return SparseTensor(indices=torch.vstack((row, col)), values=value, size=sparse_sizes)
    else:
        return sp.coo_matrix((value, (row, col)), shape=sparse_sizes)


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
