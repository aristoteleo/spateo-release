import os
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from numpy import ndarray
from scipy.linalg import pinv
from scipy.sparse import issparse
from scipy.special import psi

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
        backend = ot.backend.NumpyBackend()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if torch.cuda.is_available():
            torch.cuda.init()
            backend = ot.backend.TorchBackend()
        else:
            backend = ot.backend.NumpyBackend()
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


def check_spatial_coords(sample: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Check spatial coordinate information.

    Args:
        sample: An anndata object.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.

    Returns:
        The spatial coordinates.
    """
    coordinates = sample.obsm[spatial_key].copy()
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.values

    return np.asarray(coordinates)


def check_exp(sample: AnnData, layer: str = "X") -> np.ndarray:
    """
    Check expression matrix.

    Args:
        sample: An anndata object.
        layer: The key in `.layers` that corresponds to the expression matrix.

    Returns:
        The expression matrix.
    """

    exp_martix = sample.X.copy() if layer == "X" else sample.layers[layer].copy()
    exp_martix = to_dense_matrix(exp_martix)
    return exp_martix

def check_obs(use_rep: List[str], rep_type: List[str]) -> Optional[str]:
    """
    Check that the number of occurrences of 'obs' in the list of strings is no more than one.

    Parameters
    ----------
    use_rep : List[str]
        A list of representations to check.
    rep_type : List[str]
        A list of representation types corresponding to the representations in `use_rep`.

    Returns
    -------
    Optional[str]
        The representation key if 'obs' occurs once, otherwise None.

    Raises
    ------
    ValueError
        If 'obs' occurs more than once in the list.
    """
    for i, s in enumerate(rep_type):
        if s == 'obs':
            count += 1
            position = i
            if count > 1:
                raise ValueError(f"'obs' occurs more than once in the list. Currently Spateo only support one label consistency.")
            
    # return the obs key
    if count == 1:
        return use_rep[position]
    else:
        return None


def check_use_rep(samples: List[AnnData], use_rep: Union[str, list], rep_type: Optional[Union[str, list]] = None,) -> bool:
    """
    Check if `use_rep` exists in the `.obsm` or `.obs` attributes of AnnData objects based on `rep_type`.

    Parameters
    ----------
    samples : List[AnnData]
        A list of AnnData objects containing the data samples.
    use_rep : Union[str, list]
        The representation to check.
    rep_type : str
        The type of representation. Accept types: "obsm" and "obs".

    Returns
    -------
    bool
        True if `use_rep` exists in the specified attribute of all AnnData objects, False otherwise.
    """

    if rep_type is None:
        rep_type = "obsm"

    if isinstance(use_rep, str):
        use_rep = [use_rep]

    if isinstance(rep_type, str):
        rep_type = [rep_type] * len(use_rep)

    for sample in samples:
        for rep, rep_t in zip(use_rep, rep_type):
            if rep_t == "obsm":
                if rep not in sample.obsm:
                    raise ValueError(f"The specified representation '{rep}' not found in the '{rep_t}' attribute of some of the AnnData objects.")
                    return False
            elif rep_t == "obs":
                if rep not in sample.obs:
                    raise ValueError(f"The specified representation '{rep}' not found in the '{rep_t}' attribute of some of the AnnData objects.")
                    return False
                
                # judge if the sample.obs[rep] is categorical
                if not isinstance(sample.obs[rep].dtype, pd.CategoricalDtype):
                    raise ValueError(f"The specified representation '{rep}' found in the '{rep_t}' attribute should be categorical.")
                    return False
            else:
                raise ValueError("rep_type must be either 'obsm' or 'obs'")
    return True

# TODO: check the label in samples is consistency to the label_transfer_dict
def check_label_transfer(
    nx,
    type_as,
    samples: List[AnnData], 
    obs_key: str, 
    label_transfer_dict: Optional[List[Dict[str, Dict[str, float]]]] = None
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Check and generate label transfer matrices for the given samples.

    Parameters
    ----------
    nx : module
        Backend module (e.g., numpy or torch).
    type_as : type
        Type to which the output should be cast.
    samples : List[AnnData]
        List of AnnData objects containing the samples.
    obs_key : str
        The key in `.obs` that corresponds to the labels.
    label_transfer_dict : Optional[List[Dict[str, Dict[str, float]]]], optional
        List of dictionaries defining the label transfer cost between categories of each pair of samples.

    Returns
    -------
    List[Union[np.ndarray, torch.Tensor]]
        List of label transfer matrices, each as either a NumPy array or torch Tensor.
    
    Raises
    ------
    ValueError
        If the length of `label_transfer_dict` does not match `len(samples) - 1`.
    """

    if label_transfer_dict is not None:
        if isinstance(label_transfer_dict, dict):
            label_transfer_dict = [label_transfer_dict]
        if isinstance(label_transfer_dict, list):
            if len(label_transfer_dict) != (len(samples) - 1):
                raise ValueError("The length of label_transfer_dict must be equal to len(samples) - 1.")
        else:
            raise ValueError("label_transfer_dict should be a list or a dictionary.")

    label_transfer = []
    for i in range(len(samples)-1):
        cat1 = samples[i].obs[obs_key].cat.categories.tolist()
        cat2 = samples[i+1].obs[obs_key].cat.categories.tolist()
        cur_label_transfer = np.zeros(len(cat1), len(cat2), dtype=np.float32)

        if label_transfer_dict is not None:
            cur_label_transfer_dict = label_transfer_dict[i]
        else:
            cur_label_transfer_dict = generate_label_transfer_dict(cat1, cat2)

        for j, c2 in enumerate(cat2):
            for k, c1 in enumerate(cat1):
                cur_label_transfer[j, k] = cur_label_transfer_dict[c2][c1]
        label_transfer.append(nx.from_numpy(cur_label_transfer, type_as=type_as))

    return label_transfer


def generate_label_transfer_dict(
    cat1: List[str], 
    cat2: List[str], 
    positive_pairs: Optional[List[Dict[str, Union[List[str], float]]]] = None, 
    negative_pairs: Optional[List[Dict[str, Union[List[str], float]]]] = None,
    default_positve_value: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """
    Generate a label transfer dictionary with normalized values.

    Parameters
    ----------
    cat1 : List[str]
        List of categories from the first dataset.
    cat2 : List[str]
        List of categories from the second dataset.
    positive_pairs : Optional[List[Dict[str, Union[List[str], float]]]], optional
        List of positive pairs with transfer values. Each dictionary should have 'left', 'right', and 'value' keys. Default is None.
    negative_pairs : Optional[List[Dict[str, Union[List[str], float]]]], optional
        List of negative pairs with transfer values. Each dictionary should have 'left', 'right', and 'value' keys. Default is None.
    default_positive_value : float, optional
        Default value for positive pairs if positive pairs are "None". Default is 10.0.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A normalized label transfer dictionary.
    """

    # Initialize label transfer dictionary with default values
    label_transfer_dict = {c2: {c1: 1.0 for c1 in cat1} for c2 in cat2}

    # Generate default positive pairs if none provided
    if (positive_pairs is None) and (negative_pairs is None):
        common_cat = np.union1d(cat1, cat2)
        positive_pairs = [{'left': [c], 'right': [c], 'value': default_positve_value} for c in common_cat]

    # Apply positive pairs to the dictionary
    if positive_pairs is not None:
        for p in positive_pairs:
            for l in p['left']:
                for r in p['right']:
                    if r in label_transfer_dict and l in label_transfer_dict[r]:
                        label_transfer_dict[r][l] = p['value']

    # Apply negative pairs to the dictionary   
    if negative_pairs is not None:  
        for p in negative_pairs:
            for l in p['left']:
                for r in p['right']:
                    if r in label_transfer_dict and l in label_transfer_dict[r]:
                        label_transfer_dict[r][l] = p['value']

    # Normalize the label transfer dictionary
    norm_label_transfer_dict = dict()
    for c2 in cat2:
        norm_c = np.array([label_transfer_dict[c2][c1] for c1 in cat1]).sum()
        norm_label_transfer_dict[c2] = {c1: label_transfer_dict[c2][c1] / norm_c for c1 in cat1}

    return norm_label_transfer_dict



def get_rep(
    nx,
    type_as,
    sample: AnnData, 
    rep: str = "label",
    rep_type: str = "obsm",
    
) -> np.ndarray:
    """
    Get the specified representation from the AnnData object.

    Parameters
    ----------
    nx : module
        Backend module (e.g., numpy or torch).
    type_as : type
        Type to which the output should be cast.
    sample : AnnData
        The AnnData object containing the sample data.
    rep : str, optional
        The name of the representation to retrieve. Default is "label".
    rep_type : str, optional
        The type of representation. Accept types: "obs" and "obsm". Default is "obsm".

    Returns
    -------
    np.ndarray or torch.Tensor
        The requested representation from the AnnData object, cast to the specified type.

    Raises
    ------
    ValueError
        If `rep_type` is not one of the expected values.
    KeyError
        If the specified representation is not found in the AnnData object.
    """

    # label information stored in ".obs" field
    # TODO: currently we suppose the label corresponds to the label transfer. We can extent in the future
    if rep_type == "obs":
        # Sort categories and convert to integer codes
        representation = sample.obs[rep].cat.codes.values
        representation = nx.from_numpy(representation)
        if nx_torch(nx):
            representation = representation.to(type_as.device)

    # scalar values stored in ".obsm" field
    elif rep_type == "obsm":
        representation = nx.from_numpy(sample.obsm[rep], type_as=type_as)
    else:
        raise ValueError("rep_type must be either 'obsm' or 'obs'")

    return representation


######################
# Data preprocessing #
######################


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


def normalize_coords(
    coords: List[Union[np.ndarray, torch.Tensor]],
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend] = ot.backend.NumpyBackend,
    verbose: bool = True,
    separate_scale: bool = True,
    separate_mean: bool = True,
) -> Tuple[List[np.ndarray, torch.Tensor], List[np.ndarray, torch.Tensor], List[np.ndarray, torch.Tensor]]:
    """
    Normalize the spatial coordinate.

    Parameters
    ----------
    coords : List[Union[np.ndarray, torch.Tensor]]
        Spatial coordinates of the samples. Each element in the list can be a numpy array or a torch tensor.
    nx : Union[ot.backend.TorchBackend, ot.backend.NumpyBackend], optional
        The backend to use for computations. Default is `ot.backend.NumpyBackend`.
    verbose : bool, optional
        If `True`, print progress updates. Default is `True`.
    separate_scale : bool, optional
        If `True`, normalize each coordinate axis independently. When doing the global refinement, this weill be set to False. Default is `True`.
    separate_mean : bool, optional
        If `True`, normalize each coordinate axis to have zero mean independently. When doing the global refinement, this weill be set to False. Default is `True`.

    Returns
    -------
    Tuple[List[np.ndarray, torch.Tensor], List[np.ndarray, torch.Tensor], List[np.ndarray, torch.Tensor]]
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


def normalize_exps(
    exp_matrices: List[List[Union[np.ndarray, torch.Tensor]]],
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend] = ot.backend.NumpyBackend,
    verbose: bool = True,
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Normalize the gene expression matrices.

    Parameters
    ----------
    exp_matrices : List[List[Union[np.ndarray, torch.Tensor]]]
        Gene expression and optionally the representation matrices of the samples. Each element in the list can be a numpy array or a torch tensor.
    nx : Union[ot.backend.TorchBackend, ot.backend.NumpyBackend], optional
        The backend to use for computations. Default is `ot.backend.NumpyBackend`.
    verbose : bool, optional
        If `True`, print progress updates. Default is `True`.

    Returns
    -------
    List[Union[np.ndarray, torch.Tensor]]
        A list of normalized gene expression matrices. Each matrix in the list is a numpy array or a torch tensor.
    """

    normalize_scale = 0
    for i in range(len(exp_matrices)):
        # normalize_mean = nx.einsum("ij->j", exp_matrices[i][0]) / exp_matrices[i][0].shape[0]
        normalize_scale += nx.sqrt(
            nx.einsum("ij->", nx.einsum("ij,ij->ij", exp_matrices[i][0], exp_matrices[i][0])) / exp_matrices[i][0].shape[0]
        )

    normalize_scale /= len(exp_matrices)
    for i in range(len(exp_matrices)):
        exp_matrices[i][0] /= normalize_scale
    if verbose:
        lm.main_info(message=f"Gene expression normalization params:", indent_level=1)
        # lm.main_info(message=f"Mean: {normalize_mean}.", indent_level=2)
        lm.main_info(message=f"Scale: {normalize_scale}.", indent_level=2)

    return exp_matrices


def align_preprocess(
    samples: List[AnnData],
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    use_rep: Optional[Union[str, List[str]]] = None,
    rep_type: Optional[Union[str, List[str]]] = None,
    label_transfer_dict: Optional[Union[dict, List[dict]]] = None,
    normalize_c: bool = False,
    normalize_g: bool = False,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[
    Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    Union[torch.Tensor, np.ndarray],
    List[List[torch.Tensor, np.ndarray]],
    List[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
]:
    """
    Preprocess the data before alignment.

    Parameters
    ----------
    samples : List[AnnData]
        A list of AnnData objects containing the data samples.
    genes : Optional[Union[list, np.ndarray]], optional
        Genes used for calculation. If None, use all common genes for calculation. Default is None.
    spatial_key : str, optional
        The key in `.obsm` that corresponds to the raw spatial coordinates. Default is "spatial".
    layer : str, optional
        If 'X', uses `sample.X` to calculate dissimilarity between spots, otherwise uses the representation given by `sample.layers[layer]`. Default is "X".
    use_rep : Optional[Union[str, List[str]]], optional
        Specify the representation to use. If None, do not use the representation.
    rep_type : Optional[Union[str, List[str]]], optional
        Specify the type of representation. Accept types: "obs" and "obsm". If None, use the "obsm" type.
    normalize_c : bool, optional
        Whether to normalize spatial coordinates. Default is False.
    normalize_g : bool, optional
        Whether to normalize gene expression. Default is False.
    dtype : str, optional
        The floating-point number type. Only float32 and float64 are allowed. Default is "float64".
    device : str, optional
        The device used to run the program. Can specify the GPU to use, e.g., '0'. Default is "cpu".
    verbose : bool, optional
        If True, print progress updates. Default is True.

    Returns
    -------
    Tuple
        A tuple containing the following elements:
        - backend: The backend used for computations (TorchBackend or NumpyBackend).
        - type_as: The type used for computations which contains the dtype and device.
        - exp_layers: A list of processed expression layers.
        - spatial_coords: A list of spatial coordinates.
        - normalize_scales: Optional scaling factors for normalization.
        - normalize_means: Optional mean values for normalization.

    Raises
    ------
    ValueError
        If the specified representation is not found in the attributes of the AnnData objects.
    AssertionError
        If the spatial coordinate dimensions are different.
    """

    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)

    # Check if the representation is in the AnnData objects
    if use_rep is not None:
        if rep_type is None:
            rep_type = "obsm"

        if isinstance(use_rep, str):
            use_rep = [use_rep]

        if isinstance(rep_type, str):
            rep_type = [rep_type] * len(use_rep)

        if not check_use_rep(samples, use_rep, rep_type):
            raise ValueError(f"The specified representation is not found in the attribute of the AnnData objects.")

        obs_key = check_obs(use_rep, rep_type)
    

    # Get the common genes
    all_samples_genes = [s[0].var.index for s in samples]
    common_genes = filter_common_genes(*all_samples_genes, verbose=verbose)
    common_genes = common_genes if genes is None else intersect_lsts(common_genes, genes)

    # Extract the gene expression and optionaly representations of all samples
    # Each representation has a layer
    exp_layers = []
    for s in samples:
        cur_layer = []
        cur_layer.append(nx.from_numpy(check_exp(sample=s[:,common_genes], layer=layer), type_as=type_as))
        if use_rep is not None:
            for rep, rep_t in zip(use_rep, rep_type):
                cur_layer.append(get_rep(sample=s, rep=rep, rep_type=rep_t, nx=nx, type_as=type_as))
        exp_layers.append(cur_layer)

    # check the label tranfer dict and generate a matrix that contains the label transfer cost and cast to the specified type
    if obs_key is not None:
        label_transfer = check_label_transfer(nx, type_as, samples, obs_key, label_transfer_dict)
    else:
        label_transfer = None
            

    # Spatial coordinates of all samples
    spatial_coords = [
        nx.from_numpy(check_spatial_coords(sample=s, spatial_key=spatial_key), type_as=type_as) for s in samples
    ]

    # check the spatial coordinates dimensionality  
    coords_dims = nx.unique(_data(nx, [c.shape[1] for c in spatial_coords], type_as))
    assert len(coords_dims) == 1, "Spatial coordinate dimensions are different, please check again."

    
    if normalize_c:
        spatial_coords, normalize_scales, normalize_means = normalize_coords(
            coords=spatial_coords, nx=nx, verbose=verbose
        )
    else:
        normalize_scales, normalize_means = None, None
    if normalize_g:
        exp_layers = normalize_exps(matrices=exp_layers, nx=nx, verbose=verbose)

    # TODO: add docstring for return label_transfer
    return (
        nx,
        type_as,
        exp_layers,
        spatial_coords,
        label_transfer,
        normalize_scales,
        normalize_means,
    )

# TODO: update the function
def guidance_pair_preprocess(
    guidance_pair,
    normalize_scale_list,
    normalize_mean_list,
    nx,
    type_as,
):
    X_BI = nx.from_numpy(guidance_pair[0], type_as=type_as)
    X_AI = nx.from_numpy(guidance_pair[1], type_as=type_as)
    normalize_scale = normalize_scale_list[0]
    normalize_mean_ref = normalize_mean_list[0]
    normalize_mean_quary = normalize_mean_list[1]
    X_AI = (X_AI - normalize_mean_quary) / normalize_scale
    X_BI = (X_BI - normalize_mean_ref) / normalize_scale
    return [X_AI, X_BI]


# TODO: what does this function do?
# def _mask_from_label_prior(
#     adataA: AnnData,
#     adataB: AnnData,
#     label_key: Optional[str] = "cluster",
# ):
#     # check the label key
#     if label_key not in adataA.obs.keys():
#         raise ValueError(f"adataA does not have label key {label_key}.")
#     if label_key not in adataB.obs.keys():
#         raise ValueError(f"adataB does not have label key {label_key}.")
#     # get the label from anndata
#     labelA = pd.DataFrame(adataA.obs[label_key].values, columns=[label_key])
#     labelB = pd.DataFrame(adataB.obs[label_key].values, columns=[label_key])

#     # get the intersect and different label
#     cateA = labelA[label_key].astype("category").cat.categories
#     cateB = labelB[label_key].astype("category").cat.categories
#     intersect_cate = cateA.intersection(cateB)
#     cateA_unique = cateA.difference(cateB)
#     cateB_unique = cateB.difference(cateA)

#     # calculate the label mask
#     label_mask = np.zeros((len(labelA), len(labelB)), dtype="float32")
#     for cate in intersect_cate:
#         label_mask += (labelA[label_key] == cate).values[:, None] * (labelB[label_key] == cate).values[None, :]
#     for cate in cateA_unique:
#         label_mask += (labelA[label_key] == cate).values[:, None] * np.ones((1, len(labelB)))
#     for cate in cateB_unique:
#         label_mask += np.ones((len(labelA), 1)) * (labelB[label_key] == cate).values[None, :]
#     label_mask[label_mask > 0] = 1
#     return label_mask


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
    nx = ot.backend.get_backend(X, Y)

    # Normalize rows to sum to 1 if probabilistic is True
    if probabilistic:
        X = X / nx.sum(X, axis=1, keepdims=True)
        Y = Y / nx.sum(Y, axis=1, keepdims=True)

    
    # Compute log of X and Y
    log_X = nx.log(X + 1e-8)  # Adding epsilon to avoid log(0)
    log_Y = nx.log(Y + 1e-8)  # Adding epsilon to avoid log(0)

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
    nx = ot.backend.get_backend(X, Y)
    
    # Normalize rows to unit vectors
    X_norm = nx.sqrt(nx.sum(X**2, axis=1, keepdims=True))
    Y_norm = nx.sqrt(nx.sum(Y**2, axis=1, keepdims=True))
    X = X / nx.maximum(X_norm, eps)
    Y = Y / nx.maximum(Y_norm, eps)
    
    # Compute cosine similarity
    D = nx.dot(X, Y.T)
    
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
    nx = ot.backend.get_backend(X, Y)

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

    nx = ot.backend.get_backend(X, Y, label_transfer)

    if nx_torch(nx):
        assert not (torch.is_floating_point(X) or torch.is_floating_point(Y)), "X and Y should contain integer values."
    else:
        assert np.issubdtype(X.dtype, np.integer) and np.issubdtype(X.dtype, np.integer), "X should contain integer values."

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
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    metric: str = "euc",
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


    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert metric in [  
        "euc",
        "euclidean",
        "square_euc",
        "square_euclidean",
        "kl",
        "sym_kl",
        "cos",
        "cosine",
        "label"
    ], "``metric`` value is wrong. Available ``metric`` are: ``'euc'``, ``'euclidean'``, ``'square_euc'``, ``'square_euclidean'``, ``'kl'``, ``'sym_kl'``, ``'cos'``, ``'cosine'``, and ``'label'``."

    if metric == "label":
        assert label_transfer is not None, "label_transfer must be provided for metric 'label'."
        return _label_distance_backend(X, Y, label_transfer)
    elif metric in ["euc", "euclidean"]:
        return _euc_distance_backend(X, Y, squared=False)
    elif metric in ["square_euc", "square_euclidean"]:
        return _euc_distance_backend(X, Y, squared=True)
    elif metric == "kl":
        return _kl_distance_backend(X, Y)
    elif metric == "sym_kl":
        return (_kl_distance_backend(X, Y) + _kl_distance_backend(Y, X).T) / 2
    elif metric in ["cos", "cosine"]:
        return _cosine_distance_backend(X, Y)

def calc_probability(
    distance_matrix: Union[np.ndarray, torch.Tensor],
    probability_type: str = 'Gauss',
    probability_parameter: Optional[float] = None
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
    nx = ot.backend.get_backend(distance_matrix)
    
    if probability_type == 'Gauss':
        if probability_parameter is None:
            raise ValueError("probability_parameter must be provided for 'Gauss' probability type.") 
        probability = nx.exp(-distance_matrix / (2 * probability_parameter))
    elif probability_type == 'cos_prob':
        probability = distance_matrix * 0.5 + 0.5
    elif probability_type == 'prob':
        probability = distance_matrix
    else:
        raise ValueError(f"Unsupported probability type: {probability_type}")

    return probability


###############################
# Calculate assignment matrix #
###############################

def get_P_core(
    nx,
    type_as,
    Dim,
    spatial_dist: Union[np.ndarray, torch.Tensor],
    exp_dist: List[Union[np.ndarray, torch.Tensor]],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    model_mul: Union[np.ndarray, torch.Tensor],
    # alpha: Union[np.ndarray, torch.Tensor],
    gamma: Union[float, np.ndarray, torch.Tensor],
    # Sigma: Union[np.ndarray, torch.Tensor],
    samples_s: Optional[List[float]] = None,
    sigma2_variance: float = 1,
    probability_type: Union[str, List[str]] = 'Gauss',
    probability_parameters: Optional[List] = None,
    eps: float = 1e-8,
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
    spatial_prob = calc_probability(
        spatial_dist, 
        'Gauss', 
        probability_parameter = sigma2 / sigma2_variance
    )  # N x M

    # TODO: everytime this will generate D/2 on GPU, may influence the runtime
    spatial_outlier = _power(nx)((2 * _pi(nx) * sigma2), _data(nx, Dim / 2, type_as)) * (1 - gamma) / (gamma * outlier_s)  # scalar

    # TODO: the position of the following is unclear
    spatial_inlier = 1 - spatial_outlier / (spatial_outlier + nx.sum(spatial_prob, axis=0, keep_dims=True) + eps)  # 1 x M 

    spatial_prob = spatial_prob * model_mul

    # spatial P
    P = spatial_prob / (spatial_outlier + nx.sum(spatial_prob, axis=0, keep_dims=True) + eps)  # N x M
    K_NA_spatial = P.sum(1)

    # Calculate spatial probability without sigma2_variance
    spatial_prob = calc_probability(
        spatial_dist, 
        'Gauss', 
        probability_parameter = sigma2,
    )  # N x M
    spatial_prob = spatial_prob * model_mul

    # sigma2 P
    P = spatial_inlier * spatial_prob / (nx.sum(spatial_prob, axis=0, keep_dims=True) + eps)
    K_NA_sigma2 = P.sum(1)
    sigma2_related = (P * spatial_dist).sum()

    # Calculate probabilities for expression distances
    if probability_parameters is None:
        probability_parameters = [None] * len(exp_dist)

    for e_d, p_t, p_p in zip(exp_dist, probability_type, probability_parameters):
        spatial_prob *= calc_probability(e_d, p_t, p_p)

    P = spatial_inlier * spatial_prob / (nx.sum(spatial_prob, axis=0, keep_dims=True) + eps)

    return P, K_NA_spatial, K_NA_sigma2, sigma2_related


def get_P(
    nx,
    type_as,
    Dim,
    spatial_dist: Union[np.ndarray, torch.Tensor],
    exp_dist: List[Union[np.ndarray, torch.Tensor]],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    alpha: Union[np.ndarray, torch.Tensor],
    gamma: Union[float, np.ndarray, torch.Tensor],
    Sigma: Union[np.ndarray, torch.Tensor],
    samples_s: Optional[List[float]] = None,
    sigma2_variance: float = 1,
    probability_type: Union[str, List[str]] = 'Gauss',
    probability_parameters: Optional[List[float]] = None,
    # use_chunk: bool = False,
    # chunk_capacity_scale: int = 1,
):
    N, M = spatial_dist.shape
    # calculate vectors for model points
    model_mul = _unsqueeze(nx)(alpha * nx.exp(-Sigma / sigma2), -1)  # N x 1
    P, K_NA_spatial, K_NA_sigma2, sigma2_related = get_P_core(
        nx,
        type_as,
        Dim=Dim,
        spatial_dist=spatial_dist,
        exp_dist=exp_dist,
        sigma2=sigma2,
        model_mul=model_mul,
        gamma=gamma,
        samples_s=samples_s,
        sigma2_variance=sigma2_variance,
        probability_type=probability_type,
        probability_parameters=probability_parameters,
    )

    assignment_results = {
        "K_NA_spatial": K_NA_spatial,
        "K_NA_sigma2": K_NA_sigma2,
        "sigma2_related": sigma2_related,
    }

    return P, assignment_results


    
    # if use_chunk:
    #     # get the chunk size
    #     chunk_capacity_base = 1e8
    #     split_size = min(int(chunk_capacity_scale * chunk_capacity_base/ (N)), M)
    #     split_size = 1 if split_size == 0 else split_size

    #     spatial_dist_chunks = _split(nx, spatial_dist, split_size, dim=0)
    #     exp_dist_chunks = _split(nx, exp_dist, split_size, dim=0)

    # else:
    #     pass


def get_P_sparse(
    nx,
    type_as,
    Dim,
    spatial_XA: Union[np.ndarray, torch.Tensor],
    spatial_XB: Union[np.ndarray, torch.Tensor],
    exp_layer_A: List[Union[np.ndarray, torch.Tensor]],
    exp_layer_B: List[Union[np.ndarray, torch.Tensor]],
    label_transfer: Union[np.ndarray, torch.Tensor],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    alpha: Union[np.ndarray, torch.Tensor],
    gamma: Union[float, np.ndarray, torch.Tensor],
    Sigma: Union[np.ndarray, torch.Tensor],
    samples_s: Optional[List[float]] = None,  # TODO: now can't be None
    sigma2_variance: float = 1,  # TODO: check if float or tensor type
    probability_type: Union[str, List[str]] = 'Gauss',
    probability_parameters: Optional[List[float]] = None,
    use_chunk: bool = False,
    chunk_capacity_scale: int = 1,
    top_k: int = 1024,
    metrics: Union[str, List[str]] = 'kl',
):
    """
    Calculate sparse assignment matrix P using spatial and expression / representation distances.

    Parameters
    ----------
    nx : module
        Backend module (e.g., numpy or torch).
    type_as : type
        Type to which the output should be cast.
    Dim : int
        Dimensionality of the spatial data.
    spatial_XA : np.ndarray or torch.Tensor
        Spatial coordinates of sample A.
    spatial_XB : np.ndarray or torch.Tensor
        Spatial coordinates of sample B.
    exp_layer_A : np.ndarray or torch.Tensor
        Expression / representation data of sample A.
    exp_layer_B : np.ndarray or torch.Tensor
        Expression / representation data of sample B.
    label_transfer : np.ndarray or torch.Tensor
        Label transfer cost matrix.
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
    use_chunk : bool, optional
        Whether to use chunking for large datasets. Default is False.
    chunk_capacity_scale : int, optional
        Scale factor for chunk capacity. Default is 1.
    top_k : int, optional
        Number of top elements to keep in the sparse matrix. Default is 1024.
    metrics : Union[str, List[str]], optional
        Distance metrics to use. Default is 'kl'.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Sparse assignment matrix P.
    dict
        Additional results.
    """

    N, M = spatial_XA.shape[0], spatial_XB.shape[1]

    # calculate vectors for model points
    model_mul = _unsqueeze(nx)(alpha * nx.exp(-Sigma / sigma2), -1)  # N x 1

    if use_chunk:
        # get the chunk size
        chunk_capacity_base = 1e8
        split_size = min(int(chunk_capacity_scale * chunk_capacity_base/ (N)), M)
        split_size = 1 if split_size == 0 else split_size

        # chunk the data along the B (data points)
        spatial_XB_chunks = _split(nx, spatial_XB, split_size, dim=0)
        exp_layer_B_chunks = _split(nx, exp_layer_B, split_size, dim=0)

        # initial results for chunk
        K_NA_spatial = nx.zeros((NA,), type_as=type_as)
        K_NA_sigma2 = nx.zeros((NA,), type_as=type_as)
        Ps = []
        sigma2_related = 0

        for spatial_XB_chunk, exp_layer_B_chunk in zip(spatial_XB_chunks, exp_layer_B_chunks):
            # calculate the spatial distance
            spatial_dist = calc_distance(spatial_XA, spatial_XB_chunk, metric="euc")

            # calculate the expression / representation distances
            # TODO: calc_distance should calculate the list
            exp_dist = calc_distance(exp_layer_A, exp_layer_B_chunk, metrics, label_transfer)

            # calculate the assignment matrix within the chunk
            P, K_NA_spatial_chunk, K_NA_sigma2_chunk, sigma2_related_chunk = get_P_core(
                nx,
                type_as,
                Dim=Dim,
                spatial_dist=spatial_dist,
                exp_dist=exp_dist,
                sigma2=sigma2,
                model_mul=model_mul,
                gamma=gamma,
                samples_s=samples_s,
                sigma2_variance=sigma2_variance,
                probability_type=probability_type,
                probability_parameters=probability_parameters,
            )

            # convert to sparse matrix
            P = _dense_to_sparse(
                mat=P,
                sparse_method="topk",
                threshold=top_k,
                axis=0,
                descending=True,
            )

            # add / update chunk results
            Ps.append(P)
            K_NA_spatial += K_NA_spatial_chunk
            K_NA_sigma2 += K_NA_sigma2_chunk
            sigma2_related += sigma2_related_chunk

        # concatenate / process chunk results
        P = _sparse_concat(nx, Ps, axis=1)
        Sp_sigma2 = K_NA_sigma2.sum()
        sigma2_related = sigma2_related / (Dim * Sp_sigma2)

        assignment_results = {
            "K_NA_spatial": K_NA_spatial,
            "K_NA_sigma2": K_NA_sigma2,
            "sigma2_related": sigma2_related,
        }

        return P, assignment_results
        


    else:
        raise NotImplementedError("Non-chunking mode is not implemented yet.")




def kl_divergence_backend(X, Y, probabilistic=True):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    Takes advantage of POT backend to speed up computation.
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)
    if probabilistic:
        X = X / nx.sum(X, axis=1, keepdims=True)
        Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return D


def kl_distance(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
    symmetry: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the KL distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.
        symmetry (bool, optional): Whether to use symmetric KL divergence. Defaults to True.

    Returns:
        Union[np.ndarray, torch.Tensor]: KL distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    # Probabilistic normalization
    X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
    X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)
    while True:
        try:
            if chunk_num == 1:
                if symmetry:
                    DistMat = (kl_divergence_backend(X_A, X_B, False) + kl_divergence_backend(X_B, X_A, False).T) / 2
                else:
                    DistMat = kl_divergence_backend(X_A, X_B, False)
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            if symmetry:
                                arr2.append(
                                    (
                                        kl_divergence_backend(
                                            nx.from_numpy(x_As, type_as=type_as).cuda(),
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            False,
                                        ).cpu()
                                        + kl_divergence_backend(
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            nx.from_numpy(x_As, type_as=type_as).cuda(),
                                            False,
                                        )
                                        .cpu()
                                        .T
                                    )
                                    / 2
                                )
                            else:
                                arr2.append(
                                    kl_divergence_backend(
                                        nx.from_numpy(x_As, type_as=type_as).cuda(),
                                        nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                        False,
                                    ).cpu()
                                )
                        else:
                            if symmetry:
                                arr2.append(
                                    nx.to_numpy(
                                        kl_divergence_backend(
                                            nx.from_numpy(x_As, type_as=type_as),
                                            nx.from_numpy(x_Bs, type_as=type_as),
                                            False,
                                        )
                                        + kl_divergence_backend(
                                            nx.from_numpy(x_Bs, type_as=type_as),
                                            nx.from_numpy(x_As, type_as=type_as),
                                            False,
                                        ).T
                                    )
                                    / 2
                                )
                            else:
                                arr2.append(
                                    kl_divergence_backend(
                                        nx.from_numpy(x_As, type_as=type_as),
                                        nx.from_numpy(x_Bs, type_as=type_as),
                                        False,
                                    )
                                )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)
                break
        except:
            chunk_num = chunk_num * 2
            print("kl chunk more")
    if data_on_gpu and chunk_num != 1:
        DistMat = DistMat.cuda()
    return DistMat


def calc_exp_dissimilarity(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    dissimilarity: str = "kl",
    chunk_num: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate expression dissimilarity.
    Args:
        X_A: Gene expression matrix of sample A.
        X_B: Gene expression matrix of sample B.
        dissimilarity: Expression dissimilarity measure: ``'kl'``, ``'euclidean'``, ``'euc'``, ``'cos'``, or ``'cosine'``.

    Returns:
        Union[np.ndarray, torch.Tensor]: The dissimilarity matrix of two feature samples.
    """
    nx = ot.backend.get_backend(X_A, X_B)

    assert dissimilarity in [
        "kl",
        "euclidean",
        "euc",
        "cos",
        "cosine"
    ], "``dissimilarity`` value is wrong. Available ``dissimilarity`` are: ``'kl'``, ``'euclidean'``, ``'euc'``, ``'cos'``, and ``'cosine'``."
    if dissimilarity.lower() == "kl":
        X_A = X_A + 0.01
        X_B = X_B + 0.01
        X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
        X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)
    while True:
        try:
            if chunk_num == 1:
                DistMat = _dist(X_A, X_B, dissimilarity)
                break
            else:
                X_As = _chunk(nx, X_A, chunk_num, 0)
                X_Bs = _chunk(nx, X_B, chunk_num, 0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []
                    for x_Bs in X_Bs:
                        arr2.append(_dist(x_As, x_Bs, dissimilarity))
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)
                break
        except:
            chunk_num = chunk_num * 2
            print("chunk more")
    return DistMat


def cal_dist(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
    return_gpu: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 1.

    Returns:
        Union[np.ndarray, torch.Tensor]: Distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    while True:
        try:
            if chunk_num == 1:
                DistMat = _dist(X_A, X_B, "euc")
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as).cuda(),
                                    nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                ).cpu()
                            )
                        else:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as),
                                    nx.from_numpy(x_Bs, type_as=type_as),
                                )
                            )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)  # not convert to GPU
                break
        except:
            chunk_num = chunk_num * 2
            print("dist chunk more")
    if data_on_gpu and chunk_num != 1 and return_gpu:
        DistMat = DistMat.cuda()
    return DistMat


def cal_dot(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    use_chunk: bool = False,
    use_gpu: bool = True,
    chunk_num: int = 20,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the matrix multiplication of two matrices

    Args:
        mat1 (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        mat2 (Union[np.ndarray, torch.Tensor]): The second input matrix with shape d x m. We suppose m << n and does not require chunk.
        use_chunk (bool, optional): Whether to use chunk to reduce the GPU memory usage. Note that if set to ``True'' it will slow down the calculation. Defaults to False.
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.

    Returns:
        Union[np.ndarray, torch.Tensor]: Matrix multiplication result with shape n x m
    """
    nx = ot.backend.get_backend(mat1, mat2)
    type_as = mat1[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    if not use_chunk:
        Mat = _dot(nx)(mat1, mat2)
        return Mat
    else:
        # convert to numpy to save the GPU memory
        mat1 = nx.to_numpy(mat1)
        if use_gpu:
            mat2 = mat2.cuda()
        # chunk
        mat1s = np.array_split(mat1, chunk_num, axis=0)
        arr = []  # array for temporary storage of results
        for mat1ss in mat1s:
            if use_gpu:
                arr.append(_dot(nx)(nx.from_numpy(mat1ss, type_as=type_as).cuda(), mat2).cpu())
            else:
                arr.append(_dot(nx)(nx.from_numpy(mat1ss, type_as=type_as), mat2))
        Mat = nx.concatenate(arr, axis=0)
        return Mat


def get_optimal_R(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    P: Union[np.ndarray, torch.Tensor],
    R_init: Union[np.ndarray, torch.Tensor],
):
    """Get the optimal rotation matrix R

    Args:
        coordsA (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        coordsB (Union[np.ndarray, torch.Tensor]): The second input matrix with shape n x d
        P (Union[np.ndarray, torch.Tensor]): The optimal transport matrix with shape n x n

    Returns:
        Union[np.ndarray, torch.Tensor]: The optimal rotation matrix R with shape d x d
    """
    nx = ot.backend.get_backend(coordsA, coordsB, P, R_init)
    NA, NB, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]
    Sp = nx.einsum("ij->", P)
    K_NA = nx.einsum("ij->i", P)
    K_NB = nx.einsum("ij->j", P)
    VnA = nx.zeros(coordsA.shape, type_as=coordsA[0, 0])
    mu_XnA, mu_VnA, mu_XnB = (
        _dot(nx)(K_NA, coordsA) / Sp,
        _dot(nx)(K_NA, VnA) / Sp,
        _dot(nx)(K_NB, coordsB) / Sp,
    )
    XnABar, VnABar, XnBBar = coordsA - mu_XnA, VnA - mu_VnA, coordsB - mu_XnB
    A = -_dot(nx)(nx.einsum("ij,i->ij", VnABar, K_NA).T - _dot(nx)(P, XnBBar).T, XnABar)

    # get the optimal rotation matrix R
    svdU, svdS, svdV = _linalg(nx).svd(A)
    C = _identity(nx, D, type_as=coordsA[0, 0])
    C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
    R = _dot(nx)(_dot(nx)(svdU, C), svdV)
    t = mu_XnB - mu_VnA - _dot(nx)(mu_XnA, R.T)
    optimal_RnA = _dot(nx)(coordsA, R.T) + t
    return optimal_RnA, R, t


###############################
# Distance Matrix Calculation #
###############################
def _cal_cosine_similarity(tensor1, tensor2, dim=1, eps=1e-8):
    tensor1_norm = torch.sqrt(torch.sum(tensor1**2, dim=dim, keepdim=True))
    tensor2_norm = torch.sqrt(torch.sum(tensor2**2, dim=dim, keepdim=True))
    tensor1_norm = torch.clamp(tensor1_norm, min=eps)
    tensor2_norm = torch.clamp(tensor2_norm, min=eps)
    dot_product = torch.sum(tensor1 * tensor2, dim=dim, keepdim=True)
    cosine_similarity = dot_product / (tensor1_norm * tensor2_norm)
    cosine_similarity = cosine_similarity.squeeze(dim)
    return cosine_similarity

def _cos_similarity(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
):
    nx = ot.backend.get_backend(mat1, mat2)
    if nx_torch(nx):
        mat1_unsqueeze = mat1.unsqueeze(-1)
        mat2_unsqueeze = mat2.unsqueeze(-1).transpose(0,2)
        distMat = _cal_cosine_similarity(mat1_unsqueeze, mat2_unsqueeze) * 0.5 + 0.5
    else:
        distMat = (-ot.dist(mat1, mat2, metric='cosine')+1)*0.5 + 0.5
    return distMat

def _dist(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    metric: str = "euc",
) -> Union[np.ndarray, torch.Tensor]:
    assert metric in [
        "euc",
        "euclidean",
        "kl",
        "cos",
        "cosine"
    ], "``metric`` value is wrong. Available ``metric`` are: ``'euc'``, ``'euclidean'`` and ``'kl'``."
    nx = ot.backend.get_backend(mat1, mat2)
    if metric.lower() == "euc" or metric.lower() == "euclidean":
        distMat = nx.sum(mat1**2, 1)[:, None] + nx.sum(mat2**2, 1)[None, :] - 2 * _dot(nx)(mat1, mat2.T)
    elif metric.lower() == "kl":
        distMat = (
            nx.sum(mat1 * nx.log(mat1), 1)[:, None]
            + nx.sum(mat2 * nx.log(mat2), 1)[None, :]
            - _dot(nx)(mat1, nx.log(mat2).T)
            - _dot(nx)(mat2, nx.log(mat1).T).T
        ) / 2
    elif (metric.lower() == "cosine") or (metric.lower() == "cos"):
        distMat = _cos_similarity(mat1, mat2)
    return distMat


def PCA_reduction(
    data_mat: Union[np.ndarray, torch.Tensor],
    reduced_dim: int = 64,
    center: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor],]:
    """PCA dimensionality reduction using SVD decomposition

    Args:
        data_mat (Union[np.ndarray, torch.Tensor]): Input data matrix with shape n x k, where n is the data point number and k is the feature dimension.
        reduced_dim (int, optional): Size of dimension after dimensionality reduction. Defaults to 64.
        center (bool, optional): if True, center the input data, otherwise, assume that the input is centered. Defaults to True.

    Returns:
        projected_data (Union[np.ndarray, torch.Tensor]): Data matrix after dimensionality reduction with shape n x r.
        V_new_basis (Union[np.ndarray, torch.Tensor]): New basis with shape k x r.
        mean_data_mat (Union[np.ndarray, torch.Tensor]): The mean of the input data matrix.
    """

    nx = ot.backend.get_backend(data_mat)
    mean_data_mat = _unsqueeze(nx)(nx.mean(data_mat, axis=0), 0)
    if center:
        mean_re_data_mat = data_mat - mean_data_mat
    else:
        mean_re_data_mat = data_mat
    # SVD to perform PCA
    _, S, VH = _linalg(nx).svd(mean_re_data_mat)
    S_index = nx.argsort(-S)
    V_new_basis = VH.t()[:, S_index[:reduced_dim]]
    projected_data = nx.einsum("ij,jk->ik", mean_re_data_mat, V_new_basis)
    return projected_data, V_new_basis, mean_data_mat


def PCA_project(
    data_mat: Union[np.ndarray, torch.Tensor],
    V_new_basis: Union[np.ndarray, torch.Tensor],
    center: bool = True,
):
    nx = ot.backend.get_backend(data_mat)
    return nx.einsum("ij,jk->ik", data_mat, V_new_basis)


def PCA_recover(
    projected_data: Union[np.ndarray, torch.Tensor],
    V_new_basis: Union[np.ndarray, torch.Tensor],
    mean_data_mat: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    nx = ot.backend.get_backend(projected_data)
    return nx.einsum("ij,jk->ik", projected_data, V_new_basis.t()) + mean_data_mat


def coarse_rigid_alignment(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    transformed_points: Optional[Union[np.ndarray, torch.Tensor]] = None,
    dissimilarity: str = "kl",
    top_K: int = 10,
    allow_flip: bool = False,
    verbose: bool = True,
) -> Tuple[Any, Any, Any, Any, Union[ndarray, Any], Union[ndarray, Any]]:
    if verbose:
        lm.main_info("Performing coarse rigid alignment...")
    nx = ot.backend.get_backend(coordsA, coordsB)
    if transformed_points is None:
        transformed_points = coordsA
    N, M, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]

    coordsA, X_A = voxel_data(
        coords=coordsA,
        gene_exp=X_A,
        voxel_num=max(min(int(N / 20), 1000), 100),
    )
    coordsB, X_B = voxel_data(
        coords=coordsB,
        gene_exp=X_B,
        voxel_num=max(min(int(M / 20), 1000), 100),
    )
    DistMat = calc_exp_dissimilarity(X_A=X_A, X_B=X_B, dissimilarity=dissimilarity)

    transformed_points = nx.to_numpy(transformed_points)
    sub_coordsA = coordsA
    nx = ot.backend.NumpyBackend()

    item2 = np.argpartition(DistMat, top_K, axis=0)[:top_K, :].T
    item1 = np.repeat(np.arange(DistMat.shape[1])[:, None], top_K, axis=1)
    NN1 = np.dstack((item1, item2)).reshape((-1, 2))
    distance1 = DistMat.T[NN1[:, 0], NN1[:, 1]]

    ## construct nearest neighbor set using brute force
    item1 = np.argpartition(DistMat, top_K, axis=1)[:, :top_K]
    item2 = np.repeat(np.arange(DistMat.shape[0])[:, None], top_K, axis=1)
    NN2 = np.dstack((item1, item2)).reshape((-1, 2))
    distance2 = DistMat.T[NN2[:, 0], NN2[:, 1]]

    NN = np.vstack((NN1, NN2))
    distance = np.r_[distance1, distance2]

    train_x, train_y = sub_coordsA[NN[:, 1], :], coordsB[NN[:, 0], :]

    P, R, t, init_weight, sigma2, gamma = inlier_from_NN(train_x, train_y, distance[:, None])
    if allow_flip:
        R_flip = np.eye(D)
        R_flip[-1, -1] = -1
        P2, R2, t2, init_weight, sigma2_2, gamma_2 = inlier_from_NN(np.dot(train_x, R_flip), train_y, distance[:, None])
        if gamma_2 > gamma:
            P = P2
            R = R2
            t = t2
            sigma2 = sigma2_2
            R = np.dot(R, R_flip)
    inlier_threshold = min(P[np.argsort(-P[:, 0])[20], 0], 0.5)
    inlier_set = np.where(P[:, 0] > inlier_threshold)[0]
    inlier_x, inlier_y = train_x[inlier_set, :], train_y[inlier_set, :]
    inlier_P = P[inlier_set, :]

    transformed_points = np.dot(transformed_points, R.T) + t
    inlier_x = np.dot(inlier_x, R.T) + t
    if verbose:
        lm.main_info("Coarse rigid alignment done.")
    return transformed_points, inlier_x, inlier_y, inlier_P, R, t


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


def coarse_rigid_alignment_debug(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    DistMat: Union[np.ndarray, torch.Tensor],
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    sub_sample_num: int = -1,
    top_K: int = 10,
    transformed_points: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    assert (
        coordsA.shape[0] == DistMat.shape[0]
    ), "coordsA and the first dim of DistMat do not have the same number of features."
    assert (
        coordsB.shape[0] == DistMat.shape[1]
    ), "coordsB and the second dim of DistMat do not have the same number of features."
    nx = ot.backend.get_backend(coordsA, coordsB, DistMat)
    if transformed_points is None:
        transformed_points = coordsA
    N, M, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]
    # coordsA = nx.to_numpy(coordsA)
    # coordsB = nx.to_numpy(coordsB)
    # DistMat = nx.to_numpy(DistMat)
    # transformed_points = nx.to_numpy(transformed_points)
    sub_coordsA = coordsA

    ## subsample the data to saving time
    if N > sub_sample_num and sub_sample_num > 0:
        idxA = np.random.choice(N, sub_sample_num, replace=False)
        idxB = np.random.choice(M, sub_sample_num, replace=False)
        sub_coordsA = coordsA[idxA, :]
        coordsB = coordsB[idxB, :]
        DistMat = DistMat[idxA, :][:, idxB]
    # nx = ot.backend.NumpyBackend()

    # construct nearest neighbor set using KDTree
    # tree = KDTree(X_B)
    # K = 10
    # distance1, NN1 = tree.query(X_A, k=K, return_distance=True)
    # print(NN1)
    # print(NN1.shape)
    # tree = KDTree(X_A)
    # distance2, NN2 = tree.query(X_B, k=K, return_distance=True)
    # print(NN2)
    # print(NN2.shape)
    # NN = np.vstack((NN1, NN2))
    # distance = np.r_[distance1, distance2]

    ## construct nearest neighbor set using brute force
    # item2 = np.argsort(DistMat, axis=0)[:top_K,:].T
    # item2 = np.argpartition(DistMat, top_K, axis=0)[:top_K,:].T
    # print(_topk(nx,DistMat, top_K, 0))
    item2 = _topk(nx, DistMat, top_K, 0)[:top_K, :].T
    print(item2.shape)
    item1 = _data(nx, nx.arange(DistMat.shape[1])[:, None].repeat(1, top_K), type_as=item2)
    # item1 = np.repeat(np.arange(DistMat.shape[1])[:,None],top_K,axis=1)
    NN1 = _dstack(nx)((item1, item2)).reshape((-1, 2))
    # NN1 = np.dstack((item1,item2)).reshape((-1,2))
    distance1 = DistMat.T[NN1[:, 0], NN1[:, 1]]

    # item1 = np.argsort(DistMat, axis=1)[:,:top_K]
    # item1 = np.argpartition(DistMat, top_K, axis=1)[:,:top_K]
    item1 = _topk(nx, DistMat, top_K, 1)[:, :top_K]
    item2 = _data(nx, nx.arange(DistMat.shape[0])[:, None].repeat(1, top_K), type_as=item2)
    # item2 = np.repeat(np.arange(DistMat.shape[0])[:,None],top_K,axis=1)
    NN2 = _dstack(nx)((item1, item2)).reshape((-1, 2))
    # NN2 = np.dstack((item1,item2)).reshape((-1,2))
    distance2 = DistMat.T[NN2[:, 0], NN2[:, 1]]

    # NN = np.vstack((NN1,NN2))
    NN = _vstack(nx)((NN1, NN2))
    # distance = np.r_[distance1,distance2]
    # print(distance.shape)
    # print(nx.stack((distance1, distance2), axis=0))
    distance = nx.reshape(nx.stack((distance1, distance2), axis=0), (-1,))
    # print(distance)

    train_x, train_y = sub_coordsA[NN[:, 1], :], coordsB[NN[:, 0], :]

    P, R, t, init_weight = inlier_from_NN_debug(train_x, train_y, distance[:, None])
    inlier_threshold = nx.minimum(P[nx.argsort(-P[:, 0])[20], 0], 0.5)
    inlier_set = nx.where(P[:, 0] > inlier_threshold)[0]
    inlier_x, inlier_y = train_x[inlier_set, :], train_y[inlier_set, :]
    inlier_P = P[inlier_set, :]

    transformed_points = _dot(nx)(transformed_points, R.T) + t
    inlier_x = _dot(nx)(inlier_x, R.T) + t
    # return transformed_points, inlier_x, inlier_y, inlier_P
    return transformed_points, inlier_x, inlier_y, inlier_P, inlier_set, init_weight, P, NN


def inlier_from_NN_debug(
    train_x,
    train_y,
    distance,
):
    nx = ot.backend.get_backend(train_x, train_y, distance)
    N, D = train_x.shape[0], train_x.shape[1]
    alpha = _data(nx, 1.0, type_as=distance)
    distance = nx.maximum(0, distance)
    # normalize = np.sort(distance,0)[10]
    normalize = nx.max(distance) / nx.log(_data(nx, 10.0, type_as=distance))
    # distance = distance / (np.maximum(normalize,1e-2))
    distance = distance / (normalize)
    R = nx.eye(D, type_as=distance)
    t = nx.ones((D, 1), type_as=distance)
    y_hat = train_x
    sigma2 = nx.sum((y_hat - train_y) ** 2) / (D * N)
    weight = nx.exp(-distance * alpha)
    weight = weight / nx.max(weight)
    # weight = np.ones_like(weight)
    init_weight = weight
    P = _mul(nx)(nx.ones((N, 1), type_as=distance), weight)
    max_iter = 100
    alpha_end = 1
    alpha_decrease = nx.power(alpha_end / alpha, 1 / (max_iter - 20))
    gamma = 0.5
    a = nx.maximum(
        nx.prod(nx.max(train_x, axis=0) - nx.min(train_x, axis=0)),
        nx.prod(nx.max(train_y, axis=0) - nx.min(train_y, axis=0)),
    )
    Sp = nx.sum(P)
    for iter in range(max_iter):
        # solve rigid transformation
        mu_x = nx.sum(_mul(nx)(train_x, P), 0) / (Sp)
        mu_y = nx.sum(_mul(nx)(train_y, P), 0) / (Sp)
        t = mu_y - _dot(nx)(mu_x, R.T)
        X_mu, Y_mu = train_x - mu_x, train_y - mu_y
        A = _dot(nx)(Y_mu.T, _mul(nx)(X_mu, P))
        svdU, svdS, svdV = _linalg(nx).svd(A)
        C = _identity(nx, D, type_as=distance)
        C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
        R = _dot(nx)(_dot(nx)(svdU, C), svdV)
        y_hat = _dot(nx)(train_x, R.T) + t
        # get P
        term1 = _mul(nx)(nx.exp(-(nx.sum((train_y - y_hat) ** 2, 1, keepdims=True)) / (2 * sigma2)), weight)
        outlier_part = nx.max(weight) * (1 - gamma) * nx.power((2 * _pi(nx) * sigma2), D / 2) / (gamma * a)
        P = term1 / (term1 + outlier_part)
        Sp = nx.sum(P)
        # num_ind = np.where(P > 0.5)[0].shape[0]
        # gamma = np.minimum(np.maximum(num_ind / N, 0.05),0.95)
        gamma = nx.minimum(nx.maximum(Sp / N, 0.01), 0.99)
        P = nx.maximum(P, 1e-6)

        # update sigma2
        sigma2 = nx.sum(_mul(nx)((y_hat - train_y) ** 2, P)) / (D * Sp)
        if iter > 20:
            alpha = alpha * alpha_decrease
            weight = nx.exp(-distance * alpha)
            weight = weight / nx.max(weight)
    # print(sigma2)
    return P, R, t, init_weight


def voxel_data(
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
    nx = ot.backend.get_backend(coords, gene_exp)
    N, D = coords.shape[0], coords.shape[1]
    coords = nx.to_numpy(coords)
    gene_exp = nx.to_numpy(gene_exp)

    # create the voxel grid
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    if voxel_size is None:
        voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)
        # print(voxel_size)
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
    subsample=2000,
):
    NA, NB, D = XA.shape[0], XB.shape[0], XA.shape[1]
    sub_sample_A = np.random.choice(NA, subsample, replace=False) if NA > subsample else np.arange(NA)
    sub_sample_B = np.random.choice(NB, subsample, replace=False) if NB > subsample else np.arange(NB)
    SpatialDistMat = calc_exp_dissimilarity(
        X_A=XA[sub_sample_A, :],
        X_B=XB[sub_sample_B, :],
        dissimilarity="euc",
    )
    SpatialDistMat = SpatialDistMat ** 2
    sigma2 = 0.1 * SpatialDistMat.sum() / (D * sub_sample_A.shape[0] * sub_sample_A.shape[0])  # 2 for 3D
    return sigma2

def _init_guess_beta2(
    nx,
    XA,
    XB,
    dissimilarity="kl",
    partial_robust_level=1,
    beta2=None,
    beta2_end=None,
    subsample=5000,
    verbose=False,
):
    NA, NB, D = XA.shape[0], XB.shape[0], XA.shape[1]
    sub_sample_A = np.random.choice(NA, subsample, replace=False) if NA > subsample else np.arange(NA)
    sub_sample_B = np.random.choice(NB, subsample, replace=False) if NB > subsample else np.arange(NB)
    GeneDistMat = calc_exp_dissimilarity(
        X_A=XA[sub_sample_A, :],
        X_B=XB[sub_sample_B, :],
        dissimilarity=dissimilarity,
    )
    minGeneDistMat = nx.min(GeneDistMat, 1)
    if beta2 is None:
        beta2 = minGeneDistMat[nx.argsort(minGeneDistMat)[int(sub_sample_A.shape[0] * 0.05)]] / 5
    else:
        beta2 = _data(nx, beta2, XA)

    if beta2_end is None:
        beta2_end = nx.max(minGeneDistMat) / (nx.sqrt(_data(nx, partial_robust_level, XA)))
    else:
        beta2_end = _data(nx, beta2_end, XA)
    beta2 = nx.maximum(beta2, _data(nx, 1e-2, XA))
    if verbose:
        lm.main_info(message=f"beta2: {beta2} --> {beta2_end}.", indent_level=2)
    return beta2, beta2_end


#################################
# Funcs between Numpy and Torch #
#################################


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
