import nudged
import numpy as np
import ot
import torch

from anndata import AnnData
from scipy.spatial import distance_matrix
from scipy.sparse.csr import spmatrix
from tqdm import tqdm
from typing import List, Tuple, Union


def pairwise_align(
    slice1: AnnData,
    slice2: AnnData,
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Calculates and returns optimal alignment of two slices.

    Args:
        slice1: An AnnData object.
        slice2: An AnnData object.
        alpha: Trade-off parameter (0 < alpha < 1).
        numItermax: max number of iterations for cg.
        numItermaxEmd: Max number of iterations for emd.
        device: Equipment used to run the program.
            Can also accept a torch.device. E.g.: torch.device('cuda:0')

    Returns:
        Alignment of spots.
    """

    if device != "cpu":
        torch.cuda.init()
    # Subset for common genes
    common_genes = [value for value in slice1.var.index if value in set(slice2.var.index)]
    slice1, slice2 = slice1[:, common_genes], slice2[:, common_genes]

    # Calculate expression dissimilarity
    to_dense_array = lambda X: np.array(X.todense()) if isinstance(X, spmatrix) else X
    slice1_x, slice2_x = (
        to_dense_array(slice1.X) + 0.0000000001,
        to_dense_array(slice2.X) + 0.0000000001,
    )
    slice1_x, slice2_x = (
        slice1_x / slice1_x.sum(axis=1, keepdims=True),
        slice2_x / slice2_x.sum(axis=1, keepdims=True),
    )
    slice1_logx_slice1 = np.array([np.apply_along_axis(lambda x: np.dot(x, np.log(x).T), 1, slice1_x)])
    slice1_logx_slice2 = np.dot(slice1_x, np.log(slice2_x).T)
    M = torch.tensor(slice1_logx_slice1.T - slice1_logx_slice2, device=device, dtype=torch.float32)

    # Weight of spots
    p = torch.tensor(
        np.ones((slice1.shape[0],)) / slice1.shape[0],
        device=device,
        dtype=torch.float32,
    )
    q = torch.tensor(
        np.ones((slice2.shape[0],)) / slice2.shape[0],
        device=device,
        dtype=torch.float32,
    )

    # Calculate spatial distances
    DA = torch.tensor(
        distance_matrix(slice1.obsm["spatial"], slice1.obsm["spatial"]),
        device=device,
        dtype=torch.float32,
    )
    DB = torch.tensor(
        distance_matrix(slice2.obsm["spatial"], slice2.obsm["spatial"]),
        device=device,
        dtype=torch.float32,
    )

    # Computes the FGW transport between two slides
    pi = ot.gromov.fused_gromov_wasserstein(
        M=M,
        C1=DA,
        C2=DB,
        p=p,
        q=q,
        loss_fun="square_loss",
        alpha=alpha,
        armijo=False,
        log=False,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
    )
    if device != "cpu":
        torch.cuda.empty_cache()

    return pi.cpu().numpy()


def slice_alignment(
    slices: List[AnnData],
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
):
    """Align spatial coordinates of slices.

    Args:
        slices: List of slices (AnnData Object).
        alpha: Trade-off parameter (0 < alpha < 1).
        numItermax: max number of iterations for cg.
        numItermaxEmd: Max number of iterations for emd.
        device: Equipment used to run the program.
            Can also accept a torch.device. E.g.: torch.device('cuda:0')
        verbose: Print information along iterations.

    Returns:
        List of slices (AnnData Object) after alignment.
    """

    def _log(m):
        if verbose:
            print(m)

    if device != "cpu":
        _log(f"\nWhether CUDA is currently available: {torch.cuda.is_available()}")
        _log(f"Device: {torch.cuda.get_device_name(device=device)}")
        _log(
            f"GPU total memory: {int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024 * 1024))} GB"
        )
    else:
        _log(f"Device: CPU")

    align_slices = []
    for i in tqdm(range(len(slices) - 1), desc=" Alignment "):

        if i == 0:
            slice1 = slices[i].copy()
        else:
            slice1 = align_slices[i].copy()
        slice2 = slices[i + 1].copy()

        # Calculate and returns optimal alignment of two slices.
        pi = pairwise_align(
            slice1,
            slice2,
            alpha=alpha,
            numItermax=numItermax,
            numItermaxEmd=numItermaxEmd,
            device=device,
        )

        # Calculate new coordinates of two slices
        raw_slice1_coords, raw_slice2_coords = (
            slice1.obsm["spatial"],
            slice2.obsm["spatial"],
        )
        slice1_coords = raw_slice1_coords - pi.sum(axis=1).dot(raw_slice1_coords)
        slice2_coords = raw_slice2_coords - pi.sum(axis=0).dot(raw_slice2_coords)
        H = slice2_coords.T.dot(pi.T.dot(slice1_coords))
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)
        slice2_coords = R.dot(slice2_coords.T).T
        slice1.obsm["spatial"] = np.around(slice1_coords, decimals=2)
        slice2.obsm["spatial"] = np.around(slice2_coords, decimals=2)

        if i == 0:
            align_slices.append(slice1)
        align_slices.append(slice2)

    for i, align_slice in enumerate(align_slices):
        align_slice.obs["x"] = align_slice.obsm["spatial"][:, 0].astype(float)
        align_slice.obs["y"] = align_slice.obsm["spatial"][:, 1].astype(float)

    return align_slices


def slice_alignment_bigBin(
    slices: List[AnnData],
    slices_big: List[AnnData],
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
) -> Tuple[List[AnnData], List[AnnData]]:
    """Align spatial coordinates of slices.

    If there are too many slice coordinates to be aligned, this method can be selected.

    First select the slices with fewer coordinates for alignment, and then calculate the affine transformation matrix.
    Secondly, the required slices are aligned through the calculated affine transformation matrix.

    Args:
        slices: List of slices (AnnData Object).
        slices_big: List of slices (AnnData Object) with a small number of coordinates.
        alpha: Trade-off parameter (0 < alpha < 1).
        numItermax: max number of iterations for cg.
        numItermaxEmd: Max number of iterations for emd.
        device: Equipment used to run the program.
            Can also accept a torch.device. E.g.: torch.device('cuda:0')
        verbose: Print information along iterations.

    Returns:
        Tuple of two elements. The first contains a list of slices after alignment.
        The second contains a list of slices with a small number of coordinates
        after alignment.
    """

    # Align spatial coordinates of slices with a small number of coordinates.
    align_slices_big = slice_alignment(
        slices=slices_big,
        alpha=alpha,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        device=device,
        verbose=verbose,
    )

    align_slices = []
    for slice_big, align_slice_big, slice in zip(slices_big, align_slices_big, slices):
        # Calculate the affine transformation matrix through nudged
        slice_big_coords = slice_big.obsm["spatial"].tolist()
        align_slice_big_coords = align_slice_big.obsm["spatial"].tolist()
        trans = nudged.estimate(slice_big_coords, align_slice_big_coords)
        slice_coords = slice.obsm["spatial"].tolist()

        #  Align slices through the calculated affine transformation matrix.
        align_slice_coords = np.around(trans.transform(slice_coords), decimals=2)
        align_slice = slice.copy()
        align_slice.obsm["spatial"] = np.array(align_slice_coords)
        align_slice.obs["x"] = align_slice.obsm["spatial"][:, 0].astype(float)
        align_slice.obs["y"] = align_slice.obsm["spatial"][:, 1].astype(float)
        align_slices.append(align_slice)

    return align_slices, align_slices_big
