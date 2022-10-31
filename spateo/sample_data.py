import ntpath
import os
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.request import urlretrieve

from anndata import AnnData, read_h5ad, read_loom

from .logging import logger_manager as lm


def download_data(url: str, file_path: Optional[str] = None, dir_name: str = "./data") -> str:
    """Parse url to get the file name and then download the data to designated folders.

    Args:
        url: url that deposits the data.
        file_path: file path that will store the data locally.
        dir_name: name of the directory.

    Returns:
        the file path that points to the downloaded data.
    """

    file_path = ntpath.basename(url) if file_path is None else file_path
    file_path = os.path.join(dir_name, file_path)
    lm.main_info("Downloading data to " + file_path)

    if not os.path.exists(file_path):
        if not os.path.exists("./data/"):
            os.mkdir("data")

        # download the data
        urlretrieve(url, file_path, reporthook=lm.get_main_logger().request_report_hook)

    return file_path


def get_adata(url: str, filename: Optional[str] = None) -> AnnData:
    """Download example data to local folder.

    Args:
        url: url that deposits the data.
        filename: file name that will store the data locally.

    Returns:
        adata: :class:`~anndata.AnnData`
            an Annodata object.
    """

    file_path = download_data(url, filename)
    if Path(file_path).suffixes[-1][1:] == "loom":
        adata = read_loom(filename=file_path)
    elif Path(file_path).suffixes[-1][1:] == "h5ad":
        adata = read_h5ad(filename=file_path)

    adata.var_names_make_unique()

    return adata


def drosophila(
    url="https://www.dropbox.com/s/bvstb3en5kc6wui/E7-9h_cellbin_tdr_v2.h5ad?dl=0",
    filename="drosophila_E7-9h_v2.h5ad",
):
    """The E8-10 whole-body data of drosophila of aligned serial slices with tissue type annotations.

    Args:
        url: url that deposits the data. Available ``url`` are:

                * ``E7-9h_cellbin_tdr_v1.h5ad``: ``https://www.dropbox.com/s/ow8xkge0538309a/E7-9h_cellbin_tdr_v1.h5ad?dl=0``
                * ``E7-9h_cellbin_tdr_v2.h5ad``: ``https://www.dropbox.com/s/bvstb3en5kc6wui/E7-9h_cellbin_tdr_v2.h5ad?dl=0``
                * ``E7-9h_cellbin_tdr_v2_midgut.h5ad``: ``https://www.dropbox.com/s/q020zgxxemxl7j4/E7-9h_cellbin_tdr_v2_midgut.h5ad?dl=0``
                * ``E7-9h_cellbin_tdr_v3_midgut.h5ad``: ``https://www.dropbox.com/s/cz2nqpmoc3oo5f3/E7-9h_cellbin_tdr_v3_midgut.h5ad?dl=0``
                * ``E9-10h_cellbin_tdr_v1.h5ad``: ``https://www.dropbox.com/s/q2l8mqpn7qvz2xr/E9-10h_cellbin_tdr_v1.h5ad?dl=0``
                * ``E9-10h_cellbin_tdr_v2.h5ad``: ``https://www.dropbox.com/s/q02sx6acvcqaf35/E9-10h_cellbin_tdr_v2.h5ad?dl=0``
                * ``E9-10h_cellbin_tdr_v2_midgut.h5ad``: ``https://www.dropbox.com/s/we2fkpd1p3ww33f/E9-10h_cellbin_tdr_v2_midgut.h5ad?dl=0``
        filename: file name that will store the data locally.
    Returns:
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata


def slideseq(
    url="https://www.dropbox.com/s/d3tpusisbyzn6jk/slideseq.h5ad?dl=1",
    filename="slideseq_mouse_hippocampus.h5ad",
):
    """Saptial transcriptomic sample from the mouse hippocampus; data generated using Slide-seqV2. See:
    Stickels, R. R., Murray, E., Kumar, P., Li, J., Marshall, J. L., Di Bella, D. J., ... & Chen, F. (2021).
    Highly sensitive spatial transcriptomics at near-cellular resolution with Slide-seqV2. Nature biotechnology, 39(3),
    313-319.

    Returns:
        adata: AnnData object containing Slide-seq data
    """
    adata = get_adata(url, filename)

    return adata


def seqfish(
    url="https://www.dropbox.com/s/cm3uw8czhz5hu30/seqFISH.h5ad?dl=1",
    filename="seqfish_mouse_embryo.h5ad",
):
    """Spatial transcriptomic sample taken at one timepoint in the process of mouse organogenesis; data generated using
    seqFISH. See:
    Lohoff, T., Ghazanfar, S., Missarova, A., Koulena, N., Pierson, N., Griffiths, J. A., ... & Marioni, J. C. (2022).
    Integration of spatial and single-cell transcriptomic data elucidates mouse organogenesis.
    Nature biotechnology, 40(1), 74-85.

    Returns:
        adata: AnnData object containing Slide-seq data
    """
    adata = get_adata(url, filename)

    return adata


def merfish(
    url="https://www.dropbox.com/s/e8hwgqnrx2ob9h4/MERFISH.h5ad?dl=1",
    filename="merfish_mouse_hypothalamus.h5ad",
):
    """Spatial transcriptomic sample taken from the mouse hypothalamus; data generated using MERFISH. See:
    Moffitt, J. R., Bambah-Mukku, D., Eichhorn, S. W., Vaughn, E., Shekhar, K., Perez, J. D., ... & Zhuang, X. (2018).
    Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region.
    Science, 362(6416), eaau5324.

    Returns:
        adata: AnnData object containing Slide-seq data
    """
    adata = get_adata(url, filename)

    return adata
