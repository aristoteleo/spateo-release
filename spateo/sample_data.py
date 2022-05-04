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


def drosophila_E8_10(
    url="https://www.dropbox.com/s/z330s7a4p2w15oe/E8-10_b_all_anno_scsq.h5ad?dl=1",
    filename="drosophila_E8_10.h5ad",
):
    """The E8-10 whole-body data of drosophila of aligned serial slices with tissue type annotations.

    Returns
    -------
        Returns `adata` object
    """
    adata = get_adata(url, filename)

    return adata
