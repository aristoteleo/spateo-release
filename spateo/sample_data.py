import ntpath
import os
import shutil
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
        if not os.path.exists(dir_name):
            Path(dir_name).mkdir(parents=True, exist_ok=True)

        # download the data
        print(url)
        urlretrieve(url, file_path, reporthook=lm.get_main_logger().request_report_hook)

    return file_path


def get_adata(url: str, filename: Optional[str] = None, dir_name: str = "./data") -> AnnData:
    """Download example data to local folder.

    Args:
        url: url that deposits the data.
        filename: file name that will store the data locally.
        dir_name: name of the directory.

    Returns:
        adata: :class:`~anndata.AnnData`
            an Annodata object.
    """

    file_path = download_data(url=url, file_path=filename, dir_name=dir_name)
    if Path(file_path).suffixes[-1][1:] == "loom":
        adata = read_loom(filename=file_path)
    elif Path(file_path).suffixes[-1][1:] == "h5ad":
        adata = read_h5ad(filename=file_path)

    adata.var_names_make_unique()

    return adata


def drosophila(
    filename: str = "E7-9h_cellbin_tdr_v2.h5ad",
    dir_name: str = "./data",
):
    """Multiple drosophila spatial transcriptome data.

    Args:
        filename: file name of the data.  Available ``filename`` are:

                * ``E7-9h_cellbin_tdr_v1.h5ad``
                * ``E7-9h_cellbin_tdr_v2.h5ad``
                * ``E7-9h_cellbin_tdr_v2_midgut.h5ad``
                * ``E7-9h_cellbin_tdr_v3_midgut.h5ad``
                * ``E7-9h_cellbin_h5ad.zip``
                * ``E7-9h_bin20_h5ad.zip``
                * ``E9-10h_cellbin_tdr_v1.h5ad``
                * ``E9-10h_cellbin_tdr_v2.h5ad``
                * ``E9-10h_cellbin_tdr_v2_midgut.h5ad``
                * ``E9-10h_cellbin_tdr_v2_CNS.h5ad``
        dir_name: dir path that will store the data locally.
    Returns:
        Returns `adata` object
    """
    url_dict = {
        "E7-9h_cellbin_tdr_v1.h5ad": "https://www.dropbox.com/s/ow8xkge0538309a/E7-9h_cellbin_tdr_v1.h5ad?dl=1",
        "E7-9h_cellbin_tdr_v2.h5ad": "https://www.dropbox.com/s/bvstb3en5kc6wui/E7-9h_cellbin_tdr_v2.h5ad?dl=1",
        "E7-9h_cellbin_tdr_v2_midgut.h5ad": "https://www.dropbox.com/s/q020zgxxemxl7j4/E7-9h_cellbin_tdr_v2_midgut.h5ad?dl=1",
        "E7-9h_cellbin_tdr_v3_midgut.h5ad": "https://www.dropbox.com/s/cz2nqpmoc3oo5f3/E7-9h_cellbin_tdr_v3_midgut.h5ad?dl=1",
        "E7-9h_cellbin_h5ad.zip": "https://yq01-ct01.baidupcs.com/file/33a841239s21a64188a4dfa5ba6fc000?bkt=en-e031c0692dcd5a21cb9b7dc32086fda95e0ccccb0c3349d8cf10c6a227b624b16fa4148db441052c&fid=3627617064-250528-547706644850916&time=1670092976&sign=FDTAXUbGERLQlBHSKfWqi-DCb740ccc5511e5e8fedcff06b081203-ucK5LvvmMdebDCkUK6bIt7zl%2B84%3D&to=427&size=16158219&sta_dx=16158219&sta_cs=0&sta_ft=zip&sta_ct=0&sta_mt=0&fm2=MH%2CYangquan%2CAnywhere%2C%2C%E9%99%95%E8%A5%BF%2Cct&ctime=1670091825&mtime=1670091825&resv0=-1&resv1=0&resv2=rlim&resv3=5&resv4=16158219&vuk=3627617064&iv=0&htype=&randtype=&tkbind_id=0&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-9054aa2cad924f0d599095a94ed4b854e5970e39aa0c9444cf57152bf7053f32510488b37501bfef&sl=76480590&expires=8h&rt=pr&r=574265491&vbdid=508923211&fin=cellbin_h5ad.zip&fn=cellbin_h5ad.zip&rtype=1&dp-logid=8971527268563088775&dp-callid=0.1&hps=1&tsl=80&csl=80&fsl=-1&csign=SO8aWwhnhOnGKGYv9cWXdJamoKk%3D&so=0&ut=6&uter=4&serv=0&uc=1058262541&ti=c77e04c9862927e5cc28032e1c6abcd6f293302b741abd8e54086130d5ff933c&hflag=30&from_type=1&adg=c_e1c2310f52475804a26726f9181d2ae8&reqlabel=250528_f_68129c39a137f978b08a95b72fcc7f8f_-1_4717521fbacb8ab13b0b72daacf708b3&by=themis",
        "E7-9h_bin20_h5ad.zip": "https://ucd1e2dec0e8649689333bc8f748.dl.dropboxusercontent.com/cd/0/get/Bx9-BFgcvjpaDYK2g4JzMhDf_EFQtaKEKa-eLjVOYSGaBRYsIL7KQby9Sw75TfXkXwcRBudmW81EBV12iAeh4IX1-MbJm8zwkLJHOoWlNWV-lmqV5i8v4fwxABqMLT1Ad86AuP9nuLwlpwghAusj2Wf61bl1Gq5n-gokf5IzrO1L0xmouroXbT3Sz4dTlX6nPhg/file#",
        "E9-10h_cellbin_tdr_v1.h5ad": "https://www.dropbox.com/s/q2l8mqpn7qvz2xr/E9-10h_cellbin_tdr_v1.h5ad?dl=1",
        "E9-10h_cellbin_tdr_v2.h5ad": "https://www.dropbox.com/s/q02sx6acvcqaf35/E9-10h_cellbin_tdr_v2.h5ad?dl=1",
        "E9-10h_cellbin_tdr_v2_midgut.h5ad": "https://www.dropbox.com/s/we2fkpd1p3ww33f/E9-10h_cellbin_tdr_v2_midgut.h5ad?dl=1",
        "E9-10h_cellbin_tdr_v2_CNS.h5ad": "https://www.dropbox.com/s/a7bllwm760dmda6/E9-10h_cellbin_tdr_v2_CNS.h5ad?dl=1",
    }
    if filename.endswith(".h5ad") or filename.endswith(".loom"):
        adata = get_adata(url=url_dict[filename], filename=filename, dir_name=dir_name)
        return adata
    elif filename.endswith(".zip"):
        file_path = download_data(url=url_dict[filename], file_path=filename)
        shutil.unpack_archive(file_path, dir_name)

        zip_folder = os.path.join(dir_name, filename[:-4])
        adata_list = [
            read_h5ad(filename=os.path.join(zip_folder, filename))
            for root, dirs, files in os.walk(zip_folder)
            for filename in files
        ]
        return adata_list


def mousebrain(
    filename,
    dir_name: str = "./data",
):
    """Mouse brain spatial transcriptome data.

    Args:
        filename: file name of the data.  Available ``filename`` are:
                * ``mousebrain_bin30.h5ad``
                * ``mousebrain_bin50_raw.h5ad``
                * ``mousebrain_bin60.h5ad``
                * ``mousebrain_bin60_clustered.h5ad``
                * ``mousebrain_cellbin_clustered.h5ad``
        dir_name: dir path that will store the data locally.
    Returns:
        Returns `adata` object
    """
    url_dict = {
        "mousebrain_bin30.h5ad": "https://www.dropbox.com/s/tyvhndoyj8se5xt/mousebrain_bin30.h5ad?dl=1",
        "mousebrain_bin50_raw.h5ad": "https://www.dropbox.com/s/vtapwsccpi885l2/mousebrain_bin50_raw.h5ad?dl=0",
        "mousebrain_bin60.h5ad": "https://www.dropbox.com/s/c5tu4drxda01m0u/mousebrain_bin60.h5ad?dl=1",
        "mousebrain_bin60_clustered.h5ad": "https://www.dropbox.com/s/wxgkim87uhpaz1c/mousebrain_bin60_clustered.h5ad?dl=1",
        "mousebrain_cellbin_clustered.h5ad": "https://www.dropbox.com/s/seusnva0dgg5de5/mousebrain_cellbin_clustered.h5ad?dl=1",
    }
    adata = get_adata(url_dict[filename], filename=filename, dir_name=dir_name)

    return adata


def axolotl(
    filename,
    dir_name: str = "./data",
):
    """axolotl spatial transcriptome data.

    Args:
        filename: file name of the data.  Available ``filename`` are:
                * ``axolotl_2DPI.h5ad``
                * ``axolotl_2DPI_right.h5ad``
        dir_name: dir path that will store the data locally.
    Returns:
        Returns `adata` object
    """
    url_dict = {
        "axolotl_2DPI.h5ad": "https://www.dropbox.com/s/7w2jxf41xazrqxo/axolotl_2DPI.h5ad?dl=1",
        "axolotl_2DPI_right.h5ad": "https://www.dropbox.com/s/pm5vvqcd4leahsb/axolotl_2DPI_right.h5ad?dl=1",
    }
    adata = get_adata(url_dict[filename], filename=filename, dir_name=dir_name)

    return adata


def slideseq(
    url="https://www.dropbox.com/s/d3tpusisbyzn6jk/slideseq.h5ad?dl=1",
    filename="slideseq_mouse_hippocampus.h5ad",
):
    """Saptial transcriptomic sample from the mouse hippocampus; data generated using Slide-seqV2. See: Stickels,
    R. R., Murray, E., Kumar, P., Li, J., Marshall, J. L., Di Bella, D. J., ... & Chen, F. (2021). Highly sensitive
    spatial transcriptomics at near-cellular resolution with Slide-seqV2. Nature biotechnology, 39(3), 313-319.

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
    seqFISH. See: Lohoff, T., Ghazanfar, S., Missarova, A., Koulena, N., Pierson, N., Griffiths, J. A.,
    ... & Marioni, J. C. (2022). Integration of spatial and single-cell transcriptomic data elucidates mouse
    organogenesis. Nature biotechnology, 40(1), 74-85.

    Returns:
        adata: AnnData object containing seqFISH data
    """
    adata = get_adata(url, filename)

    return adata


def merfish(
    url="https://www.dropbox.com/s/e8hwgqnrx2ob9h4/MERFISH.h5ad?dl=1",
    filename="merfish_mouse_hypothalamus.h5ad",
):
    """Spatial transcriptomic sample taken from the mouse hypothalamus; data generated using MERFISH. See: Moffitt,
    J. R., Bambah-Mukku, D., Eichhorn, S. W., Vaughn, E., Shekhar, K., Perez, J. D., ... & Zhuang, X. (2018).
    Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region.
    Science, 362(6416), eaau5324.

    Returns:
        adata: AnnData object containing MERFISH data
    """
    adata = get_adata(url, filename)

    return adata


def seqscope(
    url="https://www.dropbox.com/s/hci9up23dkuyexb/SeqScope.h5ad?dl=1",
    filename="seqscope_mouse_liver.h5ad",
):
    """Spatial transcriptomic sample taken from the mouse liver; data generated using Seq-Scope. See: Cho, C. S.,
    Xi, J., Si, Y., Park, S. R., Hsu, J. E., Kim, M., ... & Lee, J. H. (2021). Microscopic examination of spatial
    transcriptome using Seq-Scope. Cell, 184(13), 3559-3572, and: Xi, J., Lee, J. H., Kang, H. M., & Jun, G. (2022).
    STtools: a comprehensive software pipeline for ultra-high-resolution spatial transcriptomics data.
    Bioinformatics Advances, 2(1), vbac061.

    Returns:
        adata: AnnData object containing Seq-Scope data
    """
    adata = get_adata(url, filename)

    return adata


def starmap(
    url="https://www.dropbox.com/s/zpvu387tajrwth7/STARmap.h5ad?dl=1",
    filename="starmap_mouse_brain.h5ad",
):
    """Spatial transcriptomic sample taken from the mouse brain; data generated using STARmap. See: Wang, X., Allen,
    W. E., Wright, M. A., Sylwestrak, E. L., Samusik, N., Vesuna, S., ... & Deisseroth, K. (2018). Three-dimensional
    intact-tissue sequencing of single-cell transcriptional states. Science, 361(6400), eaat5691.

    Returns:
        adata: AnnData object containing STARmap data
    """
    adata = get_adata(url, filename)

    return adata
