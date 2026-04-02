import os
import subprocess

from pathlib import Path
from typing import (
    List,
    Tuple,
    Union,
)

from .data_utils import (
    DEFAULT_PATH_LARD_ARCHIVES,
    DEFAULT_PATH_LARD_DATASETS,
    DEFAULT_TRAIN_ARCHIVES,
    DEFAULT_TESTS_ARCHIVES,
    DEFAULT_URL_LARD,
)


#######################################
##
##          DATA DOWNLOAD
##
#######################################

def dataset_download_and_unzip(
        lard_archives_dpath: Union[Path, str] = DEFAULT_PATH_LARD_ARCHIVES,
        lard_datasets_dpath: Union[Path, str] = DEFAULT_PATH_LARD_DATASETS,
        train_archives_names: List[Tuple[str, str]] = DEFAULT_TRAIN_ARCHIVES,
        tests_archives_names: List[Tuple[str, str]] = DEFAULT_TESTS_ARCHIVES,
        url_dl: str = DEFAULT_URL_LARD,
        verbose: bool = False,
):
    """
    Function to call to download the LARD dataset and unzip the archives.

    Args:
        lard_archives_dpath (Path): Path where to store the downloaded zip files.
        lard_datasets_dpath (Path): Path where to store the extracted data.
        train_archives_names (List[Tuple[str, str]]): Names of the train archives
        tests_archives_names (List[Tuple[str, str]]): Names of the tests archives
        url_dl (str): URL of the dataset.
        verbose (bool):
    """
    if not isinstance(lard_archives_dpath, Path):
        lard_archives_dpath = Path(lard_archives_dpath).resolve()
    if not isinstance(lard_datasets_dpath, Path):
        lard_datasets_dpath = Path(lard_datasets_dpath).resolve()

    # Download datasets
    os.makedirs(lard_archives_dpath.as_posix(), exist_ok=True)
    os.makedirs(lard_datasets_dpath.as_posix(), exist_ok=True)

    def _lard_download(src: str):
        if verbose:
            print(f"Downloading ... {src}")
        if not (lard_archives_dpath / src).exists():
            # subprocess.run([])
            subprocess.run(["wget", "-nc", url_dl + src, "-O", (lard_archives_dpath / src).as_posix()], check=True)
            # %time !wget -nc {"\""+url_dl+src+"\""} -O {(lard_archives_dirpath / src).as_posix()}
        elif verbose:
            print(f"Target LARD archive already downloaded.")

    def _lard_unzip(src: str, mod: str):
        if verbose:
            print(f"Unzipping ... {src}")
        if not (lard_datasets_dpath / f"LARD_{mod}" / src.rpartition('.')[0]).exists():
            subprocess.run(["unzip", "-q", "-o", (lard_archives_dpath / src).as_posix(), "-d", (lard_datasets_dpath / f"LARD_{mod}").as_posix()], check=True)
            # %time !unzip -q -o ./{(lard_archives_dirpath / src).as_posix()} -d {(lard_datasets_dirpath / f'LARD_{mod}').as_posix()}
        elif verbose:
            print(f"Target LARD dataset already exists.")

    for src,_ in train_archives_names:
        _lard_download(src)
        _lard_unzip(src, "train")

    for src,_ in tests_archives_names:
        _lard_download(src)
        _lard_unzip(src, "test")