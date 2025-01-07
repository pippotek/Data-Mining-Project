# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import logging

from recommenders.datasets.download_utils import (
    maybe_download,
    download_path,
    unzip_file,
)


URL_MIND_DEMO_TRAIN = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip"
)
URL_MIND_DEMO_VALID = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
)
URL_MIND_DEMO_UTILS = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_utils.zip"
)

URL_MIND_SMALL_TRAIN = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip"
)
URL_MIND_SMALL_VALID = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip"
)
URL_MIND_SMALL_UTILS = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_utils.zip"
)

URL_MIND_LARGE_TRAIN = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip"
)
URL_MIND_LARGE_VALID = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip"
)
URL_MIND_LARGE_TEST = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_test.zip"
)
URL_MIND_LARGE_UTILS = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_utils.zip"
)

URL_MIND = {
    "large": (URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID),
    "small": (URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID),
    "demo": (URL_MIND_DEMO_TRAIN, URL_MIND_DEMO_VALID),
}

logger = logging.getLogger()


def download_mind(size="small", dest_path=None):
    """Download MIND dataset

    Args:
        size (str): Dataset size. One of ["small", "large"]
        dest_path (str): Download path. If path is None, it will download the dataset on a temporal path

    Returns:
        str, str: Path to train and validation sets.
    """
    size_options = ["small", "large", "demo"]
    if size not in size_options:
        raise ValueError(f"Wrong size option, available options are {size_options}")
    url_train, url_valid = URL_MIND[size]
    with download_path(dest_path) as path:
        train_path = maybe_download(url=url_train, work_directory=path)
        valid_path = maybe_download(url=url_valid, work_directory=path)
    return train_path, valid_path


def extract_mind(
    train_zip,
    valid_zip,
    train_folder="train",
    valid_folder="valid",
    clean_zip_file=True,
):
    """Extract MIND dataset

    Args:
        train_zip (str): Path to train zip file
        valid_zip (str): Path to valid zip file
        train_folder (str): Destination forder for train set
        valid_folder (str): Destination forder for validation set

    Returns:
        str, str: Train and validation folders
    """
    root_folder = os.path.basename(train_zip)
    train_path = os.path.join(root_folder, train_folder)
    valid_path = os.path.join(root_folder, valid_folder)
    unzip_file(train_zip, train_path, clean_zip_file=clean_zip_file)
    unzip_file(valid_zip, valid_path, clean_zip_file=clean_zip_file)
    return train_path, valid_path

