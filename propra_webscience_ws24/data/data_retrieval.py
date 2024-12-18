"""This module provides functions to retrieve the original dataset and convert it into a DataFrame."""

import pandas as pd
import requests
import shutil
import zipfile
from loguru import logger
from propra_webscience_ws24.constants import (
    DATASETS_ROOT_PATH,
    RAW_TRAIN_DATASET_FILE_PATH,
    RAW_TEST_DATASET_FILE_PATH,
    SPLITS_LITERAL,
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
)

DATASET_COLUMNS = ["sentiment", "text"]


def download_and_convert_original_dataset_to_dataframes():
    """
    Download the original dataset and convert it into DataFrames.
    """
    _download_original_dataset()
    _create_dataframe_for_split("train")
    _create_dataframe_for_split("test")


def _download_original_dataset():
    """
    Download and extract the original dataset if it does not already exist.
    """
    if (
        not RAW_TRAIN_DATASET_FILE_PATH.is_file()
        or not RAW_TEST_DATASET_FILE_PATH.is_file()
    ):
        try:
            logger.info("Downloading original dataset files...")
            response = requests.get(
                "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
                stream=True,
                timeout=10,
            )
            response.raise_for_status()

            zip_file_path = DATASETS_ROOT_PATH / "original_file.zip"
            with open(zip_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            logger.info("Extracting original dataset files to {}", DATASETS_ROOT_PATH)
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(DATASETS_ROOT_PATH)

            shutil.move(
                DATASETS_ROOT_PATH / "training.1600000.processed.noemoticon.csv",
                RAW_TRAIN_DATASET_FILE_PATH,
            )
            shutil.move(
                DATASETS_ROOT_PATH / "testdata.manual.2009.06.14.csv",
                RAW_TEST_DATASET_FILE_PATH,
            )

            zip_file_path.unlink()
            logger.info("Downloaded and extracted original dataset files successfully")
        except Exception as e:
            logger.error("Failed to download or extract dataset: {}", e)
            raise


def _create_dataframe_for_split(split_name: SPLITS_LITERAL):
    """
    Create a DataFrame for the specified split (train or test) and save it as a Parquet file.
    """
    file_path = (
        TRAIN_DATASET_FILE_PATH if split_name == "train" else TEST_DATASET_FILE_PATH
    )

    if file_path.is_file():
        logger.info("Loading {} dataset from parquet file...", split_name)
        return pd.read_parquet(file_path)

    raw_dataset_file_path = (
        RAW_TRAIN_DATASET_FILE_PATH
        if split_name == "train"
        else RAW_TEST_DATASET_FILE_PATH
    )

    logger.info("Converting {} dataset into dataframe...", split_name)
    df = pd.read_csv(
        f"{raw_dataset_file_path}",
        encoding="latin-1",
        names=DATASET_COLUMNS,
        usecols=[0, 5],
        header=None,
    )
    df.to_parquet(file_path)
    logger.info(
        "Converted {} dataset into dataframe and saved to parquet file", split_name
    )
