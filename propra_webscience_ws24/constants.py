import pathlib
from typing import Literal


# Datasets
DATASETS_ROOT_PATH = pathlib.Path(__file__).parent.parent / "datasets" / "sentiment140"

RAW_TRAIN_DATASET_FILE_PATH = DATASETS_ROOT_PATH / "train-data-orig.csv"
RAW_TEST_DATASET_FILE_PATH = DATASETS_ROOT_PATH / "test-data-orig.csv"

TRAIN_DATASET_FILE_NAME = "train-data-orig.parquet"
TEST_DATASET_FILE_NAME = "test-data-orig.parquet"

TRAIN_DATASET_FILE_PATH = DATASETS_ROOT_PATH / TRAIN_DATASET_FILE_NAME
TEST_DATASET_FILE_PATH = DATASETS_ROOT_PATH / TEST_DATASET_FILE_NAME

SPLITS = ["train", "test"]
SPLITS_LITERAL = Literal["train", "test"]

# cleaned tweets
CLEANED_TWEETS_PATH = DATASETS_ROOT_PATH / "cleaned_tweets"
CLEANED_TWEETS_PATH.mkdir(parents=True, exist_ok=True)

CLEANED_TRAIN_TWEETS = CLEANED_TWEETS_PATH / "cleaned-train-tweets.parquet"
CLEANED_TEST_TWEETS = CLEANED_TWEETS_PATH / "cleaned-test-tweets.parquet"

# processed tweets
PROCESSED_TWEETS_PATH = DATASETS_ROOT_PATH / "processed_tweets"
PROCESSED_TWEETS_PATH.mkdir(parents=True, exist_ok=True)


def get_processed_filepath(
    normalization_strategy: str, stopwords_removal_strategy: str, split: SPLITS_LITERAL
) -> pathlib.Path:
    """
    Get the path to the processed tweets file for the given parameters.

    Args:
        normalization_strategy: The text normalization strategy to use.
        stopwords_removal_strategy: The stopword removal strategy to use.
        split: The split for which to get the processed tweets file.
    """

    filename = (
        f"{split}-tweets-{normalization_strategy}-{stopwords_removal_strategy}.parquet"
    )
    return PROCESSED_TWEETS_PATH / filename


# Models
MODELS_PATH = DATASETS_ROOT_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)


# Results
RESULTS_PATH = DATASETS_ROOT_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_RESULTS_PARQUET_PATH = RESULTS_PATH / "classification-results.parquet"
CLASSIFICATION_RESULTS_SAVED_PARQUET_PATH = (
    RESULTS_PATH / "classification-results.parquet"
)

# Visualization
PLOTS_PATH = RESULTS_PATH / "plots"
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
