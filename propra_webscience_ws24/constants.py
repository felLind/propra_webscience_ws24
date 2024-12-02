import pathlib
from typing import Literal

# Datasets
DATASETS_ROOT_PATH = pathlib.Path(__file__).parent.parent / "datasets" / "sentiment140"

TRAIN_DATASET_FILE_NAME = "train-data.parquet"
TEST_DATASET_FILE_NAME = "test-data.parquet"

TRAIN_DATASET_FILE_PATH = DATASETS_ROOT_PATH / TRAIN_DATASET_FILE_NAME
TEST_DATASET_FILE_PATH = DATASETS_ROOT_PATH / TEST_DATASET_FILE_NAME

# cleaned tweets
CLEANED_TWEETS_PATH = DATASETS_ROOT_PATH / "cleaned_tweets"
CLEANED_TWEETS_PATH.mkdir(parents=True, exist_ok=True)

CLEANED_TRAIN_TWEETS = CLEANED_TWEETS_PATH / "cleaned-train-tweets.parquet"
CLEANED_TEST_TWEETS = CLEANED_TWEETS_PATH / "cleaned-test-tweets.parquet"

# processed tweets
PROCESSED_TWEETS_PATH = DATASETS_ROOT_PATH / "processed_tweets"
PROCESSED_TWEETS_PATH.mkdir(parents=True, exist_ok=True)


def get_processed_filepath(
    normalization_strategy: str, remove_stopwords: bool, split: Literal["test", "train"]
) -> pathlib.Path:
    """Generate standardized filepath for processed tweets"""

    filename = f"{split}-tweets-{normalization_strategy}-{'wo' if remove_stopwords else 'w'}-stopwords.parquet"
    return PROCESSED_TWEETS_PATH / filename


# Word Embeddings
GLOVE_EMBEDDING_27B_25D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.25d.txt"
)
GLOVE_EMBEDDING_27B_50D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.50d.txt"
)
GLOVE_EMBEDDING_27B_100D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.100d.txt"
)
GLOVE_EMBEDDING_27B_200D_FILE_PATH = (
    DATASETS_ROOT_PATH / "word_embeddings" / "glove.twitter.27B.200d.txt"
)

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
