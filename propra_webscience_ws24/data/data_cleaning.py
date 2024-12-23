import pathlib
import re

import pandas as pd
from loguru import logger

from propra_webscience_ws24 import constants

# RegEx patterns used to sanitize tweets
URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
USER_MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#")
DIGIT_PATTERN = re.compile(r"\d+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s\']")


def load_or_clean_tweet_data(
    text_column: str = "text",
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Clean tweets by removing URLs, mentions, hashtags, digits, and special characters.

    The cleaned data is saved to a parquet file for future use.
    If the cleaned data already exists, it is loaded from the parquet file.

    Args:
        text_column: The name of the column containing the tweets in the DataFrame with the original data.

    Returns:
        dataframe_created: A boolean indicating whether the cleaned data was created or loaded.
        A tuple of DataFrames containing the cleaned tweets for the training and test data.
    """

    train_dataframe_created, df_train_cleaned = _get_or_create_cleaned_tweet_data(
        constants.CLEANED_TRAIN_TWEETS, "train", text_column
    )
    test_dataframe_created, df_test_cleaned = _get_or_create_cleaned_tweet_data(
        constants.CLEANED_TEST_TWEETS, "test", text_column
    )

    return (
        train_dataframe_created or test_dataframe_created,
        df_train_cleaned,
        df_test_cleaned,
    )


def _get_or_create_cleaned_tweet_data(
    cleaned_df_file_path: pathlib.Path, split: str, text_column: str
) -> tuple[bool, pd.DataFrame]:
    dataframe_created = False
    if cleaned_df_file_path.exists():
        logger.debug(
            f"Loading cleaned {split} tweets from "
            f"{cleaned_df_file_path.relative_to(constants.DATASETS_ROOT_PATH.parent.parent)}"
        )
        df_cleaned = pd.read_parquet(cleaned_df_file_path)
    else:
        logger.info(f"Cleaning {split} tweets...")
        dataframe_created = True
        df_cleaned = pd.read_parquet(constants.TRAIN_DATASET_FILE_PATH)
        df_cleaned["cleaned_text"] = df_cleaned[text_column].apply(_sanitize_tweets)
        df_cleaned.to_parquet(cleaned_df_file_path)

    return dataframe_created, df_cleaned


def _sanitize_tweets(text):
    text = URL_PATTERN.sub("", text)
    text = USER_MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)
    text = DIGIT_PATTERN.sub("", text)
    text = SPECIAL_CHAR_PATTERN.sub("", text)

    return text
