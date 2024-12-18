import re

from loguru import logger
from propra_webscience_ws24 import constants


import pandas as pd


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

    dataframe_created = False
    if constants.CLEANED_TRAIN_TWEETS.exists():
        logger.debug(
            "Loading cleaned training tweets from "
            f"{constants.CLEANED_TRAIN_TWEETS.relative_to(constants.DATASETS_ROOT_PATH.parent.parent)}"
        )
        df_train_cleaned = pd.read_parquet(constants.CLEANED_TRAIN_TWEETS)
    else:
        logger.info("Cleaning training tweets...")
        dataframe_created = True
        df_train_cleaned = pd.read_parquet(constants.TRAIN_DATASET_FILE_PATH)
        df_train_cleaned["cleaned_text"] = df_train_cleaned[text_column].apply(
            _sanitize_tweets
        )
        df_train_cleaned.to_parquet(constants.CLEANED_TRAIN_TWEETS)

    if constants.CLEANED_TEST_TWEETS.exists():
        logger.debug(
            "Loading cleaned test tweets from "
            f"{constants.CLEANED_TEST_TWEETS.relative_to(constants.DATASETS_ROOT_PATH.parent.parent)}"
        )
        df_test_cleaned = pd.read_parquet(constants.CLEANED_TEST_TWEETS)
    else:
        logger.info("Cleaning test tweets...")
        dataframe_created = True
        df_test_cleaned = pd.read_parquet(constants.TEST_DATASET_FILE_PATH)
        df_test_cleaned["cleaned_text"] = df_test_cleaned[text_column].apply(
            _sanitize_tweets
        )
        df_test_cleaned.to_parquet(constants.CLEANED_TEST_TWEETS)

    return dataframe_created, df_train_cleaned, df_test_cleaned


def _sanitize_tweets(text):
    text = URL_PATTERN.sub("", text)
    text = USER_MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)
    text = DIGIT_PATTERN.sub("", text)
    text = SPECIAL_CHAR_PATTERN.sub("", text)

    return text
