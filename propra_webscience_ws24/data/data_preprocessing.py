"""This module provides functions to clean and preprocess tweets."""

from functools import lru_cache, partial
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import (
    WordNetLemmatizer,
    PorterStemmer,
)
from typing import Callable, get_args
from enum import Enum
import pandas as pd
import time
import humanize
from loguru import logger

from propra_webscience_ws24 import constants
from propra_webscience_ws24.constants import (
    SPLITS_LITERAL,
    get_processed_filepath,
)
from multiprocessing import Pool

from propra_webscience_ws24.data import data_retrieval
from propra_webscience_ws24.data.data_cleaning import load_or_clean_tweet_data


_CUSTOM_STOPWORDS = None
DEFAULT_NLTK_STOPWORDS = set(stopwords.words("english"))


class TextNormalizationStrategy(Enum):
    """
    Enum to represent different text normalization strategies.
    """

    LEMMATIZER = "lemmatizer"
    PORTER = "porter"
    NONE = "none"


class StopwordRemovalStrategy(Enum):
    """
    Enum to represent different stopword removal strategies.
    """

    REMOVE_DEFAULT_NLTK = "wo-stopwords-nltk"
    REMOVE_CUSTOM = "wo-stopwords-custom"
    KEEP = "w-stopwords"


def process_and_save_tweets(
    df_train_cleaned: pd.DataFrame,
    df_test_cleaned: pd.DataFrame,
    always_create: bool,
    normalization_strategy: TextNormalizationStrategy,
    stopwords_removal_strategy: StopwordRemovalStrategy,
):
    """
    Preprocess and save the tweets for the training and test data.

    The preprocessed data is saved to a parquet file for future use.
    If the preprocessed data already exists, it is skipped.

    Args:
        df_train_cleaned: DataFrame containing the cleaned tweets for the training data.
        df_test_cleaned: DataFrame containing the cleaned tweets for the test data.
        always_create: A boolean indicating whether the preprocessed data should always be created.
        normalization_strategy: The text normalization strategy to use.
        stopwords_removal_strategy: The stopword removal strategy to use.
    """
    for split in get_args(SPLITS_LITERAL):
        output_path = get_processed_filepath(
            normalization_strategy.value, stopwords_removal_strategy.value, split
        )

        if not always_create and output_path.exists():
            logger.debug(
                "Skipping data processing as output file already exists: {}",
                output_path.relative_to(constants.DATASETS_ROOT_PATH.parent.parent),
            )
            continue

        logger.info(
            "Processing {} tweets with normalization_strategy={}, stopwords_strategy={}",
            split,
            normalization_strategy.value,
            stopwords_removal_strategy.value,
        )

        df_to_process = df_train_cleaned if split == "train" else df_test_cleaned

        start_time = time.perf_counter()
        df_processed = _preprocess_tweets(
            df_to_process,
            normalization_strategy=normalization_strategy,
            stopwords_removal_strategy=stopwords_removal_strategy,
        )
        end_time = time.perf_counter()
        elapsed_time = time.perf_counter() - start_time
        elapsed_time_delta = pd.to_timedelta(elapsed_time, unit="s")

        df_processed.to_parquet(output_path)

        elapsed_time = end_time - start_time
        logger.info(
            "Completed processing {} data with normalization_strategy={}, "
            "stopwords_strategy={} in {}.",
            split,
            normalization_strategy.value,
            stopwords_removal_strategy.value,
            humanize.precisedelta(elapsed_time_delta, minimum_unit="milliseconds"),
        )


def _get_normalization_function(
    normlization_strategy: TextNormalizationStrategy,
) -> Callable[[str], str]:
    text_normalization_functions = {
        TextNormalizationStrategy.LEMMATIZER: WordNetLemmatizer().lemmatize,
        TextNormalizationStrategy.PORTER: PorterStemmer().stem,
        TextNormalizationStrategy.NONE: lambda x: x,
    }

    return text_normalization_functions[normlization_strategy]


def _normalize_text(
    text: str,
    normalization_strategy: TextNormalizationStrategy,
    stopwords_removal_strategy: StopwordRemovalStrategy,
) -> str:
    tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tweet_tokenizer.tokenize(text)

    text_normalization_function = _get_normalization_function(normalization_strategy)
    normalized_tokens = [text_normalization_function(token) for token in tokens]

    if stopwords_removal_strategy == StopwordRemovalStrategy.KEEP:
        pass
    elif stopwords_removal_strategy == StopwordRemovalStrategy.REMOVE_DEFAULT_NLTK:
        normalized_tokens = [
            t for t in normalized_tokens if t not in DEFAULT_NLTK_STOPWORDS
        ]
    elif stopwords_removal_strategy == StopwordRemovalStrategy.REMOVE_CUSTOM:
        normalized_tokens = [
            t for t in normalized_tokens if t not in _get_custom_stopwords()
        ]
    else:
        raise NotImplementedError(
            f"Stopword removal strategy "
            f"{stopwords_removal_strategy} is not implemented."
        )

    return " ".join(normalized_tokens)


@lru_cache(maxsize=1)
def _get_custom_stopwords():
    """
    Get a set of custom stopwords that explicitly handles contractions and negations.
    """
    global _CUSTOM_STOPWORDS
    if _CUSTOM_STOPWORDS is None:
        stop_words = set(stopwords.words("english"))

        negation_words = {
            "no",
            "not",
            "nor",
            "neither",
            "never",
            "none",
            "don't",
            "doesn't",
            "didn't",
            "haven't",
            "hasn't",
            "won't",
            "wouldn't",
            "can't",
            "couldn't",
            "shouldn't",
            "weren't",
            "wasn't",
            "isn't",
            "aren't",
            "ain't",
            "mightn't",
            "mustn't",
            "needn't",
            "shan't",
            "oughtn't",
            "daren't",
            "hadn't",
        }
        _CUSTOM_STOPWORDS = stop_words - negation_words
    return _CUSTOM_STOPWORDS


def _preprocess_tweets(
    df: pd.DataFrame,
    normalization_strategy: TextNormalizationStrategy,
    stopwords_removal_strategy: StopwordRemovalStrategy,
) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, "processed_text"] = df.loc[:, "cleaned_text"].apply(
        _normalize_text,
        normalization_strategy=normalization_strategy,
        stopwords_removal_strategy=stopwords_removal_strategy,
    )
    return df


if __name__ == "__main__":
    data_retrieval.download_and_convert_original_dataset_to_dataframes()
    dfs_created, df_train_cleaned, df_test_cleaned = load_or_clean_tweet_data("text")

    args_list = [
        (normalization_strategy, stopwords_removal_strategy)
        for normalization_strategy in TextNormalizationStrategy
        for stopwords_removal_strategy in StopwordRemovalStrategy
    ]

    process_func = partial(
        process_and_save_tweets,
        df_train_cleaned,
        df_test_cleaned,
        dfs_created,
    )

    with Pool() as pool:
        logger.info("Starting multiprocessing for tweet processing")
        pool.starmap(process_func, args_list)
        logger.info("Completed multiprocessing for tweet processing")
