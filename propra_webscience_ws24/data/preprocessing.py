import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import (
    WordNetLemmatizer,
    PorterStemmer,
)
from typing import Callable
from enum import Enum
import pandas as pd
import time
import humanize

from propra_webscience_ws24 import constants
from propra_webscience_ws24.constants import get_processed_filepath
from multiprocessing import Pool

_CUSTOM_STOPWORDS = None


class TextNormalizationStrategy(Enum):
    LEMMATIZER = "lemmatizer"
    PORTER = "porter"
    # LANCASTER = "lancaster"
    NONE = "none"


def get_cleaned_tweets(text_column: str = "text") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and cache tweets if not already done"""

    if constants.CLEANED_TRAIN_TWEETS.exists():
        df_train_cleaned = pd.read_parquet(constants.CLEANED_TRAIN_TWEETS)
    else:
        df_train_cleaned = pd.read_parquet(constants.TRAIN_DATASET_FILE_PATH)
        df_train_cleaned["cleaned_text"] = df_train_cleaned[text_column].apply(
            _sanitize_tweets
        )
        df_train_cleaned.to_parquet(constants.CLEANED_TRAIN_TWEETS)

    if constants.CLEANED_TEST_TWEETS.exists():
        df_test_cleaned = pd.read_parquet(constants.CLEANED_TEST_TWEETS)
    else:
        df_test_cleaned = pd.read_parquet(constants.TEST_DATASET_FILE_PATH)
        df_test_cleaned["cleaned_text"] = df_test_cleaned[text_column].apply(
            _sanitize_tweets
        )
        df_test_cleaned.to_parquet(constants.CLEANED_TEST_TWEETS)

    return df_train_cleaned, df_test_cleaned


def process_and_save_tweets(
    df_train_cleaned: pd.DataFrame,
    df_test_cleaned: pd.DataFrame,
    normalization_strategy: TextNormalizationStrategy,
    remove_stopwords: bool,
):
    """
    Process tweets with configurable options and save the processed data.
    """
    for split in ["train", "test"]:
        output_path = get_processed_filepath(
            normalization_strategy.value, remove_stopwords, split
        )

        if output_path.exists():
            print(
                f"Skipping data processing as output file already exists: "
                f"{output_path.relative_to(constants.DATASETS_ROOT_PATH.parent.parent)}"
            )
            continue

        print(
            f"Processing {split} tweets with normalization_method="
            f"{normalization_strategy.value}, stopwords="
            f"{'removed' if remove_stopwords else 'kept'}"
        )

        df_to_process = df_train_cleaned if split == "train" else df_test_cleaned

        start_time = time.perf_counter()
        df_processed = _preprocess_tweets(
            df_to_process,
            remove_stopwords=remove_stopwords,
            normalization_strategy=normalization_strategy,
        )
        end_time = time.perf_counter()

        df_processed.to_parquet(output_path)

        elapsed_time = end_time - start_time
        print(
            f"Completed processing {split} data in {humanize.precisedelta(elapsed_time, minimum_unit='seconds')}."
        )


def _sanitize_tweets(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s\']", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text.lower().strip()


def _get_tokenizer(
    normlization_strategy: TextNormalizationStrategy,
) -> Callable[[str], str]:
    tokenizers = {
        TextNormalizationStrategy.LEMMATIZER: WordNetLemmatizer().lemmatize,
        TextNormalizationStrategy.PORTER: PorterStemmer().stem,
        # TextNormalizationStrategy.LANCASTER: LancasterStemmer().stem,
        TextNormalizationStrategy.NONE: lambda x: x,
    }
    return tokenizers[normlization_strategy]


def _normalize_text(
    text: str,
    remove_stopwords: bool = True,
    normalization_strategy: TextNormalizationStrategy = TextNormalizationStrategy.LEMMATIZER,
) -> str:
    tweet_tokenizer = TweetTokenizer(preserve_case=False)
    tokens = tweet_tokenizer.tokenize(text)
    tokenizer = _get_tokenizer(normalization_strategy)
    tokens = [tokenizer(token) for token in tokens]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _get_custom_stopwords()]

    return " ".join(tokens)


def _get_custom_stopwords():
    global _CUSTOM_STOPWORDS
    if _CUSTOM_STOPWORDS is None:
        stop_words = set(stopwords.words("english"))

        negation_words = {
            "no",
            "not",
            "nor",
            "neither",
            "never",
            "none" "don't",
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
        _CUSTOM_STOPWORDS = {
            word
            for word in stop_words
            if not any([negation in word for negation in negation_words])
        }
    return _CUSTOM_STOPWORDS


def _preprocess_tweets(
    df: pd.DataFrame,
    remove_stopwords: bool = True,
    normalization_strategy: TextNormalizationStrategy = TextNormalizationStrategy.LEMMATIZER,
) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, "processed_text"] = df.loc[:, "cleaned_text"].apply(
        _normalize_text,
        remove_stopwords=remove_stopwords,
        normalization_strategy=normalization_strategy,
    )
    return df


if __name__ == "__main__":
    df_train_cleaned, df_test_cleaned = get_cleaned_tweets("text")

    def _process_tweets_in_separate_process(args):
        (
            normalization_strategy,
            remove_stopwords,
            df_train_cleaned,
            df_test_cleaned,
        ) = args
        process_and_save_tweets(
            df_train_cleaned=df_train_cleaned,
            df_test_cleaned=df_test_cleaned,
            normalization_strategy=normalization_strategy,
            remove_stopwords=remove_stopwords,
        )

    with Pool() as pool:
        pool.map(
            _process_tweets_in_separate_process,
            [
                (
                    normalization_strategy,
                    remove_stopwords,
                    df_train_cleaned,
                    df_test_cleaned,
                )
                for normalization_strategy in TextNormalizationStrategy
                for remove_stopwords in [True, False]
            ],
        )
