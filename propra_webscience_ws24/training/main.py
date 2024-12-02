import dataclasses
from functools import lru_cache
import os
from typing import List, Tuple
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from enum import Enum
import pandas as pd
import concurrent.futures
from multiprocessing import cpu_count

from propra_webscience_ws24 import constants
from propra_webscience_ws24.data.preprocessing import TextNormalizationStrategy
from propra_webscience_ws24.training import svm
from propra_webscience_ws24.training.svm import (
    ClassificationResult,
    NGramRange,
    TrainingCombination,
)


class VectorizerConfig(Enum):
    TFIDF = TfidfVectorizer
    # COUNT = CountVectorizer
    HASHING = HashingVectorizer


N_PARALLEL_JOBS = os.cpu_count() // 2
NGRAM_RANGES = [
    NGramRange.UNIGRAMS,
    NGramRange.UNI_AND_BIGRAMS,
    # NGramRange.UNI_AND_BI_AND_TRIGRAMS,
]
MAX_FEATURES_LIST = [None, 10_000, 25_000, 50_000, 100_000, 150_000]


def train_all_combinations() -> pd.DataFrame:
    trained_combinations_df = _get_existing_classification_results()

    for tokenizer in TextNormalizationStrategy:
        for remove_stopwords in [True, False]:
            print(f"\nProcessing {tokenizer.value}: {remove_stopwords=}")

            df_train, df_test = _load_preprocessed_datasets(
                tokenizer.value, remove_stopwords
            )

            for vectorizer_config in VectorizerConfig:
                results = _train_with_vectorizer_parameter_combinations_parallel(
                    tokenizer,
                    remove_stopwords,
                    df_train,
                    df_test,
                    vectorizer_config,
                    trained_combinations_df,
                )
                trained_combinations_df = _add_classification_results_to_df(
                    trained_combinations_df, results
                )

                trained_combinations_df.to_parquet(
                    constants.CLASSIFICATION_RESULTS_PARQUET_PATH
                )

    return trained_combinations_df


def _get_existing_classification_results() -> pd.DataFrame | None:
    output_file = constants.CLASSIFICATION_RESULTS_PARQUET_PATH
    if os.path.exists(output_file):
        return pd.read_parquet(output_file)

    return pd.DataFrame(
        columns=[field.name for field in dataclasses.fields(TrainingCombination)]
        + [
            "report_training_data_test_split",
            "report_test_data",
            "processing_duration",
        ]
    )


def _load_preprocessed_datasets(
    tokenizer: str, remove_stopwords: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = constants.get_processed_filepath(tokenizer, remove_stopwords, "train")
    test_path = constants.get_processed_filepath(tokenizer, remove_stopwords, "test")

    df_train = pd.read_parquet(train_path)

    df_test = pd.read_parquet(test_path)
    df_test = df_test.loc[(df_test.sentiment == 0) | (df_test.sentiment == 4), :]

    return df_train, df_test


def _add_classification_results_to_df(
    df: pd.DataFrame, classification_results: list[ClassificationResult]
):
    entries_to_add = []
    for result in classification_results:
        training_combination = result.training_combination

        result_dict = dataclasses.asdict(result)
        del result_dict["training_combination"]

        training_combination_dict = dataclasses.asdict(training_combination)
        training_combination_dict["ngram_range"] = training_combination.ngram_range.name
        training_combination_dict["vectorizer"] = training_combination.vectorizer_name

        entries_to_add.append({**training_combination_dict, **result_dict})

    new_results_df = pd.DataFrame(entries_to_add)
    return pd.concat([df, new_results_df], ignore_index=True)


def _train_with_vectorizer_parameter_combinations_parallel(
    normalization_strategy: TextNormalizationStrategy,
    remove_stopwords: bool,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    vectorizer_config: VectorizerConfig,
    trained_combinations_df: pd.DataFrame,
) -> List[ClassificationResult]:
    training_combinations = (
        TrainingCombination(
            normalization_strategy=normalization_strategy.value,
            remove_stopwords=remove_stopwords,
            vectorizer=_instantiate_vectorizer(
                vectorizer_config.value,
                max_features,
                ngram_range.value,
            ),
            max_features=max_features,
            ngram_range=ngram_range,
        )
        for ngram_range in NGRAM_RANGES
        for max_features in MAX_FEATURES_LIST
    )

    training_combinations_to_train = [
        training_combination
        for training_combination in training_combinations
        if not _was_combination_already_processed(
            df=trained_combinations_df,
            training_combination=training_combination,
        )
        and not training_combination.should_combination_be_discarded
    ]
    print(f"Training {len(training_combinations_to_train)} combinations...")

    parallel_ngram_max_features_results = train_linear_svc_parallel(
        df_train, df_test, training_combinations_to_train, n_jobs=N_PARALLEL_JOBS
    )

    return parallel_ngram_max_features_results


@lru_cache
def _instantiate_vectorizer(
    vectorizer_class: TfidfVectorizer | CountVectorizer | HashingVectorizer,
    max_features: int | None,
    ngram_range_tuple: tuple[int, int],
) -> CountVectorizer | TfidfVectorizer | HashingVectorizer:
    kwargs = {"max_features": max_features} if max_features else {}
    if vectorizer_class == HashingVectorizer:
        kwargs = {"n_features": (max_features if max_features else 2**20)}
    kwargs |= {"ngram_range": ngram_range_tuple}

    return vectorizer_class(**kwargs)


def _was_combination_already_processed(
    df: pd.DataFrame,
    training_combination: TrainingCombination,
) -> bool:
    result = df.loc[
        (df.normalization_strategy == training_combination.normalization_strategy)
        & (df.remove_stopwords == training_combination.remove_stopwords)
        & (df.vectorizer == training_combination.vectorizer_name)
        & (df.ngram_range == training_combination.ngram_range.name)
        & (
            df.max_features == training_combination.max_features
            if training_combination.max_features is not None
            else df.max_features.isna()
        )
    ]
    return len(result) >= 1


def _train_linear_svc_wrapper(args):
    df_train, df_test, training_combination = args
    return svm.train_linear_svc(
        df_train,
        df_test,
        training_combination,
    )


def train_linear_svc_parallel(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    training_combinations_to_train: list[TrainingCombination],
    n_jobs: int | None = None,
):
    if n_jobs is None:
        n_jobs = cpu_count()  # Standardmäßig alle verfügbaren CPU-Kerne nutzen

    args_list = [
        (df_train, df_test, training_combination)
        for training_combination in training_combinations_to_train
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_combination = {
            executor.submit(_train_linear_svc_wrapper, args): args for args in args_list
        }

        for future in concurrent.futures.as_completed(future_to_combination):
            args = future_to_combination[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Training abgeschlossen für Kombination: {args[2]}")
            except Exception as exc:
                print(
                    f"Training erzeugte eine Ausnahme für Kombination {args[2]}: {exc}"
                )

    return results


if __name__ == "__main__":
    classification_results_df = train_all_combinations()
    classification_results_df.to_parquet(constants.CLASSIFICATION_RESULTS_PARQUET_PATH)
