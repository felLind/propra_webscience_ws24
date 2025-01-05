"""
This module provides functionality to train different ml models.
"""

from typing import Iterator
import pandas as pd
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
import click

from propra_webscience_ws24 import constants
from propra_webscience_ws24.data import data_preprocessing
from propra_webscience_ws24.training.model.model_base import ModelType
from propra_webscience_ws24.training.model.model_factory import  model_factory
from propra_webscience_ws24.training.training_combinations import (
    USE_ALL_FEATURES_SPECIFIER,
    TrainingCombination,
    was_combination_already_processed,
    NGramRange,
    VectorizerConfig,
    MAX_FEATURES_LIST,
)
from propra_webscience_ws24.training.training_results import (
    ClassificationResult,
    add_classification_results_to_df,
    get_existing_classification_results,
)



MAX_WORKERS_DEFAULT = 1


@click.command()
@click.option(
    "--model-type",
    required=True,
    type=click.Choice([e.name for e in ModelType]),
    help="Type of model to train.",
)
@click.option(
    "--normalization-strategy",
    required=False,
    type=click.Choice([e.name for e in data_preprocessing.TextNormalizationStrategy]),
    help="Normalization strategy to use.",
)
@click.option(
    "--stopword-removal-strategy",
    required=False,
    type=click.Choice([e.name for e in data_preprocessing.StopwordRemovalStrategy]),
    help="Stopword removal strategy to use.",
)
@click.option(
    "--vectorizer",
    required=False,
    type=click.Choice([e.name for e in VectorizerConfig]),
    help="Vectorizer to use.",
)
@click.option(
    "--max-features",
    required=False,
    type=click.Choice([str(e) for e in MAX_FEATURES_LIST]),
    help="Maximum number of features for the vectorizer.",
)
@click.option(
    "--ngram-range",
    required=False,
    type=click.Choice([e.name for e in NGramRange]),
    help="N-gram range for the vectorizer.",
)
@click.option(
    "--max-workers",
    default=MAX_WORKERS_DEFAULT,
    type=int,
    help="Number of workers to use.",
)

@click.option(
    "--model-args",
    required=False,
    type=str,
    help="Arguments to pass to the model used for training.(Key1:value,Key2:value,...)",
)

def main(
    model_type: str,
    normalization_strategy: str | None,
    stopword_removal_strategy: str | None,
    vectorizer: str | None,
    max_features: str | None,
    ngram_range: str | None,
    max_workers: int,
    model_args: str | None
):
    """
    Train a model using the specified configurations.

    Args:
        model_type (str): Type of model to train.
        normalization_strategy (str | None): Normalization strategy to use.
        stopword_removal_strategy (str | None): Stopword removal strategy to use.
        vectorizer (str | None): Vectorizer to use.
        max_features (str | None): Maximum number of features for the vectorizer.
        ngram_range (str | None): N-gram range for the vectorizer.
        max_workers (int): Number of workers to use, the default is one.
        model_args (str | None): Arguments to pass to the model used for training. (Key1:value,Key2:value,...)

    Returns:
        pd.DataFrame: DataFrame containing the results of the training.
    """
    model_type_ = ModelType[model_type]
    max_features_ = (
        int(max_features)
        if max_features is not None and max_features != USE_ALL_FEATURES_SPECIFIER
        else max_features
    )
    max_workers = max_workers if max_workers > 0 else MAX_WORKERS_DEFAULT

    logger.info("Given parameters:")
    logger.info(f"\tModel type: {model_type_}")
    logger.info(f"\tNormalization strategy: {normalization_strategy}")
    logger.info(f"\tStopword removal strategy: {stopword_removal_strategy}")
    logger.info(f"\tVectorizer: {vectorizer}")
    logger.info(f"\tMax features: {max_features_}")
    logger.info(f"\tN-gram range: {ngram_range}")
    logger.info(f"\tModel Arguments: {model_args}")

    training_combinations = TrainingCombination.create_training_combination_subset(
        normalization_strategy=normalization_strategy,
        stopword_removal_strategy=stopword_removal_strategy,
        vectorizer=vectorizer,
        max_features=max_features_,
        ngram_range=ngram_range,
    )

    return _train_all_combinations(
        model_type_,
        training_combinations,
        _parse_model_args(model_args),
        max_workers=max_workers,
    )

def _parse_model_args(model_args: str) -> dict:
    result = {}
    if model_args is not None:
        model_args = model_args.replace(" ", "").replace("=", ":")
        for key_value in model_args.split(","):
            kv = key_value.split(":")
            result[kv[0]] = kv[1]
    return result

def _train_all_combinations(
    model_type: ModelType,
    training_combinations: Iterator[TrainingCombination],
    model_args: dict,
    max_workers: int = MAX_WORKERS_DEFAULT,
) -> pd.DataFrame:
    trained_combinations_df = get_existing_classification_results()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        all_training_combinations = set(training_combinations)
        skipped_combinations = set(
            training_combination
            for training_combination in all_training_combinations
            if was_combination_already_processed(
                trained_combinations_df, model_type.value, training_combination
            )
        )
        logger.info(
            f"Training {len(all_training_combinations - skipped_combinations)} combinations."
        )
        logger.info(
            f"Skipping {len(skipped_combinations)} already processed combinations."
        )

        futures = {
            executor.submit(
                _train_single_combination, training_combination, model_type, model_args
            ): training_combination
            for training_combination in all_training_combinations - skipped_combinations
        }

        for future in as_completed(futures):
            training_combination = futures[future]
            try:
                result = future.result()
                trained_combinations_df = add_classification_results_to_df(
                    trained_combinations_df, result
                )
            except Exception as exc:
                logger.error(f"{training_combination} generated an exception: {exc}")

    return trained_combinations_df


def _train_single_combination(
    training_combination: TrainingCombination, model_type: ModelType, model_args: dict
) -> ClassificationResult:
    df_train, df_test = _load_preprocessed_datasets(
        training_combination.normalization_strategy,
        training_combination.stopword_removal_strategy,
    )

    return model_factory(model_type, df_train, df_test, training_combination, model_args).train_model()


def _load_preprocessed_datasets(
    normalization_strategy: data_preprocessing.TextNormalizationStrategy,
    stopword_strategy: data_preprocessing.StopwordRemovalStrategy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = constants.get_processed_filepath(
        normalization_strategy.value, stopword_strategy.value, "train"
    )
    test_path = constants.get_processed_filepath(
        normalization_strategy.value, stopword_strategy.value, "test"
    )
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "File with pre-processed train or test data not found. "
            "Please preprocess the data first."
        )

    df_train = pd.read_parquet(train_path)

    df_test = pd.read_parquet(test_path)
    df_test = df_test.loc[(df_test.sentiment == 0) | (df_test.sentiment == 4), :]

    return df_train, df_test


if __name__ == "__main__":
    main()
