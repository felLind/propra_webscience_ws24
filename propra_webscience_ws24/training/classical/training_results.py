"""
This module provides functionality to store and retrieve classification results.
"""

import os
import pandas as pd
from propra_webscience_ws24 import constants
from propra_webscience_ws24.training.classical.training_combinations import (
    TrainingCombination,
)


import dataclasses


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    """
    A dataclass to store the results of a classification run.
    """

    model_type: str
    training_combination: TrainingCombination
    report_training_data: dict
    report_test_data: dict
    processing_duration: float
    y_pred: list[int]

    @property
    def test_accuracy(self) -> float:
        return self.report_test_data["accuracy"]

    @property
    def train_accuracy(self) -> float:
        return self.report_training_data["accuracy"]


def add_classification_results_to_df(
    df: pd.DataFrame,
    classification_result: ClassificationResult,
) -> pd.DataFrame:
    """
    Add the classification results to the existing DataFrame and save it.

    Args:
        df: The existing DataFrame.
        classification_result: The classification results to add.

    Returns:
        The updated DataFrame.
    """
    new_row = {
        "model_type": classification_result.model_type,
        "normalization_strategy": classification_result.training_combination.normalization_strategy.value,
        "stopword_removal_strategy": classification_result.training_combination.stopword_removal_strategy.value,
        "vectorizer": classification_result.training_combination.vectorizer_name,
        "max_features": str(classification_result.training_combination.max_features),
        "ngram_range": classification_result.training_combination.ngram_range.name,
        "vocab_size": classification_result.training_combination.vocab_size,
        "training_accuracy": classification_result.train_accuracy,
        "test_accuracy": classification_result.test_accuracy,
        "processing_duration": classification_result.processing_duration,
        "report_training_data": classification_result.report_training_data,
        "report_test_data": classification_result.report_test_data,
    }

    result = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    result.to_parquet(constants.CLASSICAL_ML_RESULTS_PARQUET_PATH)

    return result


def get_existing_classification_results() -> pd.DataFrame | None:
    """
    Get the existing classification results from the parquet file.

    If the file does not exist, return an empty DataFrame.

    Returns:
        The existing classification results.
    """
    output_file = constants.CLASSICAL_ML_RESULTS_PARQUET_PATH
    if os.path.exists(output_file):
        return pd.read_parquet(output_file)

    df = pd.DataFrame(
        columns=[
            "model_type",
            "normalization_strategy",
            "stopword_removal_strategy",
            "vectorizer",
            "max_features",
            "ngram_range",
            "vocab_size",
            "training_accuracy",
            "test_accuracy",
            "processing_duration",
            "report_training_data",
            "report_test_data",
        ]
    )
    df.max_features = df.max_features.astype(str)
    return df
