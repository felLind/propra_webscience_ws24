"""
This module provides functions for training SVM models using scikit-learn’s LinearSVC.
"""

import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd
import time
import humanize

from propra_webscience_ws24 import constants
from propra_webscience_ws24.training.training_results import ClassificationResult
from propra_webscience_ws24.training.training_combinations import TrainingCombination


def train_linear_svc(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model_type: str,
    training_combination: TrainingCombination,
) -> ClassificationResult:
    """
    Train a linear SVC model using the specified training combination.

    Args:
        df_train (pd.DataFrame): Training data with 'processed_text' and 'sentiment' columns.
        df_test (pd.DataFrame): Test data with 'processed_text' for evaluation.
        training_combination (TrainingCombination): Holds vectorizer and related configurations.

    Returns:
        ClassificationResult: Object containing metrics, model details, and runtime information.
    """
    start_time = time.perf_counter()

    X_train = training_combination.vectorizer.fit_transform(df_train["processed_text"])
    X_test = training_combination.vectorizer.transform(df_test["processed_text"])

    report, model = _train_linear_svc(X_train, df_train.sentiment)
    y_pred = model.predict(X_test)
    test_report = classification_report(df_test.sentiment, y_pred, output_dict=True)

    processing_duration = time.perf_counter() - start_time

    logger.info(
        f"{training_combination} | "
        f"processing_duration={humanize.precisedelta(processing_duration, minimum_unit='seconds')}"
        f" | test_accuracy={test_report['accuracy']:.3f}"
    )

    _save_model(model_type, model, training_combination)

    return ClassificationResult(
        model_type="LinearSVC",
        training_combination=training_combination,
        processing_duration=processing_duration,
        report_training_data=report,
        report_test_data=test_report,
        y_pred=y_pred,
    )


def _train_linear_svc(X, y) -> tuple[dict, LinearSVC]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True), model


def _save_model(
    model_type: str, model: LinearSVC, training_combination: TrainingCombination
):
    model_path = f"{constants.MODELS_PATH}/{model_type}-{training_combination.model_name}_model.joblib"
    joblib.dump(model, model_path)
