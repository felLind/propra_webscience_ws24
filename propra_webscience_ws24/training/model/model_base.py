"""
This module provides functions for training SVM models using scikit-learn’s LinearSVC.
"""
from enum import Enum

import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import time
import humanize
from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

from propra_webscience_ws24 import constants
from propra_webscience_ws24.training.training_results import ClassificationResult
from propra_webscience_ws24.training.training_combinations import TrainingCombination


class ModelType(Enum):
    LINEAR_SVC = "LinearSVC"
    DECISION_TREE = "DecisionTreeClassifier"
    KNN = "KNeighborsClassifier"
    LOGISTIC_REGRESSION = "LogisticRegression"
    NAIVE_BAYES = "NaiveBayes"


T = TypeVar('T', bound='BaseEstimator')

class ModelBase(Generic[T], metaclass=ABCMeta):
    def __init__(self,
                df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                training_combination: TrainingCombination):
        self.df_train = df_train
        self.df_test = df_test
        self.model_type = self._get_model_type()
        self.model = self._get_model()
        self.training_combination = training_combination

    @abstractmethod
    def _get_model(self) -> T:
        pass

    @abstractmethod
    def _get_model_type(self) -> ModelType:
        pass

    def train_model(self) -> ClassificationResult:
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

        X_train = self.training_combination.vectorizer.fit_transform(self.df_train["processed_text"])
        X_test = self.training_combination.vectorizer.transform(self.df_test["processed_text"])

        report, model = self._train_model(X_train, self.df_train.sentiment)
        y_pred = model.predict(X_test)
        test_report = classification_report(self.df_test.sentiment, y_pred, output_dict=True)

        processing_duration = time.perf_counter() - start_time

        logger.info(
            f"{self.training_combination} | "
            f"processing_duration={humanize.precisedelta(processing_duration, minimum_unit='seconds')}"
            f" | test_accuracy={test_report['accuracy']:.3f}"
        )

        self._save_model()

        return ClassificationResult(
            model_type=self.model_type,
            training_combination=self.training_combination,
            processing_duration=processing_duration,
            report_training_data=report,
            report_test_data=test_report,
            y_pred=y_pred,
        )

    def _train_model(self, X, y) -> tuple[dict, T]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred, output_dict=True), self.model

    def _save_model(self):
        model_path = f"{constants.MODELS_PATH}/{self.model_type}-{self.training_combination.model_name}_model.joblib"
        joblib.dump(self.model, model_path)

