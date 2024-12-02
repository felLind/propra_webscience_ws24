import dataclasses
from enum import Enum
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd
import time
import humanize

from propra_webscience_ws24 import constants


class NGramRange(Enum):
    UNIGRAMS = (1, 1)
    UNI_AND_BIGRAMS = (1, 2)
    UNI_AND_BI_AND_TRIGRAMS = (1, 3)


@dataclasses.dataclass(frozen=True)
class TrainingCombination:
    normalization_strategy: str
    remove_stopwords: bool
    vectorizer: CountVectorizer | TfidfVectorizer | HashingVectorizer
    max_features: int | None
    ngram_range: NGramRange

    @property
    def vectorizer_name(self) -> str:
        return self.vectorizer.__class__.__name__

    @property
    def should_combination_be_discarded(self) -> bool:
        if (
            self.ngram_range == NGramRange.UNIGRAMS
            or self.ngram_range == NGramRange.UNI_AND_BIGRAMS
        ):
            return self.max_features is not None and self.max_features > 100_000
        if self.ngram_range == NGramRange.UNI_AND_BI_AND_TRIGRAMS:
            return self.max_features is not None and self.max_features > 300_000
        raise ValueError(f"Unknown ngram_range: {self.ngram_range}")

    def __hash__(self) -> int:
        return hash(
            (
                self.normalization_strategy,
                self.remove_stopwords,
                self.vectorizer_name,
                self.max_features,
                self.ngram_range,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingCombination):
            return NotImplemented
        return (
            self.normalization_strategy == other.normalization_strategy
            and self.remove_stopwords == other.remove_stopwords
            and self.vectorizer_name == other.vectorizer_name
            and self.max_features == other.max_features
            and self.ngram_range == other.ngram_range
        )

    @property
    def model_name(self) -> str:
        return (
            f"{self.normalization_strategy}_{'wo' if self.remove_stopwords else 'w'}"
            f"_stopwords_{self.vectorizer_name}_"
            f"{self.ngram_range.name.lower()}_{self.max_features}"
        )


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    training_combination: TrainingCombination
    report_training_data_test_split: dict
    report_test_data: dict
    processing_duration: float
    y_pred: list[int]
    num_tokens: int | None = None

    @property
    def test_accuracy(self) -> float:
        return self.report_test_data["accuracy"]

    @property
    def train_accuracy(self) -> float:
        return self.report_training_data_test_split["accuracy"]


def train_linear_svc(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    training_combination: TrainingCombination,
) -> ClassificationResult:
    print(f"{training_combination}", end=" |")

    start_time = time.perf_counter()

    X_train = training_combination.vectorizer.fit_transform(df_train["processed_text"])
    X_test = training_combination.vectorizer.transform(df_test["processed_text"])

    report, model = _train_linear_svc(X_train, df_train.sentiment)
    y_pred = model.predict(X_test)
    test_report = classification_report(df_test.sentiment, y_pred, output_dict=True)

    processing_duration = time.perf_counter() - start_time

    print(
        f" processing_duration={humanize.precisedelta(processing_duration, minimum_unit='seconds')} | test_accuracy={test_report['accuracy']:.3f}"
    )

    _save_model(model, training_combination)

    return ClassificationResult(
        training_combination=training_combination,
        num_tokens=len(getattr(training_combination.vectorizer, "vocabulary_", {})),
        processing_duration=processing_duration,
        report_training_data_test_split=report,
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


def _save_model(model: LinearSVC, training_combination: TrainingCombination):
    """Save the trained LinearSVC model to a file."""
    model_path = (
        f"{constants.MODELS_PATH}/{training_combination.model_name}_model.joblib"
    )
    joblib.dump(model, model_path)
