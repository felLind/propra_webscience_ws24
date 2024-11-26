import dataclasses
import json
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from propra_webscience_ws24.constants import CLASSIFICATION_RESULTS_OUTPUT_FILE_PATH


@dataclasses.dataclass
class ClassificationResult:
    column_name: str
    report_training_data_test_split: dict
    report_test_data: dict
    embedding: Literal["TFIDF", "GLOVE"]
    max_tokens: int | None = None
    num_tokens: int | None = None

    @property
    def test_accuracy(self) -> float:
        return self.report_test_data["accuracy"]

    @property
    def train_accuracy(self) -> float:
        return self.report_training_data_test_split["accuracy"]


@dataclasses.dataclass
class ClassificationResults:
    results: list[ClassificationResult]


def train_linear_svc(X, y) -> tuple[dict, LinearSVC]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True), model


def train_with_tf_idf_vectorizer(
    df_train, df_test
) -> tuple[list[ClassificationResult], str, LinearSVC, TfidfVectorizer]:
    best_model = None
    best_accuracy = 0.0
    classification_results = []
    col_name_for_model = ""
    for max_tokens in [
        None,
        1_000,
        5_000,
        10_000,
        15_000,
    ]:
        for column in [
            "lemmatized_tokens_wo_stop_words",
            "lemmatized_tokens_w_stop_words",
            "stemmed_tokens_wo_stop_words",
            "stemmed_tokens_w_stop_words",
        ]:
            print(f"{max_tokens=} | {column=}", end="")
            tfidf_vectorizer = TfidfVectorizer(max_features=max_tokens)

            X_train = tfidf_vectorizer.fit_transform(df_train[column])
            y_train = df_train["sentiment"]
            report, trained_model = train_linear_svc(X_train, y_train)

            X_test = tfidf_vectorizer.transform(df_test[column])
            y_test = df_test["sentiment"]
            y_pred = trained_model.predict(X_test)
            report_test_data = classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )

            print(
                f" | accuracy_train={report['accuracy']:.3f} | accuracy_test={report_test_data['accuracy']:.3f}"
            )
            if report_test_data["accuracy"] > best_accuracy:
                best_accuracy = report_test_data["accuracy"]
                best_model = trained_model
                vectorizer = tfidf_vectorizer
                col_name_for_model = column
            classification_results.append(
                ClassificationResult(
                    max_tokens=max_tokens,
                    num_tokens=len(tfidf_vectorizer.vocabulary_),
                    column_name=column,
                    embedding="TFIDF",
                    report_training_data_test_split=report,
                    report_test_data=report_test_data,
                )
            )

    return classification_results, col_name_for_model, best_model, vectorizer


def dump_classification_results(classification_results):
    with open(CLASSIFICATION_RESULTS_OUTPUT_FILE_PATH, "w") as f:
        json.dump(
            dataclasses.asdict(ClassificationResults(results=classification_results)), f
        )
