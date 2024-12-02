import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from propra_webscience_ws24 import constants

sns.set_theme(style="whitegrid", palette="viridis")


def plot_classification_results():
    results_df = pd.read_parquet(constants.CLASSIFICATION_RESULTS_SAVED_PARQUET_PATH)
    results_df["test_accuracy"] = results_df.loc[:, "report_test_data"].apply(
        lambda x: x.get("accuracy")
    )
    _plot_accuracy_by_stopwords(results_df)
    _plot_accuracy_by_vectorizer_and_ngrams(results_df)
    _plot_accuracy_by_normalization_strategy_and_ngrams(results_df)
    _plot_accuracy_by_normalization_strategy_and_max_features(results_df)
    _plot_accuracy_by_vectorizer_and_max_features(results_df)
    _plot_accuracy_by_stopwords_and_normalization_strategy(results_df)
    _plot_accuracy_by_vectorizer_and_normalization_strategy(results_df)
    _plot_accuracy_by_ngrams_and_vectorizer(results_df)
    _plot_accuracy_by_normalization_strategy_and_stopwords(results_df)
    _plot_accuracy_by_max_features_and_normalization_strategy(results_df)
    _plot_accuracy_by_max_features_and_vectorizer(results_df)
    _plot_training_duration_by_max_features_and_vectorizer(results_df)
    _plot_accuracy_by_normalization_strategy_and_ngram_range(results_df)
    _plot_accuracy_by_vectorizer_and_ngram_range(results_df)


def _plot_accuracy_by_normalization_strategy_and_ngram_range(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="normalization_strategy",
        y="test_accuracy",
        hue="ngram_range",
        title="Accuracy by Normalization Strategy and n-Gram Range",
        xlabel="Nrmalization Strategy",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_vectorizer_and_ngram_range(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="vectorizer",
        y="test_accuracy",
        hue="ngram_range",
        title="Accuracy by Vectorizer and n-Gram Range",
        xlabel="Vectorizer",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_vectorizer_and_normalization_strategy(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="vectorizer",
        y="test_accuracy",
        hue="normalization_strategy",
        title="Accuracy by Vectorizer and Normalization Strategy",
        xlabel="Vectorizer",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_ngrams_and_vectorizer(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="ngram_range",
        y="test_accuracy",
        hue="normalization_strategy",
        title="Accuracy by n-Gram Range and Normalization Strategy",
        xlabel="N-Gram Range",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_max_features_and_normalization_strategy(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="max_features",
        y="test_accuracy",
        hue="normalization_strategy",
        title="Accuracy by Max Features and Normalization Strategy",
        xlabel="Max Features",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_vectorizer_and_ngrams(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="vectorizer",
        y="test_accuracy",
        hue="max_features",
        title="Accuracy by Vectorizer and Max Features",
        xlabel="Vectorizer",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_normalization_strategy_and_ngrams(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="normalization_strategy",
        y="test_accuracy",
        hue="max_features",
        title="Accuracy by Normalization Strategy and Max Features",
        xlabel="Normalization Strategy",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_normalization_strategy_and_max_features(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="normalization_strategy",
        y="test_accuracy",
        hue="max_features",
        title="Accuracy by Normalization Strategy and Max Features",
        xlabel="Normalization Strategy",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_vectorizer_and_max_features(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="vectorizer",
        y="test_accuracy",
        hue="max_features",
        title="Accuracy by Vectorizer and Max Features",
        xlabel="Vectorizer",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_max_features_and_vectorizer(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="max_features",
        y="test_accuracy",
        hue="vectorizer",
        title="Accuracy by Max Features and Vectorizer",
        xlabel="Max Features",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_stopwords_and_normalization_strategy(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="remove_stopwords",
        y="test_accuracy",
        hue="normalization_strategy",
        title="Accuracy by Stopwords and Normalization Strategy",
        xlabel="Remove Stopwords",
        ylabel="Test Accuracy",
    )


def _plot_accuracy_by_normalization_strategy_and_stopwords(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="normalization_strategy",
        y="test_accuracy",
        hue="remove_stopwords",
        title="Accuracy by Normalization Strategy and Stopwords",
        xlabel="Normalization Strategy",
        ylabel="Test Accuracy",
    )


def _plot_training_duration_by_max_features_and_vectorizer(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="max_features",
        y="processing_duration",
        hue="vectorizer",
        title="Training Duration by Max Features and Vectorizer",
        xlabel="Max Features",
        ylabel="Training Duration",
    )


def _plot_accuracy_by_stopwords(results_df: pd.DataFrame):
    _plot_catplot(
        data=results_df,
        x="remove_stopwords",
        y="test_accuracy",
        title="Accuracy by Stopwords Removal",
        xlabel="Remove Stopwords",
        ylabel="Test Accuracy",
    )


def _plot_catplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    hue: str | None = None,
):
    g = sns.catplot(data=data, x=x, y=y, hue=hue, kind="bar", height=8, aspect=1.5)

    g.figure.suptitle(title)
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    def _sanitize_title(title: str) -> str:
        return title.lower().replace(" ", "-")

    g.figure.savefig(f"{constants.PLOTS_PATH}/{_sanitize_title(title)}.png", dpi=300)

    plt.close(g.figure)


if __name__ == "__main__":
    plot_classification_results()
