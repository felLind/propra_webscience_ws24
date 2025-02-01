"""
This module provides functionality to create and manage training combinations.
"""

from enum import Enum
from functools import lru_cache
from typing import Iterator
import pandas as pd
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfVectorizer,
)


import dataclasses

from propra_webscience_ws24.data import data_preprocessing


class NGramRange(Enum):
    UNIGRAMS = (1, 1)
    UNI_AND_BIGRAMS = (1, 2)
    UNI_AND_BI_AND_TRIGRAMS = (1, 3)


class VectorizerConfig(Enum):
    TFIDF = TfidfVectorizer
    HASHING = HashingVectorizer


USE_ALL_FEATURES_SPECIFIER = "ALL_FEATURES"
MAX_FEATURES_LIST: list[str | int] = [
    USE_ALL_FEATURES_SPECIFIER,
    10_000,
    50_000,
    250_000,
]
HASHING_VECTORIZER_MAX_FEATURES_MAPPING = {
    (1, 1): 300_000,
    (1, 2): 3_500_000,
    (1, 3): 10_000_000,
}


@dataclasses.dataclass(frozen=True)
class TrainingCombination:
    """
    Represents a combination of text preprocessing and vectorization strategies.
    """

    normalization_strategy: data_preprocessing.TextNormalizationStrategy
    stopword_removal_strategy: data_preprocessing.StopwordRemovalStrategy
    vectorizer: TfidfVectorizer | HashingVectorizer

    @property
    def vectorizer_name(self) -> str:
        return self.vectorizer.__class__.__name__

    @property
    def vocab_size(self) -> int:
        if isinstance(self.vectorizer, HashingVectorizer):
            return self.vectorizer.n_features

        return len(getattr(self.vectorizer, "vocabulary_", {}))

    @property
    def max_features(self) -> int | str:
        if isinstance(self.vectorizer, HashingVectorizer):
            n_features = self.vectorizer.n_features
            ngram_range = self.vectorizer.ngram_range
            if n_features == HASHING_VECTORIZER_MAX_FEATURES_MAPPING[ngram_range]:
                return USE_ALL_FEATURES_SPECIFIER

            return n_features

        if self.vectorizer.max_features is None:
            return USE_ALL_FEATURES_SPECIFIER

        return self.vectorizer.max_features

    @property
    def ngram_range(self) -> NGramRange:
        ngram_range = self.vectorizer.ngram_range
        if ngram_range == (1, 1):
            return NGramRange.UNIGRAMS
        elif ngram_range == (1, 2):
            return NGramRange.UNI_AND_BIGRAMS
        elif ngram_range == (1, 3):
            return NGramRange.UNI_AND_BI_AND_TRIGRAMS
        raise ValueError(f"Unknown ngram_range: {ngram_range}")

    def __hash__(self) -> int:
        return hash(
            (
                self.normalization_strategy,
                self.stopword_removal_strategy,
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
            and self.stopword_removal_strategy == other.stopword_removal_strategy
            and self.vectorizer_name == other.vectorizer_name
            and self.max_features == other.max_features
            and self.ngram_range == other.ngram_range
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"normalization_strategy={self.normalization_strategy.name}, "
            f"stopword_removal_strategy={self.stopword_removal_strategy.name}, "
            f"vectorizer_name={self.vectorizer_name}, "
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range.name}"
            ")"
        )

    @property
    def model_name(self) -> str:
        return (
            f"{self.normalization_strategy.value}-{self.stopword_removal_strategy.value}"
            f"-{self.vectorizer_name.lower()}-"
            f"{self.ngram_range.name.lower()}-{self.max_features}"
        )

    @classmethod
    def get_all_combinations(cls) -> Iterator["TrainingCombination"]:

        def _get_all_vectorizer_combinations() -> (
            Iterator[TfidfVectorizer | HashingVectorizer]
        ):
            return (
                _instantiate_vectorizer(
                    vectorizer.value, max_features, ngram_range.value
                )
                for vectorizer in VectorizerConfig
                for max_features in MAX_FEATURES_LIST
                for ngram_range in NGramRange
            )

        return (
            TrainingCombination(
                normalization_strategy=normalization_strategy,
                stopword_removal_strategy=stopword_removal_strategy,
                vectorizer=vectorizer,
            )
            for normalization_strategy in data_preprocessing.TextNormalizationStrategy
            for stopword_removal_strategy in data_preprocessing.StopwordRemovalStrategy
            for vectorizer in _get_all_vectorizer_combinations()
        )

    @classmethod
    def create_training_combination_subset(
        cls,
        normalization_strategy: str | None,
        stopword_removal_strategy: str | None,
        vectorizer: str | None,
        max_features: int | str | None,
        ngram_range: str | None,
    ) -> Iterator["TrainingCombination"]:
        normalization_options = (
            [data_preprocessing.TextNormalizationStrategy[normalization_strategy]]
            if normalization_strategy
            else [strategy for strategy in data_preprocessing.TextNormalizationStrategy]
        )
        stopword_options = (
            [data_preprocessing.StopwordRemovalStrategy[stopword_removal_strategy]]
            if stopword_removal_strategy
            else [strategy for strategy in data_preprocessing.StopwordRemovalStrategy]
        )
        vectorizer_options = (
            [VectorizerConfig[vectorizer]]
            if vectorizer
            else [v for v in VectorizerConfig]
        )
        max_features_options = [max_features] if max_features else MAX_FEATURES_LIST
        ngram_range_options = (
            [NGramRange[ngram_range]]
            if ngram_range
            else [ngram_range for ngram_range in NGramRange]
        )

        def _vectorizer_subset(
            vectorizer_options: list[VectorizerConfig],
            ngram_range_options: list[NGramRange],
            max_features_options: list[str | int],
        ) -> Iterator[TfidfVectorizer | HashingVectorizer]:
            return (
                _instantiate_vectorizer(
                    vectorizer.value, max_features, ngram_range.value
                )
                for vectorizer in vectorizer_options
                for max_features in max_features_options
                for ngram_range in ngram_range_options
            )

        return (
            TrainingCombination(
                normalization_strategy=normalization_strategy,
                stopword_removal_strategy=stopword_removal_strategy,
                vectorizer=vectorizer,
            )
            for normalization_strategy in normalization_options
            for stopword_removal_strategy in stopword_options
            for vectorizer in _vectorizer_subset(
                vectorizer_options=vectorizer_options,
                ngram_range_options=ngram_range_options,
                max_features_options=max_features_options,
            )
        )


@lru_cache
def _instantiate_vectorizer(
    vectorizer_class: TfidfVectorizer | HashingVectorizer,
    max_features: int | str,
    ngram_range_tuple: tuple[int, int],
) -> TfidfVectorizer | HashingVectorizer:
    kwargs: dict[str, int | str | tuple[int, int]] = (
        {}
        if max_features == USE_ALL_FEATURES_SPECIFIER
        else {"max_features": max_features}
    )
    if vectorizer_class == HashingVectorizer:
        n_features = max_features
        if max_features == USE_ALL_FEATURES_SPECIFIER:
            # manually set the n_features parameter approx. to the number of vocabulary
            # entries of the tfidf vectorizer to be able to compare the results
            n_features = HASHING_VECTORIZER_MAX_FEATURES_MAPPING[ngram_range_tuple]
        kwargs = {"n_features": n_features}
    kwargs |= {"ngram_range": ngram_range_tuple}

    return vectorizer_class(**kwargs)


def was_combination_already_processed(
    df: pd.DataFrame,
    model_type: str,
    training_combination: TrainingCombination,
) -> bool:
    """
    Check if a training combination was already processed.

    Args:
        df: The DataFrame containing the processed training combinations.
        model_type: The type of the model.
        training_combination: The training combination to check.

    Returns:
        True if the training combination was already processed, False otherwise.
    """
    result = df.loc[
        (df.model_type == model_type)
        & (
            df.normalization_strategy
            == training_combination.normalization_strategy.value
        )
        & (
            df.stopword_removal_strategy
            == training_combination.stopword_removal_strategy.value
        )
        & (df.vectorizer == training_combination.vectorizer_name)
        & (df.ngram_range == training_combination.ngram_range.name)
        & (df.max_features == str(training_combination.max_features))
    ]
    return len(result) >= 1
