import pandas as pd

from propra_webscience_ws24.training.classical.model import RandomForest
from propra_webscience_ws24.training.classical.model.model_base import (
    ModelType,
    ModelBase,
)
from propra_webscience_ws24.training.classical.model.svm import SVM
from propra_webscience_ws24.training.classical.model.decision_tree import DecisionTree
from propra_webscience_ws24.training.classical.model.knn import KNN
from propra_webscience_ws24.training.classical.model.naive_bayes import NaiveBayes
from propra_webscience_ws24.training.classical.model.logistic_regression import (
    LogisticRegressionModel,
)
from propra_webscience_ws24.training.classical.training_combinations import (
    TrainingCombination,
)


models = {
    ModelType.LINEAR_SVC: SVM,
    ModelType.DECISION_TREE: DecisionTree,
    ModelType.KNN: KNN,
    ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
    ModelType.NAIVE_BAYES: NaiveBayes,
    ModelType.RANDOM_FOREST: RandomForest,
}


def model_factory(
    model_type: ModelType,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    training_combination: TrainingCombination,
    model_args: dict,
) -> ModelBase:
    if model_type in models:
        return models[model_type](df_train, df_test, training_combination, model_args)  # type: ignore[operator]
    raise ValueError(f"Unsupported model type: {model_type}")
