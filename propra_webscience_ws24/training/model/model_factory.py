import pandas as pd

from propra_webscience_ws24.training.model.model_base import ModelType, ModelBase
from propra_webscience_ws24.training.model.svm import SVM
from propra_webscience_ws24.training.model.decision_tree import DecisionTree
from propra_webscience_ws24.training.model.knn import KNN
from propra_webscience_ws24.training.model.naive_bayes import NaiveBayes
from propra_webscience_ws24.training.model.logistic_regression import LogisticRegressionModel
from propra_webscience_ws24.training.training_combinations import TrainingCombination


models = {
    ModelType.LINEAR_SVC: lambda df_train, df_test, training_combination : SVM(df_train, df_test, training_combination),
    ModelType.DECISION_TREE: lambda df_train, df_test, training_combination : DecisionTree(df_train, df_test, training_combination),
    ModelType.KNN: lambda df_train, df_test, training_combination : KNN(df_train, df_test, training_combination),
    ModelType.LOGISTIC_REGRESSION: lambda df_train, df_test, training_combination : LogisticRegressionModel(df_train, df_test, training_combination),
    ModelType.NAIVE_BAYES: lambda df_train, df_test, training_combination : NaiveBayes(df_train, df_test, training_combination),
}


def model_factory(model_type: ModelType,
                  df_train: pd.DataFrame,
                  df_test: pd.DataFrame,
                  training_combination: TrainingCombination) -> ModelBase:
    if model_type in models:
        return models[model_type](df_train, df_test, training_combination)
    raise ValueError(f"Unsupported model type: {model_type}")