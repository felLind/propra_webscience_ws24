import pandas as pd

from propra_webscience_ws24.training.model.model_base import ModelType, ModelBase
from propra_webscience_ws24.training.model.svm import SVM
from propra_webscience_ws24.training.model.decision_tree import DecisionTree
from propra_webscience_ws24.training.model.knn import KNN
from propra_webscience_ws24.training.model.naive_bayes import NaiveBayes
from propra_webscience_ws24.training.model.logistic_regression import LogisticRegressionModel
from propra_webscience_ws24.training.training_combinations import TrainingCombination


models = {
    ModelType.LINEAR_SVC: lambda df_train, df_test, training_combination, model_args : SVM(df_train, df_test, training_combination, model_args),
    ModelType.DECISION_TREE: lambda df_train, df_test, training_combination, model_args : DecisionTree(df_train, df_test, training_combination, model_args),
    ModelType.KNN: lambda df_train, df_test, training_combination, model_args : KNN(df_train, df_test, training_combination, model_args),
    ModelType.LOGISTIC_REGRESSION: lambda df_train, df_test, training_combination, model_args : LogisticRegressionModel(df_train, df_test, training_combination, model_args),
    ModelType.NAIVE_BAYES: lambda df_train, df_test, training_combination, model_args : NaiveBayes(df_train, df_test, training_combination, model_args),
}


def model_factory(model_type: ModelType,
                  df_train: pd.DataFrame,
                  df_test: pd.DataFrame,
                  training_combination: TrainingCombination,
                  model_args: dict,) -> ModelBase:
    if model_type in models:
        return models[model_type](df_train, df_test, training_combination, model_args)
    raise ValueError(f"Unsupported model type: {model_type}")