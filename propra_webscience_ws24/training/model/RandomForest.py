from sklearn.ensemble import RandomForestClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class NaiveBayes(ModelBase[RandomForestClassifier]):

    def _get_model(self, model_args) -> RandomForestClassifier:
        return RandomForestClassifier(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.NAIVE_BAYES
