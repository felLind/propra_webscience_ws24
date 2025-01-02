from sklearn.naive_bayes import GaussianNB

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class NaiveBayes(ModelBase[GaussianNB]):

    def _get_model(self) -> GaussianNB:
        return GaussianNB()

    def _get_model_type(self) -> ModelType:
        return ModelType.NAIVE_BAYES
