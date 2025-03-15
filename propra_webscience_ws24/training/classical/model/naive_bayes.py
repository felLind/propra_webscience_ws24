from sklearn.naive_bayes import BernoulliNB

from propra_webscience_ws24.training.classical.model.model_base import (
    ModelBase,
    ModelType,
)


class NaiveBayes(ModelBase[BernoulliNB]):

    def _get_model(self, model_args) -> BernoulliNB:
        return BernoulliNB(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.NAIVE_BAYES
