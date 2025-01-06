from sklearn.linear_model import LogisticRegression

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class LogisticRegressionModel(ModelBase[LogisticRegression]):

    def _get_model(self, model_args) -> LogisticRegression:
        return LogisticRegression(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.LOGISTIC_REGRESSION

    def _get_default_model_args(self) -> dict:
        return {
            'random_state':42
        }