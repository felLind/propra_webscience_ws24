from sklearn.linear_model import LogisticRegression

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class LogisticRegressionModel(ModelBase[LogisticRegression]):

    def _get_model(self, model_args) -> LogisticRegression:
        if "random_state" not in model_args:
            model_args["random_state"] = 42
        return LogisticRegression(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.LOGISTIC_REGRESSION
