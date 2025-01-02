from sklearn.linear_model import LogisticRegression

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class LogisticRegressionModel(ModelBase[LogisticRegression]):

    def _get_model(self) -> LogisticRegression:
        return LogisticRegression(random_state=42)

    def _get_model_type(self) -> ModelType:
        return ModelType.LOGISTIC_REGRESSION
