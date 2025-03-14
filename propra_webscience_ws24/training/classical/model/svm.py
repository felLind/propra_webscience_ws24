from sklearn.svm import LinearSVC
from propra_webscience_ws24.training.classical.model.model_base import (
    ModelBase,
    ModelType,
)


class SVM(ModelBase[LinearSVC]):

    def _get_model(self, model_args) -> LinearSVC:
        return LinearSVC(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.LINEAR_SVC

    def _get_default_model_args(self) -> dict:
        return {"random_state": 42}
