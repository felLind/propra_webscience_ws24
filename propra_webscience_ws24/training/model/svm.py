from sklearn.svm import LinearSVC
from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType

class SVM(ModelBase[LinearSVC]):

    def _get_model(self, model_args) -> LinearSVC:
        if "random_state" not in model_args:
            model_args["random_state"] = 42
        return LinearSVC(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.LINEAR_SVC
