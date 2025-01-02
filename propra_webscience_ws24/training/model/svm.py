from sklearn.svm import LinearSVC
from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType

class SVM(ModelBase[LinearSVC]):

    def _get_model(self) -> LinearSVC:
        return LinearSVC(random_state=42)

    def _get_model_type(self) -> ModelType:
        return ModelType.LINEAR_SVC
