from sklearn.neighbors import KNeighborsClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class KNN(ModelBase[KNeighborsClassifier]):

    def _get_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=5)

    def _get_model_type(self) -> ModelType:
        return ModelType.KNN
