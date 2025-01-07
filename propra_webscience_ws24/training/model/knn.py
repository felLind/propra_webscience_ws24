from sklearn.neighbors import KNeighborsClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class KNN(ModelBase[KNeighborsClassifier]):

    def _get_model(self, model_args) -> KNeighborsClassifier:
        return KNeighborsClassifier(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.KNN

    def _get_default_model_args(self) -> dict:
        return {"n_neighbors": 5}
