from sklearn.neighbors import KNeighborsClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class KNN(ModelBase[KNeighborsClassifier]):

    def _get_model(self, model_args) -> KNeighborsClassifier:
        if "n_neighbors" not in model_args:
            model_args["n_neighbors"] = 5
        return KNeighborsClassifier(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.KNN
