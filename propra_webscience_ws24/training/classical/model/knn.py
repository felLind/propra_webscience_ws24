"""
Module containing the implementation of a K-Nearest Neighbors model.
"""

from sklearn.neighbors import KNeighborsClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class KNN(ModelBase[KNeighborsClassifier]):
    """
    K-Nearest Neighbors classifier model wrapper.
    """

    def _get_model(self, model_args) -> KNeighborsClassifier:
        """
        Get a K-Nearest Neighbors model with the given arguments.

        Args:
            model_args: Arguments to pass to the K-Nearest Neighbors model.

        Returns:
            A K-Nearest Neighbors model.
        """
        return KNeighborsClassifier(**model_args)

    def _get_model_type(self) -> ModelType:
        """
        Get the model type.

        Returns:
            The model type enumeration value.
        """
        return ModelType.KNN

    def _get_default_model_args(self) -> dict:
        """
        Get the default model arguments.

        Returns:
            The default model arguments.
        """
        return {"n_neighbors": 5}
