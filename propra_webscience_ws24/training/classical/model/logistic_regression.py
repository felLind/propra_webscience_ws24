"""
Module containing the implementation of a Logistic Regression model.
"""

from sklearn.linear_model import LogisticRegression

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class LogisticRegressionModel(ModelBase[LogisticRegression]):
    """
    Logistic Regression classifier model wrapper.
    """

    def _get_model(self, model_args) -> LogisticRegression:
        """
        Get a logistic regression model with the given arguments.

        Args:
            model_args: Arguments to pass to the logistic regression model.

        Returns:
            A logistic regression model.
        """
        return LogisticRegression(**model_args)

    def _get_model_type(self) -> ModelType:
        """
        Get the model type.

        Returns:
            The model type enumeration value.
        """
        return ModelType.LOGISTIC_REGRESSION

    def _get_default_model_args(self) -> dict:
        """
        Get the default model arguments.

        Returns:
            The default model arguments.
        """
        return {"random_state": 42}
