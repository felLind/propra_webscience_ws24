from sklearn.tree import DecisionTreeClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class DecisionTree(ModelBase[DecisionTreeClassifier]):

    def _get_model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=42)

    def _get_model_type(self) -> ModelType:
        return ModelType.DECISION_TREE
