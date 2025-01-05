from sklearn.tree import DecisionTreeClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class DecisionTree(ModelBase[DecisionTreeClassifier]):

    def _get_model(self, model_args) -> DecisionTreeClassifier:
        if "random_state" not in model_args:
            model_args["random_state"] = 42
        if "ccp_alpha" not in model_args:
            model_args["ccp_alpha"] = 0.0
        return DecisionTreeClassifier(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.DECISION_TREE
