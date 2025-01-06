from sklearn.tree import DecisionTreeClassifier

from propra_webscience_ws24.training.model.model_base import ModelBase, ModelType


class DecisionTree(ModelBase[DecisionTreeClassifier]):
    def _get_model(self, model_args) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**model_args)

    def _get_model_type(self) -> ModelType:
        return ModelType.DECISION_TREE


    def _get_default_model_args(self) -> dict:
        return {
            'random_state':42,
            'ccp_alpha':0.0
        }