import numpy as np
from lazypredict.Supervised import CLASSIFIERS
from sklearn.metrics import accuracy_score, f1_score

from src.utils.helpers import load_tabular_models
from src.utils.constansts import CONFIG_CLOUD_MODELS_TOKEN, CONFIG_CLOUD_MODELS_PATH_TOKEN
from collections import Counter
from mlxtend.classifier import EnsembleVoteClassifier



class CloudModels:
    """
    This is a mockup of a models that are trained on the organization data and are deployed on the cloud
    """

    def __init__(self, config: dict):
        models_names = config[CONFIG_CLOUD_MODELS_TOKEN]
        # path = config[CONFIG_CLOUD_MODELS_PATH_TOKEN]
        self.cloud_models = models_names
        self.name = "-".join(models_names)

        models = []
        for name_and_model in CLASSIFIERS:
            name, model = name_and_model
            if name in models_names:
                models.append(model())

        self.models = EnsembleVoteClassifier(models)
        # self.models = [model[1] for model in load_tabular_models(path) if
        #                model[0] in models_names]  # Use default models

    def predict(self, X) -> np.ndarray:

        return self.models.predict_proba(X)

    def evaluate(self, X, y) -> tuple:
        """Evaluate using majority voting over predictions from multiple models.

        Returns:
            tuple: accuracy and F1 score of the ensemble model.
        """

        majority_vote = self.models.predict(X)
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y, majority_vote)
        f1 = f1_score(y, majority_vote, average='weighted')

        return accuracy, f1

    def fit(self, X_train, y_train):
        self.models.fit(X_train, y_train)
