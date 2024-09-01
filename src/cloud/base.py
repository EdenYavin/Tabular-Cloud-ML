from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score


from src.utils.constansts import CONFIG_CLOUD_MODELS_TOKEN


class CloudModels:
    """
    This is a mockup of a models that are trained on the organization data and are deployed on the cloud
    """
    name: str
    def __init__(self, **kwargs):
        models_names = kwargs.get(CONFIG_CLOUD_MODELS_TOKEN)
        self.cloud_models = models_names
        self.models = None

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X, y) -> tuple:
        """Evaluate using majority voting over predictions from multiple models.

        Returns:
            tuple: accuracy and F1 score of the ensemble model.
        """

        predictions = self.models.predict(X)
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        return accuracy, f1

    def fit(self, X_train, y_train, **kwargs):
        self.models.fit(X_train, y_train)














