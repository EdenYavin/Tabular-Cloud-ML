
import numpy as np

from src.utils.helpers import load_tabular_models
from src.utils.constansts import CONFIG_CLOUD_MODELS_TOKEN, CONFIG_CLOUD_MODELS_PATH_TOKEN

class CloudModels:
    """
    This is a mockup of a models that are trained on the organization data and are deployed on the cloud
    """

    def __init__(self, config: dict):
        models_names = config[CONFIG_CLOUD_MODELS_TOKEN]
        path = config[CONFIG_CLOUD_MODELS_PATH_TOKEN]
        self.name = "-".join(models_names)
        self.models = [model[1] for model in load_tabular_models(path) if model[0] in models_names]

    def predict(self, X) -> np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        return np.vstack(predictions)
