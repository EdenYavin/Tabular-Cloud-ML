
import numpy as np

from src.utils.helpers import load_tabular_models
from src.utils.constansts import CONFIG_CLOUD_MODELS_TOKEN, CONFIG_CLOUD_MODELS_PATH_TOKEN

class CloudModels:

    def __init__(self, config: dict):
        models_names = config[CONFIG_CLOUD_MODELS_TOKEN]
        path = config[CONFIG_CLOUD_MODELS_PATH_TOKEN]
        self.name = models_names
        self.models = list(filter(lambda model: model[0] in models_names, load_tabular_models(path)))

    def predict(self, X) -> np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))

        return np.vstack(predictions)
