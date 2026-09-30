from abc import abstractmethod, ABC
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from keras import Model
try:
    from keras.api.layers import GlobalAveragePooling2D, Flatten
except ModuleNotFoundError:
    try:
        from keras.src.layers import GlobalAveragePooling2D, Flatten
    except ModuleNotFoundError:
        from keras.layers import GlobalAveragePooling2D, Flatten

from src.utils.config import config
from src.utils.constansts import CONFIG_CLOUD_MODELS_TOKEN


def as_embedding_model(model):
    """
    Extract the layer before softmax output as an embedding/feature vector model.
    Ensures that the output is a flat 2D tensor (batch_size, feature_dim).
    If the penultimate layer output is 4D (spatial tensor), applies GlobalAveragePooling2D.
    If it is 3D or higher, applies Flatten.
    """
    try:
        penultimate = model.layers[-1].input
    except Exception:
        penultimate = model.layers[-2].output

    if len(penultimate.shape) == 4:
        penultimate = GlobalAveragePooling2D()(penultimate)
    elif len(penultimate.shape) > 2:
        penultimate = Flatten()(penultimate)

    return Model(inputs=model.inputs, outputs=penultimate)


class CloudModel:
    """
    This is a mockup of a models that are trained on the organization data and are deployed on the cloud
    """
    name: str
    def __init__(self, **kwargs):
        models_names = kwargs.get(CONFIG_CLOUD_MODELS_TOKEN)
        self.cloud_models = models_names
        self.model = None
        self.output_shape = None
        self.input_shape = None

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def preprocess(self, X):
        pass

    def evaluate(self, X, y) -> tuple:
        """Evaluate using majority voting over predictions from multiple models.

        Returns:
            tuple: accuracy and F1 score of the ensemble model.
        """

        predictions = self.model.predict(X)
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        return accuracy, f1

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)



class KerasApplicationCloudModel(ABC):

    input_shape: tuple

    def __init__(self, **kwargs):
        self.output_shape = (1, 1000)
        self.preprocess_input = kwargs.get("preprocess_input")
        self.model = self.get_model()
        if config.cloud_config.use_embedding:
            self.model = self._as_embedding_model(self.model)
            self.output_shape = (1, self.model.output_shape[-1])

    @staticmethod
    def _as_embedding_model(model):
        return as_embedding_model(model)

    @abstractmethod
    def get_model(self):
        """Abstract method to be implemented by subclasses to return the model."""
        pass

    def fit(self, X_train, y_train, **kwargs):
        """Fit the model. This can be overridden if needed."""
        pass

    def preprocess(self, X):
        """Preprocess the input data using the specified preprocessing function."""
        if any(s < self.input_shape[0] for s in X.shape[1:3]):
            # Pad the input to make its size equal to the required input shape
            padded_X = tf.image.resize_with_crop_or_pad(X, self.input_shape[0], self.input_shape[1])
            X = self.preprocess_input(padded_X.numpy())
        else:
            # If no padding is needed, directly preprocess the input
            X = self.preprocess_input(X)
        return X

    def predict(self, X):
        """Predict using the model."""
        X = self.preprocess(X)
        return self.model.predict(X, verbose=None)

    def evaluate(self, X, y, **kwargs):
        """Evaluate the model. This can be overridden if needed."""
        return -1, -1











