from abc import abstractmethod

import keras
import numpy as np
from keras import Model
from lazypredict.Supervised import CLASSIFIERS
from sklearn.metrics import accuracy_score, f1_score

from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, GlobalAveragePooling2D, AveragePooling2D

from src.utils.constansts import CONFIG_CLOUD_MODELS_TOKEN
from mlxtend.classifier import EnsembleVoteClassifier
from keras.api.applications import ResNet50V2, VGG16, EfficientNetB2
from keras.api.applications.resnet50 import preprocess_input, decode_predictions
from keras.api.applications.vgg16 import preprocess_input, decode_predictions


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

    def fit(self, X_train, y_train):
        self.models.fit(X_train, y_train)



class NeuralNetCloudModels(CloudModels):
    name = "dense"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_classes = kwargs.get("num_classes", 2)
        self.epochs = kwargs.get("epochs", 10)
        self.models = self.get_model(num_classes)

    def get_model(self, num_classes):
        # Build the model
        model = Sequential()
        model.add(Dense(units=128, activation='leaky_relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=64, activation='leaky_relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        # Dynamic input shape

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def fit(self, X_train, y_train):
        y_train = keras.utils.to_categorical(y_train)
        self.models.fit(X_train, y_train, epochs=self.epochs, batch_size=8)

    def predict(self, X):
        return self.models.predict(X, verbose=None)

    def evaluate(self, X, y):
        predictions = np.argmax(self.models.predict(X), axis=1)
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        return accuracy, f1


class TabularCloudModels(CloudModels):
    name = "tabular"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        models = []
        for name_and_model in CLASSIFIERS:
            name, model = name_and_model
            if name in self.cloud_models:
                try:
                    models.append(model(verbose=0))
                except TypeError:
                    models.append(model())

        self.models = EnsembleVoteClassifier(models)

    def predict(self, X):
        predictions = []
        for model in self.models.clfs_:
            predictions.append(model.predict_proba(X))

        return np.hstack(predictions)

class EnsembleCloudModels(CloudModels):
    """
    This is a mockup of a models that are trained on the organization data and are deployed on the cloud
    """
    name = "ensemble"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.name = "-".join(self.cloud_models)

        models = []
        for name_and_model in CLASSIFIERS:
            name, model = name_and_model
            if name in self.cloud_models:
                try:
                    models.append(model(verbose=0))
                except TypeError:
                    models.append(model())

        self.models = EnsembleVoteClassifier(models)


    def predict(self, X) -> np.ndarray:
        return self.models.predict_proba(X)




class ImageCloudModels(CloudModels):
    name = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = self.get_model()

    def fit(self, X_train, y_train):
        pass

    def get_model(self):
        # Load the pretrained ResNet50 model with ImageNet weights
        model = EfficientNetB2(weights='imagenet')
        return model

    def predict(self, X):
        # Ensure the input is properly preprocessed for ResNet50
        X = preprocess_input(X)
        predictions = self.models.predict(X, verbose=None)
        return predictions

    def evaluate(self, X, y):
        return -1, -1


class ResNetEmbeddingCloudModel:
    name = "resnet_embedding"

    def __init__(self, **kwargs):
        self.output_shape = kwargs.get('output_shape', 2048)  # Default embedding size for ResNet50
        self.model = self.get_model()

    def get_model(self):
        base_model = ResNet50V2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global average pooling to obtain embedding
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def predict(self, X):
        """Predict using the ResNet model to get embeddings."""
        return self.model.predict(X, verbose=None)

    def fit(self, X_train, y_train):
        # This model does not require fitting as it's using pretrained ResNet
        pass

    def evaluate(self, X, y):
        """Evaluate the model by using a downstream classifier after getting embeddings."""
        return -1, -1