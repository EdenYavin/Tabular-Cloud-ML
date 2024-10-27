
from lazypredict.supervised import CLASSIFIERS
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from mlxtend.classifier import EnsembleVoteClassifier
import keras
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.utils.config import config
from src.cloud.base import CloudModel


class NeuralNetCloudModel(CloudModel):
    name = "dense"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_classes = kwargs.get("num_classes", 2)
        self.epochs = config.neural_net_config.epochs
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


    def fit(self, X_train, y_train, **kwargs):
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


class TabularCloudModel(CloudModel):
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

class EnsembleCloudModel(CloudModel):
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
