from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from src.utils.constansts import CONFIG_IMM_NAME_TOKEN
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras import backend as K
from keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import Metric
from keras.src.metrics import F1Score
import numpy as np

models = {
    "catboost": CatBoostClassifier(),
    "xgboost": XGBClassifier(),
}


class TabularInternalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        config = kwargs.get("config")
        self.name = config[CONFIG_IMM_NAME_TOKEN]
        self.model = models.get(self.name, "xgboost")
        print(f"{self.name} IIM Model Loaded")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred), f1_score(y, pred, average='weighted')


class NeuralNetInternalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        config = kwargs.get("config")
        self.name = config[CONFIG_IMM_NAME_TOKEN]
        num_classes = kwargs.get("num_classes")
        self.batch_size = config.get("batch_size", 8)
        self.dropout_rate = config.get("dropout_rate", 0.5)
        self.epochs = config.get("epochs", 10)
        self.model = self.get_model(num_classes=num_classes)

    def get_model(self, num_classes):
        # Build the model
        model = Sequential()

        # Input layer and a hidden layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        # Additional hidden layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()])

        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X):
        prediction = self.model.predict(X)
        return np.argmax(prediction, axis=1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        if len(y.shape) == 2 and y.shape[1] == 2:
            y = np.argmax(y, axis=1)

        pred = self.predict(X)
        return accuracy_score(y, pred), f1_score(y, pred, average='weighted')


class InternalInferenceModelFactory:

    @staticmethod
    def get_model(**kwargs):
        config = kwargs.get("config")
        name = config[CONFIG_IMM_NAME_TOKEN]
        return (
            NeuralNetInternalModel(**kwargs)
            if name == "neural_net"
            else TabularInternalModel(**kwargs)
        )
