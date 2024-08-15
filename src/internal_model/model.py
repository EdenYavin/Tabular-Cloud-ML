from typing import Union

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from src.utils.constansts import CONFIG_IMM_NAME_TOKEN
from catboost import CatBoostClassifier
from tensorflow.keras.layers import Input, Attention, Concatenate, Lambda
from tensorflow.keras.models import Model


from keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.src.metrics import F1Score
import numpy as np

models = {
    "catboost": CatBoostClassifier(),
    "xgboost": XGBClassifier(),
}


class TabularInternalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "xgboost")
        self.model = kwargs.get("model", models['xgboost'])
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



class NeuralNetworkInternalModel(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 8)
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.epochs = kwargs.get("epochs", 10)
        self.features_split = kwargs.get("features_split")

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



class AttentionNeuralNetInternalModel(NeuralNetworkInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_classes = kwargs.get("num_classes")
        self.name = "attention"
        self.model = self.get_model(num_classes=num_classes)


    def get_model(self, num_classes, raw_features_split):
        # Assume input is a concatenation of raw features, prediction vector, and noise samples
        input_layer = Input(shape=(None,))  # Dynamic input shape

        # Split the input into raw features and prediction vector
        raw_features = Lambda(lambda x: x[:, :raw_features_split])(input_layer)
        prediction_vector = Lambda(lambda x: x[:, raw_features_split:])(input_layer)

        # Process raw features
        x1 = Dense(128, activation='relu')(raw_features)
        x1 = Dropout(self.dropout_rate)(x1)

        # Process prediction vector
        x2 = Dense(64, activation='relu')(prediction_vector)
        x2 = Dropout(self.dropout_rate)(x2)

        # Attention mechanism
        attention_output = Attention()([x1, x2])

        # Combine attention output with processed features
        combined = Concatenate()([attention_output, x1, x2])

        # Final dense layers
        x = Dense(64, activation='relu')(combined)
        x = Dropout(self.dropout_rate)(x)

        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()])

        return model



class DenseInternalModel(NeuralNetworkInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "dense"
        num_classes = kwargs.get("num_classes")
        self.model = self.get_model(num_classes=num_classes)

    def get_model(self, num_classes):
        # Build the model
        model = Sequential()

        # Input layer and a hidden layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout_rate))

        # Additional hidden layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()])

        return model



class InternalInferenceModelFactory:

    @staticmethod
    def get_model(**kwargs) -> Union[NeuralNetworkInternalModel, TabularInternalModel]:
        config = kwargs.get("config")
        name = config[CONFIG_IMM_NAME_TOKEN]

        model = models.get(name)

        if model:
            return TabularInternalModel(**dict(model=model, **config))

        if name == "attention":
            return AttentionNeuralNetInternalModel(**config)

        else:
            return DenseInternalModel(**config)
