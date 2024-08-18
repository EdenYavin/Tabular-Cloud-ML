from typing import Union

from keras import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from keras.src.models import Model
from keras.src.layers import Dense, Dropout, Input, Attention, Concatenate, Lambda, Conv1D, Flatten, MaxPooling1D
from keras.src.metrics import F1Score
import numpy as np

from src.utils.constansts import CONFIG_IMM_NAME_TOKEN


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
        self.model: Model = None

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X):
        prediction = self.model.predict(X)
        return np.argmax(prediction, axis=1)


    def evaluate(self, X, y):
        if len(y.shape) == 2 and y.shape[1] == 2:
            y = np.argmax(y, axis=1)

        pred = self.predict(X)
        return accuracy_score(y, pred), f1_score(y, pred, average='weighted')



class AttentionNeuralNetInternalModel(NeuralNetworkInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_classes = kwargs.get("num_classes")
        input_shape = kwargs.get("input_shape")
        self.name = "attention"
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)


    def get_model(self, num_classes, input_shape):
        # Assume input is a concatenation of raw features, prediction vector, and noise samples
        input_layer = Input(shape=(input_shape,))  # Dynamic input shape

        # # Split the input into raw features and prediction vector
        # raw_features = Lambda(lambda x: x[:, :raw_features_split])(input_layer)
        # prediction_vector = Lambda(lambda x: x[:, raw_features_split:])(input_layer)

        # Process raw features
        # x1 = Dense(128, activation='relu')(input_layer)
        # x1 = Dropout(self.dropout_rate)(x1)
        #
        # # Process prediction vector
        # x2 = Dense(64, activation='relu')(prediction_vector)
        # x2 = Dropout(self.dropout_rate)(x2)
        #
        # # Attention mechanism
        # attention_output = Attention()([x1, x2])
        attention = Attention()([input_layer, input_layer])

        # Combine attention output with processed features
        # combined = Concatenate()([attention_output, x1, x2])

        # Final dense layers
        x = Dense(64, activation='relu')(attention)
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
        input_shape = kwargs.get("input_shape")
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)

    def get_model(self, num_classes, input_shape):
        # Build the model
        inputs = Input(shape=(input_shape,))  # Dynamic input shape

        # Define the hidden layers
        x = Dense(units=128, activation='leaky_relu')(inputs)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(units=64, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Define the output layer
        outputs = Dense(units=num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()])

        return model


class ConvInternalModel(NeuralNetworkInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "conv"
        num_classes = kwargs.get("num_classes")
        input_shape = kwargs.get("input_shape")
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)

    def fit(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def get_model(self, num_classes, input_shape):
        # Build the model

        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=num_classes, activation='softmax'))

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()])

        return model



class InternalInferenceModelFactory:

    @staticmethod
    def get_model(**kwargs) -> Union[NeuralNetworkInternalModel, TabularInternalModel]:

        name = kwargs.get(CONFIG_IMM_NAME_TOKEN)
        model = models.get(name)

        if model:
            return TabularInternalModel(**dict(model=model, **kwargs))

        if name == "attention":
            return AttentionNeuralNetInternalModel(**kwargs)

        if name == "conv":
            return ConvInternalModel(**kwargs)

        else:
            return DenseInternalModel(**kwargs)
