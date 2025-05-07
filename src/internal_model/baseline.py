from typing import Union

import numpy as np
from keras.src.layers import Flatten, Input, Dense,Dropout, BatchNormalization
from keras.src import Model
from keras.src.metrics import F1Score
from xgboost import XGBClassifier

from src.internal_model import TabularInternalModel, DenseInternalModel
from src.utils.constansts import IIM_MODELS

class EmbeddingBaseline(DenseInternalModel):

    def get_model(self, num_classes, input_shape):

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        inputs = Input(shape=input_shape)
        x = Flatten()(inputs)
        x = BatchNormalization()(x)
        x = Dense(units=128, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Define the output layer
        outputs = Dense(units=num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()]
                      )

        return model


class TreeEmbeddingBaseModel(TabularInternalModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = np.squeeze(X, axis=1)
        self.model.fit(X, y)
        return self

    def evaluate(self, X, y):
        if len(X.shape) == 1:
            X = np.squeeze(X, axis=1)
        return super().evaluate(X, y)

class EmbeddingBaselineModelFactory:

    @staticmethod
    def get_model(**kwargs) -> Union[EmbeddingBaseline, TreeEmbeddingBaseModel]:

        iim_type: IIM_MODELS = kwargs.get("type", IIM_MODELS.XGBOOST)
        model = XGBClassifier()

        if iim_type == IIM_MODELS.XGBOOST:
            return TreeEmbeddingBaseModel(**dict(model=model, **kwargs))

        else:
            return EmbeddingBaseline(**kwargs)