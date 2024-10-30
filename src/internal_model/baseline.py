from keras.src.layers import Flatten, Input, Dense,Dropout, BatchNormalization
from keras.src import Model
from keras.src.metrics import F1Score

from src.internal_model.model import DenseInternalModel

class EmbeddingBaseline(DenseInternalModel):

    def get_model(self, num_classes, input_shape):
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