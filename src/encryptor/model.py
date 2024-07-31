from keras.src.layers import Input, Dense,  Flatten
from keras.src.layers import BatchNormalization, Activation, Conv2DTranspose
from keras.src.models import Model
import numpy as np

class Encryptor:
    def __init__(self):
        self.model = None

    def build_generator(self, input_shape, output_shape):
        return build_dense_generator(input_shape, output_shape)

    def encode(self, inputs) -> np.array:
        # Ensure inputs are in the correct shape (batch_size, height, width, channels)
        # inputs = np.expand_dims(inputs, axis=-1)  # Add the channel dimension
        inputs = np.expand_dims(inputs, axis=0)  # Add the batch dimension
        if self.model is None:
            input_shape = inputs.shape[1:]  # Exclude batch size
            output_shape = (1, inputs.shape[2])  # Output shape should be (1, width)
            self.model = self.build_generator(input_shape, output_shape)

        return self.model(inputs).numpy()


def build_dense_generator(input_shape, output_shape):
    input_layer = Input(shape=(input_shape[0], input_shape[1]))  # Input shape is dynamic

    x = Flatten()(input_layer)  # Flatten the 2D input into 1D

    # Adding Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    output_vector = Dense(output_shape[1], activation='tanh')(
        x)  # Output vector length matches the specified output shape

    model = Model(inputs=input_layer, outputs=output_vector)
    return model

def build_dc_generator(input_shape, output_shape):
    input_layer = Input(shape=(input_shape[0], input_shape[1], 1))  # Add channel dimension to input shape

    x = Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Flatten the output and reduce to 1D array matching the width of the input
    x = Flatten()(x)

    output_vector = Dense(output_shape[1])(x)  # Output vector length matches the width of the input

    model = Model(inputs=input_layer, outputs=output_vector)
    return model