from keras.src.layers import Input, Dense,  Flatten
from keras.src.layers import BatchNormalization, Activation, Conv2DTranspose
from keras.src.models import Model
import numpy as np
from keras.src.layers import Concatenate, LeakyReLU
from keras.src.layers import Reshape, Conv2D
from keras.src.layers import UpSampling2D


class Encryptor:
    def __init__(self, output_shape=None, **kwargs):
        self.model = None
        self.generator_type = kwargs.get('name')
        self.output_shape = output_shape

    def build_generator(self, input_shape, output_shape):
        generators = {
            "dc": build_dc_generator,
            "dense_complex": build_dense_complex_generator,
            "dense": build_dense_generator,
            "resnet": build_resnet_generator,  # Add the new generator
            "efficientnet": build_efficientnet_generator,

        }
        return generators[self.generator_type](input_shape, output_shape)

    def encode(self, inputs) -> np.array:
        # Ensure inputs are in the correct shape (batch_size, height, width, channels)

        inputs = np.expand_dims(inputs, axis=0)  # Add the batch dimension
        if self.model is None:
            input_shape = inputs.shape[1:]  # Exclude batch size
            output_shape = self.output_shape or (1, inputs.shape[2])  # Output shape should be (1, width)
            self.model = self.build_generator(input_shape, output_shape)

        return self.model(inputs).numpy()



def build_resnet_generator(input_shape, output_shape):
    input_layer = Input(shape=(input_shape[0], input_shape[1]))  # Dynamic input shape

    x = Flatten()(input_layer)  # Flatten the input

    # Dense layers to upscale the input to a larger vector
    x = Dense(256 * 56 * 56)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Reshape to a small 2D image shape that can be progressively upsampled
    x = Reshape((56, 56, 256))(x)

    # Upsample to 112x112
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Upsample to 224x224
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Final Conv2D layer to create the 3-channel output image
    output_image = Conv2D(3, kernel_size=3, activation='tanh', padding='same')(x)

    model = Model(inputs=input_layer, outputs=output_image)
    return model


def build_dense_complex_generator(input_shape, output_shape):
    input_layer = Input(shape=(input_shape[0], input_shape[1]))
    x = Flatten()(input_layer)

    # First dense block
    x1 = Dense(256)(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    # Second dense block with skip connection
    x2 = Dense(128)(x1)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Concatenate()([x2, x1])

    # Third dense block with skip connection
    x3 = Dense(64)(x2)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(alpha=0.2)(x3)
    x3 = Concatenate()([x3, x2])

    # Output layer
    output_vector = Dense(output_shape[1], activation='tanh')(x3)

    model = Model(inputs=input_layer, outputs=output_vector)
    return model


def build_dense_generator(input_shape, output_shape):
    input_layer = Input(shape=(input_shape[0], input_shape[1]))  # Input shape is dynamic

    x = Flatten()(input_layer)  # Flatten the 2D input into 1D

    # Adding Dense layers
    x = Dense(256, activation='leaky_relu')(x)
    x = Dense(128, activation='leaky_relu')(x)
    x = Dense(64, activation='leaky_relu')(x)

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


def build_efficientnet_generator(input_shape, output_shape):
    """Builds a generator model compatible with EfficientNetB2."""
    output_shape = (260, 260, 3)
    input_layer = Input(shape=(input_shape[0], input_shape[1]))  # Assuming input shape as (height, width)
    x = Flatten()(input_layer)

    # Dense layers with activations
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    # Final dense layer to match the flattened EfficientNetB2 shape
    x = Dense(np.prod(output_shape))(x)

    # Reshape the output to (260, 260, 3) for EfficientNetB2
    output_vector = Reshape(output_shape)(x)

    model = Model(inputs=input_layer, outputs=output_vector)
    return model