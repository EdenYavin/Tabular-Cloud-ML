from keras.src.layers import Input, Dense,  Flatten
from keras.src.layers import BatchNormalization, Activation, Conv2DTranspose
from keras.src.models import Model
import numpy as np
from keras.src.layers import LeakyReLU
from keras.src.layers import Reshape



class BaseEncryptor:

    name: str

    def __init__(self, output_shape=None):
        self.model = None
        self.output_shape = output_shape

    def build_generator(self, input_shape, output_shape):
        raise NotImplementedError("Subclasses should implement this method")

    def encode(self, inputs) -> np.array:
        inputs = np.expand_dims(inputs, axis=0)
        if self.model is None:
            input_shape = inputs.shape[1:]
            output_shape = self.output_shape or (1, inputs.shape[2])
            self.model = self.build_generator(input_shape, output_shape)
        return self.model(inputs).numpy()

class TabularDCEncryptor(BaseEncryptor):

    name = "tabular_dc"

    def build_generator(self, input_shape, output_shape):
        input_layer = Input(shape=(input_shape[0], input_shape[1], 1))
        # x = Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")(input_layer)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)

        # x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)

        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Flatten()(x)
        output_vector = Dense(output_shape[1])(x)
        return Model(inputs=input_layer, outputs=output_vector)


class DenseEncryptor(BaseEncryptor):

    name = "dense"

    def build_generator(self, input_shape, output_shape):
        input_layer = Input(shape=(input_shape[0], input_shape[1]))
        x = Flatten()(input_layer)

        x = Dense(256, activation='leaky_relu')(x)
        x = Dense(128, activation='leaky_relu')(x)
        x = Dense(64, activation='leaky_relu')(x)

        output_vector = Dense(output_shape[1], activation='linear')(x)
        return Model(inputs=input_layer, outputs=output_vector)

class DCEncryptor(BaseEncryptor):

    name = "dc"

    def build_generator(self, input_shape, output_shape):

        input_layer = Input(shape=input_shape)
        x = Flatten()(input_layer)

        x = Dense(7*7*256, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Reshape((7, 7, 256))(x)
        x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        output_image = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
        # assert output_image.shape[1:] == (224, 224, 3)

        return Model(inputs=input_layer, outputs=output_image)

class EfficientNetEncryptor(BaseEncryptor):

    name = "efficientnet"

    def build_generator(self, input_shape, output_shape):
        output_shape = (260, 260, 3)
        input_layer = Input(shape=(input_shape[0], input_shape[1]))
        x = Flatten()(input_layer)

        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)

        x = Dense(np.prod(output_shape))(x)
        output_vector = Reshape(output_shape)(x)
        return Model(inputs=input_layer, outputs=output_vector)