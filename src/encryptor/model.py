import tensorflow as tf
from keras.src.layers import Input, Dense, Reshape, Flatten, Lambda
from keras.src.layers import BatchNormalization, Activation, Conv2DTranspose
from keras.src.models import Model
import numpy as np

class Encryptor(Model):
    def __init__(self, config):
        super(Encryptor, self).__init__()
        self.model = self.build_generator()
        self.config = config

    def build_generator(self):
        def reshape_input(x):
            batch_size = tf.shape(x)[0]
            height = tf.shape(x)[1]
            width = tf.shape(x)[2]
            reshaped = tf.reshape(x, (batch_size, height, width, 1))
            return reshaped

        input_layer = Input(shape=(None, None))  # Dynamic shape for 2D input
        x = Lambda(reshape_input)(input_layer)

        x = Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")(x)
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

        output_vector = Dense(tf.shape(input_layer)[-1])(x)  # Output vector length matches the width of the input

        model = Model(inputs=input_layer, outputs=output_vector)
        return model

    def call(self, inputs):
        inputs = np.vstack(inputs)
        return self.model(inputs)