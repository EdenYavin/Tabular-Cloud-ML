from keras.src.layers import Input, Dense,  Flatten
from keras.src.layers import BatchNormalization, Activation, Conv2DTranspose
from keras.src.models import Model, Sequential
from keras.src.layers import LeakyReLU, Reshape, Conv2D, UpSampling2D, ReLU
from src.encryptor.base import BaseEncryptor



class TabularDCEncryptor(BaseEncryptor):

    name = "tabular_dc"

    def build_generator(self, input_shape, output_shape):
        input_layer = Input(shape=(input_shape[0], input_shape[1], 1))

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

    def build_generator_from_ziv(self, input_shape, output_shape):

        G = Sequential()

        G.add(Reshape(target_shape=[1, *input_shape], input_shape=input_shape))
        # No weights or activations here

        # 1x1x4096
        G.add(Conv2DTranspose(filters=64, kernel_size=4))
        G.add(Activation('relu'))
        # Weights index: 0, Activations index: 1

        # 4x4x64
        G.add(Conv2D(filters=64, kernel_size=4, padding='same'))
        G.add(BatchNormalization(momentum=0.7))
        G.add(Activation('relu'))
        # Weights index: 2, Activations index: 5
        G.add(UpSampling2D())
        # No weights or activations here

        # 8x8x64
        G.add(Conv2D(filters=32, kernel_size=4, padding='same'))
        G.add(BatchNormalization(momentum=0.7))
        G.add(Activation('relu'))
        # Weights index: 8, Activations index: 9
        G.add(UpSampling2D())
        # No weights or activations here

        # 16x16x32
        G.add(Conv2D(filters=16, kernel_size=4, padding='same'))
        G.add(BatchNormalization(momentum=0.7))
        G.add(Activation('relu'))
        # Weights index: 14, Activations index: 13
        G.add(UpSampling2D())
        # No weights or activations here

        # 32x32x16
        G.add(Conv2D(filters=8, kernel_size=4, padding='same'))
        G.add(BatchNormalization(momentum=0.7))
        G.add(Activation('relu'))
        # Weights index: 20, Activations index: 17
        G.add(UpSampling2D())
        # No weights or activations here

        # 64x64x8
        G.add(Conv2D(filters=4, kernel_size=4, padding='same'))
        G.add(BatchNormalization(momentum=0.7))
        G.add(Activation('relu'))
        # Weights index: 26, Activations index: 21
        G.add(UpSampling2D())
        # No weights or activations here

        # 128x128x4
        G.add(Conv2D(filters=3, kernel_size=4, padding='same'))
        G.add(Activation('relu'))
        # Weights index: 32, Activations index: 25

        return G


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

        output_image = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')(x)

        return Model(inputs=input_layer, outputs=output_image)


class DC32x32Encryptor(BaseEncryptor):

    name = "dc32x32"

    def build_generator(self, input_shape, output_shape):

        input_layer = Input(shape=input_shape)
        x = Flatten()(input_layer)
        x = Dense(3 * 3 * 256, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((3, 3, 256))(x)
        x = Conv2DTranspose(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(32, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = UpSampling2D()(x)
        output_image = Conv2DTranspose(3, 4, padding='same', activation='sigmoid')(x)


        return Model(inputs=input_layer, outputs=output_image)
