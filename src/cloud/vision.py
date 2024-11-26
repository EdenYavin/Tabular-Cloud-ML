
from keras.api.applications import ResNet50V2, VGG16, Xception
from keras.api.applications.xception import preprocess_input as xception_preprocess_input
from keras.api.applications.vgg16 import preprocess_input as vgg_preprocess
from keras import Model, Sequential
from keras.src.layers import GlobalAveragePooling2D, Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D, \
    Flatten, Dense
import tensorflow as tf
from keras import regularizers
from keras.api.models import load_model

from src.cloud.base import CloudModel
from src.utils.constansts import VGG16_CIFAR10_MODEL_PATH, CIFAR_100_VGG16_MODEL_PATH

class ResNetEmbeddingCloudModel:
    name = "resnet"

    def __init__(self, **kwargs):
        self.output_shape = kwargs.get('output_shape', 2048)  # Default embedding size for ResNet50
        self.model = self.get_model()

    def get_model(self):
        base_model = ResNet50V2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Global average pooling to obtain embedding
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def predict(self, X):
        """Predict using the ResNet model to get embeddings."""
        return self.model.predict(X, verbose=None)

    def fit(self, X_train, y_train, **kwargs):
        # This model does not require fitting as it's using pretrained ResNet
        pass

    def evaluate(self, X, y, **kwargs):
        """Evaluate the model by using a downstream classifier after getting embeddings."""
        return -1, -1


class XceptionCloudModel(CloudModel):
    name = "xception"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = Xception(weights='imagenet')
        self.input_shape = (299, 299, 3)
        self.output_shape = (1,1000)


    def fit(self, X_train, y_train, **kwargs):
        pass

    def preprocess(self, X):

        if any(s < 224 for s in X.shape):
            # Pad the input to make its size equal to 224
            padded_X = tf.image.resize_with_crop_or_pad(X, 299, 299)

            # Ensure the input is properly preprocessed for VGG16
            X = xception_preprocess_input(padded_X.numpy())
        else:
            # If no padding is needed, directly preprocess the input
            X = xception_preprocess_input(X)

        return X

    def predict(self, X):
        X = self.preprocess(X)
        predictions = self.models.predict(X, verbose=None)
        return predictions

    def evaluate(self, X, y):
        return -1, -1


class VGG16CloudModel(CloudModel):
    name = "vgg16"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.get_model()
        self.input_shape = (224, 224, 3)
        self.output_shape = (1,1000)

    def fit(self, X_train, y_train, **kwargs):
        pass

    def get_model(self):
        # Load the pretrained VGG16 model with ImageNet weights
        model = VGG16(weights='imagenet')
        return model

    def predict(self, X):
        X = self.preprocess(X)
        predictions = self.model.predict(X, verbose=None)
        return predictions

    def preprocess(self, X):

        if any(s < 224 for s in X.shape[1:3]):
            # Pad the input to make its size equal to 224
            padded_X = tf.image.resize_with_crop_or_pad(X, 224, 224)

            # Ensure the input is properly preprocessed for VGG16
            X = vgg_preprocess(padded_X.numpy())
        else:
            # If no padding is needed, directly preprocess the input
            X = vgg_preprocess(X)

        return X

    def evaluate(self, X, y):
        return -1, -1


class VGG16Cifar100CloudModel(CloudModel):
    name = "vgg16_cifar100"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.get_model()
        self.input_shape = (32, 32, 3)
        self.output_shape = (1,100)

    def fit(self, X_train, y_train, **kwargs):
        pass

    def get_model(self):
        # Load the pretrained VGG16 model with ImageNet weights
        model = load_model(CIFAR_100_VGG16_MODEL_PATH)
        return model

    def predict(self, X):
        X = self.preprocess(X)
        predictions = self.model.predict(X, verbose=None)
        return predictions


    def preprocess(self, X):

        padded_X = tf.image.resize_with_crop_or_pad(X, 32, 32)

        # Ensure the input is properly preprocessed for VGG16
        X = vgg_preprocess(padded_X.numpy())

        return X

    def evaluate(self, X, y):
        return -1, -1



class VGG16Cifer10CloudModel(CloudModel):
    name = "vgg16_cifar10"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.get_model()
        self.input_shape = (32, 32, 3)
        self.output_shape = (1,10)

    def fit(self, X_train, y_train, **kwargs):
        pass

    def predict(self, X):
        X = self.preprocess(X)
        predictions = self.model.predict(X, verbose=None)
        return predictions

    def preprocess(self, X):

        if any(s < 32 for s in X.shape[1:3]):
            # Pad the input to make its size equal to 224
            X = tf.image.resize_with_crop_or_pad(X, 32, 32)

        return X

    def evaluate(self, X, y):
        return -1, -1

    def get_model(self):
        # Load the pretrained VGG16 model with ImageNet weights
        model = self.build_model()
        model.load_weights(VGG16_CIFAR10_MODEL_PATH)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=[32, 32, 3], kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        return model