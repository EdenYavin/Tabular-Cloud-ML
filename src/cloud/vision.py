
from keras.api.applications import ResNet50V2, VGG16, Xception, MobileNetV2
from keras.api.applications.xception import preprocess_input as xception_preprocess_input
from keras.api.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.api.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess, EfficientNetV2B3
from keras.api.applications.densenet import preprocess_input as densenet_preprocess, DenseNet201
from keras.api.applications.inception_v3 import preprocess_input as inception_v3_preprocess, InceptionV3
from keras.api.applications.mobilenet import preprocess_input as mobilenet_preprocess

from keras import Model, Sequential
from keras.src.layers import GlobalAveragePooling2D, Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D, \
    Flatten, Dense
import tensorflow as tf
from keras import regularizers
from keras.api.models import load_model

from src.cloud.base import CloudModel, KerasApplicationCloudModel
from src.utils.constansts import VGG16_CIFAR10_MODEL_PATH, CIFAR_100_VGG16_MODEL_PATH
from src.utils.config import config

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


class MobileNetCloudModel(KerasApplicationCloudModel):
    name = "mobile_net"
    input_shape = (224, 224, 3)

    def __init__(self, **kwargs):
        super().__init__(preprocess_input=mobilenet_preprocess, **kwargs)

    def get_model(self):
        return MobileNetV2(weights='imagenet')

class InceptionCloudModel(KerasApplicationCloudModel):
    name = "inception"
    input_shape = (299, 299, 3)

    def __init__(self, **kwargs):
        super().__init__(preprocess_input=inception_v3_preprocess, **kwargs)

    def get_model(self):
        return InceptionV3(weights='imagenet')

class EfficientNetCloudModel(KerasApplicationCloudModel):
    name = "efficientnet"
    input_shape = (300, 300, 3)

    def __init__(self, **kwargs):
        super().__init__( preprocess_input=efficientnet_v2_preprocess)

    def get_model(self):
        return EfficientNetV2B3(weights='imagenet')

class DenseNetCloudModel(KerasApplicationCloudModel):
    name = "densenet"
    input_shape = (224, 224, 3)

    def __init__(self, **kwargs):
        super().__init__( preprocess_input=densenet_preprocess)

    def get_model(self):
        return DenseNet201(weights='imagenet')

class XceptionCloudModel(KerasApplicationCloudModel):
    name = "xception"
    input_shape = (299, 299, 3)

    def __init__(self, **kwargs):
        super().__init__(preprocess_input=xception_preprocess_input)

    def get_model(self):
        return Xception(weights='imagenet')


class VGG16CloudModel(KerasApplicationCloudModel):
    name = "vgg16"
    input_shape = (224, 224, 3)

    def __init__(self, **kwargs):
        super().__init__(preprocess_input=vgg_preprocess)

    def get_model(self):
        return VGG16(weights='imagenet')


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