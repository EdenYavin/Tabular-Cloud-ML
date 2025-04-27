import pandas as pd
from huggingface_hub.keras_mixin import keras
from keras.src.applications import resnet
from keras.src.applications.resnet import preprocess_input
from keras.src.layers import Dense, BatchNormalization, Input, LeakyReLU
from keras.src import Sequential
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tab2img.converter import Tab2Img
from keras.src.callbacks import EarlyStopping
from keras.src.utils import to_categorical
from loguru import logger
from transformers import CLIPProcessor, CLIPModel
import torch
from src.utils.helpers import create_image_from_numbers, expand_matrix_to_img_size
from src.utils.config import config
from src.utils.constansts import CPU_DEVICE, EMBEDDING_MODEL_PATH

class DNNEmbedding(nn.Module):

    name = "dnn_embedding"

    def __init__(self, **kwargs):
        super(DNNEmbedding, self).__init__()
        X, y = kwargs.get("X"), kwargs.get("y")
        dataset_name = kwargs.get("dataset_name", None)
        path = EMBEDDING_MODEL_PATH / f"{dataset_name}.h5" or ""
        if path.exists():
            model = keras.models.load_model(path)
        else:
            model = self._get_trained_model(X,y)
            model.save(path)

        self.model = model.layers[0]
        self.output_shape = (1, X.shape[1]//2)


    def forward(self, x):

        if type(x) is pd.DataFrame:
            x = x.to_numpy()

        embedding = self.model(x)
        return embedding

    def _get_trained_model(self, X:np.ndarray | pd.DataFrame, y:np.ndarray | pd.DataFrame):

        num_classes = len(set(y))
        y = to_categorical(y, num_classes)

        model = Sequential()
        model.add(Dense(units=X.shape[1] // 2, activation='relu', name="embedding"))
        model.add(BatchNormalization())
        model.add(Dense(units=num_classes, activation='softmax', name="output"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stop = EarlyStopping(patience=2, monitor="loss")

        with tf.device(CPU_DEVICE):
            # Dense networks run faster on CPU
            model.fit(X, y, epochs=50, batch_size=64, callbacks=[early_stop])

        return model

class SparseAE(nn.Module):
    name: str = "sparse_ae"

    def __init__(self, **kwargs):
        super(SparseAE, self).__init__()
        X = kwargs.get("X")
        force = kwargs.get("force", False) # Flag to force creating a new model.
        dataset_name = kwargs.get("dataset_name", None)
        path = EMBEDDING_MODEL_PATH / f"{dataset_name}_sparse.h5" or ""

        if path.exists() and not force:
            self.model = keras.models.load_model(path)
        else:
            logger.info("Creating new sparse autoencoder model}")
            self.model = self._get_trained_model(X.astype(float))
            self.model.save(path)

        self.output_shape = (1, 64)

    def forward(self, x):

        if type(x) is pd.DataFrame:
            x = x.to_numpy()

        embedding = self.model(x)
        return embedding

    def _get_trained_model(self, X:np.ndarray | pd.DataFrame):

        def sparse_loss(y_true, y_pred):
            sparsity_level = 0.05
            lambda_sparse = 0.1
            mse_loss = tf.reduce_mean(keras.losses.MeanSquaredError()(y_true, y_pred))
            hidden_layer_output = encoder(y_true)
            mean_activation = tf.reduce_mean(hidden_layer_output, axis=0)

            kl_divergence = tf.reduce_sum(sparsity_level * tf.math.log(sparsity_level / (mean_activation + 1e-10)) +
                                          (1 - sparsity_level) * tf.math.log(
                (1 - sparsity_level) / (1 - mean_activation + 1e-10)))

            return mse_loss + lambda_sparse * kl_divergence

        input_dim = X.shape[1]
        encoding_dim = max(int(input_dim * 1.5), 64)
        inputs = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation="relu")(inputs)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = keras.Model(inputs, decoded)
        encoder = keras.Model(inputs, encoded)
        early_stop = EarlyStopping(patience=2, monitor="loss")

        autoencoder.compile(optimizer='adam', loss=sparse_loss)
        autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, callbacks=[early_stop])

        return encoder


class ImageEmbedding(nn.Module):

    name = "image_embedding"

    def __init__(self, **kwargs):

        super(ImageEmbedding, self).__init__()

        base_model = config.embedding_config.base_model
        # Load a pre-trained ResNet model
        if base_model == 'resnet50':
            self.model = resnet.ResNet50(weights="imagenet", include_top=False)
        elif base_model == 'resnet101':
            self.model = resnet.ResNet101(weights="imagenet", include_top=False)
        elif base_model == 'resnet152':
            self.model = resnet.ResNet152(weights="imagenet", include_top=False)
        else:
            raise ValueError("Unsupported ResNet model")

        self.input_shape = (224, 224)
        self.output_shape = (7, 7, 2048)
        self.image_transformer = Tab2Img()
        self.image_transformer.fit(kwargs.get("X").values, kwargs.get("y").values)


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape(1,224,224,3)
        # Extract embeddings

        image = self.image_transformer.transform(x) # expand from 6x6 to 224x224
        image = expand_matrix_to_img_size(image[0], self.input_shape)

        embeddings = self.model(preprocess_input(image))
        return embeddings.numpy()[0]

class ClipEmbedding(nn.Module):

    name = "clip_embedding"

    def __init__(self, **kwargs):
        super(ClipEmbedding, self).__init__()
        model_id = "openai/clip-vit-base-patch32"
        self.input_shape = (224, 224)
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.output_shape = (1, 512)


    def forward(self, x):

        image_input = self.processor(images=x, return_tensors="pt", do_rescale=False)["pixel_values"]
        with torch.no_grad():
            embeddings = self.model.get_image_features(pixel_values=image_input)
        # embedding is your flat vector (e.g., 512-dim)[6]
        return embeddings.numpy()


class RawDataEmbedding(nn.Module):
    name = "raw_data_embedding"

    def __init__(self, **kwargs):
        super(RawDataEmbedding, self).__init__()

    def forward(self, x):
        return x.astype(np.float32)