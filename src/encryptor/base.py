import gc

import numpy as np
from keras.api.models import load_model
import os

from keras.src import initializers
from keras.src import layers
from src.utils.constansts import ENCRYPTOR_MODELS_DIR_PATH
from src.utils.config import config
import tensorflow as tf


embedding_name = config.embedding_config.name

class BaseEncryptor:

    name: str

    def __init__(self, dataset_name: str,output_shape=None):
        self.model = None
        self.output_shape = output_shape
        self.input_shape = None
        self.dataset_name = dataset_name
        self.seed = 1

    def build_generator(self, input_shape, output_shape, seed=None):
        raise NotImplementedError("Subclasses should implement this method")

    def save_model(self, filename):
        if self.model is not None:
            self.model.save(filename)  # For Keras models

    def load_model(self, filename):
        self.model = load_model(filename)  # For Keras models

    def switch_key(self):
        del self.model
        gc.collect()
        self.model = self.build_generator(self.input_shape, self.output_shape, seed=self.seed)
        self.seed += 1

    def reset_weights_in_place(self, seed):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                self.reset_weights_in_place(seed)
                continue

            # Dense / Conv2D layers
            if hasattr(layer, 'kernel'):
                init = initializers.GlorotUniform(seed=seed)
                layer.kernel.assign(init(layer.kernel.shape, layer.kernel.dtype))

            if hasattr(layer, 'bias') and layer.bias is not None:
                init = initializers.Zeros()
                layer.bias.assign(init(layer.bias.shape, layer.bias.dtype))

            # BatchNormalization
            if isinstance(layer, layers.BatchNormalization):
                # Trainable parameters
                if hasattr(layer, 'gamma') and layer.gamma is not None:
                    gamma_init = initializers.Ones()  # Typically initialized to 1
                    layer.gamma.assign(gamma_init(layer.gamma.shape, layer.gamma.dtype))

                if hasattr(layer, 'beta') and layer.beta is not None:
                    beta_init = initializers.Zeros()  # Typically initialized to 0
                    layer.beta.assign(beta_init(layer.beta.shape, layer.beta.dtype))

                # Non-trainable moving stats
                if hasattr(layer, 'moving_mean') and layer.moving_mean is not None:
                    layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))

                if hasattr(layer, 'moving_variance') and layer.moving_variance is not None:
                    layer.moving_variance.assign(tf.ones_like(layer.moving_variance))


    def encode(self, inputs) -> np.array:

        self.input_shape = inputs.shape[1:]
        self.output_shape = self.output_shape or (1, inputs.shape[2])

        model_path = os.path.join(ENCRYPTOR_MODELS_DIR_PATH, f"{self.dataset_name}_{embedding_name}.keras")
        if self.model is None:
            if os.path.exists(model_path) and not config.encoder_config.rotating_key:
                self.model = load_model(model_path)
            else:
                self.model = self.build_generator(self.input_shape, self.output_shape)
                self.seed += 1
                if not os.path.exists(model_path):
                    # We don't want to trigger saving each time, only once.
                    # No need to save the same model multiple times
                    self.save_model(model_path)

        return self.model(inputs).numpy()


class Encryptors:
    """
    Ensemble class to join together numerous encryptors from the same type.
    """
    name: str

    def __init__(self, dataset_name: str, input_shape=None, output_shape=None, number_of_encryptors_to_init=1, enc_base_cls=BaseEncryptor):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.number_of_encryptors_to_init = number_of_encryptors_to_init
        self.models = None
        self.enc_base_cls = enc_base_cls
        self.name =  enc_base_cls.name
        self.dataset_name = dataset_name

    def switch_key(self):
        for model in self.models:
            model.switch_key()

    def encode(self, inputs, number_of_encoder_to_use=1) -> np.array:
        if self.models is None:
            self.models = [
                self.enc_base_cls(dataset_name=self.dataset_name,output_shape=self.output_shape)
                for _ in range(self.number_of_encryptors_to_init)
            ]

        assert number_of_encoder_to_use <= len(self.models), \
            f"Error: number_of_encoder_to_use ({number_of_encoder_to_use}) exceeds the number of available models ({len(self.models)})"

        outputs = []
        for encoder in self.models[:number_of_encoder_to_use]:
            outputs.append(encoder.encode(inputs))

        return np.vstack(outputs)
