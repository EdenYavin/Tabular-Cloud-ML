
import numpy as np
from keras.api.models import load_model
import os

from keras.src import initializers

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
        self.reset_weights_in_place(seed=self.seed)
        self.seed += 1

    def reset_weights_in_place(self, seed):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                self.reset_weights_in_place(seed)
                continue

            # You can expand this based on layer types or your model structure
            if hasattr(layer, 'kernel'):
                init = initializers.GlorotUniform(seed=seed)
                layer.kernel.assign(init(layer.kernel.shape, layer.kernel.dtype))

            if hasattr(layer, 'bias') and layer.bias is not None:
                init = initializers.Zeros()  # Bias is usually Zeros by default
                layer.bias.assign(init(layer.bias.shape, layer.bias.dtype))

            # if isinstance(layer, tf.keras.layers.BatchNormalization):
            #     # Reset batch norm non-trainable params if needed
            #     for attr in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
            #         var = getattr(layer, attr, None)
            #         if var is not None:
            #             init = tf.keras.initializers.Ones() if 'gamma' in attr else tf.keras.initializers.Zeros()
            #             var.assign(init(var.shape, var.dtype))
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
