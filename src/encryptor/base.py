
import numpy as np
from keras.api.models import load_model
import os

from src.utils.constansts import ENCRYPTOR_MODELS_DIR_PATH
from src.utils.config import config

embedding_name = config.embedding_config.name

class BaseEncryptor:

    name: str

    def __init__(self, dataset_name: str,output_shape=None):
        self.model = None
        self.output_shape = output_shape
        self.input_shape = None
        self.dataset_name = dataset_name

    def build_generator(self, input_shape, output_shape):
        raise NotImplementedError("Subclasses should implement this method")

    def save_model(self, filename):
        if self.model is not None:
            self.model.save(filename)  # For Keras models

    def load_model(self, filename):
        self.model = load_model(filename)  # For Keras models

    def switch_key(self):
        del self.model
        self.model = self.build_generator(self.input_shape, self.output_shape)

    def encode(self, inputs) -> np.array:

        self.input_shape = inputs.shape[1:]
        self.output_shape = self.output_shape or (1, inputs.shape[2])

        model_path = os.path.join(ENCRYPTOR_MODELS_DIR_PATH, f"{self.dataset_name}_{embedding_name}.h5")
        if self.model is None:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
            else:

                self.model = self.build_generator(self.input_shape, self.output_shape)
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
