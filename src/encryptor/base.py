import numpy as np


class BaseEncryptor:

    name: str

    def __init__(self, input_shape=None, output_shape=None):
        self.model = None
        self.output_shape = output_shape
        self.input_shape = input_shape

    def build_generator(self, input_shape, output_shape):
        raise NotImplementedError("Subclasses should implement this method")

    def encode(self, inputs) -> np.array:
        inputs = np.expand_dims(inputs, axis=0)
        if self.model is None:
            input_shape = inputs.shape[1:]
            output_shape = self.output_shape or (1, inputs.shape[2])
            self.model = self.build_generator(input_shape, output_shape)
        return self.model(inputs).numpy()