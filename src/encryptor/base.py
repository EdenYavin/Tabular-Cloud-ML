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
        # inputs = np.resize(inputs, (100, 11, 11, 1))  # Resize by repeating, turning the samples to an images
        if self.model is None:
            input_shape = inputs.shape[1:]
            output_shape = self.output_shape or (1, inputs.shape[2])
            self.model = self.build_generator(input_shape, output_shape)
        return self.model(inputs).numpy()


class Encryptors:
    """
    Ensemble class to join together numerous encryptors from the same type.
    """
    name: str

    def __init__(self, input_shape=None, output_shape=None, number_of_encryptors_to_init=1, enc_base_cls=BaseEncryptor):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.number_of_encryptors_to_init = number_of_encryptors_to_init
        self.models = None
        self.enc_base_cls = enc_base_cls
        self.name =  enc_base_cls.name

    def encode(self, inputs, number_of_encoder_to_use=1) -> np.array:
        if self.models is None:
            self.models = [self.enc_base_cls(output_shape=self.output_shape) for _ in range(self.number_of_encryptors_to_init)]

        assert number_of_encoder_to_use <= len(self.models)

        outputs = []
        for encoder in self.models[:number_of_encoder_to_use]:
            outputs.append(encoder.encode(inputs))

        return np.vstack(outputs)
