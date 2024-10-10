from src.utils.constansts import CONFIG_ENCRYPTOR_NAME_TOKEN
from src.encryptor.model import DCEncryptor, DenseEncryptor, ResNetEncryptor, EfficientNetEncryptor, BaseEncryptor


class EncryptorFactory:

    ENCRYPTORS = {
        DCEncryptor.name: DCEncryptor,
        DenseEncryptor.name: DenseEncryptor,
        ResNetEncryptor.name: ResNetEncryptor,
        EfficientNetEncryptor.name: EfficientNetEncryptor
    }

    @staticmethod
    def get_model(**kwargs):
        return EncryptorFactory.ENCRYPTORS.get(kwargs.pop(CONFIG_ENCRYPTOR_NAME_TOKEN))(**kwargs)
