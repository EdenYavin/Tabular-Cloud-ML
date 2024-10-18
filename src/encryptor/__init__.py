from src.utils.config import config
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
        return EncryptorFactory.ENCRYPTORS.get(config.encoder_config.name)(**kwargs)
