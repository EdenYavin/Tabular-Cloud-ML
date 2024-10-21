from src.utils.config import config
from src.encryptor.model import DCEncryptor, DenseEncryptor, EfficientNetEncryptor, BaseEncryptor, TabularDCEncryptor


class EncryptorFactory:

    ENCRYPTORS = {
        DCEncryptor.name: DCEncryptor,
        DenseEncryptor.name: DenseEncryptor,
        TabularDCEncryptor.name: TabularDCEncryptor,
        EfficientNetEncryptor.name: EfficientNetEncryptor
    }

    @staticmethod
    def get_model(**kwargs):
        return EncryptorFactory.ENCRYPTORS.get(config.encoder_config.name)(**kwargs)
