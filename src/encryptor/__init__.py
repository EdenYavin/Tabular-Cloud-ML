from src.utils.config import config
from src.encryptor.model import DCEncryptor, DenseEncryptor, BaseEncryptor, TabularDCEncryptor


class EncryptorFactory:

    ENCRYPTORS = {
        DCEncryptor.name: DCEncryptor,
        DenseEncryptor.name: DenseEncryptor,
        TabularDCEncryptor.name: TabularDCEncryptor,
    }

    @staticmethod
    def get_model(**kwargs):
        return EncryptorFactory.ENCRYPTORS.get(config.encoder_config.name)(**kwargs)
