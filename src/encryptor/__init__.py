from src.utils.config import config
from src.encryptor.model import DCEncryptor, DenseEncryptor, TabularDCEncryptor, DC32x32Encryptor

class EncryptorFactory:

    ENCRYPTORS = {
        DCEncryptor.name: DCEncryptor,
        DenseEncryptor.name: DenseEncryptor,
        TabularDCEncryptor.name: TabularDCEncryptor,
        DC32x32Encryptor.name: DC32x32Encryptor
    }

    @staticmethod
    def get_model(**kwargs):
        return EncryptorFactory.ENCRYPTORS.get(config.encoder_config.name)(**kwargs)

    @staticmethod
    def get_model_cls():
        return EncryptorFactory.ENCRYPTORS.get(config.encoder_config.name)