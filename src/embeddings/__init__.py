from src.utils.config import config
from src.embeddings.model import IdentityEmbedding, NumericalTableEmbeddings


class EmbeddingsFactory:

    EMBEDDINGS = {
        IdentityEmbedding.name: IdentityEmbedding,
        NumericalTableEmbeddings.name: NumericalTableEmbeddings,

    }

    @staticmethod
    def get_model(**kwargs):
        return EmbeddingsFactory.EMBEDDINGS.get(config.embedding_config.name)(**kwargs)
