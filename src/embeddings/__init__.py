from src.utils.config import config
from src.embeddings.model import DNNEmbedding, ImageEmbedding, SparseAE


class EmbeddingsFactory:

    EMBEDDINGS = {
        DNNEmbedding.name: DNNEmbedding,
        ImageEmbedding.name: ImageEmbedding,
        SparseAE.name: SparseAE
    }

    @staticmethod
    def get_model(**kwargs):
        return EmbeddingsFactory.EMBEDDINGS.get(config.embedding_config.name)(**kwargs)
