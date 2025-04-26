from src.utils.config import config
from src.embeddings.model import DNNEmbedding, ImageEmbedding, SparseAE, RawDataEmbedding, ClipEmbedding


class EmbeddingsFactory:

    EMBEDDINGS = {
        DNNEmbedding.name: DNNEmbedding,
        ImageEmbedding.name: ImageEmbedding,
        SparseAE.name: SparseAE,
        RawDataEmbedding.name: RawDataEmbedding,
        ClipEmbedding.name: ClipEmbedding
    }

    @staticmethod
    def get_model(**kwargs):
        return EmbeddingsFactory.EMBEDDINGS.get(config.embedding_config.name)(**kwargs)
