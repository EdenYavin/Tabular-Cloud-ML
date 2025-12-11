from src.utils.config import config
from src.embeddings.model import DNNEmbedding, ImageEmbedding, SparseAE, RawDataEmbedding, ClipEmbedding, DinoEmbedding


class EmbeddingsFactory:

    EMBEDDINGS = {
        DNNEmbedding.name: DNNEmbedding,
        ImageEmbedding.name: ImageEmbedding,
        SparseAE.name: SparseAE,
        RawDataEmbedding.name: RawDataEmbedding,
        ClipEmbedding.name: ClipEmbedding,
        DinoEmbedding.name: DinoEmbedding
    }

    @staticmethod
    def get_model(**kwargs):
        return EmbeddingsFactory.EMBEDDINGS.get(config.embedding_config.name)(**kwargs)
