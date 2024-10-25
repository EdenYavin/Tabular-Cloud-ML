from src.utils.config import config
from src.embeddings.model import DNNEmbedding, NumericalTableEmbeddings, ImageEmbedding, w2vEmbedding


class EmbeddingsFactory:

    EMBEDDINGS = {
        DNNEmbedding.name: DNNEmbedding,
        NumericalTableEmbeddings.name: NumericalTableEmbeddings,
        ImageEmbedding.name: ImageEmbedding,
        w2vEmbedding.name: w2vEmbedding

    }

    @staticmethod
    def get_model(**kwargs):
        return EmbeddingsFactory.EMBEDDINGS.get(config.embedding_config.name)(**kwargs)
