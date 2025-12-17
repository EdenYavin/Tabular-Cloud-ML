import gc
from abc import ABC, abstractmethod
import numpy as np
from keras.src.utils import to_categorical
from loguru import logger
from src.cloud import CloudModelManager
from src.domain.dataset import IIMFeatures, IIMDataset, PredictionBaselineFeatures, PredictionBaselineDataset, \
    EmbeddingBaselineFeatures, EmbeddingBaselineDataset
from src.embeddings import DinoEmbedding, ClipEmbedding
from src.encryptor.base import BaseEncryptor
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory
from src.utils.traingulations import get_triangulation_samples_clustering, get_class_representative_samples
from src.utils.constansts import EMBEDDING_TYPES

class FeatureEngineeringPipeline(ABC):

    def __init__(self, dataset_name, encryptor: BaseEncryptor, embeddings_model,
                 metadata = None):

        self.name = dataset_name
        self.use_embedding = config.experiment_config.use_embedding
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.embedding_db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)
        self.encryptor = encryptor
        self.original_train_size = None
        self.cloud_model_manager = CloudModelManager()

        if config.encoder_config.embedding == EMBEDDING_TYPES.DINO:
            logger.info(f"Triangulation model is ON, using: DINO")
            self.triangulation_embedding = DinoEmbedding()
        elif config.encoder_config.embedding == EMBEDDING_TYPES.CLIP:
            logger.info(f"Triangulation model is ON, using: CLIP")
            self.triangulation_embedding = ClipEmbedding()
        else:
            raise ValueError(f"Unknown triangulation embedding model: {config.encoder_config.embedding}")

    def _get_triangulation_samples(self, embeddings, labels=None, how_to_choose: list[str] | str=None, n_samples=None):

        if type(how_to_choose) == str:
            how_to_choose = [how_to_choose]

        if not how_to_choose:
            how_to_choose = [config.experiment_config.triangulation_choosing]

        if not n_samples:
            n_samples = config.experiment_config.n_triangulation_samples

        logger.info(f"Choosing the {how_to_choose} {n_samples} triangulation samples")

        triangulation_samples = []
        for method in how_to_choose:

            if method == 'random':
                indices = np.random.choice(len(embeddings), size=n_samples,
                                           replace=False)
                triangulation_samples.append(embeddings[indices])

            elif method == 'first':
                triangulation_samples.append(embeddings[:n_samples])

            elif method == 'kmeans':
                triangulation_samples.append(get_triangulation_samples_clustering(n_samples, embeddings))
            elif method == 'classes':
                triangulation_samples.append(get_class_representative_samples(embeddings, labels))
            else:
                # Last method
                triangulation_samples.append(embeddings[-n_samples:])

        return np.vstack(triangulation_samples) # Stack all the triangulation vertically

    def create(self, X_train, y_train, X_test, y_test) -> tuple[list[IIMDataset] | IIMDataset, EmbeddingBaselineDataset]:#, PredictionBaselineDataset]:

        X_emb_train = self._get_embeddings(X_train, is_test=False)
        X_emb_test = self._get_embeddings(X_test, is_test=True)

        # One hot encode the labels
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=num_classes)
        gc.collect()
        y_test = to_categorical(y_test, num_classes=num_classes)
        gc.collect()

        embeddings_baseline = EmbeddingBaselineDataset(
            train=EmbeddingBaselineFeatures(embeddings=X_emb_train, labels=y_train),
            test=EmbeddingBaselineFeatures(embeddings=X_emb_test, labels=y_test)
        )


        Xs_train, new_y_train, X_pred_train = self._get_features(X_train, X_emb_train, y_train, is_test=False)
        Xs_test, new_y_test, X_pred_test = self._get_features(X_test, X_emb_test, y_test, is_test=True)

        # pred_baseline = PredictionBaselineDataset(
        #     train=PredictionBaselineFeatures(predictions=X_pred_train, labels=new_y_train),
        #     test=PredictionBaselineFeatures(predictions=X_pred_test, labels=new_y_test)
        # )

        if config.iim_config.stacking:
            # For stacking we create a new dataset for each stack model
            dataset = [
                        IIMDataset(
                        train=IIMFeatures(features=x_train, labels=y_train),
                        test=IIMFeatures(features=x_test, labels=y_test)
                    )
                for x_train, x_test in zip(Xs_train, Xs_test)
                ]
        else:
            # For non stacking we create only one dataset for one internal model
            dataset = IIMDataset(train=IIMFeatures(features=Xs_train, labels=new_y_train),
                                 test=IIMFeatures(features=Xs_test, labels=new_y_test))

        self.embedding_db.save()
        return dataset, embeddings_baseline#, pred_baseline

    def _get_embeddings(self, X, is_test=False):
        embeddings = self.embedding_db.get_embedding(X, is_test)
        return np.vstack(embeddings)

    @abstractmethod
    def _get_features(self, X, embeddings, y, is_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
