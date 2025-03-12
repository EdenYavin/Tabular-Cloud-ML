from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from tqdm import tqdm

from src.domain.dataset import IIMFeatures, IIMDataset, PredictionBaselineFeatures, PredictionBaselineDataset, \
    EmbeddingBaselineFeatures, EmbeddingBaselineDataset
from src.encryptor.base import Encryptors
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory, CloudPredictionDataDatabase


class FeatureEngineeringPipeline(ABC):

    def __init__(self, dataset_name, encryptor: Encryptors, embeddings_model,
                 n_pred_vectors, metadata = None):

        self.n_pred_vectors = n_pred_vectors
        self.name = dataset_name
        self.use_embedding = config.experiment_config.use_embedding
        self.use_predictions = config.experiment_config.use_preds
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.embedding_db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)
        self.encryptor = encryptor
        self.cloud_db = CloudPredictionDataDatabase(dataset_name)

    def create(self, X_train, y_train, X_test, y_test) -> tuple[list[IIMDataset] | IIMDataset, EmbeddingBaselineDataset, PredictionBaselineDataset]:

        X_emb_train = self._get_embeddings(X_train)
        X_emb_test = self._get_embeddings(X_test)

        Xs_train, new_y_train, X_pred_train = self._get_features(X_emb_train, y_train, is_test=False)
        Xs_test, new_y_test, X_pred_test = self._get_features(X_emb_test, y_test, is_test=True)

        # One hot encode the labels
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=num_classes)
        new_y_train = to_categorical(new_y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)
        new_y_test = to_categorical(new_y_test, num_classes=num_classes)

        embeddings_baseline = EmbeddingBaselineDataset(
            train=EmbeddingBaselineFeatures(embeddings=X_emb_train, labels=y_train),
            test=EmbeddingBaselineFeatures(embeddings=X_emb_test, labels=y_test)
        )

        pred_baseline = PredictionBaselineDataset(
            train=PredictionBaselineFeatures(predictions=X_pred_train, labels=new_y_train),
            test=PredictionBaselineFeatures(predictions=X_pred_test, labels=new_y_test)
        )

        if config.experiment_config.stacking:
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
            dataset = IIMDataset(train=IIMFeatures(features=Xs_train, labels=y_train),
                                 test=IIMFeatures(features=Xs_test, labels=y_test))

        self.embedding_db.save()
        return dataset, embeddings_baseline, pred_baseline

    def _get_embeddings(self, X):
        X = pd.DataFrame(X)

        embeddings = []
        for idx, row in tqdm(X.iterrows(), total=len(X), position=0, leave=True, desc="Embedding Dataset"):

            embeddings.append(self.embedding_db.get_embedding(pd.DataFrame(row).T))
        return np.vstack(embeddings)

    @abstractmethod
    def _get_features(self, embeddings, y, is_test):
        pass

