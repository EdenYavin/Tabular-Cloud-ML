
import pandas as pd
from keras.src.utils import to_categorical
import numpy as np
import tensorflow as tf

from src.domain.dataset import Batch, IIMFeatures
from src.encryptor.base import Encryptors
from src.cloud.base import CloudModel
from src.utils.constansts import GPU_DEVICE
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory


class FeatureEngineeringPipeline(object):

    def __init__(self, dataset_name, cloud_models, encryptor, embeddings_model, metadata = None):

        self.cloud_model: CloudModel = cloud_models
        self.encryptor: Encryptors = encryptor
        self.name = dataset_name
        self.split_ratio = config.dataset_config.split_ratio
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)


    def create(self, X, y) -> IIMFeatures:

        X = self._get_new_features(X, y)

        # One hot encode the labels
        num_classes = len(np.unique(y))
        y = to_categorical(y, num_classes=num_classes)

        self.db.save()
        return IIMFeatures(
            features=X,
            labels=y,
        )


    def _get_new_features(self, X, y, is_test):

        observations = []

        X = pd.DataFrame(X)

        embeddings = X.apply(self.db.get_embedding)

        batch = Batch(X=embeddings,y=y, size=config.dataset_config.batch_size)
        for mini_batch, labels in batch:


            with tf.device(GPU_DEVICE):

                images = self.encryptor.encode(mini_batch, 1)

                # We are then creating a prediction vector for each new encoded sample (image)
                predictions = self.cloud_model.predict(images)

            observations.append(np.vstack(predictions))

        return np.vstack(observations)