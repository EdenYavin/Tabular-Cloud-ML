
import pandas as pd
from keras.src.utils import to_categorical
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.domain.dataset import (EmbeddingBaselineDataset,
                                EmbeddingBaselineFeatures,
                                PredictionBaselineFeatures,
                                PredictionBaselineDataset,
                                IIMFeatures,
                                IIMDataset,
                                Batch
                                )
from src.encryptor.base import Encryptors
from src.cloud.base import CloudModel
from src.utils.constansts import GPU_DEVICE
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory


class FeatureEngineeringPipeline(object):

    def __init__(self, dataset_name, cloud_models, encryptor, embeddings_model,
                 n_pred_vectors, metadata = None):

        self.cloud_models: list[CloudModel] = cloud_models
        self.encryptor: Encryptors = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.name = dataset_name
        self.use_embedding = config.experiment_config.use_embedding
        self.use_predictions = config.experiment_config.use_preds
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)


    def create(self, X_train, y_train, X_test, y_test) -> tuple[list[IIMDataset], EmbeddingBaselineDataset, PredictionBaselineDataset]:

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

        datasets = [
                    IIMDataset(
                    train=IIMFeatures(features=x_train, labels=y_train),
                    test=IIMFeatures(features=x_test, labels=y_test)
                )
            for x_train, x_test in zip(Xs_train, Xs_test)
            ]

        self.db.save()
        return datasets, embeddings_baseline, pred_baseline


    def _get_embeddings(self, X):
        X = pd.DataFrame(X)

        embeddings = []
        for idx, row in tqdm(X.iterrows(), total=len(X), position=0, leave=True, desc="Embedding Dataset"):

            embeddings.append(self.db.get_embedding(pd.DataFrame(row).T))
        return np.vstack(embeddings)


    def _get_features(self, embeddings, y, is_test):

        datasets = [[] for _ in range(len(self.cloud_models))]
        predictions_for_baseline = [] # Will be used for the baseline
        embeddings_for_baseline = [] # Will be used for embedding baseline
        new_y = []

        batch = Batch(X=embeddings,y=y, size=config.dataset_config.batch_size)
        for mini_batch, labels in batch:

            # For the training set creation we can create multiple encoded images for each input
            # and by doing so augmenting the training set beyond the original size.
            # For the testing, we can't create new samples
            number_of_new_samples = (
                self.n_pred_vectors if not is_test
                else 1
            )

            embeddings_samples = np.vstack([mini_batch for _ in range(
                number_of_new_samples)])  # Duplicate the embeddings as the number of predictions
            embeddings_for_baseline.append(embeddings_samples)

            with tf.device(GPU_DEVICE):

                images = self.encryptor.encode(mini_batch,
                                               number_of_new_samples)  # We are encrypting each sample N times, where N is the number of prediction vectors we want to use as features
                for idx, cloud_model in enumerate(self.cloud_models):

                    # We are then creating a prediction vector for each new encoded sample (image)
                    predictions = cloud_model.predict(images)

                    predictions = np.vstack(list(predictions)) # Create one feature vector of all concatenated predictions

                    datasets[idx].append(np.hstack([predictions, embeddings_samples]))  # Shape - |CMLS|, (1,|Embedding| * Number of noise samples)

                    predictions_for_baseline.append(predictions)

                # Because we are potentially expanding X_train we should also expand y_train
                # with the same number of labels. We do so by duplicate the labels for each new augmentation.
                # For example if we encode x_1 to 3 new samples x_enc_1_1, x_enc_1_2, x_enc_1_2 and the original
                # label for x_1 was 1, than we add [1,1,1] to y_train.
                # This is also true in regard to the number of cloud models we use, each new pred vector will be
                # a new sample
                # For y_test, we just duplicate it based on the cloud models alone
                for label in labels:
                    if not is_test:
                        [new_y.append(label) for _ in range(number_of_new_samples * len(self.cloud_models))]
                    else:
                        [new_y.append(label) for _ in range(len(self.cloud_models))]

        return [np.vstack(d) for d in datasets], np.vstack(new_y), np.vstack(predictions_for_baseline)