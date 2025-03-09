import pandas as pd
from keras.src.utils import to_categorical
from tqdm import tqdm
import numpy as np
from loguru import logger
import tensorflow as tf

from src.domain.dataset import PredictionsDataset,IIMFeatures,PredictionBaselineFeatures,EmbeddingBaselineFeatures, Batch
from src.encryptor.base import Encryptors
from src.cloud.base import CloudModel
from src.utils.constansts import GPU_DEVICE
from src.utils.helpers import sample_noise, load_cache_file, save_cache_file
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory


class FeatureEngineeringPipeline(object):

    def __init__(self, dataset_name, cloud_models, encryptor, embeddings_model,
                 n_pred_vectors, n_noise_samples, metadata = None):

        self.cloud_models: list[CloudModel] = cloud_models
        self.encryptor: Encryptors = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.n_noise_samples = n_noise_samples
        self.split_ratio = config.dataset_config.split_ratio
        self.use_embedding = config.experiment_config.use_embedding
        self.use_noise_labels = config.experiment_config.use_labels
        self.use_predictions = config.experiment_config.use_preds
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)


    def create(self, X_train, y_train, X_test, y_test) -> PredictionsDataset:


        X_train, new_y_train, X_emb_train, X_pred_train = self._get_new_features(X_train, y_train, is_test=False)
        X_test, new_y_test, X_emb_test, X_pred_test = self._get_new_features(X_test, y_test, is_test=True)

        # One hot encode the labels
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes=num_classes)
        new_y_train = to_categorical(new_y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)
        new_y_test = to_categorical(new_y_test, num_classes=num_classes)

        dataset = PredictionsDataset(
            train_iim_features=IIMFeatures(features=X_train, labels=new_y_train),
            train_embeddings=EmbeddingBaselineFeatures(embeddings=X_emb_train, labels=y_train),
            train_predictions=PredictionBaselineFeatures(predictions=X_pred_train, labels=new_y_train),
            test_iim_features=IIMFeatures(features=X_test, labels=new_y_test),
            test_embeddings=EmbeddingBaselineFeatures(embeddings=X_emb_test, labels=y_test),
            test_predictions=PredictionBaselineFeatures(predictions=X_pred_test, labels=new_y_test),
        )

        self.db.save()
        return dataset

    def _prepare_embedding_data(self, X, y, is_test=False):
        new_y = []
        noise_labels_data = []
        embeddings = []
        for idx, row in tqdm(X.iterrows(), total=len(X), position=0, leave=True, desc="Embedding Dataset"):

            samples, noise_labels = sample_noise(row=row, X=X, y=pd.Series(y), sample_n=self.n_noise_samples)
            noise_labels_data.append(noise_labels)
            new_y.append(y[idx])
            if not is_test and noise_labels.shape[0] > 0:
                # For train we will create new labels based on the number of noise samples
                new_y.append(noise_labels)

            embeddings.append(self.db.get_embedding(samples))

        return embeddings, np.vstack(new_y), np.vstack(noise_labels_data)

    def _get_new_features(self, X, y, is_test):

        observations = []
        embeddings_for_baseline = [] # Will be used for the baseline
        predictions_for_baseline = [] # Will be used for the baseline

        X = pd.DataFrame(X)

        embeddings, y, noise_labels = self._prepare_embedding_data(X, y, is_test)

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
            # Because we are potentially expanding X_train we should also expand y_train
            # with the same number of labels. We do so by duplicate the labels for each new augmentation.
            # For example if we encode x_1 to 3 new samples x_enc_1_1, x_enc_1_2, x_enc_1_2 and the original
            # label for x_1 was 1, than we add [1,1,1] to y_train.
            # For y_test, we just duplicate it
            for label in labels:
                if not is_test:
                    [new_y.append(label) for _ in range(number_of_new_samples)]
                else:
                    new_y.append(label)

            observation = []

            embeddings_samples = np.vstack([mini_batch for _ in range(
                number_of_new_samples)])  # Duplicate the embeddings as the number of predictions
            embeddings_for_baseline.append(mini_batch)

            with tf.device(GPU_DEVICE):

                images = self.encryptor.encode(mini_batch, number_of_new_samples) # We are encrypting each sample N times, where N is the number of prediction vectors we want to use as feautres
                # image = (encrypted_data * 10000).astype(np.uint8)

                for cloud_model in self.cloud_models:
                    # We are then creating a prediction vector for each new encoded sample (image)
                    predictions = cloud_model.predict(images)
                    predictions = np.vstack(predictions) # Create one feature vector of all concatenated predictions

                    observation.append(predictions) # Shape - |CMLS|
                    observation.append(embeddings_samples) # Shape - (1,|Embedding| * Number of noise samples)

                    observations.append(np.hstack(observation))
                    predictions_for_baseline.append(predictions)

        return np.vstack(observations), np.vstack(new_y), np.vstack(embeddings_for_baseline), np.vstack(predictions_for_baseline)