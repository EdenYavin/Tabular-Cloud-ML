import pandas as pd
from keras.src.utils import to_categorical
from tqdm import tqdm
import numpy as np

from src.encryptor.base import BaseEncryptor
from src.cloud.base import CloudModel
from src.utils.helpers import sample_noise, load_cache_file, save_cache_file
from src.utils.config import config
from src.utils.db import EmbeddingDBFactory
import src.utils.constansts as consts

class Pipeline(object):

    def __init__(self, dataset_name, cloud_models, encryptor, embeddings_model,
                 n_pred_vectors, n_noise_samples, metadata = None):

        self.cloud_model: CloudModel = cloud_models
        self.encryptor: BaseEncryptor = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.n_noise_samples = n_noise_samples
        self.one_hot = config.dataset_config.one_hot
        self.name = dataset_name
        self.split_ratio = config.dataset_config.split_ratio
        self.use_embedding = config.experiment_config.use_embedding
        self.use_noise_labels = config.experiment_config.use_labels
        self.use_predictions = config.experiment_config.use_preds
        self.force_run = config.pipeline_config.force_to_create_again
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.db = EmbeddingDBFactory.get_db(dataset_name, self.embeddings_model)


    def create(self, X_train, y_train, X_test, y_test) -> dict:

        name = f"{self.name}_{'one_hot' if self.one_hot else ''}"

        if dataset := load_cache_file(dataset_name=name, split_ratio=self.split_ratio):
            if not self.force_run:
                print(f"Dataset {self.name} was already processed before, loading cache")
                return dataset

        X_train, y_train, X_emb_train = self._create_train(X_train, y_train)
        X_test, X_emb_test = self._create_test(X_test, y_test)

        if self.one_hot:
            num_classes = len(np.unique(y_train))
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)


        train = [X_train, y_train]
        test = [X_test, y_test]

        dataset = {
            consts.IIM_TRAIN_SET_TOKEN: train,
            consts.IIM_BASELINE_TRAIN_SET_TOKEN: [X_emb_train, y_train],
            consts.IIM_TEST_SET_TOKEN: test,
            consts.IIM_BASELINE_TEST_SET_TOKEN: [X_emb_test, y_test]
        }

        save_cache_file(dataset_name=name, split_ratio=self.split_ratio, data=dataset)
        self.db.save()
        return dataset

    def _create_train(self, X, y):

        new_y = []
        observations = []
        embeddings_for_baseline = [] # Will be used for the baseline

        X = pd.DataFrame(X)

        for idx, row in tqdm(X.iterrows(), total=len(X), leave=True, position=0):

            for _ in range(self.n_pred_vectors):

                # Because we are expanding the dataset to more samples (depending on n_pred) we need to expand the labels as well
                new_y.append(y[idx])

                observation = []

                # For each new pred vector we will sample new noise to be used. This will cause
                # The prediction vector to be different each time
                samples, noise_labels = sample_noise(row=row, X=X, y=pd.Series(y), sample_n=self.n_noise_samples)

                embeddings = self.db.get_embedding(samples)

                encrypted_data = self.encryptor.encode(embeddings)
                image = (encrypted_data * 10000).astype(np.uint8)

                predictions = self.cloud_model.predict(image)

                if self.use_predictions:
                    observation.append(predictions) # Shape - |CMLS|
                if self.use_embedding:
                    observation.append(embeddings.reshape(1,-1)) # Shape - (1,|Embedding| * Number of noise samples)
                if self.use_noise_labels and noise_labels.shape[0] > 0:
                    observation.append(noise_labels) # Shape - |V| * Number of noise samples

                observations.append(np.hstack(observation))
                embeddings_for_baseline.append(embeddings)

        return np.vstack(observations), np.array(new_y), np.stack(embeddings_for_baseline)

    def _create_test(self, X, y):
        observations = []
        embeddings_for_baseline = []  # Will be used for the baseline

        X = pd.DataFrame(X)

        for idx, row in tqdm(X.iterrows(), total=len(X), leave=True, position=0):

            # We can't touch the test set, i.e. expand it to more samples. So we do it only once
            sample = []

            # For each new pred vector we will sample new noise to be used. This will cause
            # The prediction vector to be different each time
            samples, noise_labels = sample_noise(row=row, X=X, y=pd.Series(y), sample_n=self.n_noise_samples)

            embeddings = self.db.get_embedding(samples)

            encrypted_data = self.encryptor.encode(embeddings)
            image = (encrypted_data * 10000).astype(np.uint8)

            predictions = self.cloud_model.predict(image)

            if self.use_predictions:
                sample.append(predictions)
            if self.use_embedding:
                sample.append(embeddings.reshape(1, -1))  # Shape - (1,|Embedding| * Number of noise samples)
            if self.use_noise_labels and noise_labels.shape[0] > 0:
                sample.append(noise_labels)

            observations.append(np.hstack(sample))
            embeddings_for_baseline.append(embeddings)

        return np.vstack(observations), np.stack(embeddings_for_baseline)
