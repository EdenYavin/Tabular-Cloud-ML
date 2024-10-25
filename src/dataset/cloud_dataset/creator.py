import pandas as pd
from openai import embeddings
from tqdm import tqdm
import numpy as np

from src.encryptor import BaseEncryptor
from src.cloud.base import CloudModels
from src.utils.helpers import sample_noise, one_hot_labels, load_cache_file, save_cache_file, expand_matrix_to_img_size
from src.utils.config import config
from src.utils.cache import DBFactory

class Dataset(object):

    def __init__(self, dataset_name, cloud_models, encryptor, embeddings_model,
                 n_pred_vectors, n_noise_samples, metadata = None):

        self.cloud_models: CloudModels = cloud_models
        self.encryptor: BaseEncryptor = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.n_noise_samples = n_noise_samples
        self.one_hot = config.dataset_config.one_hot
        self.name = dataset_name
        self.split_ratio = config.dataset_config.split_ratio
        self.use_embedding = config.experiment_config.use_embedding
        self.use_noise_labels = config.experiment_config.use_labels
        self.use_predictions = config.experiment_config.use_preds
        self.force_run = config.dataset_config.force_to_create_again
        self.raw_metadata = metadata
        self.embeddings_model = embeddings_model
        self.db = DBFactory.get_db(dataset_name, self.embeddings_model)


    def create(self, X_train, y_train, X_test, y_test) -> dict:

        name = f"{self.name}_{'one_hot' if self.one_hot else ''}"

        if dataset := load_cache_file(dataset_name=name, split_ratio=self.split_ratio):
            if not self.force_run:
                print(f"Dataset {self.name} was already processed before, loading cache")
                return dataset

        X_train, y_train = self._create_train(X_train, y_train)
        X_test = self._create_test(X_test, y_test)

        if self.one_hot:
            num_classes = len(np.unique(y_train))
            y_train = one_hot_labels(labels=y_train, num_classes=num_classes)
            y_test = one_hot_labels(labels=y_test, num_classes=num_classes)


        train = [X_train, y_train]
        test = [X_test, y_test]

        dataset = {
            "train": train,
            "test": test
        }

        save_cache_file(dataset_name=name, split_ratio=self.split_ratio, data=dataset)
        self.db.save()
        return dataset

    def _create_train(self, X, y):

        new_y = []
        examples = []

        X = pd.DataFrame(X)

        for idx, row in tqdm(X.iterrows(), total=len(X), leave=True, position=0):

            for _ in range(self.n_pred_vectors):

                # Because we are expanding the dataset to more samples we need to expand the labels as well
                new_y.append(y[idx])

                example = []

                # For each new pred vector we will sample new noise to be used. This will cause
                # The prediction vector to be different each time
                samples, noise_labels = sample_noise(row=row, X=X, y=pd.Series(y), sample_n=self.n_noise_samples)

                embeddings = self.db.get_embedding(samples)

                encrypted_data = self.encryptor.encode(embeddings)
                encrypted_data = (encrypted_data * 10000).astype(np.uint8)
                predictions = self.cloud_models.predict(encrypted_data)

                if self.use_predictions:
                    example.append(predictions) # Shape - |CMLS|
                if self.use_embedding:
                    example.append(embeddings.reshape(1,-1)) # Shape - (1,|Embedding| * Number of noise samples)
                if self.use_noise_labels and noise_labels.shape[0] > 0:
                    example.append(noise_labels) # Shape - |V| * Number of noise samples

                examples.append(np.hstack(example))

        return np.vstack(examples), np.array(new_y)

    def _create_test(self, X, y):
        examples = []

        X = pd.DataFrame(X)

        for idx, row in tqdm(X.iterrows(), total=len(X), leave=True, position=0):

            # We can't touch the test set, i.e. expand it to more samples. So we do it only once
            example = []

            # For each new pred vector we will sample new noise to be used. This will cause
            # The prediction vector to be different each time
            samples, noise_labels = sample_noise(row=row, X=X, y=pd.Series(y), sample_n=self.n_noise_samples)

            embeddings = self.db.get_embedding(samples)

            encrypted_data = self.encryptor.encode(embeddings)
            encrypted_data = (encrypted_data * 10000).astype(np.uint8)

            predictions = self.cloud_models.predict(encrypted_data)

            if self.use_predictions:
                example.append(predictions)
            if self.use_embedding:
                example.append(embeddings.reshape(1, -1))  # Shape - (1,|Embedding| * Number of noise samples)
            if self.use_noise_labels and noise_labels.shape[0] > 0:
                example.append(noise_labels)

            examples.append(np.hstack(example))

        return np.vstack(examples)
