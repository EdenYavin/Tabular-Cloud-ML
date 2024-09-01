import pandas as pd


from src.encryptor.model import Encryptor
from src.cloud.base import CloudModels
from src.utils.helpers import sample_noise, one_hot_labels, load_cache_file, save_cache_file
from tqdm import tqdm
import numpy as np

np.random.seed(42)


class Dataset(object):

    def __init__(self, dataset_name, cloud_models, encryptor, n_pred_vectors, n_noise_samples,
                 use_embedding=True, use_noise_labels=True,use_predictions=False, ratio=0.2,
                 one_hot=False, force=False, raw_metadata = None
                 ):

        self.cloud_models: CloudModels = cloud_models
        self.encryptor: Encryptor = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.n_noise_samples = n_noise_samples
        self.one_hot = one_hot
        self.name = dataset_name
        self.split_ratio = ratio
        self.use_embedding = use_embedding
        self.use_noise_labels = use_noise_labels
        self.use_predictions = use_predictions
        self.force_run = force
        self.raw_metadata = raw_metadata

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


        train = [X_train, y_train]
        test = [X_test, y_test]

        dataset = {
            "train": train,
            "test": test
        }

        save_cache_file(dataset_name=name, split_ratio=self.split_ratio, data=dataset)

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
                encrypted_data = self.encryptor.encode(samples)

                predictions = self.cloud_models.predict(encrypted_data)

                if self.use_predictions:
                    example.append(predictions)
                if self.use_embedding:
                    example.append(row.values.reshape(1, -1))
                if self.use_noise_labels and noise_labels:
                    example.append(noise_labels)

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
            encrypted_data = self.encryptor.encode(samples)

            predictions = self.cloud_models.predict(encrypted_data)

            if self.use_predictions:
                example.append(predictions)
            if self.use_embedding:
                example.append(row.values.reshape(1, -1))
            if self.use_noise_labels and noise_labels:
                example.append(noise_labels)

            examples.append(np.hstack(example))

        return np.vstack(examples)
