import pandas as pd

from src.utils.helpers import load_data
import src.utils.constansts as consts
from src.encryptor.model import Encryptor
from src.cloud.models import CloudModels
from src.utils.helpers import sample_noise, one_hot_labels, load_cache_file, save_cache_file
from tqdm import tqdm
import numpy as np


class Dataset(object):

    def __init__(self, dataset_name, config, cloud_models, encryptor, n_pred_vectors, n_noise_samples,
                 use_embedding=True, use_noise_labels=True):
        self.config = config
        self.cloud_models: CloudModels = cloud_models
        self.encryptor: Encryptor = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.n_noise_samples = n_noise_samples

        self.name = dataset_name
        self.split_ratio = self.config[consts.CONFIG_DATASET_SPLIT_RATIO_TOKEN]
        self.use_embedding = use_embedding
        self.use_noise_labels = use_noise_labels

    def create(self) -> dict:

        if dataset := load_cache_file(dataset_name=self.name, split_ratio=self.split_ratio):
            if not self.config[consts.CONFIG_DATASET_FORCE_CREATION_TOKEN]:
                print(f"Dataset {self.name} was already processed before, loading cache")
                return dataset

        X_train, y_train, X_test, y_test = load_data(dataset_name=self.name,
                                                     split_ratio=self.split_ratio
                                                     )

        if self.config[consts.CONFIG_DATASET_ONEHOT_TOKEN]:
            num_classes = int(np.unique(y_train))
            y_train = one_hot_labels(labels=y_train, num_classes=num_classes)
            y_test = one_hot_labels(labels=y_test, num_classes=num_classes)

        X_train = self._create(X_train, y_train)
        X_test = self._create(X_test, y_test)

        train = [X_train, y_train]
        test = [X_test, y_test]

        dataset = {
            "train": train,
            "test": test
        }

        save_cache_file(dataset_name=self.name, split_ratio=self.split_ratio, data=dataset)

        return dataset

    def _create(self, X, y):
        examples = []

        print(f"CREATING THE META-DATASET FROM {self.name}")
        print(f"ORIGINAL DATASET SIZE {X.shape}")

        X = pd.DataFrame(X)

        for idx, row in tqdm(X.iterrows(), total=len(X)):

            for _ in range(self.n_pred_vectors):
                example = []

                # For each new pred vector we will sample new noise to be used. This will cause
                # The prediction vector to be different each time
                samples, noise_labels = sample_noise(row=row, X=X, y=pd.Series(y), sample_n=self.n_noise_samples)
                encrypted_data = self.encryptor.encode(samples)

                predictions = self.cloud_models.predict(encrypted_data)

                example.append(predictions)
                if self.use_embedding:
                    example.append(row.values.reshape(1, -1))
                if self.use_noise_labels:
                    example.append(noise_labels)

                examples.append(np.hstack(example))

        return np.vstack(examples)
