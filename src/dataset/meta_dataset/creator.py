from src.utils.helpers import load_data
import src.utils.constansts as consts
from src.encryptor.model import Encryptor
from src.cloud.models import CloudModels
from src.utils.helpers import sample_noise, one_hot_labels
from tqdm import tqdm
import numpy as np


class Dataset(object):

    def __init__(self, config, cloud_models, encryptor, n_pred_vectors, n_noise_samples):
        self.config = config
        self.cloud_models: CloudModels = cloud_models
        self.encryptor: Encryptor = encryptor
        self.n_pred_vectors = n_pred_vectors
        self.n_noise_samples = n_noise_samples
        self.train = []
        self.test = []

        self.name = self.config[consts.CONFIG_DATASET_NAME_TOKEN]
        self.split_ratio = self.config[consts.CONFIG_DATASET_SPLIT_RATIO_TOKEN]

    def create(self):
        X_train, y_train, X_test, y_test = load_data(dataset_name=self.name,
                                                     split_ratio=self.split_ratio
                                                     )

        if self.config[consts.CONFIG_DATASET_ONEHOT_TOKEN]:
            num_classes = int(np.unique(y_train))
            y_train = one_hot_labels(labels=y_train, num_classes=num_classes)
            y_test = one_hot_labels(labels=y_test, num_classes=num_classes)

        X_train = self._create(X_train, y_train)
        X_test = self._create(X_test, y_test)

        self.train = [X_train, y_train]
        self.test = [X_test, y_test]
        return self

    def _create(self, X, y):
        examples = []

        print(f"CREATING THE META-DATASET FROM {self.name}")
        print(f"ORIGINAL DATASET SIZE {X.shape}")
        print(f"ENCRYPTED DATA SIZE {self.encryptor.shape}")

        for idx, row in tqdm(enumerate(X), total=len(X)):

            samples, noise_labels = sample_noise(row=row, X=X,y=y, sample_n=self.n_noise_samples)

            predictions = []

            for _ in range(self.n_pred_vectors):
                encrypted_data = self.encryptor(samples)

                predictions.append(self.cloud_models.predict(encrypted_data))
                predictions = np.hstack(predictions)

            examples.append(
                np.hstack([
                    samples,
                    noise_labels,
                    predictions
                ]
                )
            )

        return np.vstack(examples)







