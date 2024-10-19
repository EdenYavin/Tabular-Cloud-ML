import numpy as np
from tqdm import tqdm

from src.embeddings.model import NumericalTableEmbeddings

from src.dataset.cloud_dataset.creator import Dataset
from src.cloud import CloudModels, CLOUD_MODELS
from src.encryptor import BaseEncryptor, EncryptorFactory
from src.internal_model.model import InternalInferenceModelFactory

import src.utils.constansts as consts
from src.dataset.raw import DATASETS, RawDataset
from src.utils.config import config
import pandas as pd


class KFoldExperimentHandler:

    def __init__(self):
        self.experiment_name = config.experiment_config.name
        self.n_pred_vectors = config.experiment_config.n_pred_vectors
        self.n_noise_samples = config.experiment_config.n_noise_samples
        self.k_folds = config.experiment_config.k_folds


    def run_experiment(self):

        if type(self.n_noise_samples) is int:
            self.n_noise_samples = [self.n_noise_samples]

        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]

        reports = []
        datasets = config.dataset_config.names

        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            raw_dataset: RawDataset = DATASETS[dataset_name]()

            progress = tqdm(total=self.k_folds, desc="K-Folds", unit="Fold")

            cloud_models: CloudModels = CLOUD_MODELS[config.cloud_config.name](
                num_classes=raw_dataset.get_n_classes()
            )

            embedding_model = NumericalTableEmbeddings()
            encryptor: BaseEncryptor = EncryptorFactory().get_model(
                output_shape=(1, *embedding_model.output_shape),
            )

            # Initialize lists to store metrics across folds
            cloud_acc_scores = []
            cloud_f1_scores = []
            baseline_acc_scores = []
            baseline_f1_scores = []
            train_acc_scores = []
            train_f1_scores = []
            test_acc_scores = []
            test_f1_scores = []

            for X_train, X_test, X_sample, y_sample, y_train, y_test in raw_dataset.k_fold_iterator(n_splits=self.k_folds):

                for n_noise_samples in self.n_noise_samples:
                    for n_pred_vectors in self.n_pred_vectors:

                        cloud_models.fit(X_train, y_train)

                        cloud_acc, cloud_f1 = cloud_models.evaluate(X_test, y_test)
                        baseline_acc, baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)

                        dataset_creator = Dataset(
                            dataset_name=dataset_name,
                            cloud_models=cloud_models,
                            encryptor=encryptor,
                            embeddings_model=embedding_model,
                            n_noise_samples=n_noise_samples,
                            n_pred_vectors=n_pred_vectors,
                            metadata=raw_dataset.metadata
                        )
                        dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)

                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=raw_dataset.get_n_classes(),
                            input_shape=dataset['train'][0].shape[1],  # Only give the number of features
                        )

                        internal_model.fit(*dataset['train'])
                        train_acc, train_f1 = internal_model.evaluate(*dataset['train'])
                        test_acc, test_f1 = internal_model.evaluate(*dataset['test'])

                        # Append scores for each fold
                        cloud_acc_scores.append(cloud_acc)
                        cloud_f1_scores.append(cloud_f1)
                        baseline_acc_scores.append(baseline_acc)
                        baseline_f1_scores.append(baseline_f1)
                        train_acc_scores.append(train_acc)
                        train_f1_scores.append(train_f1)
                        test_acc_scores.append(test_acc)
                        test_f1_scores.append(test_f1)

                progress.update(1)

            # Create a final report with average metrics
            average_cloud_acc = np.mean(cloud_acc_scores)
            average_cloud_f1 = np.mean(cloud_f1_scores)
            average_baseline_acc = np.mean(baseline_acc_scores)
            average_baseline_f1 = np.mean(baseline_f1_scores)
            average_train_acc = np.mean(train_acc_scores)
            average_train_f1 = np.mean(train_f1_scores)
            average_test_acc = np.mean(test_acc_scores)
            average_test_f1 = np.mean(test_f1_scores)

            final_report = pd.DataFrame(
                {
                    "exp_name": [self.experiment_name],
                    "dataset": [dataset_name],
                    "train_size_ratio": [config.dataset_config.split_ratio],
                    "n_pred_vectors": [self.n_pred_vectors],
                    "n_noise_sample": [self.n_noise_samples],
                    "iim_model": [config.iim_config.name],
                    "encryptor": [config.encoder_config.name],
                    "cloud_model": [cloud_models.name],
                    "iim_train_acc": [average_train_acc],
                    "iim_train_f1": [average_train_f1],
                    "iim_test_acc": [average_test_acc],
                    "iim_test_f1": [average_test_f1],
                    "baseline_acc": [average_baseline_acc],
                    "baseline_f1": [average_baseline_f1],
                    "cloud_acc": [average_cloud_acc],
                    "cloud_f1": [average_cloud_f1],
                }
            )

            reports.append(final_report)


        return pd.concat(reports)
