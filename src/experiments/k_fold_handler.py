import numpy as np
from tqdm import tqdm

from src.embeddings import EmbeddingsFactory
import src.utils.constansts as consts
from src.internal_model.baseline import EmbeddingBaseline
from src.pipeline.encoding_pipeline import FeatureEngineeringPipeline
from src.cloud import CloudModel, CLOUD_MODELS
from src.encryptor import EncryptorFactory
from src.encryptor.base import BaseEncryptor
from src.internal_model.model import InternalInferenceModelFactory

from src.dataset import DATASETS, RawDataset
from src.utils.config import config
import pandas as pd

from src.utils.db import RawSplitDBFactory


class KFoldExperimentHandler:

    def __init__(self):
        w_emb = "w_emb" if config.experiment_config.use_embedding else "wo_emb"
        w_noise_labels = "w_noise_labels" if config.experiment_config.use_labels else "wo_noise_labels"
        w_pred = "w_pred" if config.experiment_config.use_preds else "wo_pred"
        self.experiment_name = f"{w_emb}_{w_noise_labels}_{w_pred}"
        self.n_pred_vectors = config.experiment_config.n_pred_vectors
        self.n_noise_samples = config.experiment_config.n_noise_samples
        self.k_folds = config.experiment_config.k_folds


    def run_experiment(self):
        print(f"##### STARTING {self.k_folds}-FOLD EXPERIMENT ########\n\n")
        if type(self.n_noise_samples) is int:
            self.n_noise_samples = [self.n_noise_samples]

        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]

        reports = []
        datasets = config.dataset_config.names

        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):

            progress = tqdm(total=self.k_folds, desc="K-Folds", unit="Fold")


            raw_dataset: RawDataset = DATASETS[dataset_name]()

            cloud_model: CloudModel = CLOUD_MODELS[config.cloud_config.name](
                num_classes=raw_dataset.get_n_classes()
            )
            embedding_model = EmbeddingsFactory().get_model(X=raw_dataset.X, y=raw_dataset.y)
            encryptor: BaseEncryptor = EncryptorFactory().get_model(
                output_shape=(1, *cloud_model.input_shape),
            )

            X_train, X_test, X_sample, y_train, y_test, y_sample = RawSplitDBFactory.get_db(raw_dataset).get_split()
            print(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")


            cloud_model.fit(X_train, y_train, **raw_dataset.metadata)

            print("#### GETTING CLOUD DATASET FULL BASELINE####")
            cloud_acc, cloud_f1 = raw_dataset.get_cloud_model_baseline(X_train, X_test, y_train, y_test)

            print("#### GETTING RAW BASELINE PREDICTION ####")
            raw_baseline_acc, raw_baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)

            # Initialize lists to store metrics across folds
            cloud_acc_scores = []
            cloud_f1_scores = []
            baseline_acc_scores = []
            baseline_emb_acc_scores = []
            baseline_f1_scores = []
            baseline_emb_f1_scores = []
            train_acc_scores = []
            train_f1_scores = []
            test_acc_scores = []
            test_f1_scores = []

            for X_train, X_test, X_sample, y_sample, y_train, y_test in raw_dataset.k_fold_iterator(n_splits=self.k_folds):

                for n_noise_samples in self.n_noise_samples:
                    for n_pred_vectors in self.n_pred_vectors:

                        print(f"CREATING THE CLOUD-TRAINSET FROM {dataset_name},"
                              f" WITH {n_noise_samples} NOISE SAMPLES AND {n_pred_vectors} PREDICTION VECTORS")

                        dataset_creator = FeatureEngineeringPipeline(
                            dataset_name=dataset_name,
                            cloud_models=cloud_model,
                            encryptor=encryptor,
                            embeddings_model=embedding_model,
                            n_noise_samples=n_noise_samples,
                            n_pred_vectors=n_pred_vectors,
                            metadata=raw_dataset.metadata
                        )
                        dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)
                        print("Finished Creating the dataset")

                        print("#### GETTING EMBEDDING BASELINE PREDICTION ####")
                        baseline_model = EmbeddingBaseline(
                            num_classes=raw_dataset.get_n_classes(),
                            input_shape=dataset[consts.IIM_BASELINE_TRAIN_SET_TOKEN][0].shape[1:],
                        )
                        baseline_model.fit(
                            *dataset[consts.IIM_BASELINE_TRAIN_SET_TOKEN]
                        )
                        baseline_emb_acc, baseline_emb_f1 = baseline_model.evaluate(
                            *dataset[consts.IIM_BASELINE_TEST_SET_TOKEN])

                        internal_model = InternalInferenceModelFactory().get_model(
                            num_classes=raw_dataset.get_n_classes(),
                            input_shape=dataset[consts.IIM_TRAIN_SET_TOKEN][0].shape[1],
                            # Only give the number of features
                        )

                        print(f"Training the IIM {internal_model.name} Model")
                        internal_model.fit(
                            *dataset[consts.IIM_TRAIN_SET_TOKEN]
                        )
                        train_acc, train_f1 = internal_model.evaluate(*dataset[consts.IIM_TRAIN_SET_TOKEN])
                        test_acc, test_f1 = internal_model.evaluate(*dataset[consts.IIM_TEST_SET_TOKEN])

                        # Append scores for each fold
                        cloud_acc_scores.append(cloud_acc)
                        cloud_f1_scores.append(cloud_f1)
                        baseline_acc_scores.append(raw_baseline_acc)
                        baseline_emb_acc_scores.append(baseline_emb_acc)
                        baseline_f1_scores.append(raw_baseline_f1)
                        baseline_emb_f1_scores.append(baseline_emb_f1)
                        train_acc_scores.append(train_acc)
                        train_f1_scores.append(train_f1)
                        test_acc_scores.append(test_acc)
                        test_f1_scores.append(test_f1)

                progress.update(1)

            # Create a final report with average metrics
            average_cloud_acc = np.mean(cloud_acc_scores)
            average_cloud_f1 = np.mean(cloud_f1_scores)
            average_baseline_acc = np.mean(baseline_acc_scores)
            average_emb_acc = np.mean(baseline_emb_acc_scores)
            average_baseline_f1 = np.mean(baseline_f1_scores)
            average_emb_f1 = np.mean(baseline_emb_f1_scores)
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
                    "cloud_model": [cloud_model.name],
                    "iim_train_acc": [average_train_acc],
                    "iim_train_f1": [average_train_f1],
                    "iim_test_acc": [average_test_acc],
                    "iim_test_f1": [average_test_f1],
                    "raw_baseline_acc": [average_baseline_acc],
                    "raw_baseline_f1": [average_baseline_f1],
                    "emb_baseline_acc": [average_emb_acc],
                    "emb_baseline_f1": [average_emb_f1],
                    "cloud_acc": [average_cloud_acc],
                    "cloud_f1": [average_cloud_f1],
                }
            )

            reports.append(final_report)


        return pd.concat(reports)
