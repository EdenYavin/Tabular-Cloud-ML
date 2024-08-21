import numpy as np
from tensorflow.core.config.flags import config
from tqdm import tqdm

from src.dataset.cloud_dataset.creator import Dataset
from src.cloud import CloudModels, CLOUD_MODELS
from src.encryptor.model import Encryptor
from src.internal_model.model import InternalInferenceModelFactory

import src.utils.constansts as consts
from src.dataset.raw import DATASETS, RawDataset
import pandas as pd


class ExperimentHandler:

    def __init__(self, experiment_config: dict):
        self.experiment_name = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get("name")
        self.n_pred_vectors = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get("n_pred_vectors", 1)
        self.n_noise_samples = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get("n_noise_samples", 3)
        self.k_folds = experiment_config[consts.CONFIG_EXPERIMENT_SECTION].get(consts.K_FOLDS_TOKEN, 10)
        self.config = experiment_config

    def run_experiment(self):

        # For dynamic run time - the list type is preferred
        if type(self.n_noise_samples) is int:
            self.n_noise_samples = [self.n_noise_samples]

        if type(self.n_pred_vectors) is int:
            self.n_pred_vectors = [self.n_pred_vectors]


        # Create a final report with average metrics
        final_report = pd.DataFrame()

        datasets = self.config[consts.CONFIG_DATASET_SECTION][consts.CONFIG_DATASET_NAME_TOKEN]

        for dataset_name in tqdm(datasets, total=len(datasets), desc="Datasets Progress", unit="dataset"):
            raw_dataset: RawDataset = DATASETS[dataset_name](**self.config[consts.CONFIG_DATASET_SECTION])
            for n_noise_samples in self.n_noise_samples:
                for n_pred_vectors in self.n_pred_vectors:

                    X_train, X_test, X_sample, y_train, y_test, y_sample = raw_dataset.get_split()
                    print(f"SAMPLE_SIZE {X_sample.shape}, TRAIN_SIZE: {X_train.shape}, TEST_SIZE: {X_test.shape}")
                    num_classes = len(np.unique(y_train))

                    encryptor = Encryptor(
                        output_shape=(1, X_train.shape[1]),
                        **self.config[consts.CONFIG_ENCRYPTOR_SECTION],
                    )

                    print("#### TRAINING CLOUD MODELS ####")
                    cloud_models: CloudModels = CLOUD_MODELS[self.config[consts.CONFIG_CLOUD_MODEL_SECTION]['name']](
                        **self.config[consts.CONFIG_CLOUD_MODEL_SECTION],
                        num_classes = num_classes
                    )
                    cloud_models.fit(X_train, y_train)

                    print("#### GETTING CLOUD MODELS PREDICTION ####")
                    cloud_acc, cloud_f1 = cloud_models.evaluate(X_test, y_test)


                    print("#### GETTING BASELINE PREDICTION ####")
                    baseline_acc, baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)


                    print(f"CREATING THE CLOUD-TRAINSET FROM {dataset_name},"
                          f" WITH {n_noise_samples} NOISE SAMPLES AND {n_pred_vectors} PREDICTION VECTORS")

                    dataset_creator = Dataset(
                        dataset_name=dataset_name,
                        cloud_models=cloud_models,
                        encryptor=encryptor,
                        n_pred_vectors=n_pred_vectors,
                        n_noise_samples=n_noise_samples,
                        use_embedding=True if "w_emb" in self.experiment_name else False,
                        use_noise_labels=True if "w_label" in self.experiment_name else False,
                        use_predictions=True if "w_pred" in self.experiment_name else False,
                        one_hot=self.config[consts.CONFIG_DATASET_SECTION]['one_hot'],
                        shuffle=self.config[consts.CONFIG_DATASET_SECTION]['shuffle'],
                        ratio=self.config[consts.CONFIG_DATASET_SECTION]['ratio'],
                        force=self.config[consts.CONFIG_DATASET_SECTION]['force'],
                        feature_reduction_config=self.config[consts.CONFIG_DATASET_SECTION]['feature_reduction'],
                    )
                    dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)

                    print("Finished Creating the dataset")
                    print(f"##### CLOUD DATASET SIZE - {dataset['train'][0].shape} ###########")

                    internal_model = InternalInferenceModelFactory().get_model(
                        **self.config[consts.CONFIG_INN_SECTION],
                        num_classes=num_classes,
                        input_shape=dataset['train'][0].shape[1],  # Only give the number of features
                    )

                    print(f"Training the IIM {internal_model.name} Model")
                    internal_model.fit(
                        *dataset['train']
                    )
                    train_acc, train_f1 = internal_model.evaluate(*dataset['train'])
                    test_acc, test_f1 = internal_model.evaluate(*dataset['test'])


                    print(f"""
                          Cloud: {cloud_acc}, {cloud_f1}\n
                          Baseline: {baseline_acc}, {baseline_f1}\n
                          IIM: {test_acc}, {test_f1}\n
                          """)

                    final_report = pd.concat(
                        [
                            final_report,
                            pd.DataFrame(
                                {
                                    "exp_name": [self.experiment_name],
                                    "dataset": [dataset_name],
                                    "train_size_ratio": [dataset_creator.split_ratio],
                                    "n_pred_vectors": [n_pred_vectors],
                                    "n_noise_sample": [n_noise_samples],
                                    "iim_model": [internal_model.name],
                                    "encryptor": [encryptor.generator_type],
                                    "cloud_model": [cloud_models.name],
                                    "iim_train_acc": [train_acc],
                                    "iim_train_f1": [train_f1],
                                    "iim_test_acc": [test_acc],
                                    "iim_test_f1": [test_f1],
                                    "baseline_acc": [baseline_acc],
                                    "baseline_f1": [baseline_f1],
                                    "cloud_acc": [cloud_acc],
                                    "cloud_f1": [cloud_f1],

                                }
                            )
                        ])


        return final_report


    def run_k_fold_experiment(self):

        reports = []

        for dataset_name in self.config[consts.CONFIG_DATASET_SECTION][consts.CONFIG_DATASET_NAME_TOKEN]:
            raw_dataset: RawDataset = DATASETS[dataset_name](**self.config[consts.CONFIG_DATASET_SECTION])

            # Initialize lists to store metrics across folds
            cloud_acc_scores = []
            cloud_f1_scores = []
            baseline_acc_scores = []
            baseline_f1_scores = []
            train_acc_scores = []
            train_f1_scores = []
            test_acc_scores = []
            test_f1_scores = []

            progress = tqdm(total=self.k_folds, desc="K-Folds", unit="Fold")

            for X_train, X_test, X_sample, y_sample, y_train, y_test in raw_dataset.k_fold_iterator(
                    n_splits=self.k_folds):
                encryptor = Encryptor(output_shape=(1,X_train.shape[1]))

                print("#### TRAINING CLOUD MODELS ####")
                cloud_models = CloudModels(self.config[consts.CONFIG_CLOUD_MODEL_SECTION])
                cloud_models.fit(X_train, y_train)

                print("#### GETTING CLOUD MODELS PREDICTION ####")
                cloud_acc, cloud_f1 = cloud_models.evaluate(X_test, y_test)
                cloud_acc_scores.append(cloud_acc)
                cloud_f1_scores.append(cloud_f1)

                print("#### GETTING BASELINE PREDICTION ####")
                baseline_acc, baseline_f1 = raw_dataset.get_baseline(X_sample, X_test, y_sample, y_test)
                baseline_acc_scores.append(baseline_acc)
                baseline_f1_scores.append(baseline_f1)

                print(f"CREATING THE CLOUD-TRAINSET FROM {dataset_name}")
                print(f"ORIGINAL SAMPLE SIZE {X_sample.shape}")
                dataset_creator = Dataset(
                    dataset_name=dataset_name,
                    cloud_models=cloud_models,
                    encryptor=encryptor,
                    n_pred_vectors=self.n_pred_vectors,
                    n_noise_samples=self.n_noise_samples,
                    use_embedding=True if "w_emb" in self.experiment_name else False,
                    use_noise_labels=True if "w_label" in self.experiment_name else False,
                    one_hot=self.config[consts.CONFIG_DATASET_SECTION]['one_hot'],
                    shuffle=self.config[consts.CONFIG_DATASET_SECTION]['shuffle'],
                    ratio=self.config[consts.CONFIG_DATASET_SECTION]['ratio'],
                    force=self.config[consts.CONFIG_DATASET_SECTION]['force'],
                    feature_reduction_config=self.config[consts.CONFIG_DATASET_SECTION]['feature_reduction'],
                )
                dataset = dataset_creator.create(X_sample, y_sample, X_test, y_test)

                print("Finished Creating the dataset")
                print(f"##### CLOUD DATASET SIZE - {dataset['train'][0].shape} ###########")

                internal_model = InternalInferenceModelFactory().get_model(
                    **self.config[consts.CONFIG_INN_SECTION],
                    num_classes=len(
                        np.unique(dataset['train'][1])
                    ),
                    input_shape=dataset['train'][0].shape[1], # Only give the number of features
                )

                print(f"Training the IIM {internal_model.name} Model")
                internal_model.fit(
                    *dataset['train']
                )
                train_acc, train_f1 = internal_model.evaluate(*dataset['train'])
                test_acc, test_f1 = internal_model.evaluate(*dataset['test'])

                # Append IIM scores for each fold
                train_acc_scores.append(train_acc)
                train_f1_scores.append(train_f1)
                test_acc_scores.append(test_acc)
                test_f1_scores.append(test_f1)

                print(f"""
                Cloud: {cloud_acc}, {cloud_f1}\n
                Baseline: {baseline_acc}, {baseline_f1}\n
                IIM: {test_acc}, {test_f1}\n
                """)

                progress.update(1)

            # Compute the average metrics across all folds
            average_cloud_acc = np.mean(cloud_acc_scores)
            average_cloud_f1 = np.mean(cloud_f1_scores)
            average_baseline_acc = np.mean(baseline_acc_scores)
            average_baseline_f1 = np.mean(baseline_f1_scores)
            average_train_acc = np.mean(train_acc_scores)
            average_train_f1 = np.mean(train_f1_scores)
            average_test_acc = np.mean(test_acc_scores)
            average_test_f1 = np.mean(test_f1_scores)

            # Create a final report with average metrics
            final_report = pd.DataFrame()

            final_report["dataset"] = [dataset_name]
            final_report["train_size_ratio"] = [dataset_creator.split_ratio]
            final_report["iim_model"] = [internal_model.name]
            final_report["cloud_models"] = [cloud_models.name]
            final_report["n_pred_vectors"] = [self.n_pred_vectors]
            final_report["n_noise_sample"] = [self.n_noise_samples]
            final_report["exp_name"] = [self.experiment_name]
            final_report["iim_train_accuracy"] = [average_train_acc]
            final_report["iim_train_f1"] = [average_train_f1]
            final_report["iim_test_accuracy"] = [average_test_acc]
            final_report["iim_test_f1"] = [average_test_f1]
            final_report["baseline_test_accuracy"] = [average_baseline_acc]
            final_report["baseline_test_f1"] = [average_baseline_f1]
            final_report["cloud_test_accuracy"] = [average_cloud_acc]
            final_report["cloud_test_f1"] = [average_cloud_f1]

            reports.append(final_report)

        return pd.concat(reports)


