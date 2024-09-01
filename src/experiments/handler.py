import numpy as np
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
                    cloud_models.fit(X_train, y_train, **raw_dataset.metadata)

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
                        ratio=self.config[consts.CONFIG_DATASET_SECTION]['ratio'],
                        force=self.config[consts.CONFIG_DATASET_SECTION]['force'],
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




